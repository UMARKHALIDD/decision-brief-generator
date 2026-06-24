"""
Decision Brief Generator
========================
Turns raw CSV data into an executive-ready brief: headline, findings,
recommended actions, and risk flags.

Design principle: the LLM never sees raw rows. It only sees a deterministic
statistical *profile* computed in pandas — so numbers in the brief are
always correct, and the LLM is used only for what it's good at:
qualitative reasoning and recommendation framing.

Run locally:    streamlit run app.py
"""

import json
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Decision Brief Generator",
    page_icon="📊",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Data profiling — deterministic stats, computed in pandas
# ---------------------------------------------------------------------------
def profile_dataframe(df: pd.DataFrame, outcome_col: str) -> Dict[str, Any]:
    """Compute a structured statistical profile of the dataframe.

    The LLM will only see this profile, never raw rows. This guarantees
    that any number appearing in the final brief is one we actually
    computed — eliminating hallucinated metrics.
    """
    profile: Dict[str, Any] = {
        "shape": {"rows": int(len(df)), "columns": int(len(df.columns))},
        "outcome_column": outcome_col,
        "outcome_stats": {},
        "segment_analysis": {},
        "correlations": {},
        "data_quality": {},
    }

    # --- Outcome distribution ---
    outcome = pd.to_numeric(df[outcome_col], errors="coerce").dropna()
    profile["outcome_stats"] = {
        "total": round(float(outcome.sum()), 2),
        "mean": round(float(outcome.mean()), 2),
        "median": round(float(outcome.median()), 2),
        "std": round(float(outcome.std()), 2),
        "min": round(float(outcome.min()), 2),
        "max": round(float(outcome.max()), 2),
        "negative_count": int((outcome < 0).sum()),
        "negative_pct": round(float((outcome < 0).mean() * 100), 2),
    }

    # --- Segment analysis: outcome by each useful categorical column ---
    candidate_cols: List[str] = []
    for col in df.columns:
        if col == outcome_col:
            continue
        n_unique = df[col].nunique(dropna=True)
        if n_unique <= 1 or n_unique > 50:
            continue  # skip constant columns and high-cardinality ones
        if df[col].dtype == "object" or n_unique < 20:
            candidate_cols.append(col)

    for col in candidate_cols:
        agg = (
            df.groupby(col)
            .agg(total=(outcome_col, "sum"), mean=(outcome_col, "mean"), count=(outcome_col, "count"))
            .round(2)
            .sort_values("total", ascending=False)
        )
        n_groups = len(agg)
        top_n = min(3, n_groups)
        top = {str(k): v for k, v in agg.head(top_n).to_dict("index").items()}
        bottom = (
            {str(k): v for k, v in agg.tail(min(3, n_groups - top_n)).to_dict("index").items()}
            if n_groups > top_n
            else {}
        )
        profile["segment_analysis"][col] = {
            "unique_values": int(n_groups),
            "top_segments": top,
            "bottom_segments": bottom,
        }

    # --- Numeric correlations with the outcome ---
    num_cols = [
        c for c in df.select_dtypes(include=np.number).columns
        if c != outcome_col and df[c].nunique() > 2
    ]
    if num_cols:
        corrs = df[num_cols + [outcome_col]].corr()[outcome_col].drop(outcome_col)
        profile["correlations"] = {
            c: round(float(v), 3) for c, v in corrs.items() if not pd.isna(v)
        }

    # --- Data quality flags ---
    missing_by_col = df.isna().sum()
    profile["data_quality"] = {
        "missing_values_total": int(missing_by_col.sum()),
        "missing_by_column": {c: int(v) for c, v in missing_by_col.items() if v > 0},
        "duplicate_rows": int(df.duplicated().sum()),
        "constant_columns": [c for c in df.columns if df[c].nunique(dropna=True) <= 1],
    }

    return profile


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a senior business analyst writing decision briefs for executives. "
    "You reason carefully from the data provided and never invent numbers. "
    "You return only valid JSON matching the schema requested."
)


def build_user_prompt(profile: Dict[str, Any], business_context: str) -> str:
    return f"""Generate an executive decision brief from the statistical PROFILE below.

STRICT RULES:
1. Every number you cite must appear in the PROFILE. Do not invent figures.
2. Findings must reference specific named segments (e.g., a specific region, category, or sub-category), not generic statements.
3. Actions must be concrete: WHAT to do, WHERE/to which segment, and WHY it follows from a finding.
4. Risk flags must name what the data CANNOT tell you — confounders, missing context, small-sample groups, single-snapshot data, etc.
5. Return ONLY valid JSON matching the schema. No preamble, no markdown.

JSON SCHEMA:
{{
  "headline": "Single sentence (<=25 words) capturing the most important business insight",
  "key_findings": [
    {{
      "claim": "Specific finding referencing a named segment",
      "supporting_metric": "The exact figure(s) from the profile that support this claim"
    }}
    // exactly 3 findings
  ],
  "recommended_actions": [
    {{
      "action": "Concrete next step with a named target",
      "rationale": "Which finding this addresses and the expected effect",
      "priority": "high" | "medium" | "low"
    }}
    // 2 or 3 actions
  ],
  "risk_flags": [
    "Specific limitation, uncertainty, or thing the data cannot tell you"
    // 2 or 3 flags
  ]
}}

BUSINESS CONTEXT (optional):
{business_context.strip() if business_context.strip() else "(none provided — infer the business setting from the columns and outcome)"}

PROFILE:
{json.dumps(profile, indent=2)}

Return the JSON object now."""


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------
def generate_brief(
    profile: Dict[str, Any],
    business_context: str,
    api_key: str,
    model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """Call the LLM and parse the JSON response."""
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        temperature=0.3,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(profile, business_context)},
        ],
    )
    return json.loads(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# Brief → Markdown (for download)
# ---------------------------------------------------------------------------
def brief_to_markdown(brief: Dict[str, Any], outcome_col: str) -> str:
    findings_md = "\n".join(
        f"{i}. **{f['claim']}**  \n   *Supporting metric:* {f['supporting_metric']}"
        for i, f in enumerate(brief["key_findings"], 1)
    )
    actions_md = "\n".join(
        f"{i}. **[{a.get('priority', 'medium').upper()}]** {a['action']}  \n   *Rationale:* {a['rationale']}"
        for i, a in enumerate(brief["recommended_actions"], 1)
    )
    flags_md = "\n".join(f"- {flag}" for flag in brief["risk_flags"])
    return f"""# Decision Brief

**Outcome analysed:** `{outcome_col}`

## Headline
{brief['headline']}

## Key Findings
{findings_md}

## Recommended Actions
{actions_md}

## Risk Flags
{flags_md}

---
*Generated by Decision Brief Generator. Numbers computed deterministically in Python; narrative generated by LLM grounded in the computed profile.*
"""


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("📊 Decision Brief Generator")
st.caption(
    "Upload a CSV. Get an executive brief — headline, findings, actions, risk flags — "
    "with numbers computed deterministically and narrative generated by an LLM."
)

with st.sidebar:
    st.subheader("⚙️ Configuration")
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Not stored. Set OPENAI_API_KEY env var or Streamlit secret to skip this.",
    )
    model = st.selectbox(
        "Model",
        ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
        index=0,
        help="gpt-4o-mini is the cheap/fast default and works well for this task.",
    )
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown(
        "1. **Profile** the data in pandas (totals, segment breakdowns, correlations).  \n"
        "2. **Send only the profile** to the LLM — never raw rows.  \n"
        "3. LLM returns **structured JSON**: headline, findings, actions, risks.  \n"
        "4. Render as an executive brief.  \n\n"
        "This separation is what guarantees numbers are correct."
    )
    st.markdown("---")
    st.markdown(
        "**Sample dataset:** [Superstore Sales](https://github.com/Wunmi-O/Superstore/blob/master/SampleSuperstore.csv) "
        "(included in `sample_data/`)"
    )

uploaded = st.file_uploader("Upload a CSV", type=["csv"])

# Demo fallback: load bundled sample if it exists and user hasn't uploaded
if uploaded is None and os.path.exists("sample_data/superstore.csv"):
    if st.button("📦 Load sample Superstore data"):
        uploaded = "sample_data/superstore.csv"

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    st.success(f"Loaded **{len(df):,}** rows × **{len(df.columns)}** columns")

    with st.expander("📋 Preview data (first 20 rows)"):
        st.dataframe(df.head(20), use_container_width=True)

    # Pick outcome column
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns found. The outcome must be numeric.")
        st.stop()

    outcome_hints = ["profit", "revenue", "sales", "value", "amount", "margin", "conversion"]
    default_idx = 0
    for i, c in enumerate(numeric_cols):
        if any(h in c.lower() for h in outcome_hints):
            default_idx = i
            break

    col_a, col_b = st.columns([1, 2])
    with col_a:
        outcome_col = st.selectbox(
            "Outcome metric to analyse",
            numeric_cols,
            index=default_idx,
            help="The metric you want the brief to focus on — usually the one tied to a business goal.",
        )
    with col_b:
        business_context = st.text_area(
            "Business context (optional, ~1–2 sentences)",
            placeholder="e.g., US office-supplies retailer. Leadership wants to improve margins ahead of Q4 planning.",
            height=80,
        )

    if st.button("🚀 Generate Brief", type="primary", disabled=not api_key):
        with st.spinner("Profiling data…"):
            profile = profile_dataframe(df, outcome_col)

        with st.spinner("Generating brief with LLM…"):
            try:
                brief = generate_brief(profile, business_context, api_key, model)
            except Exception as e:
                st.error(f"LLM call failed: {e}")
                st.stop()

        # ---- Render brief ----
        st.markdown("---")
        st.markdown(f"## 🎯 {brief['headline']}")

        col_findings, col_actions = st.columns(2)

        with col_findings:
            st.markdown("### 🔍 Key Findings")
            for i, f in enumerate(brief["key_findings"], 1):
                st.markdown(f"**{i}.** {f['claim']}")
                st.caption(f"📊 {f['supporting_metric']}")
                st.markdown("")

        with col_actions:
            st.markdown("### ✅ Recommended Actions")
            for i, a in enumerate(brief["recommended_actions"], 1):
                priority = a.get("priority", "medium").lower()
                priority_dot = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(priority, "🟡")
                st.markdown(f"**{i}.** {priority_dot} {a['action']}")
                st.caption(f"💡 *{a['rationale']}*")
                st.markdown("")

        st.markdown("### ⚠️ Risk Flags")
        for flag in brief["risk_flags"]:
            st.warning(flag)

        st.markdown("---")
        col_dl, col_view = st.columns([1, 1])
        with col_dl:
            md = brief_to_markdown(brief, outcome_col)
            st.download_button(
                "📥 Download brief (Markdown)",
                md,
                file_name="decision_brief.md",
                mime="text/markdown",
            )
        with col_view:
            with st.popover("View underlying profile (what the LLM saw)"):
                st.json(profile)

        with st.expander("Raw LLM JSON output"):
            st.json(brief)

elif not api_key:
    st.info("👈 Add your OpenAI API key in the sidebar, then upload a CSV.")
else:
    st.info("👆 Upload a CSV to get started.")
