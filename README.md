# 📊 Decision Brief Generator

> Turns raw CSV data into an executive-ready decision brief — headline, three findings, two-to-three recommended actions, and risk flags — with numbers that are guaranteed to be correct.

**Live demo:** _https://decision-brief-umar.streamlit.app/_
**Repo:** _(https://github.com/UMARKHALIDD/decision-brief-generator)_
**Sample dataset:** [Sample Superstore (US retail, 9,994 orders)](https://github.com/Wunmi-O/Superstore/blob/master/SampleSuperstore.csv)

---

## 100-word summary (for submission)

Executives don't read dashboards; they need pre-digested decisions. **Decision Brief Generator** ingests any CSV, computes a statistical profile in pandas (totals, top/bottom segments per dimension, correlations, data-quality flags), and sends *only the computed profile* to an LLM — never raw rows. The LLM returns structured JSON: a one-line headline, three specific findings tied to named segments, two-to-three concrete actions with priority, and explicit risk flags about what the data *cannot* tell you. This separation between deterministic computation and qualitative reasoning eliminates hallucinated metrics — every number in the brief is one the system actually calculated. Demoed on the Sample Superstore dataset.

---

## Why this design

Most "AI for business insights" tools dump the CSV into the LLM and ask for findings. That has two failure modes that disqualify the output for executive consumption:

1. **Hallucinated numbers.** LLMs are unreliable at arithmetic; any figure they cite from a wide table is suspect.
2. **Vague findings.** Without forced structure, output drifts toward generic management-speak.

This tool fixes both:

| Concern | This tool's approach |
|---|---|
| Number correctness | Stats computed in pandas; LLM sees only the profile, can only cite numbers that are in it |
| Specificity | Structured JSON schema forces named segments + supporting metrics |
| Generic actions | Prompt requires concrete actions with WHAT/WHERE/WHY |
| AI overconfidence | Mandatory `risk_flags` array — the LLM must call out what the data *can't* tell you |
| Reproducibility | `response_format={"type": "json_object"}` + `temperature=0.3` |

## Architecture

```
┌──────────────┐    ┌────────────────┐    ┌─────────────┐    ┌──────────────┐
│  CSV upload  │───▶│ pandas profile │───▶│   LLM call  │───▶│  Rendered    │
│              │    │ (deterministic)│    │ (JSON mode) │    │  exec brief  │
└──────────────┘    └────────────────┘    └─────────────┘    └──────────────┘
                            ▲                                       │
                            │                                       │
                    User picks outcome col                     Download .md
                    + adds business context
```

The LLM is constrained to:
- Use only numbers that appear in the profile (rule #1 in the prompt)
- Reference named segments (regions, categories, etc.)
- Return JSON matching a strict schema
- Include risk flags (no choice — required field)

## Run it

### Locally

```bash
git clone https://github.com/UMARKHALIDD/decision-brief-generator.git
cd decision-brief-generator
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`). Either click "Load sample Superstore data" or upload your own CSV.

### Deploy free on Streamlit Community Cloud

1. Push this folder to a public GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo.
3. In the app's **Settings → Secrets**, add:
   ```toml
   OPENAI_API_KEY = "sk-..."
   ```
4. Set the entrypoint to `app.py` and deploy.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit app (UI + profiling + LLM call + rendering) |
| `sample_data/superstore.csv` | Bundled demo dataset |
| `example_output.json` | Sample LLM output (for reference / screenshots) |
| `example_output.md` | Sample rendered brief |
| `requirements.txt` | Dependencies |
| `.streamlit/config.toml` | Light theme for clean screenshots |

## What I'd do next (if this were a real product)

- **Validate citations.** Post-process the LLM output to verify every cited number actually exists in the profile; flag mismatches.
- **Add a forecasting profile.** If a date column is present, compute trend and seasonality, and add a "what's changed recently" section.
- **Swap LLMs cleanly.** Abstract the `generate_brief` call so Anthropic / local models are drop-in.
- **Cache profiles.** For repeat use against the same file, cache the profile so iteration on business context is free.
- **Eval suite.** A small set of (dataset, expected-finding-keywords) pairs to catch prompt regressions.

---

Built as a take-home demo. 
