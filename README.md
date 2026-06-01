# AnnotEval — QA Annotation Quality & LLM Evaluation Pipeline

> A production-grade pipeline for measuring inter-annotator agreement, validating annotation schemas, and benchmarking LLM output quality on question-answering datasets using an LLM-as-judge evaluation framework.

---

## Overview

AnnotEval is an end-to-end data quality system built for teams working with human-annotated QA datasets and LLM-generated outputs. It automates the full evaluation lifecycle — from synthetic data generation and schema validation through inter-annotator agreement analysis and GPT-4o-mini-powered answer scoring — and surfaces results via a Streamlit monitoring dashboard backed by SQLite and AWS S3.

This project addresses a real and growing problem in ML pipelines: **how do you systematically measure and improve the quality of both human annotations and LLM-generated answers at scale?**

---

## Architecture

```
raw_qa.csv ──► annotation_pipeline ──► agreement_metrics ──► llm_evaluator
                        │                      │                     │
                        ▼                      ▼                     ▼
                 Schema Validation      Fleiss / Cohen's       LLM-as-Judge
                 Conflict Metrics         Kappa Scores        (faithfulness,
                 Annotator Stats       Entropy Analysis        relevance,
                                                             completeness)
                                                                     │
                                            ┌────────────────────────┘
                                            ▼
                                     SQLite Logger ◄──── AWS S3 Storage
                                            │
                                            ▼
                                   Streamlit Dashboard
```

---

## Key Features

| Feature | Details |
|---|---|
| **Synthetic Data Generation** | Generates 500 SQuAD-style QA records across 5 domains with realistic annotator disagreement profiles (60% consensus, 30% moderate conflict, 10% high conflict) |
| **Schema Validation** | Validates labels, confidence levels, score ranges (1–5), annotator IDs, and timestamps; exports all violations to CSV |
| **Conflict Metrics** | Per-question majority voting, Shannon entropy, and conflict scoring; flags questions where no label has a majority |
| **Inter-Annotator Agreement** | Fleiss Kappa (multi-annotator), pairwise Cohen's Kappa for all 10 annotator pairs, and percent-agreement matrix |
| **Annotator Outlier Detection** | Flags annotators whose error rate exceeds mean + 1.5σ of the group |
| **LLM-as-Judge Evaluation** | GPT-4o-mini generates answers then scores them on faithfulness, relevance, and completeness (1–5 scale) with structured JSON output |
| **AWS S3 Integration** | Uploads all pipeline artifacts to a configurable S3 bucket under `pipeline_outputs/` |
| **SQLite Run Logging** | Persists run-level statistics (kappa scores, conflict rates, avg LLM scores) and annotator stats across pipeline executions |
| **Streamlit Dashboard** | Interactive dashboard for inspecting agreement metrics, LLM judge scores, and dataset quality summaries |

---

## Tech Stack

- **Language:** Python 3.10+
- **Data:** Pandas, NumPy
- **ML / Stats:** Scikit-learn (`cohen_kappa_score`), custom Fleiss Kappa implementation
- **LLM:** OpenAI API (`gpt-4o-mini`) — generation + LLM-as-judge scoring
- **Cloud:** AWS S3 via `boto3`
- **Persistence:** SQLite via Python `sqlite3`
- **Dashboard:** Streamlit
- **Testing:** Pytest
- **Config:** `python-dotenv`

---

## Project Structure

```
annoteval-pipeline/
├── src/
│   ├── generate_data.py        # Synthetic QA + annotation dataset generation
│   ├── annotation_pipeline.py  # Schema validation, conflict metrics, annotator stats
│   ├── agreement_metrics.py    # Fleiss Kappa, Cohen's Kappa, percent agreement
│   ├── llm_evaluator.py        # GPT-4o-mini answer generation + LLM-as-judge scoring
│   ├── aws_storage.py          # S3 upload / download / listing
│   └── db.py                   # SQLite schema init, run logging, annotator stat logging
├── dashboard/                  # Streamlit app
├── tests/                      # Pytest test suite
├── docs/
│   └── runbook.md              # Operational runbook
├── data/                       # Generated artifacts (gitignored)
├── run_pipeline.py             # Single-command full pipeline runner
├── requirements.txt
└── .env                        # API keys and AWS credentials (not committed)
```

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/bish-ds/annoteval-pipeline.git
cd annoteval-pipeline
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
AWS_ACCESS_KEY_ID=your_aws_access_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
AWS_BUCKET_NAME=your_s3_bucket_name
AWS_REGION=us-east-1
```

> **Note:** The pipeline runs fully without AWS credentials. S3 upload steps are skipped gracefully if credentials are absent.

---

## Running the Pipeline

### Full pipeline (recommended)

```bash
python run_pipeline.py
```

This executes all 6 steps in order:

| Step | Module | Output |
|------|--------|--------|
| 1 | `generate_data` | `data/raw_qa.csv`, `data/annotations.csv`, `data/llm_responses.csv` |
| 2 | `annotation_pipeline` | `data/validated_annotations.csv`, `data/validation_errors.csv`, `data/annotator_stats.csv` |
| 3 | `agreement_metrics` | `data/cohen_kappa_scores.csv`, `data/percent_agreement.csv`, `data/agreement_summary.json`, `data/contested_questions.csv` |
| 4 | `llm_evaluator` | `data/llm_eval_scores.csv` |
| 5 | `aws_storage` | Uploads all artifacts to S3 |
| 6 | `db` | Logs run stats to `pipeline.db` (SQLite) |

### Individual modules

```bash
python -m src.generate_data
python -m src.annotation_pipeline
python -m src.agreement_metrics
python -m src.llm_evaluator
python -m src.aws_storage
```

### Dashboard

```bash
streamlit run dashboard/app.py
```

### Tests

```bash
pytest tests/
```

---

## Sample Output

**Agreement Metrics Report**
```
=== AGREEMENT METRICS REPORT ===
Fleiss Kappa: 0.47 (Moderate)

Pairwise Cohen's Kappa Summary:
Best pair:  annotator_1 vs annotator_4 = 0.61
Worst pair: annotator_2 vs annotator_5 = 0.31
Average kappa: 0.46

Annotator Error Rates:
annotator_1: 18.4%
annotator_2: 31.2%   ← flagged as outlier
annotator_3: 22.1%
annotator_4: 19.8%
annotator_5: 20.5%
================================
```

**LLM Evaluation Report**
```
=== LLM EVALUATION REPORT ===
Total evaluated: 100
Average Scores:
  Faithfulness:  4.21 / 5
  Relevance:     4.38 / 5
  Completeness:  3.94 / 5
  Overall Avg:   4.18 / 5

Score Distribution:
  4.0–5.0 (High quality):   73 questions
  3.0–3.9 (Medium quality): 19 questions
  Below 3.0 (Low quality):   8 questions
==============================
```

---

## Evaluation Methodology

The LLM-as-judge scoring approach follows the **G-Eval** paradigm:

1. **Generation phase** — GPT-4o-mini generates a concise 1–2 sentence answer given only the passage context.
2. **Evaluation phase** — A separate judge prompt asks GPT-4o-mini to compare the generated answer against the reference answer and return a structured JSON score on three axes:
   - **Faithfulness** — Does the answer match the reference facts?
   - **Relevance** — Does the answer directly address the question?
   - **Completeness** — Does the answer cover the key information?
3. Scores outside the 1–5 integer range are discarded; only `eval_status = success` rows are included in summary statistics.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
