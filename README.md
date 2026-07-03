# Simple Baseline — BIRD-Interact Interpretation Sampling Agent

A Text-to-SQL baseline agent for the [BIRD-Interact](https://bird-interact.github.io/) benchmark. The pipeline explicitly enumerates semantic interpretations of ambiguous user queries, generates one SQL candidate per interpretation at temperature 0, and records per-token log-probabilities on every candidate to study the relationship between model confidence and SQL correctness.

---

## Overview

BIRD-Interact queries are intentionally ambiguous — terms like "high standards," "usable signals," or "recent activity" can resolve to different SQL conditions depending on context. Standard approaches either guess a single interpretation or sample multiple outputs at high temperature, neither of which systematically captures the space of plausible meanings.

This baseline introduces **INTERPRET_SAMPLE**: a three-call pipeline that identifies ambiguous terms, enumerates concrete resolutions with an explicit coverage rule, and generates a structurally diverse set of SQL candidates — each with full logprob data attached.

---

## Pipeline

```
User query + Schema + KB
        │
        ▼
 ┌─────────────┐
 │  1. Explore │  Identify relevant tables/columns, flag every ambiguous term
 └──────┬──────┘
        │
        ▼
 ┌──────────────────┐
 │  2. Interpret    │  List 2 options per ambiguity, build assignment table
 │  (coverage rule) │  with mandatory coverage: every option ≥ 1 appearance
 └──────┬───────────┘
        │
        ├─ interpretation 1 ──▶  3. Generate SQL  (temp=0, logprobs recorded)
        ├─ interpretation 2 ──▶  3. Generate SQL  (temp=0, logprobs recorded)
        ├─ interpretation 3 ──▶  3. Generate SQL  (temp=0, logprobs recorded)
        └─ ...up to 6          ──▶  3. Generate SQL  (temp=0, logprobs recorded)
```

### Call 1 — Explore
Reads the full schema, column descriptions, visible KB entries, and user query. Returns a free-text analysis identifying relevant tables/columns and flagging every vague term with an explanation of why it matters for SQL.

### Call 2 — Interpret
Takes the exploration output. For each ambiguous term, lists 2 concrete options grounded in the schema. Builds an **assignment table** where every listed option must appear in at least one interpretation's assignments (**coverage rule** — prevents collapse to identical candidates). Returns structured JSON:

```json
{
  "ambiguities": {
    "high standards": {
      "option_a": "IRS > 8.0",
      "option_b": "IRS > 9.0"
    }
  },
  "interpretations": [
    {
      "id": 1,
      "assignments": {"high standards": "option_a"},
      "interpretation_text": "Retrieve controllers where IRS > 8.0 ..."
    }
  ]
}
```

### Call 3 — Generate (×N)
One call per interpretation, always at temperature 0. The `interpretation_text` is passed as authoritative context — the model implements it exactly. Per-token logprobs are captured via `logprobs=True` and three statistics are computed per SQL span: `avg_token_logprob`, `min_token_logprob`, and `std_token_logprob`.

---

## Key Research Findings (25-task run)

| Metric | Value |
|--------|-------|
| Tasks evaluated | 15 unique across 15 databases |
| Total SQL candidates | 92 |
| Tasks with ≥1 correct interpretation | 3/15 (20%) |
| Exec error rate | 30/92 (32.6%) |

**Logprob signal is real.** Exec-errored SQL had avg logprob −0.042 vs −0.027 for clean queries, and higher std (0.135 vs 0.101). Both signals move in the right direction at temperature 0.

**Std is more informative than avg.** The proportional gap between errored and clean is larger for standard deviation, which captures local token-level uncertainty even when global confidence is high.

**Confident-but-wrong is the key failure mode.** Some high-confidence candidates (avg_logprob > −0.03) still exec-errored with wrong column references inherited from the explore step. Logprob measures self-consistency, not factual correctness — this is an important calibration limitation.

**Structural diversity is working.** Mean logprob spread within a task of 0.020 confirms interpretations are structurally different (different GROUP BY, metrics, filters), not just lexical paraphrases.

---

## Setup

### Prerequisites

- Python 3.10+
- PostgreSQL running locally with BIRD-Interact databases loaded (see [env/](../env/))
- OpenAI API key

### Install dependencies

```bash
pip install openai psycopg2-binary python-dotenv sqlparse
```

### Configure environment

Create a `.env` file at the repo root:

```
OPENAI_API_KEY=sk-...
```

### Database connection

Edit `db.py` to match your local PostgreSQL config:

```python
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "user": "your_user",
    "password": "your_password",
}
```

---

## Usage

### Run the pipeline

```bash
cd simple_baseline

# 10 tasks sampled round-robin across databases
python agent.py --sample 10 --output runs/results.jsonl

# Specific database only
python agent.py --database gaming --limit 5 --output runs/results_gaming.jsonl

# Append to an existing run
python agent.py --sample 15 --output runs/results.jsonl --append

# Filter to SELECT-only tasks
python agent.py --sample 10 --task-type select --output runs/results.jsonl
```

### CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--sample N` | — | Pick N tasks round-robin across databases |
| `--limit N` | — | Take first N tasks from the task list |
| `--database DB` | all | Only run tasks from this database |
| `--task-type` | all | `select`, `management`, or `all` |
| `--output FILE` | `runs/results.jsonl` | Base path for all output files |
| `--append` | off | Append to existing output files instead of overwriting |
| `--pause` | off | Pause after each management task for DB inspection |

### Output files

Every run produces four files derived from the `--output` base path:

| File | Contents |
|------|----------|
| `results.jsonl` | Full results per task — all pipeline outputs, candidate matches, ground truth |
| `results_sql_outputs.jsonl` | Compact per-task SQL candidates and LLM call traces |
| `results.csv` | One row per interpretation — all logprob and interpretation columns |
| `results_token_logprobs.jsonl` | Per-token logprob data for every SQL span |

### CSV columns

```
# Task identification
instance_id, database, task_type, user_query

# SQL comparison
pipeline, rank, sql, ground_truth_sql, exact_match, sql_exec_error

# Logprob data
avg_token_logprob, min_token_logprob, std_token_logprob, sql_token_count

# Interpretation metadata
interpretation_id, interpretation_text, assignments, ambiguities_found
```

---

## Analysis

```bash
# Summary: logprob signal, diversity, calibration
python analyze.py runs/results.csv

# Per-interpretation detail
python analyze.py runs/results.csv --verbose
```

The analysis script prints three sections:

- **Logprob signal** — avg and std for exec-errored vs clean SQL
- **Diversity** — logprob spread within each task
- **Calibration** — high-confidence queries that still exec-errored

---

## Module Reference

| File | Responsibility |
|------|---------------|
| `agent.py` | Entry point — CLI, task loading, DB connections, output writing |
| `pipeline.py` | Pipeline orchestration — `run_pipeline_interpret()`, `summarize_results()` |
| `llm.py` | LLM client — all step functions, logprob tracking, JSON parsing |
| `data.py` | Data loading — tasks, ground truth, schema, KB entries |
| `db.py` | Database utilities — SQL execution, result comparison, DB reset |
| `analyze.py` | Post-run analysis script for logprob and diversity metrics |

### Key constants (pipeline.py)

```python
INTERPRET_MAX = 6      # max interpretations per task
```

### Key constants (llm.py)

```python
SQL_CANDIDATE_COUNT = 3   # used only by legacy top-3 pipeline (inactive)
MAX_REPAIR_ATTEMPTS = 2   # used only by legacy repair pipeline (inactive)
```

---

## Limitations

**Shared exploration ceiling.** All interpretations share one explore call. If the explore step gets a table or column wrong, every interpretation inherits the error. Tasks with 100% exec error rates (e.g. `mental_1`, `gaming_1`) typically exhibit this pattern.

**Hidden KB entries.** The benchmark masks specific knowledge base entries per task. These often contain the exact formulas and thresholds used in ground truth SQL. Without them, the agent must guess — lowering accuracy is by benchmark design, not a pipeline flaw.

**Low-ambiguity collapse.** Queries with few genuine ambiguities (e.g. `robot_1`, `credit_1`) produce near-identical interpretations with logprob spread near 0.003. The coverage rule helps but can't force diversity when the model identifies only one meaningful ambiguity.

---

## Results Directory

Pre-run results are stored in `runs/`:

| File | Description |
|------|-------------|
| `runs/results_sample10.*` | 25-task run across 15 databases (10 + 15 appended) |

---

## Citation

If you use BIRD-Interact data or evaluation infrastructure, please cite:

```bibtex
@inproceedings{
huo2026birdinteract,
title={{BIRD}-{INTERACT}: Re-imagining Text-to-{SQL} Evaluation via Lens of Dynamic Interactions},
author={Nan Huo and Xiaohan Xu and Jinyang Li and Per Jacobsson and Shipei Lin and Bowen Qin and Binyuan Hui and Xiaolong Li and Ge Qu and Shuzheng Si and Linheng Han and Edward Alexander and Xintong Zhu and Rui Qin and Ruihan Yu and Yiyao Jin and Feige Zhou and Weihao Zhong and Yun Chen and Hongyu Liu and Chenhao Ma and Fatma Ozcan and Yannis Papakonstantinou and Reynold Cheng},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=nHrYBGujps}
}
```
