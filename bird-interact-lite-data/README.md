---
license: cc-by-sa-4.0
configs:
- config_name: bird-interact-base=lite
  data_files:
  - path: bird_interact_data.jsonl
    split: dev
viewer: true
tags:
- text-to-sql
- database
task_categories:
- TABLE_QUESTION_ANSWERING
---

[🌐 Website](https://bird-interact.github.io) • [📄 Paper (ICLR 2026 **Oral**)](https://arxiv.org/abs/2510.05318) • [💻 GitHub](https://github.com/bird-bench/BIRD-Interact) • [🗄️ bird-interact-lite](https://huggingface.co/datasets/birdsql/bird-interact-lite) • [🗄️ bird-interact-full](https://huggingface.co/datasets/birdsql/bird-interact-full) • [🗄️ LiveSQLBench](https://livesqlbench.ai)


## 🧸 Overview

BIRD-INTERACT, an interactive text-to-SQL benchmark, **re-imagines Text-to-SQL evaluation via lens of dynamic interactions**, which is built on top of single-turn unambiguous T2S tasks from [LiveSQLBench](https://livesqlbench.ai).
The environment blends a hierarchical knowledge base, database documentation and a function-driven user simulator to recreate authentic enterprise environments across full **CRUD** operations.
It offers two rigorous test modes: (1) passive **Conversational Interaction** and (2) active **Agentic Interaction**, spanning 600 annotated tasks including Business Intelligence (BI), CRUD operations and etc., each guarded by executable test cases.
Typical evaluations trigger 1,968-5,496 interaction turns between model and user simulator, while state-of-the-art reasoning models currently solve only **≈24%** and **≈18%** of tasks, underscoring the benchmark's challenge.

### ✅ Two Evaluation Modes

BIRD-INTERACT supports two evaluation modes as mentioned above:

   - **c-Interact**: Conversational Interaction which is a passive mode and the workflow is fixed. The code and detailed information can be found in `bird_interact_conv`.
   - **a-Interact**: Agentic Interaction which is an embodied active mode where the workflow is dynamic and led by models. The code and detailed information can be found in `bird_interact_agent`.


### 🐣 Lite Version

This repo is a lite version of BIRD-INTERACT, `bird-interact-lite`, which includes 300 high-quality real-world tasks specifically for PostgreSQL. This is a good starting point for quick experimentation. This data is built on top of single-turn unambiguous T2S tasks from [LiveSQLBench-Base-Lite](https://huggingface.co/datasets/birdsql/livesqlbench-base-lite).

### 🦜 Full Version

The full version of BIRD-INTERACT, `bird-interact-full`, is a comprehensive benchmark that includes 600 tasks for PostgreSQL. It covers a wide range of SQL operations and user queries. The full version is also released. This data is built on top of single-turn unambiguous T2S tasks from [LiveSQLBench-Base-Full-v1](https://huggingface.co/datasets/birdsql/livesqlbench-base-full-v1).


### Dataset Uses

1. Download the task file and DB metafiles (including schema, HKB, column meaning files) by cloning this entire repo:
```bash
git clone https://huggingface.co/datasets/birdsql/bird-interact-lite
```

2. To avoid data leakage by auto-crawling, we do not include GT solution sqls and test cases along with data in `bird_interact_data.jsonl`.
please email [bird.bench25@gmail.com](mailto:bird.bench25@gmail.com) with the tag `[bird-interact-lite GT&Test Cases]` in title for full set, which will be sent automatically.

    Then refer to [Combine the Public Data with the Ground Truth and Test Cases](https://github.com/bird-bench/BIRD-Interact?tab=readme-ov-file#combine-the-public-data-with-the-ground-truth-and-test-cases) Section in our Github Repo to integrate the ground truth fields into the public data.

3. Refer to [bird-interact repo](https://github.com/bird-bench/BIRD-Interact) for details of DB building, usage and evaluation.

## Sample Usage

You can download the dataset from Hugging Face using the following command:

```python
from datasets import load_dataset

# Load the lite version of the dataset
dataset = load_dataset("birdsql/bird-interact-lite")
print(dataset["dev"][0])

# Load the full version of the dataset (coming soon)
# dataset = load_dataset("birdsql/bird-interact-full")
# print(dataset["full"][0])
```

## 📦 Dataset Details

- Database: The complete PostgreSQL database can be downloaded from [the Google Drive](https://drive.google.com/file/d/1QIGQlRKbkqApAOrQXPqFJgUg8rQ7HRRZ/view?usp=sharing). Check the [Github repo](https://github.com/bird-bench/BIRD-Interact) to build the database.
- data: Each data instance contain the following main parts:
   - `selected_database`: The name of the database.  
   - `query`: The unambiguous user query (comes from `query` field in [LiveSQLBench-Base-Lite](https://huggingface.co/datasets/birdsql/livesqlbench-base-lite)).
   - `amb_user_query`: The user query with injected ambiguities.
   - `user_query_ambiguity`: The ambiguities injected into the user query.
   - `non_critical_ambiguity`: The non-critical ambiguities like order, limit, etc.
   - `knowledge_ambiguity`: The ambiguities created by masked external knowledges. 
   - `sol_sql`: The ground truth SQL solution.  
   - `preprocess_sql`: SQL queries to run before executing the solution or prediction.  
   - `clean_up_sql`: SQL queries to run after the test cases to revert any changes made to the database.  
   - `test_cases`: A set of test cases to validate the predicted corrected SQL.
   - `follow_up`: The labeled follow up questions.
   - `external_knowledge`: The external knowledge related to the specific task.

- Evaluation: The evaluation code is available in the [github](https://github.com/bird-bench/BIRD-Interact).
- **Curated by:** BIRD Team & Google Cloud
- **License:** [cc-by-sa-4.0](https://creativecommons.org/licenses/by-sa/4.0/)

## 📋 Todo Lists

- [x] Release lite version, bird-interact-lite (300).
- [x] Release conversational version, bird-interact-conv.
- [x] Release agent version, bird-interact-agent.
- [x] Release Full bird-interact-full (600).
- [ ] SFT / RL an User Simulator

## Created By:
BIRD Team & Google Cloud