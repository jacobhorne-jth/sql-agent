"""
Multi-step BIRD-Interact baseline agent.

Loads the real bird-interact-lite dataset and runs a 3-step agent per task:
  Step 1 (Explore)  - reason about the schema and identify ambiguous terms
  Step 2 (Clarify)  - agent asks one clarifying question; user simulator answers
                      using the actual ambiguity data from the dataset
  Step 3 (Generate) - write the final SQL using all accumulated context

Usage:
    python agent.py [--limit N] [--output results.jsonl]

OPENAI_API_KEY is loaded from the .env file at the repo root.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from dotenv import load_dotenv
import psycopg2

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DATA_DIR = Path(__file__).parent.parent / "bird-interact-lite-data"
GT_FILE = Path(__file__).parent.parent / "GT-stuff" / "bird_interact_gt_kg_testcases_1008.jsonl"
TASKS_FILE = DATA_DIR / "bird_interact_data.jsonl"

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "user": "root",
    "password": "123123",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_tasks() -> list[dict]:
    """
    Reads the main dataset file (bird_interact_data.jsonl) line by line.
    Each line is one task — a JSON object with fields like:
      - instance_id: unique task identifier (e.g. 'alien_1')
      - selected_database: which database this task uses (e.g. 'alien')
      - amb_user_query: the ambiguous natural language question the agent must answer
      - user_query_ambiguity: which terms in the query are vague and what they mean
      - knowledge_ambiguity: which KB entries are intentionally hidden from the agent
    Returns a list of all task dicts.
    """
    tasks = []
    with open(TASKS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def load_gt() -> dict[str, dict]:
    """
    Reads the ground truth file. Returns a dict mapping:
      instance_id -> {"sol_sql": str, "test_cases": list[str]}
    test_cases is a list of Python function strings for management tasks,
    empty list for pure query tasks.
    """
    gt = {}
    with open(GT_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                sql_list = r.get("sol_sql", [])
                gt[r["instance_id"]] = {
                    "sol_sql": sql_list[0] if sql_list else "",
                    "test_cases": r.get("test_cases", []),
                }
    return gt


def load_db_context(db_name: str) -> tuple[str, str, list[dict]]:
    """
    Loads all three context files for a given database from bird-interact-lite-data/:
      - {db}_schema.txt       : the CREATE TABLE statements (DDL) for every table
      - {db}_column_meaning_base.json : plain-English explanation for every column
      - {db}_kb.jsonl         : the external knowledge base — domain-specific formulas
                                and definitions the agent may need (e.g. SNQI formula)
    Returns a tuple of (schema_text, col_meanings_text, kb_as_list_of_dicts).
    The column meanings are formatted into a single readable string.
    """
    db_dir = DATA_DIR / db_name

    schema = (db_dir / f"{db_name}_schema.txt").read_text()

    col_meanings: dict = json.loads((db_dir / f"{db_name}_column_meaning_base.json").read_text())
    col_text = "\n".join(f"  {k}: {v}" for k, v in col_meanings.items())

    kb: list[dict] = []
    with open(db_dir / f"{db_name}_kb.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                kb.append(json.loads(line))

    return schema, col_text, kb


def build_kb_text(kb: list[dict], exclude_ids: list[int]) -> str:
    """
    Formats the knowledge base into a readable string for the LLM prompt,
    but intentionally skips any entries whose IDs are in exclude_ids.

    This mirrors how the real benchmark works: each task specifies certain
    KB entries as 'deleted_knowledge' — they are hidden from the agent to
    simulate the agent not having access to key domain definitions upfront.
    The agent can only learn about them by asking the user for clarification.

    Returns a newline-separated string of [id] name: definition entries.
    """
    lines = []
    for entry in kb:
        if entry["id"] not in exclude_ids:
            lines.append(f"[{entry['id']}] {entry['knowledge']}: {entry['definition']}")
    return "\n".join(lines) if lines else "(no external knowledge available)"


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_llm(messages: list[dict]) -> str:
    """
    Makes a single call to GPT-4o-mini with the given message list and returns
    the model's response as a plain string.
    Temperature is set to 0 for deterministic, consistent outputs.
    """
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Agent steps
# ---------------------------------------------------------------------------

def step1_explore(query: str, schema: str, col_text: str, kb_text: str) -> str:
    """
    STEP 1 — EXPLORE

    The agent's first look at the problem. It receives:
      - the database schema (table/column structure)
      - plain-English column descriptions
      - the visible subset of the knowledge base
      - the user's ambiguous query

    The agent is asked to reason about which tables and columns are relevant
    to the query, and to flag any terms that are vague or undefined.

    This output is NOT SQL — it's the agent thinking out loud before asking
    for clarification. It feeds into both Step 2 and Step 3.
    """
    return call_llm([
        {
            "role": "system",
            "content": (
                "You are a SQL analyst. Given a database schema and a user query, "
                "identify which tables and columns are relevant, and flag any terms "
                "in the query that are vague or need clarification. Be concise."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Schema:\n{schema}\n\n"
                f"Column descriptions:\n{col_text}\n\n"
                f"Knowledge definitions:\n{kb_text}\n\n"
                f"User query:\n{query}\n\n"
                "Which tables/columns are relevant, and what terms need clarification?"
            ),
        },
    ])


def step2_clarify(
    query: str,
    schema: str,
    exploration: str,
    ambiguity_data: dict,
) -> tuple[str, str]:
    """
    STEP 2 — CLARIFY (two LLM calls)

    This step simulates the back-and-forth between the agent and the user
    that happens in the real BIRD-Interact benchmark.

    Call 2a — Agent asks a question:
      The agent uses its Step 1 exploration to identify the single most
      important ambiguous term and asks the user one focused question about it.

    Call 2b — User simulator answers:
      Instead of a real human, we simulate the user with another LLM call.
      Crucially, the user simulator is given the real ambiguity resolution data
      from the dataset (user_query_ambiguity and knowledge_ambiguity fields),
      so its answer is grounded in what the benchmark says the terms actually mean.
      This is what makes this a faithful simulation rather than a made-up answer.

    Returns (question_string, answer_string).
    """
    # --- Call 2a: agent generates a clarifying question ---
    question = call_llm([
        {
            "role": "system",
            "content": (
                "You are a SQL analyst. Ask ONE short clarifying question about "
                "the single most important ambiguous term in the user's query."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Schema:\n{schema}\n\n"
                f"User query:\n{query}\n\n"
                f"Your analysis:\n{exploration}\n\n"
                "Ask one clarifying question:"
            ),
        },
    ])

    # --- Call 2b: user simulator answers using real ambiguity data ---
    answer = call_llm([
        {
            "role": "system",
            "content": (
                "You are the user who submitted a database query. "
                "Answer the analyst's clarifying question based on what you actually meant. "
                "Use the ambiguity resolution details provided — they describe exactly what "
                "the ambiguous terms mean in this context."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Your original query: {query}\n\n"
                f"What the ambiguous terms actually mean:\n"
                f"{json.dumps(ambiguity_data, indent=2)}\n\n"
                f"The analyst asks: {question}\n\n"
                "Answer their question clearly and specifically:"
            ),
        },
    ])

    return question, answer


def step3_generate(
    query: str,
    schema: str,
    col_text: str,
    kb_text: str,
    exploration: str,
    question: str,
    answer: str,
) -> str:
    """
    STEP 3 — GENERATE

    The final LLM call. The agent now has everything it needs:
      - the full schema and column descriptions
      - the visible knowledge base
      - its own Step 1 reasoning about which tables/columns matter
      - the clarification exchange from Step 2

    It uses all of this to write a single PostgreSQL SQL query.
    The model is instructed to return only SQL — no explanation, no markdown.
    This output is what gets compared against the ground truth.
    """
    return call_llm([
        {
            "role": "system",
            "content": (
                "You are an expert PostgreSQL analyst. "
                "Write a single SQL query that answers the user's question. "
                "Return ONLY the SQL with no explanation or markdown fences."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Schema:\n{schema}\n\n"
                f"Column descriptions:\n{col_text}\n\n"
                f"Knowledge definitions:\n{kb_text}\n\n"
                f"User query:\n{query}\n\n"
                f"Your earlier analysis:\n{exploration}\n\n"
                f"Clarification:\n"
                f"  Q: {question}\n"
                f"  A: {answer}\n\n"
                "Write the SQL query:"
            ),
        },
    ])


# ---------------------------------------------------------------------------
# Comparison via database execution
# ---------------------------------------------------------------------------

def preprocess_results(results: list) -> list[tuple]:
    """
    Mirrors bird_interact_agent test_utils.preprocess_results:
    normalizes dates to YYYY-MM-DD strings and converts rows to tuples.
    """
    processed = []
    for row in results:
        new_row = []
        for item in row:
            if isinstance(item, (date, datetime)):
                new_row.append(item.strftime("%Y-%m-%d"))
            else:
                new_row.append(item)
        processed.append(tuple(new_row))
    return processed


def execute_sql(sql: str, conn) -> list | None:
    """Runs a single SQL query on conn and returns rows, or None on error."""
    try:
        cursor = conn.cursor()
        cursor.execute("SET statement_timeout = '60s';")
        cursor.execute(sql)
        conn.commit()
        lower = sql.strip().lower()
        if lower.startswith("select") or lower.startswith("with"):
            rows = cursor.fetchmany(10001)
            return rows[:10000] if len(rows) > 10000 else rows
        return cursor.fetchall()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()


def execute_queries(sql: str, db_name: str, conn):
    """Wrapper used by exec'd test case functions. Returns (rows, error, timeout)."""
    try:
        return execute_sql(sql, conn), False, False
    except Exception as e:
        return None, str(e), False


def reset_db(db_name: str):
    """Drops db_name and recreates it from {db_name}_template, restoring original state."""
    env = os.environ.copy()
    env["PGPASSWORD"] = DB_CONFIG["password"]
    h, p, u = DB_CONFIG["host"], str(DB_CONFIG["port"]), DB_CONFIG["user"]
    template = f"{db_name}_template"

    subprocess.run(
        ["psql", "-h", h, "-p", p, "-U", u, "-d", "postgres", "-c",
         f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '{db_name}' AND pid <> pg_backend_pid();"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["dropdb", "--if-exists", "-h", h, "-p", p, "-U", u, db_name],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["createdb", "-h", h, "-p", p, "-U", u, db_name, "--template", template],
        check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def compare(pred: str, gold: str, test_cases: list, db_name: str, conn) -> bool:
    """
    For query tasks (no test_cases): execute both SQLs and compare result sets.
    For management tasks (test_cases present): execute predicted SQL then run
    each test case Python function to verify the DB state, mirroring the real eval.
    """
    if not test_cases:
        try:
            pred_rows = execute_sql(pred, conn)
            gold_rows = execute_sql(gold, conn)
        except Exception:
            return False
        if pred_rows is None or gold_rows is None:
            return False
        pred_rows = preprocess_results(pred_rows)
        gold_rows = preprocess_results(gold_rows)
        return set(pred_rows) == set(gold_rows)
    else:
        try:
            execute_sql(pred, conn)
        except Exception:
            return False
        for tc_str in test_cases:
            try:
                namespace = {"execute_queries": execute_queries}
                exec(tc_str, namespace)
                namespace["test_case"]([pred], [gold], db_name, conn)
            except Exception:
                return False
        return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only run first N tasks")
    parser.add_argument("--output", default="results.jsonl")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Error: OPENAI_API_KEY not found. Add it to the .env file at the repo root.")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  BIRD-Interact Baseline Agent")
    print("  3-step: Explore → Clarify → Generate")
    print("=" * 60)

    print("\n[SETUP] Loading tasks from bird_interact_data.jsonl ...")
    tasks = load_tasks()
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"  → {len(tasks)} tasks loaded")

    print("\n[SETUP] Loading ground truth SQL answers ...")
    gt = load_gt()
    print(f"  → {len(gt)} ground truth entries loaded")

    db_cache: dict[str, tuple] = {}
    conn_cache: dict[str, object] = {}
    results = []
    exact_matches = 0

    # ------------------------------------------------------------------
    # Task loop
    # ------------------------------------------------------------------
    for i, task in enumerate(tasks):
        instance_id = task["instance_id"]
        db = task["selected_database"]
        query = task["amb_user_query"]
        gt_entry = gt.get(instance_id, {"sol_sql": "", "test_cases": []})
        sol_sql = gt_entry["sol_sql"]
        test_cases = gt_entry["test_cases"]

        print(f"\n{'─' * 60}")
        print(f"  TASK {i+1}/{len(tasks)}  |  {instance_id}  |  database: {db}")
        print(f"{'─' * 60}")
        print(f"  User query: {query}")

        # Load DB files and open a connection the first time we see this database
        if db not in db_cache:
            print(f"\n  [DB] Loading schema, column meanings, and KB for '{db}' ...")
            db_cache[db] = load_db_context(db)
            schema, col_text, kb = db_cache[db]
            print(f"       {len(kb)} KB entries available")
            conn_cache[db] = psycopg2.connect(dbname=db, **DB_CONFIG)
        else:
            schema, col_text, kb = db_cache[db]
        conn = conn_cache[db]

        # Some KB entries are intentionally hidden per task (deleted_knowledge)
        exclude_ids = [item["deleted_knowledge"] for item in task.get("knowledge_ambiguity", [])]
        kb_text = build_kb_text(kb, exclude_ids)
        if exclude_ids:
            print(f"  [KB] Masking KB entries {exclude_ids} from agent (hidden per task rules)")

        # The real ambiguity info — used by the user simulator in Step 2
        ambiguity_data = {
            "user_query_ambiguity": task.get("user_query_ambiguity", {}),
            "knowledge_ambiguity": task.get("knowledge_ambiguity", []),
        }

        try:
            # --- Step 1 ---
            print(f"\n  [STEP 1 / EXPLORE] Sending schema + query to GPT-4o-mini ...")
            print(f"                     Asking: which tables are relevant? what's ambiguous?")
            exploration = step1_explore(query, schema, col_text, kb_text)
            print(f"\n  Agent exploration:")
            print(f"  {exploration}")

            # --- Step 2 ---
            print(f"\n  [STEP 2 / CLARIFY] Agent generating a clarifying question ...")
            question, answer = step2_clarify(query, schema, exploration, ambiguity_data)
            print(f"\n  Agent asks:  {question}")
            print(f"  User answers: {answer}")

            # --- Step 3 ---
            print(f"\n  [STEP 3 / GENERATE] Agent writing final SQL with full context ...")
            predicted_sql = step3_generate(query, schema, col_text, kb_text, exploration, question, answer)
            print(f"\n  Predicted SQL:")
            print(f"  {predicted_sql}")

        except Exception as e:
            print(f"\n  [ERROR] {e}")
            results.append({
                "instance_id": instance_id,
                "database": db,
                "user_query": query,
                "predicted_sql": "",
                "ground_truth_sql": sol_sql,
                "exact_match": False,
                "error": str(e),
            })
            continue

        # --- Comparison ---
        match = compare(predicted_sql, sol_sql, test_cases, db, conn)
        if match:
            exact_matches += 1

        # Management tasks mutate the DB — reset to template state before next task
        if test_cases:
            print(f"\n  [DB] Resetting '{db}' to original state ...")
            conn.close()
            del conn_cache[db]
            reset_db(db)
            conn_cache[db] = psycopg2.connect(dbname=db, **DB_CONFIG)
            conn = conn_cache[db]

        print(f"\n  Ground truth SQL:")
        print(f"  {sol_sql[:200]}{'...' if len(sol_sql) > 200 else ''}")
        print(f"\n  Exact match: {'✓ PASS' if match else '✗ FAIL'}")

        results.append({
            "instance_id": instance_id,
            "database": db,
            "user_query": query,
            "exploration": exploration,
            "clarifying_question": question,
            "user_answer": answer,
            "predicted_sql": predicted_sql,
            "ground_truth_sql": sol_sql,
            "exact_match": match,
        })

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    total = len(results)
    accuracy = exact_matches / total * 100 if total else 0
    print(f"\n{'=' * 60}")
    print(f"  FINAL RESULTS")
    print(f"{'=' * 60}")
    print(f"  Tasks run:        {total}")
    print(f"  Exact matches:    {exact_matches}")
    print(f"  Accuracy:         {accuracy:.1f}%")
    print(f"  Results saved to: {args.output}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
