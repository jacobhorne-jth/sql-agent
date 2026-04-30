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
import re
import subprocess
import sys
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
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
# Logprobs JSONL logging
# ---------------------------------------------------------------------------

# Each entry written to the logprobs file is one LLM call:
# {instance_id, step, call_idx, model, input_tokens, output_tokens,
#  input: [...messages], output: "...", token_logprobs: [{token, logprob}, ...]}

_lp_ctx: dict = {"instance_id": None, "task_type": None, "file": None, "call_idx": 0}


def init_logprobs_jsonl(path: str, append: bool = False) -> None:
    _lp_ctx["file"] = open(path, "a" if append else "w")


# ---------------------------------------------------------------------------
# Row serialization helpers
# ---------------------------------------------------------------------------

def _safe_val(v):
    if isinstance(v, (date, datetime)):
        return v.strftime("%Y-%m-%d")
    if isinstance(v, Decimal):
        return float(v)
    return v


def serialize_rows(rows, max_rows: int = 20) -> list | None:
    if rows is None:
        return None
    return [[_safe_val(v) for v in row] for row in rows[:max_rows]]


def format_rows(rows, max_rows: int = 10) -> str:
    if rows is None:
        return "  (execution error)"
    if not rows:
        return "  (no rows returned)"
    lines = []
    for row in rows[:max_rows]:
        lines.append("  " + " | ".join(str(_safe_val(v)) for v in row))
    if len(rows) > max_rows:
        lines.append(f"  ... ({len(rows) - max_rows} more rows)")
    return "\n".join(lines)


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

def call_llm(messages: list[dict], step: str = "unknown") -> str:
    """
    Makes a single call to GPT-4o-mini. If a logprobs file is open, writes one
    JSONL entry per call with the full input, output, and per-token logprobs.
    """
    from openai import OpenAI
    client = OpenAI()
    use_logprobs = _lp_ctx["file"] is not None
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0,
        logprobs=use_logprobs,
    )
    content = response.choices[0].message.content.strip()
    if use_logprobs:
        lp_content = response.choices[0].logprobs.content if response.choices[0].logprobs else []
        _lp_ctx["file"].write(json.dumps({
            "instance_id": _lp_ctx["instance_id"] or "",
            "task_type": _lp_ctx["task_type"] or "",
            "step": step,
            "call_idx": _lp_ctx["call_idx"],
            "model": response.model,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "input": messages,
            "output": content,
            "token_logprobs": [{"token": t.token, "logprob": t.logprob} for t in lp_content],
        }) + "\n")
        _lp_ctx["file"].flush()
        _lp_ctx["call_idx"] += 1
    return content


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
    ], step="explore")


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
    ], step="clarify_question")

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
    ], step="clarify_answer")

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
    ], step="generate")


# ---------------------------------------------------------------------------
# Comparison via database execution
# ---------------------------------------------------------------------------


def execute_sql(sql: str, conn) -> list | None:
    """Runs a single SQL query on conn and returns rows, or None on error."""
    try:
        cursor = conn.cursor()
        cursor.execute("SET statement_timeout = '60s';")
        cursor.execute(sql)
        conn.commit()
        if cursor.description is None:
            return []
        rows = cursor.fetchmany(10001)
        return rows[:10000] if len(rows) > 10000 else rows
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


def _preprocess_sql(sql: str) -> str:
    """Strip comments, DISTINCT, and ROUND() — mirroring the real BIRD-Interact eval.
    ORDER BY is intentionally NOT stripped; set-based comparison handles it instead."""
    # Remove block comments /* ... */
    sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
    # Remove line comments -- ...
    sql = re.sub(r'--.*?(\r\n|\r|\n)', r'\1', sql)
    # Remove DISTINCT
    sql = ' '.join(t for t in sql.split() if t.lower() != 'distinct')
    # Remove ROUND(..., n) keeping the inner expression
    def _strip_round(s):
        pat = re.compile(r'ROUND\s*\(', re.IGNORECASE)
        while True:
            m = pat.search(s)
            if not m:
                break
            depth, i = 0, m.end() - 1
            first_arg_end = None
            for j in range(i, len(s)):
                if s[j] == '(':
                    depth += 1
                elif s[j] == ')':
                    depth -= 1
                    if depth == 0:
                        close = j
                        break
                elif s[j] == ',' and depth == 1:
                    first_arg_end = j
            end = first_arg_end if first_arg_end is not None else close
            s = s[:m.start()] + s[i + 1:end].strip() + s[close + 1:]
        return s
    sql = _strip_round(sql)
    return re.sub(r'\s+', ' ', sql).strip()


def _normalize_rows(rows, decimal_places: int = 2) -> list[tuple]:
    """Normalize dates, round floats/Decimals, convert unhashable types to strings."""
    quantizer = Decimal(1).scaleb(-decimal_places)
    out = []
    for row in rows:
        new_row = []
        for v in row:
            if isinstance(v, (date, datetime)):
                new_row.append(v.strftime('%Y-%m-%d'))
            elif isinstance(v, Decimal):
                new_row.append(v.quantize(quantizer, rounding=ROUND_HALF_UP))
            elif isinstance(v, float):
                new_row.append(round(v, decimal_places))
            elif isinstance(v, (dict, list)):
                new_row.append(json.dumps(v, sort_keys=True))
            else:
                new_row.append(v)
        out.append(tuple(new_row))
    return out


def compare(
    pred: str, gold: str, test_cases: list, db_name: str, conn
) -> tuple[bool, dict, dict]:
    """
    For query tasks (no test_cases): execute both SQLs and compare result sets.
    For management tasks (test_cases present): execute predicted SQL then run
    each test case Python function to verify the DB state, mirroring the real eval.

    Returns (match, pred_exec, gold_exec) where each exec dict has "rows" and "error".
    """
    pred_exec: dict = {"rows": None, "error": None}
    gold_exec: dict = {"rows": None, "error": None}

    if not test_cases:
        try:
            pred_exec["rows"] = execute_sql(_preprocess_sql(pred), conn)
        except Exception as e:
            pred_exec["error"] = str(e)
        try:
            gold_exec["rows"] = execute_sql(_preprocess_sql(gold), conn)
        except Exception as e:
            gold_exec["error"] = str(e)

        if pred_exec["error"] or gold_exec["error"]:
            return False, pred_exec, gold_exec
        if pred_exec["rows"] is None or gold_exec["rows"] is None:
            return False, pred_exec, gold_exec

        return (
            set(_normalize_rows(pred_exec["rows"])) == set(_normalize_rows(gold_exec["rows"])),
            pred_exec,
            gold_exec,
        )
    else:
        try:
            execute_sql(pred, conn)
        except Exception as e:
            pred_exec["error"] = str(e)
            return False, pred_exec, gold_exec
        for tc_str in test_cases:
            try:
                namespace = {"execute_queries": execute_queries}
                exec(tc_str, namespace)
                namespace["test_case"]([pred], [gold], db_name, conn)
            except Exception as e:
                return False, pred_exec, gold_exec
        return True, pred_exec, gold_exec


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only run first N tasks")
    parser.add_argument("--sample", type=int, default=None, help="Pick N tasks spread across different databases (round-robin)")
    parser.add_argument("--output", default="results.jsonl")
    parser.add_argument(
        "--logprobs", default=None, metavar="FILE",
        help="If set, write full input/output/token logprobs to this JSONL file",
    )
    parser.add_argument(
        "--task-type", default="all", choices=["all", "select", "management"],
        help="Filter tasks: 'select' (no test_cases), 'management' (has test_cases), or 'all'",
    )
    parser.add_argument(
        "--pause", action="store_true",
        help="Pause after each task's SQL executes (before DB reset) so you can inspect the DB",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="Append to output and logprobs files instead of overwriting",
    )
    parser.add_argument(
        "--database", default=None, metavar="DB",
        help="Only run tasks from this database (e.g. 'gaming')",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Error: OPENAI_API_KEY not found. Add it to the .env file at the repo root.")

    if args.logprobs:
        init_logprobs_jsonl(args.logprobs, append=args.append)
        mode = "Appending to" if args.append else "Writing"
        print(f"\n[LOGPROBS] {mode} full input/output/token logprobs to {args.logprobs}")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  BIRD-Interact Baseline Agent")
    print("  3-step: Explore → Clarify → Generate")
    print("=" * 60)

    print("\n[SETUP] Loading tasks from bird_interact_data.jsonl ...")
    tasks = load_tasks()

    print("\n[SETUP] Loading ground truth SQL answers ...")
    gt = load_gt()
    print(f"  → {len(gt)} ground truth entries loaded")

    if args.task_type != "all":
        before = len(tasks)
        if args.task_type == "select":
            tasks = [t for t in tasks if not gt.get(t["instance_id"], {}).get("test_cases")]
        else:
            tasks = [t for t in tasks if gt.get(t["instance_id"], {}).get("test_cases")]
        print(f"  → filtered to {len(tasks)} '{args.task_type}' tasks (from {before})")

    if args.database:
        before = len(tasks)
        tasks = [t for t in tasks if t["selected_database"] == args.database]
        print(f"  → filtered to {len(tasks)} tasks for database '{args.database}' (from {before})")

    if args.sample:
        from collections import defaultdict
        by_db: dict[str, list] = defaultdict(list)
        for t in tasks:
            by_db[t["selected_database"]].append(t)
        sampled, db_order, idx = [], list(by_db.keys()), 0
        while len(sampled) < args.sample:
            db = db_order[idx % len(db_order)]
            if by_db[db]:
                sampled.append(by_db[db].pop(0))
            idx += 1
            if idx > args.sample * len(db_order):
                break
        tasks = sampled
        print(f"  → sampled {len(tasks)} tasks across {len(set(t['selected_database'] for t in tasks))} databases")
    elif args.limit:
        tasks = tasks[: args.limit]
    print(f"  → {len(tasks)} tasks to run")

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

        _lp_ctx["instance_id"] = instance_id
        task_type = "management" if test_cases else "select"
        _lp_ctx["task_type"] = task_type

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

        except Exception as e:
            print(f"\n  [ERROR] {e}")
            results.append({
                "instance_id": instance_id,
                "task_type": task_type,
                "database": db,
                "user_query": query,
                "predicted_sql": "",
                "ground_truth_sql": sol_sql,
                "exact_match": False,
                "error": str(e),
            })
            continue

        # --- Comparison ---
        match, pred_exec, gold_exec = compare(predicted_sql, sol_sql, test_cases, db, conn)
        if match:
            exact_matches += 1

        # Management tasks mutate the DB — reset to template state before next task
        if test_cases:
            if args.pause:
                input(f"\n  [PAUSE] DB '{db}' has been modified. Inspect it now, then press Enter to reset...")
            print(f"\n  [DB] Resetting '{db}' to original state ...")
            conn.close()
            del conn_cache[db]
            reset_db(db)
            conn_cache[db] = psycopg2.connect(dbname=db, **DB_CONFIG)
            conn = conn_cache[db]

        print(f"\n  Predicted SQL:")
        print(f"  {predicted_sql}")
        if pred_exec["error"]:
            print(f"\n  Predicted SQL result: ERROR — {pred_exec['error']}")
        else:
            print(f"\n  Predicted SQL result ({len(pred_exec['rows'] or [])} rows):")
            print(format_rows(pred_exec["rows"]))

        print(f"\n  Ground truth SQL:")
        print(f"  {sol_sql}")
        if not test_cases:
            if gold_exec["error"]:
                print(f"\n  Ground truth SQL result: ERROR — {gold_exec['error']}")
            else:
                print(f"\n  Ground truth SQL result ({len(gold_exec['rows'] or [])} rows):")
                print(format_rows(gold_exec["rows"]))

        print(f"\n  Exact match: {'✓ PASS' if match else '✗ FAIL'}")

        results.append({
            "instance_id": instance_id,
            "task_type": task_type,
            "database": db,
            "user_query": query,
            "exploration": exploration,
            "clarifying_question": question,
            "user_answer": answer,
            "predicted_sql": predicted_sql,
            "predicted_sql_rows": serialize_rows(pred_exec["rows"]),
            "predicted_sql_error": pred_exec["error"],
            "ground_truth_sql": sol_sql,
            "ground_truth_sql_rows": serialize_rows(gold_exec["rows"]),
            "ground_truth_sql_error": gold_exec["error"],
            "exact_match": match,
        })

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    with open(args.output, "a" if args.append else "w") as f:
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
    if args.logprobs:
        print(f"  Logprobs saved to: {args.logprobs}")
    print(f"{'=' * 60}\n")

    if _lp_ctx["file"]:
        _lp_ctx["file"].close()
        _lp_ctx["file"] = None


if __name__ == "__main__":
    main()
