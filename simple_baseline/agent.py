"""
Multi-step BIRD-Interact baseline agent.

Runs two no-clarification pipelines per task:
  NO_CLARIFY_SAMPLE  - Explore + Generate 10x at temperature 0.7
  NO_CLARIFY_TOP3    - Explore + Generate top-3 candidates in one call

Usage:
    python agent.py [--limit N] [--output results.jsonl] [--logprobs logprobs.jsonl]

Outputs:
  <output>.jsonl                  full results
  <output>_sql_outputs.jsonl      compact SQL + call outputs
  <output>.csv                    summary table (sql, accuracy, avg/min logprob)
  <output>_token_logprobs.jsonl   per-token logprobs for SQL spans

OPENAI_API_KEY is loaded from the .env file at the repo root.
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

from data import load_tasks, load_gt, load_db_context, build_kb_text
from db import DB_CONFIG, compare, reset_db, serialize_rows, format_rows
from llm import _lp_ctx, init_logprobs_jsonl, init_token_logprobs_jsonl
from pipeline import run_pipeline, run_pipeline_top3, run_pipeline_interpret, summarize_results

def _strip_gt_comments(sql: str) -> str:
    import re
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    sql = re.sub(r"--[^\n]*", "", sql)
    sql = "\n".join(line for line in sql.splitlines() if line.strip())
    return sql.strip()


CSV_FIELDS = [
    "instance_id", "database", "task_type", "user_query",
    "pipeline", "rank", "sql", "ground_truth_sql",
    "exact_match", "sql_exec_error",
    "avg_token_logprob", "min_token_logprob", "std_token_logprob", "sql_token_count",
    "interpretation_id", "interpretation_text", "assignments", "ambiguities_found",
]

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only run first N tasks")
    parser.add_argument("--sample", type=int, default=None, help="Pick N tasks spread across different databases (round-robin)")
    parser.add_argument("--output", default="runs/results.jsonl")
    parser.add_argument(
        "--sql-output", default=None, metavar="FILE",
        help="Write compact JSONL with output SQLs and call outputs. Defaults to '<output stem>_sql_outputs.jsonl'.",
    )
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
        "--resume", action="store_true",
        help="Skip tasks already present in the output file (use with --append to extend a previous run)",
    )
    parser.add_argument(
        "--database", default=None, metavar="DB",
        help="Only run tasks from this database (e.g. 'gaming')",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sql_output_path = args.sql_output
    if sql_output_path is None:
        sql_output_path = str(output_path.with_name(f"{output_path.stem}_sql_outputs{output_path.suffix}"))

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Error: OPENAI_API_KEY not found. Add it to the .env file at the repo root.")

    csv_path = str(output_path.with_name(f"{output_path.stem}.csv"))
    token_logprobs_path = str(output_path.with_name(f"{output_path.stem}_token_logprobs.jsonl"))

    if args.logprobs:
        init_logprobs_jsonl(args.logprobs, append=args.append)
        mode = "Appending to" if args.append else "Writing"
        print(f"\n[LOGPROBS] {mode} full input/output/token logprobs to {args.logprobs}")

    init_token_logprobs_jsonl(token_logprobs_path, append=args.append)
    mode = "Appending to" if args.append else "Writing"
    print(f"[TOKEN LOGPROBS] {mode} per-token SQL logprobs to {token_logprobs_path}")

    print("=" * 60)
    print("  BIRD-Interact Baseline Agent")
    print("  Pipeline:   INTERPRET_SAMPLE    (Explore + Interpret + Generate per interpretation)")
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

    if args.resume and output_path.exists():
        done_ids: set[str] = set()
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done_ids.add(json.loads(line)["instance_id"])
                    except Exception:
                        pass
        before = len(tasks)
        tasks = [t for t in tasks if t["instance_id"] not in done_ids]
        print(f"  → resuming: skipped {before - len(tasks)} already-done tasks, {len(tasks)} remaining")

    print(f"  → {len(tasks)} tasks to run")

    db_cache: dict[str, tuple] = {}
    conn_cache: dict[str, object] = {}
    results = []
    sql_output_results = []
    csv_rows: list[dict] = []
    exact_matches = 0

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

        if db not in db_cache:
            print(f"\n  [DB] Loading schema, column meanings, and KB for '{db}' ...")
            db_cache[db] = load_db_context(db)
            schema, col_text, kb = db_cache[db]
            print(f"       {len(kb)} KB entries available")
            conn_cache[db] = psycopg2.connect(dbname=db, **DB_CONFIG)
        else:
            schema, col_text, kb = db_cache[db]
        conn = conn_cache[db]

        exclude_ids = [item["deleted_knowledge"] for item in task.get("knowledge_ambiguity", [])]
        kb_text = build_kb_text(kb, exclude_ids)
        if exclude_ids:
            print(f"  [KB] Masking KB entries {exclude_ids} from agent (hidden per task rules)")

        try:
            print("\n  [PIPELINE] INTERPRET_SAMPLE  (Explore + Interpret + Generate per interpretation)")
            interpret_pipeline = run_pipeline_interpret(
                name="INTERPRET_SAMPLE",
                query=query,
                schema=schema,
                col_text=col_text,
                kb_text=kb_text,
            )

            pipelines = [interpret_pipeline]
            predicted_sql = (
                interpret_pipeline["sql_candidates"][0]["sql"]
                if interpret_pipeline["sql_candidates"]
                else ""
            )
            exploration = interpret_pipeline["exploration"]

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
            sql_output_results.append({
                "instance_id": instance_id,
                "task_type": task_type,
                "database": db,
                "user_query": query,
                "pipelines": [],
                "output_sqls": [],
                "ground_truth_sql": sol_sql,
                "error": str(e),
            })
            continue

        primary_match, pred_exec, gold_exec = compare(predicted_sql, sol_sql, test_cases, db, conn)
        match = primary_match
        candidate_matches = []
        if not test_cases:
            for pipeline in pipelines:
                for candidate in pipeline["sql_candidates"]:
                    candidate_match, candidate_pred_exec, _ = compare(
                        candidate["sql"], sol_sql, test_cases, db, conn
                    )
                    candidate_matches.append({
                        "pipeline": pipeline["name"],
                        "rank": candidate["rank"],
                        "exact_match": candidate_match,
                        "predicted_sql_error": candidate_pred_exec["error"],
                        "avg_token_logprob": candidate.get("avg_token_logprob"),
                        "min_token_logprob": candidate.get("min_token_logprob"),
                    })
            match = any(c["exact_match"] for c in candidate_matches)
        if match:
            exact_matches += 1

        if test_cases:
            if args.pause:
                input(f"\n  [PAUSE] DB '{db}' has been modified. Inspect it now, then press Enter to reset...")
            print(f"\n  [DB] Resetting '{db}' to original state ...")
            conn.close()
            del conn_cache[db]
            reset_db(db)
            conn_cache[db] = psycopg2.connect(dbname=db, **DB_CONFIG)
            conn = conn_cache[db]

        print(f"\n  Primary predicted SQL (INTERPRET_SAMPLE rank 1):")
        print(f"  {predicted_sql}")
        print(f"\n  All output SQL candidates:")
        for pipeline in pipelines:
            for candidate in pipeline["sql_candidates"]:
                avg = candidate.get("avg_token_logprob")
                std = candidate.get("std_token_logprob")
                lp_str = f"  avg_logprob={avg:.4f}" if avg is not None else ""
                std_str = f"  std={std:.4f}" if std is not None else ""
                interp_str = ""
                if candidate.get("interpretation_text"):
                    interp_str = f"\n  [interpretation] {candidate['interpretation_text'][:120]}{'...' if len(candidate['interpretation_text']) > 120 else ''}"
                print(f"\n  [{pipeline['name']} rank {candidate['rank']}]{lp_str}{std_str}{interp_str}")
                print(f"  {candidate['sql']}")
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

        match_by_key = {(m["pipeline"], m["rank"]): m for m in candidate_matches}
        for pipeline in pipelines:
            for candidate in pipeline["sql_candidates"]:
                cm = match_by_key.get((pipeline["name"], candidate["rank"]), {})
                csv_rows.append({
                    "instance_id": instance_id,
                    "database": db,
                    "task_type": task_type,
                    "user_query": query,
                    "ground_truth_sql": _strip_gt_comments(sol_sql),
                    "pipeline": pipeline["name"],
                    "rank": candidate["rank"],
                    "sql": candidate["sql"],
                    "exact_match": cm.get("exact_match", ""),
                    "sql_exec_error": cm.get("predicted_sql_error") or "",
                    "avg_token_logprob": candidate.get("avg_token_logprob"),
                    "min_token_logprob": candidate.get("min_token_logprob"),
                    "std_token_logprob": candidate.get("std_token_logprob"),
                    "sql_token_count": candidate.get("sql_token_count"),
                    "interpretation_id": candidate.get("interpretation_id", ""),
                    "interpretation_text": candidate.get("interpretation_text", ""),
                    "assignments": json.dumps(candidate.get("assignments", {})) if candidate.get("assignments") else "",
                    "ambiguities_found": " | ".join(pipeline.get("ambiguities_found", [])),
                })

        results.append({
            "instance_id": instance_id,
            "task_type": task_type,
            "database": db,
            "user_query": query,
            "exploration": exploration,
            "predicted_sql": predicted_sql,
            "predicted_sql_rows": serialize_rows(pred_exec["rows"]),
            "predicted_sql_error": pred_exec["error"],
            "ground_truth_sql": sol_sql,
            "ground_truth_sql_rows": serialize_rows(gold_exec["rows"]),
            "ground_truth_sql_error": gold_exec["error"],
            "exact_match": match,
            "primary_exact_match": primary_match,
            "candidate_matches": candidate_matches,
            "pipelines": pipelines,
            "sql_outputs": [
                {
                    "pipeline": pipeline["name"],
                    "rank": candidate["rank"],
                    "sql": candidate["sql"],
                    "avg_token_logprob": candidate.get("avg_token_logprob"),
                    "min_token_logprob": candidate.get("min_token_logprob"),
                    "std_token_logprob": candidate.get("std_token_logprob"),
                    "sql_token_count": candidate.get("sql_token_count"),
                    "interpretation_id": candidate.get("interpretation_id"),
                    "interpretation_text": candidate.get("interpretation_text"),
                }
                for pipeline in pipelines
                for candidate in pipeline["sql_candidates"]
            ],
        })
        sql_output_results.append({
            "instance_id": instance_id,
            "task_type": task_type,
            "database": db,
            "user_query": query,
            "output_sqls": [
                {
                    "pipeline": pipeline["name"],
                    "rank": candidate["rank"],
                    "sql": candidate["sql"],
                    "avg_token_logprob": candidate.get("avg_token_logprob"),
                    "min_token_logprob": candidate.get("min_token_logprob"),
                    "std_token_logprob": candidate.get("std_token_logprob"),
                    "sql_token_count": candidate.get("sql_token_count"),
                    "interpretation_id": candidate.get("interpretation_id"),
                    "interpretation_text": candidate.get("interpretation_text"),
                }
                for pipeline in pipelines
                for candidate in pipeline["sql_candidates"]
            ],
            "call_outputs": [
                {"pipeline": pipeline["name"], "calls": pipeline["calls"]}
                for pipeline in pipelines
            ],
            "ground_truth_sql": sol_sql,
        })

    with open(args.output, "a" if args.append else "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    with open(sql_output_path, "a" if args.append else "w") as f:
        for r in sql_output_results:
            f.write(json.dumps(r) + "\n")

    write_header = not args.append or not Path(csv_path).exists()
    with open(csv_path, "a" if args.append else "w", newline="") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(csv_rows)

    total = len(results)
    summary = summarize_results(results)
    print(f"\n{'=' * 60}")
    print(f"  FINAL RESULTS")
    print(f"{'=' * 60}")
    print(f"  Tasks run:        {total}")
    print(f"  Exact matches:    {summary['exact']}")
    print(f"  Accuracy:         {summary['accuracy']:.1f}%")
    print(f"  Pred exec errors: {summary['pred_exec_errors']}")
    print(f"  Run/setup errors: {summary['setup_errors']}")
    for task_type, bucket in summary["by_type"].items():
        bucket_acc = bucket["exact"] / bucket["total"] * 100 if bucket["total"] else 0
        print(
            f"  {task_type}: {bucket['exact']}/{bucket['total']} exact "
            f"({bucket_acc:.1f}%), {bucket['pred_exec_errors']} pred exec errors"
        )
    if summary["by_pipeline"]:
        by_name: dict[str, list] = {}
        for b in summary["by_pipeline"].values():
            by_name.setdefault(b["pipeline"], []).append(b)

        print(f"\n  Per-sample accuracy (select tasks):")
        for pipeline_name in sorted(by_name):
            samples = sorted(by_name[pipeline_name], key=lambda x: x["rank"])
            n = samples[0]["total"] if samples else 0
            print(f"\n  {pipeline_name}  (n={n} tasks)")
            print(f"  {'Sample':>6}  {'Correct':>9}  {'Accuracy':>9}  {'Exec Errs':>9}")
            print(f"  {'------':>6}  {'---------':>9}  {'---------':>9}  {'---------':>9}")
            for b in samples:
                acc = b["exact"] / b["total"] * 100 if b["total"] else 0
                print(f"  {b['rank']:>6}  {b['exact']:>4}/{b['total']:<4}  {acc:>8.1f}%  {b['exec_errors']:>9}")
            oracle = summary["by_pipeline_oracle"].get(pipeline_name, {})
            if oracle:
                oracle_acc = oracle["oracle_exact"] / oracle["total"] * 100 if oracle["total"] else 0
                print(f"  {'oracle':>6}  {oracle['oracle_exact']:>4}/{oracle['total']:<4}  {oracle_acc:>8.1f}%  (any of {len(samples)} matched)")
    print(f"  Results saved to:       {args.output}")
    print(f"  SQL outputs saved to:   {sql_output_path}")
    print(f"  CSV saved to:           {csv_path}")
    print(f"  Token logprobs saved to: {token_logprobs_path}")
    if args.logprobs:
        print(f"  Full logprobs saved to: {args.logprobs}")
    print(f"{'=' * 60}\n")

    if _lp_ctx["file"]:
        _lp_ctx["file"].close()
        _lp_ctx["file"] = None
    if _lp_ctx["token_lp_file"]:
        _lp_ctx["token_lp_file"].close()
        _lp_ctx["token_lp_file"] = None


if __name__ == "__main__":
    main()
