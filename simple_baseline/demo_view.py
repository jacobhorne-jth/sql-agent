"""
Print a compact with-vs-without-clarification demo view.

Usage:
    python simple_baseline/demo_view.py \
      --results simple_baseline/results_one.jsonl \
      --logprobs simple_baseline/logprobs_one.jsonl
"""

import argparse
import json
from pathlib import Path

from llm import sql_logprob_stats


GENERATE_STEPS = {
    **{f"no_clarify_sample_generate_{i}": "NO_CLARIFY_SAMPLE" for i in range(1, 11)},
    "no_clarify_top3_generate_top3": "NO_CLARIFY_TOP3",
}


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def one_line_sql(sql: str, max_len: int = 180) -> str:
    text = " ".join(sql.split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def strip_sql_comments(sql: str) -> str:
    import re

    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    sql = re.sub(r"--[^\n]*", "", sql)
    return "\n".join(line for line in sql.splitlines() if line.strip()).strip()


def index_generation_confidence(logprob_rows: list[dict]) -> dict[tuple[str, str, int], dict]:
    by_candidate = {}
    counters: dict[tuple[str, str], int] = {}

    for row in logprob_rows:
        pipeline = GENERATE_STEPS.get(row.get("step"))
        if not pipeline:
            continue
        instance_id = row.get("instance_id", "")
        key = (instance_id, pipeline)
        counters.setdefault(key, 0)
        stats = row.get("sql_logprob_stats")
        if stats is None:
            class TokenInfo:
                def __init__(self, token: str, logprob: float | None):
                    self.token = token
                    self.logprob = logprob

            token_infos = [
                TokenInfo(t.get("token", ""), t.get("logprob"))
                for t in row.get("token_logprobs", [])
            ]
            stats = sql_logprob_stats(row.get("output", ""), token_infos)

        for stat in stats:
            counters[key] += 1
            by_candidate[(instance_id, pipeline, counters[key])] = stat

    return by_candidate


def print_task(task: dict, confidence: dict[tuple[str, str, int], dict]) -> None:
    instance_id = task["instance_id"]
    print("=" * 100)
    print(f"{instance_id} | db={task['database']} | task_type={task.get('task_type', 'unknown')}")
    print(f"overall exact_match={task.get('exact_match')} | primary_exact_match={task.get('primary_exact_match')}")
    print()
    print("User query:")
    print(f"  {task['user_query']}")
    print()

    if task.get("clarifying_question") or task.get("user_answer"):
        print("Clarification:")
        print(f"  Q: {task.get('clarifying_question', '')}")
        answer = " ".join(task.get("user_answer", "").split())
        print(f"  A: {answer[:500]}{'...' if len(answer) > 500 else ''}")
        print()

    match_by_candidate = {
        (m["pipeline"], m["rank"]): m
        for m in task.get("candidate_matches", [])
    }

    for pipeline_name in ("NO_CLARIFY_SAMPLE", "NO_CLARIFY_TOP3"):
        print("-" * 100)
        print(f"{pipeline_name}")
        print("-" * 100)
        candidates = [
            c for c in task.get("sql_outputs", [])
            if c.get("pipeline") == pipeline_name
        ]
        for candidate in candidates:
            rank = candidate["rank"]
            match = match_by_candidate.get((pipeline_name, rank), {})
            conf = confidence.get((instance_id, pipeline_name, rank), {})
            avg = conf.get("avg_token_logprob") if conf else candidate.get("avg_token_logprob")
            min_lp = conf.get("min_token_logprob") if conf else candidate.get("min_token_logprob")
            token_count = conf.get("token_count") if conf else candidate.get("token_count")
            avg_text = f"{avg:.4f}" if isinstance(avg, (int, float)) else "n/a"
            min_text = f"{min_lp:.4f}" if isinstance(min_lp, (int, float)) else "n/a"
            tokens_text = str(token_count) if token_count is not None else "n/a"

            print(
                f"rank {rank} | match={match.get('exact_match', 'n/a')} "
                f"| avg_logprob={avg_text} | min_logprob={min_text} | sql_tokens={tokens_text}"
            )
            print("  raw_sql:")
            print(f"    {one_line_sql(candidate.get('raw_sql') or candidate.get('sql', ''))}")
            reviewed = candidate.get("reviewed_sql") or candidate.get("sql", "")
            if reviewed and reviewed != candidate.get("raw_sql"):
                print("  reviewed_sql:")
                print(f"    {one_line_sql(reviewed)}")
            error = match.get("predicted_sql_error")
            if error:
                print(f"  error: {error}")
            print()

    print("Ground truth:")
    ground_truth_sql = strip_sql_comments(task.get("ground_truth_sql", ""))
    print(f"  {one_line_sql(ground_truth_sql, max_len=300)}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Results JSONL from agent.py")
    parser.add_argument("--logprobs", default=None, help="Logprobs JSONL from agent.py")
    parser.add_argument("--instance-id", default=None, help="Only show one task")
    args = parser.parse_args()

    results = load_jsonl(args.results)
    if args.instance_id:
        results = [r for r in results if r.get("instance_id") == args.instance_id]

    confidence = {}
    if args.logprobs and Path(args.logprobs).exists():
        confidence = index_generation_confidence(load_jsonl(args.logprobs))

    for task in results:
        print_task(task, confidence)


if __name__ == "__main__":
    main()
