"""
Analysis script for INTERPRET_SAMPLE pipeline results.

Usage:
    python analyze.py runs/results_sample10.csv
    python analyze.py runs/results_sample10.csv --verbose
"""

import argparse
import csv
import json
import statistics
from collections import defaultdict


def load(path: str) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def avg(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Path to results CSV")
    parser.add_argument("--verbose", action="store_true", help="Show per-interpretation detail")
    args = parser.parse_args()

    rows = load(args.csv)
    by_task: dict[str, list] = defaultdict(list)
    for r in rows:
        by_task[r["instance_id"]].append(r)

    n_tasks = len(by_task)
    n_rows = len(rows)
    n_exec_err = sum(1 for r in rows if r["sql_exec_error"])
    n_correct = sum(1 for r in rows if r["exact_match"] == "True")
    oracle_correct = sum(
        1 for task_rows in by_task.values()
        if any(r["exact_match"] == "True" for r in task_rows)
    )

    # ------------------------------------------------------------------ #
    section("OVERVIEW")
    print(f"  Tasks:            {n_tasks}")
    print(f"  Interpretations:  {n_rows}  ({n_rows/n_tasks:.1f} avg per task)")
    print(f"  Exact matches:    {n_correct}/{n_rows}  ({100*n_correct/n_rows:.1f}%)")
    print(f"  Oracle accuracy:  {oracle_correct}/{n_tasks}  ({100*oracle_correct/n_tasks:.1f}%  — any interp correct per task)")
    print(f"  Exec errors:      {n_exec_err}/{n_rows}  ({100*n_exec_err/n_rows:.1f}%)")

    # ------------------------------------------------------------------ #
    section("LOGPROB SIGNAL: errored vs clean SQL")

    err_rows = [r for r in rows if r["sql_exec_error"]]
    ok_rows  = [r for r in rows if not r["sql_exec_error"]]

    def lp_stats(subset, label):
        avgs = [float(r["avg_token_logprob"]) for r in subset if r["avg_token_logprob"]]
        stds = [float(r["std_token_logprob"]) for r in subset if r["std_token_logprob"]]
        toks = [float(r["sql_token_count"]) for r in subset if r["sql_token_count"]]
        print(f"  {label} (n={len(subset)}):")
        print(f"    avg_logprob      = {avg(avgs):+.4f}" if avgs else "    avg_logprob = N/A")
        print(f"    std_logprob      = {avg(stds):.4f}" if stds else "    std_logprob = N/A")
        print(f"    avg token count  = {avg(toks):.0f}" if toks else "    avg token count = N/A")

    lp_stats(err_rows, "Exec-errored")
    print()
    lp_stats(ok_rows, "Clean (runnable)")

    if err_rows and ok_rows:
        err_avgs = [float(r["avg_token_logprob"]) for r in err_rows if r["avg_token_logprob"]]
        ok_avgs  = [float(r["avg_token_logprob"]) for r in ok_rows  if r["avg_token_logprob"]]
        err_stds = [float(r["std_token_logprob"]) for r in err_rows if r["std_token_logprob"]]
        ok_stds  = [float(r["std_token_logprob"]) for r in ok_rows  if r["std_token_logprob"]]
        avg_gap = avg(err_avgs) - avg(ok_avgs)
        std_gap = avg(err_stds) - avg(ok_stds)
        print(f"\n  avg_logprob gap  (err - clean) = {avg_gap:+.4f}  (negative = errors are less confident)")
        print(f"  std_logprob gap  (err - clean) = {std_gap:+.4f}  (positive = errors have higher local uncertainty)")

    # ------------------------------------------------------------------ #
    section("DIVERSITY: logprob spread within each task")

    spreads = []
    std_spreads = []
    for task_id, task_rows in sorted(by_task.items()):
        avgs = [float(r["avg_token_logprob"]) for r in task_rows if r["avg_token_logprob"]]
        stds = [float(r["std_token_logprob"]) for r in task_rows if r["std_token_logprob"]]
        spread = max(avgs) - min(avgs) if len(avgs) > 1 else 0
        std_spread = max(stds) - min(stds) if len(stds) > 1 else 0
        spreads.append(spread)
        std_spreads.append(std_spread)

    print(f"  Mean avg_logprob spread per task:  {avg(spreads):.4f}")
    print(f"  Max  avg_logprob spread per task:  {max(spreads):.4f}")
    print(f"  Mean std_logprob spread per task:  {avg(std_spreads):.4f}")
    print()
    print(f"  {'Task':<20} {'DB':<12} {'N':>3}  {'AvgLp spread':>13}  {'StdLp spread':>13}  {'ExecErrs':>9}")
    print(f"  {'-'*20} {'-'*12} {'-':>3}  {'-'*13}  {'-'*13}  {'-'*9}")
    for task_id, task_rows in sorted(by_task.items()):
        db = task_rows[0]["database"]
        avgs = [float(r["avg_token_logprob"]) for r in task_rows if r["avg_token_logprob"]]
        stds = [float(r["std_token_logprob"]) for r in task_rows if r["std_token_logprob"]]
        spread = max(avgs) - min(avgs) if len(avgs) > 1 else 0
        std_spread = max(stds) - min(stds) if len(stds) > 1 else 0
        n_err = sum(1 for r in task_rows if r["sql_exec_error"])
        print(f"  {task_id:<20} {db:<12} {len(task_rows):>3}  {spread:>13.4f}  {std_spread:>13.4f}  {n_err:>4}/{len(task_rows)}")

    # ------------------------------------------------------------------ #
    section("CALIBRATION: confident-but-wrong cases")

    confident_wrong = [
        r for r in rows
        if r["sql_exec_error"]
        and r["avg_token_logprob"]
        and float(r["avg_token_logprob"]) > -0.03
    ]
    print(f"  Interpretations with avg_logprob > -0.03 but exec error: {len(confident_wrong)}")
    if confident_wrong:
        for r in confident_wrong:
            print(f"\n    [{r['instance_id']} rank {r['rank']}]  avg={float(r['avg_token_logprob']):.4f}  std={float(r['std_token_logprob']) if r['std_token_logprob'] else 'N/A':.4f}")
            print(f"    Error: {r['sql_exec_error'][:100]}")
            print(f"    Interp: {r['interpretation_text'][:100]}...")

    # ------------------------------------------------------------------ #
    if args.verbose:
        section("PER-TASK DETAIL")
        for task_id, task_rows in sorted(by_task.items()):
            print(f"\n  {task_id}  ({task_rows[0]['database']})")
            for r in task_rows:
                avg_lp = float(r["avg_token_logprob"]) if r["avg_token_logprob"] else None
                std_lp = float(r["std_token_logprob"]) if r["std_token_logprob"] else None
                err = "ERR" if r["sql_exec_error"] else "ok "
                match = "CORRECT" if r["exact_match"] == "True" else "       "
                avg_str = f"{avg_lp:+.4f}" if avg_lp is not None else "  N/A  "
                std_str = f"{std_lp:.4f}" if std_lp is not None else " N/A  "
                interp = r["interpretation_text"][:90]
                print(f"    [{err}] {match}  avg={avg_str}  std={std_str}  {interp}...")

    print()


if __name__ == "__main__":
    main()
