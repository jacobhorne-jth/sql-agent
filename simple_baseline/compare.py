"""
Prints predicted vs ground truth SQL side by side for each result in results.jsonl.

Usage:
    python compare.py [--input results.jsonl]
"""

import argparse
import json


def print_side_by_side(label_a: str, text_a: str, label_b: str, text_b: str, width: int = 60):
    """
    Prints two blocks of text in two columns side by side.
    Each column is `width` characters wide.
    """
    lines_a = text_a.splitlines()
    lines_b = text_b.splitlines()

    # Pad the shorter side so both columns have the same number of rows
    max_lines = max(len(lines_a), len(lines_b))
    lines_a += [""] * (max_lines - len(lines_a))
    lines_b += [""] * (max_lines - len(lines_b))

    # Header row
    print(f"  {label_a:<{width}}  {label_b}")
    print(f"  {'-' * width}  {'-' * width}")

    for a, b in zip(lines_a, lines_b):
        print(f"  {a:<{width}}  {b}")


def strip_comments(sql: str) -> str:
    """Remove /* ... */ block comments and -- line comments for cleaner display."""
    import re
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    sql = re.sub(r"--[^\n]*", "", sql)
    # Remove blank lines left behind
    lines = [l for l in sql.splitlines() if l.strip()]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results.jsonl")
    args = parser.parse_args()

    results = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    total = len(results)
    passes = sum(1 for r in results if r.get("exact_match"))

    print("=" * 124)
    print("  BIRD-Interact Baseline — Predicted vs Ground Truth SQL")
    print("=" * 124)

    for i, r in enumerate(results):
        status = "✓ PASS" if r.get("exact_match") else "✗ FAIL"
        error = r.get("error", "")

        print(f"\n  Task {i+1}/{total}  |  {r['instance_id']}  |  db: {r['database']}  |  {status}")
        print(f"  Query: {r['user_query']}")

        if error:
            print(f"\n  ERROR: {error}")
            print("─" * 124)
            continue

        predicted = strip_comments(r.get("predicted_sql", ""))
        ground_truth = strip_comments(r.get("ground_truth_sql", ""))

        print()
        print_side_by_side("PREDICTED SQL", predicted, "GROUND TRUTH SQL", ground_truth)
        print("─" * 124)

    print(f"\n  SUMMARY: {passes}/{total} exact matches ({passes/total*100:.1f}%)")
    print(f"  (Exact match requires character-for-character identical SQL after normalization)")
    print("=" * 124)


if __name__ == "__main__":
    main()
