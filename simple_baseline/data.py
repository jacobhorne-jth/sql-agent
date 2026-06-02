# Data loading: load_tasks, load_gt, load_db_context, build_kb_text

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "bird-interact-lite-data"
GT_FILE = Path(__file__).parent.parent / "GT-stuff" / "bird_interact_gt_kg_testcases_1008.jsonl"
TASKS_FILE = DATA_DIR / "bird_interact_data.jsonl"


def load_tasks() -> list[dict]:
    tasks = []
    with open(TASKS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def load_gt() -> dict[str, dict]:
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
    lines = []
    for entry in kb:
        if entry["id"] not in exclude_ids:
            lines.append(f"[{entry['id']}] {entry['knowledge']}: {entry['definition']}")
    return "\n".join(lines) if lines else "(no external knowledge available)"
