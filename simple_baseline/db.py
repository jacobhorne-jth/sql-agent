# Database execution, SQL comparison, DB reset, and row serialization helpers

import json
import os
import re
import subprocess
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "user": "root",
    "password": "123123",
}


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
# SQL execution
# ---------------------------------------------------------------------------

def execute_sql(sql: str, conn) -> list | None:
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


def probe_sql(sql: str, conn) -> tuple[list | None, str | None]:
    cursor = None
    try:
        cursor = conn.cursor()
        cursor.execute("SET statement_timeout = '60s';")
        cursor.execute(sql)
        rows = cursor.fetchmany(10001) if cursor.description is not None else []
        conn.rollback()
        return rows[:10000] if len(rows) > 10000 else rows, None
    except Exception as e:
        conn.rollback()
        return None, str(e)
    finally:
        if cursor is not None:
            cursor.close()


def execute_queries(sql: str, db_name: str, conn):
    """Wrapper used by exec'd test case functions."""
    try:
        return execute_sql(sql, conn), False, False
    except Exception as e:
        return None, str(e), False


def reset_db(db_name: str):
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


# ---------------------------------------------------------------------------
# SQL comparison
# ---------------------------------------------------------------------------

def _preprocess_sql(sql: str) -> str:
    """Strip comments, DISTINCT, and ROUND() — mirroring the real BIRD-Interact eval."""
    sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
    sql = re.sub(r'--.*?(\r\n|\r|\n)', r'\1', sql)
    sql = ' '.join(t for t in sql.split() if t.lower() != 'distinct')

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
