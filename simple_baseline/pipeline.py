# Pipeline orchestration (run_pipeline, run_pipeline_top3) and results summarization

import sqlparse

from llm import _lp_ctx, _strip_markdown_fence, step1_explore, step2_clarify, step3_generate_sample, step3_generate_top3, step_interpret


def _format_sql(sql: str) -> str:
    try:
        return sqlparse.format(sql, reindent=True, keyword_case="upper").strip()
    except Exception:
        return sql

GENERATE_SAMPLE_COUNT = 10
GENERATE_TEMPERATURE = 0.7
INTERPRET_MAX = 6


def run_pipeline(
    *,
    name: str,
    skip_clarification: bool,
    query: str,
    schema: str,
    col_text: str,
    kb_text: str,
    ambiguity_data: dict,
) -> dict:
    _lp_ctx["step_pipeline"] = name

    print(f"\n  [{name} / STEP 1] Exploring schema + query ...")
    exploration = step1_explore(query, schema, col_text, kb_text)
    print(f"\n  [{name}] Agent exploration:")
    print(f"  {exploration}")

    calls = [{"step": "explore", "output": exploration}]
    question = ""
    answer = ""

    if skip_clarification:
        print(f"\n  [{name} / STEP 2] Skipping clarification call ...")
    else:
        print(f"\n  [{name} / STEP 2] Agent generating a clarifying question ...")
        question, answer = step2_clarify(query, schema, exploration, ambiguity_data)
        print(f"\n  [{name}] Agent asks:  {question}")
        print(f"  [{name}] User answers: {answer}")
        calls.extend([
            {"step": "clarify_question", "output": question},
            {"step": "clarify_answer", "output": answer},
        ])

    print(f"\n  [{name} / STEP 3] Generating {GENERATE_SAMPLE_COUNT} SQL samples (temp={GENERATE_TEMPERATURE}) ...")
    sql_candidates = []
    for sample_idx in range(1, GENERATE_SAMPLE_COUNT + 1):
        print(f"  [{name} / GENERATE {sample_idx}/{GENERATE_SAMPLE_COUNT}]", end=" ", flush=True)
        sql, sql_stats = step3_generate_sample(
            query=query,
            schema=schema,
            col_text=col_text,
            kb_text=kb_text,
            exploration=exploration,
            question=None if skip_clarification else question,
            answer=None if skip_clarification else answer,
            step_name=f"{name.lower()}_generate_{sample_idx}",
            temperature=GENERATE_TEMPERATURE,
        )
        sql = _strip_markdown_fence(sql)
        avg_lp = sql_stats[0]["avg_token_logprob"] if sql_stats else None
        min_lp = sql_stats[0]["min_token_logprob"] if sql_stats else None
        print(f"done  (avg_logprob={avg_lp:.4f})" if avg_lp is not None else "done")
        calls.append({"step": f"generate_{sample_idx}", "output": sql})
        sql_candidates.append({
            "rank": sample_idx,
            "raw_sql": sql,
            "sql": sql,
            "avg_token_logprob": avg_lp,
            "min_token_logprob": min_lp,
            "std_token_logprob": sql_stats[0].get("std_token_logprob") if sql_stats else None,
            "sql_token_count": sql_stats[0].get("token_count") if sql_stats else None,
        })

    return {
        "name": name,
        "skipped_clarification": skip_clarification,
        "exploration": exploration,
        "clarifying_question": question,
        "user_answer": answer,
        "sql_candidates": sql_candidates,
        "calls": calls,
    }


def run_pipeline_top3(
    *,
    name: str,
    query: str,
    schema: str,
    col_text: str,
    kb_text: str,
) -> dict:
    _lp_ctx["step_pipeline"] = name

    print(f"\n  [{name} / STEP 1] Exploring schema + query ...")
    exploration = step1_explore(query, schema, col_text, kb_text)
    print(f"\n  [{name}] Agent exploration:")
    print(f"  {exploration}")

    calls = [{"step": "explore", "output": exploration}]

    print(f"\n  [{name} / STEP 2] Generating top-3 SQL candidates (single call) ...")
    candidates, _ = step3_generate_top3(
        query=query,
        schema=schema,
        col_text=col_text,
        kb_text=kb_text,
        exploration=exploration,
        question=None,
        answer=None,
        step_name=f"{name.lower()}_generate_top3",
    )
    sql_stats = _lp_ctx["last_sql_stats"]

    sql_candidates = []
    for i, sql in enumerate(candidates):
        sql = _format_sql(_strip_markdown_fence(sql))
        stat = sql_stats[i] if i < len(sql_stats) else {}
        avg_lp = stat.get("avg_token_logprob")
        min_lp = stat.get("min_token_logprob")
        rank = i + 1
        print(f"  [{name} rank {rank}] done  (avg_logprob={avg_lp:.4f})" if avg_lp is not None else f"  [{name} rank {rank}] done")
        calls.append({"step": f"generate_top3_rank{rank}", "output": sql})
        sql_candidates.append({
            "rank": rank,
            "raw_sql": sql,
            "sql": sql,
            "avg_token_logprob": avg_lp,
            "min_token_logprob": min_lp,
            "std_token_logprob": stat.get("std_token_logprob"),
            "sql_token_count": stat.get("token_count"),
        })

    return {
        "name": name,
        "exploration": exploration,
        "clarifying_question": "",
        "user_answer": "",
        "sql_candidates": sql_candidates,
        "calls": calls,
    }


def run_pipeline_interpret(
    *,
    name: str,
    query: str,
    schema: str,
    col_text: str,
    kb_text: str,
) -> dict:
    _lp_ctx["step_pipeline"] = name

    print(f"\n  [{name} / STEP 1] Exploring schema + query ...")
    exploration = step1_explore(query, schema, col_text, kb_text)
    print(f"\n  [{name}] Agent exploration:")
    print(f"  {exploration}")
    calls = [{"step": "explore", "output": exploration}]

    print(f"\n  [{name} / STEP 2] Generating up to {INTERPRET_MAX} interpretations ...")
    ambiguities, interpretations, raw_interpret = step_interpret(
        query=query,
        schema=schema,
        col_text=col_text,
        kb_text=kb_text,
        exploration=exploration,
        max_n=INTERPRET_MAX,
    )
    print(f"\n  [{name}] Ambiguities found ({len(ambiguities)}):")
    for a in ambiguities:
        print(f"    - {a}")
    print(f"\n  [{name}] Interpretations:")
    for interp in interpretations:
        assignments = interp.get("assignments", {})
        assign_str = "  |  ".join(f"{k}: {v}" for k, v in assignments.items()) if assignments else ""
        if assign_str:
            print(f"    [{interp['id']}] {assign_str}")
        print(f"         {interp['interpretation_text'][:120]}{'...' if len(interp['interpretation_text']) > 120 else ''}")
    calls.append({"step": "interpret", "output": raw_interpret})

    sql_candidates = []
    for interp in interpretations:
        print(f"\n  [{name} / STEP 3 interp {interp['id']}] Generating SQL (temp=0) ...", end=" ", flush=True)
        sql, sql_stats = step3_generate_sample(
            query=query,
            schema=schema,
            col_text=col_text,
            kb_text=kb_text,
            exploration=exploration,
            interpretation_text=interp["interpretation_text"],
            step_name=f"{name.lower()}_generate_interp{interp['id']}",
            temperature=0,
        )
        sql = _strip_markdown_fence(sql)
        avg_lp = sql_stats[0]["avg_token_logprob"] if sql_stats else None
        min_lp = sql_stats[0]["min_token_logprob"] if sql_stats else None
        std_lp = sql_stats[0].get("std_token_logprob") if sql_stats else None
        token_count = sql_stats[0].get("token_count") if sql_stats else None
        print(f"done  (avg_logprob={avg_lp:.4f})" if avg_lp is not None else "done")
        calls.append({"step": f"generate_interp{interp['id']}", "output": sql})
        sql_candidates.append({
            "rank": interp["id"],
            "interpretation_id": interp["id"],
            "interpretation_text": interp["interpretation_text"],
            "assignments": interp.get("assignments", {}),
            "raw_sql": sql,
            "sql": sql,
            "avg_token_logprob": avg_lp,
            "min_token_logprob": min_lp,
            "std_token_logprob": std_lp,
            "sql_token_count": token_count,
        })

    return {
        "name": name,
        "exploration": exploration,
        "ambiguities_found": ambiguities,
        "interpretations": interpretations,
        "clarifying_question": "",
        "user_answer": "",
        "sql_candidates": sql_candidates,
        "calls": calls,
    }


def summarize_results(results: list[dict]) -> dict:
    total = len(results)
    exact = sum(1 for r in results if r.get("exact_match"))
    pred_exec_errors = sum(1 for r in results if r.get("predicted_sql_error"))
    gt_exec_errors = sum(1 for r in results if r.get("ground_truth_sql_error"))
    setup_errors = sum(1 for r in results if r.get("error"))
    by_type: dict[str, dict] = {}
    for r in results:
        task_type = r.get("task_type", "unknown")
        bucket = by_type.setdefault(task_type, {"total": 0, "exact": 0, "pred_exec_errors": 0})
        bucket["total"] += 1
        bucket["exact"] += 1 if r.get("exact_match") else 0
        bucket["pred_exec_errors"] += 1 if r.get("predicted_sql_error") else 0
    by_pipeline: dict[str, dict] = {}
    by_pipeline_oracle: dict[str, dict] = {}
    for r in results:
        cms = r.get("candidate_matches", [])
        for cm in cms:
            key = f"{cm['pipeline']}_rank{cm['rank']}"
            b = by_pipeline.setdefault(key, {
                "pipeline": cm["pipeline"], "rank": cm["rank"],
                "total": 0, "exact": 0, "exec_errors": 0,
            })
            b["total"] += 1
            b["exact"] += 1 if cm["exact_match"] else 0
            b["exec_errors"] += 1 if cm["predicted_sql_error"] else 0
        by_pipeline_name: dict[str, list] = {}
        for cm in cms:
            by_pipeline_name.setdefault(cm["pipeline"], []).append(cm)
        for pipeline_name, pipeline_cms in by_pipeline_name.items():
            ob = by_pipeline_oracle.setdefault(pipeline_name, {"total": 0, "oracle_exact": 0})
            ob["total"] += 1
            if any(cm["exact_match"] for cm in pipeline_cms):
                ob["oracle_exact"] += 1
    return {
        "total": total,
        "exact": exact,
        "accuracy": exact / total * 100 if total else 0,
        "pred_exec_errors": pred_exec_errors,
        "gt_exec_errors": gt_exec_errors,
        "setup_errors": setup_errors,
        "by_type": by_type,
        "by_pipeline": by_pipeline,
        "by_pipeline_oracle": by_pipeline_oracle,
    }
