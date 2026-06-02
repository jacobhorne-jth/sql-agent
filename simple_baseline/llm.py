# LLM client, logprob tracking, and all agent step functions (explore, clarify, generate, repair, review)

import json
import re
import statistics

SQL_CANDIDATE_COUNT = 3
MAX_REPAIR_ATTEMPTS = 2

_lp_ctx: dict = {
    "instance_id": None, "task_type": None, "file": None,
    "call_idx": 0, "last_sql_stats": [],
    "token_lp_file": None,
    "step_pipeline": None,
}

SQL_START_RE = re.compile(
    r"^\s*(?:WITH|SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|TRUNCATE|MERGE|DO)\b",
    re.IGNORECASE | re.DOTALL,
)


# ---------------------------------------------------------------------------
# Logprobs
# ---------------------------------------------------------------------------

def init_logprobs_jsonl(path: str, append: bool = False) -> None:
    _lp_ctx["file"] = open(path, "a" if append else "w")


def init_token_logprobs_jsonl(path: str, append: bool = False) -> None:
    _lp_ctx["token_lp_file"] = open(path, "a" if append else "w")


def _json_string_value(raw_json_string: str) -> str:
    try:
        return json.loads(f'"{raw_json_string}"')
    except json.JSONDecodeError:
        return raw_json_string


def extract_sql_spans(text: str) -> list[dict]:
    spans: list[dict] = []

    for match in re.finditer(r'"(?:sql|query)"\s*:\s*"((?:\\.|[^"\\])*)"', text, flags=re.DOTALL):
        raw_sql_payload = match.group(1)
        sql = _json_string_value(raw_sql_payload).strip()
        if sql and SQL_START_RE.match(sql):
            spans.append({"sql": sql, "start": match.start(1), "end": match.end(1)})

    for match in re.finditer(r"```sql\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL):
        sql = match.group(1).strip()
        if sql and SQL_START_RE.match(sql):
            spans.append({"sql": sql, "start": match.start(1), "end": match.end(1)})

    stripped = _strip_markdown_fence(text)
    if SQL_START_RE.match(stripped):
        start = text.find(stripped)
        if start == -1:
            start = 0
        spans.append({"sql": stripped, "start": start, "end": start + len(stripped)})

    deduped = []
    seen = set()
    for span in spans:
        key = (span["start"], span["end"], span["sql"])
        if key not in seen:
            seen.add(key)
            deduped.append(span)
    return deduped


def sql_logprob_stats(content: str, lp_content: list) -> list[dict]:
    spans = extract_sql_spans(content)
    if not spans or not lp_content:
        return []

    token_ranges = []
    pos = 0
    for token_info in lp_content:
        token = token_info.token
        start = pos
        end = start + len(token)
        token_ranges.append((start, end, token_info))
        pos = end

    stats = []
    for span in spans:
        span_tokens = [
            {"token": token_info.token, "logprob": token_info.logprob}
            for start, end, token_info in token_ranges
            if token_info.logprob is not None and start < span["end"] and end > span["start"]
        ]
        selected = [t["logprob"] for t in span_tokens]
        stats.append({
            "sql": span["sql"],
            "avg_token_logprob": sum(selected) / len(selected) if selected else None,
            "min_token_logprob": min(selected) if selected else None,
            "std_token_logprob": statistics.stdev(selected) if len(selected) > 1 else None,
            "token_count": len(selected),
            "tokens": span_tokens,
        })
    return stats


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_llm(messages: list[dict], step: str = "unknown", temperature: float = 0) -> str:
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=temperature,
        logprobs=True,
        max_tokens=2000,
    )
    content = response.choices[0].message.content.strip()
    lp_content = response.choices[0].logprobs.content if response.choices[0].logprobs else []
    stats = sql_logprob_stats(content, lp_content)
    _lp_ctx["last_sql_stats"] = stats

    if _lp_ctx["file"]:
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
            "sql_logprob_stats": [{k: v for k, v in s.items() if k != "tokens"} for s in stats],
            "token_logprobs": [{"token": t.token, "logprob": t.logprob} for t in lp_content],
        }) + "\n")
        _lp_ctx["file"].flush()

    if _lp_ctx["token_lp_file"] and stats:
        for span_idx, stat in enumerate(stats):
            tokens = stat.get("tokens", [])
            if tokens:
                _lp_ctx["token_lp_file"].write(json.dumps({
                    "instance_id": _lp_ctx["instance_id"] or "",
                    "pipeline": _lp_ctx.get("step_pipeline") or "",
                    "step": step,
                    "span_idx": span_idx,
                    "sql_tokens": tokens,
                }) + "\n")
        _lp_ctx["token_lp_file"].flush()

    _lp_ctx["call_idx"] += 1
    return content


def _strip_markdown_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|sql)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_sql_candidates(raw_output: str) -> list[str]:
    text = _strip_markdown_fence(raw_output)
    candidates: list[str] = []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            parsed = (
                parsed.get("sql_candidates")
                or parsed.get("queries")
                or parsed.get("sql")
                or parsed.get("candidates")
            )
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    sql = item.get("sql") or item.get("query")
                else:
                    sql = item
                if isinstance(sql, str) and sql.strip():
                    candidates.append(_strip_markdown_fence(sql))
    except json.JSONDecodeError:
        pass

    if not candidates:
        fenced = re.findall(r"```sql\s*(.*?)```", raw_output, flags=re.IGNORECASE | re.DOTALL)
        candidates = [_strip_markdown_fence(sql) for sql in fenced if sql.strip()]

    if not candidates:
        for match in re.finditer(r'"(?:sql|query)"\s*:\s*"((?:\\.|[^"\\])*)"', raw_output, flags=re.DOTALL):
            try:
                sql = json.loads(f'"{match.group(1)}"').strip()
            except json.JSONDecodeError:
                sql = match.group(1).strip()
            if sql and SQL_START_RE.match(sql):
                candidates.append(sql)

    if not candidates:
        parts = re.split(r"(?im)^\s*(?:SQL\s*)?(?:Option|Candidate|Query)?\s*[1-3]\s*[:.)-]\s*", text)
        candidates = [p.strip() for p in parts if p.strip()]

    if not candidates:
        candidates = [text]

    candidates = candidates[:SQL_CANDIDATE_COUNT]
    while len(candidates) < SQL_CANDIDATE_COUNT and candidates:
        candidates.append(candidates[-1])
    return candidates


# ---------------------------------------------------------------------------
# Human / simulator interaction
# ---------------------------------------------------------------------------

def ask_human_user(question: str, ambiguity_data: dict | None = None) -> str:
    print("\n  [HUMAN USER] Please answer the analyst's question.")
    print(f"  [AGENT ASKS] {question}")
    if ambiguity_data:
        print("  [REFERENCE] Hidden ambiguity hints, shown to help you act as the user:")
        print(indent_text(format_human_reference(ambiguity_data), prefix="    "))
        print("  Please answer in natural user language, not as SQL.")
    print("\n  Type your answer. Press Enter on an empty line when done:")

    lines = []
    while True:
        try:
            line = input("  > ")
        except EOFError:
            break
        if not line and lines:
            break
        if not line:
            print("  Please enter an answer, or press Ctrl-D to cancel.")
            continue
        lines.append(line)

    answer = "\n".join(lines).strip()
    if not answer:
        raise RuntimeError("No human clarification answer was provided.")
    return answer


def indent_text(text: str, prefix: str = "  ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def format_human_reference(ambiguity_data: dict) -> str:
    lines = []
    user_ambiguity = ambiguity_data.get("user_query_ambiguity", {})
    for group_name in ("critical_ambiguity", "non_critical_ambiguity"):
        items = user_ambiguity.get(group_name, [])
        if items:
            label = group_name.replace("_", " ")
            lines.append(f"{label}:")
            for item in items:
                term = item.get("term", "unknown term")
                ambiguity_type = item.get("type", "unknown type")
                lines.append(f"  - {term} ({ambiguity_type})")
    knowledge_items = ambiguity_data.get("knowledge_ambiguity", [])
    if knowledge_items:
        lines.append("knowledge hints:")
        for item in knowledge_items:
            term = item.get("term", "unknown knowledge")
            lines.append(f"  - {term}")
    return "\n".join(lines) if lines else "(no ambiguity hints available)"


def simulate_user_answer(query: str, question: str, ambiguity_data: dict) -> str:
    return call_llm([
        {
            "role": "system",
            "content": (
                "You are the benchmark user simulator for a Text-to-SQL task. Use the "
                "ambiguity metadata as the source of truth and answer the analyst's "
                "question with a structured clarification that gives the SQL agent all "
                "implementation-relevant intent details. Be concise, but include exact "
                "metric definitions, formulas, thresholds, output identifier choices, "
                "filters, joins, precision, grouping, ranking, and sorting whenever they "
                "appear in the metadata, even if the analyst asked a narrower question. "
                "It is acceptable to include SQL expressions or snippets from the metadata "
                "as formulas/criteria, but do not provide a complete final SQL query. "
                "If metadata contains multiple linked ambiguities, answer all linked "
                "ambiguities together."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original user query:\n{query}\n\n"
                f"Analyst question:\n{question}\n\n"
                f"Ambiguity metadata:\n{json.dumps(ambiguity_data, indent=2)}\n\n"
                "Answer in this format:\n"
                "Clarified intent:\n"
                "- ...\n"
                "Implementation details:\n"
                "- Metric/formula: ...\n"
                "- Threshold/filter: ...\n"
                "- Output fields/IDs: ...\n"
                "- Join/sort/precision/grouping details: ...\n\n"
                "Do not invent details that are absent from metadata."
            ),
        },
    ], step="clarify_answer")


# ---------------------------------------------------------------------------
# Agent steps
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# HERE_NEW
# CURRENT EXPLORE (LLM CALL 1)
#
# ---------------------------------------------------------------------------



def step1_explore(query: str, schema: str, col_text: str, kb_text: str) -> str:
    return call_llm([
        {
            "role": "system",
            "content": (
                "You are a SQL analyst. Given a database schema and a user query, "
                "identify which tables and columns are relevant, and flag every term "
                "in the query that is vague or needs clarification. Pay special "
                "attention to terms that affect formulas, thresholds, filters, counts, "
                "ranking, sorting, grouping, or whether the task is SELECT vs DDL/DML. "
                "Use exact table and column names from the schema only. Be concise."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Schema:\n{schema}\n\n"
                f"Column descriptions:\n{col_text}\n\n"
                f"Knowledge definitions:\n{kb_text}\n\n"
                f"User query:\n{query}\n\n"
                "List the relevant tables/columns, then list unresolved ambiguities. "
                "For each ambiguity, state why it matters for the SQL."
            ),
        },
    ], step="explore")


def step2_clarify(
    query: str,
    schema: str,
    exploration: str,
    ambiguity_data: dict,
) -> tuple[str, str]:
    question = call_llm([
        {
            "role": "system",
            "content": (
                "You are a SQL analyst. Ask ONE short clarifying question about "
                "the single most important ambiguity for producing correct SQL. "
                "Prefer a question that resolves formulas, thresholds, filters, "
                "counts, sorting/ranking, or DDL/DML details. If several terms are "
                "linked, ask one compact question that covers the linked definition."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Schema:\n{schema}\n\n"
                f"User query:\n{query}\n\n"
                f"Your analysis:\n{exploration}\n\n"
                "Ask one clarifying question. It should be specific enough that the "
                "answer can be used directly in SQL:"
            ),
        },
    ], step="clarify_question")

    answer = simulate_user_answer(query, question, ambiguity_data)
    # To switch to interactive human answers replace the line above with:
    # answer = ask_human_user(question, ambiguity_data=ambiguity_data)

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
    return call_llm([
        {
            "role": "system",
            "content": (
                "You are an expert PostgreSQL analyst. "
                "Write a single SQL query that answers the user's question. "
                "Use only tables and columns that appear in the provided schema, "
                "spelled exactly as they appear there. Do not invent columns, "
                "thresholds, filters, joins, or aliases. If the clarification gives "
                "a formula or threshold, preserve its meaning exactly while adapting "
                "table aliases to your query. For derived metrics, prefer a CTE so "
                "aggregates, medians, filters, and counts reuse the same expression. "
                "Use explicit JOIN ... ON ... clauses for foreign-key relationships; "
                "do not use a scalar subquery as a join key unless the schema guarantees "
                "the subquery returns exactly one row. Use only join predicates whose "
                "columns exist on both joined tables and are supported by the schema. "
                "Do not add extra equality predicates just because names seem related. "
                "When the user asks for an ID, use the table's primary identifier, "
                "registry/key/code column, not a tag/name/label column unless the user "
                "explicitly asks for a tag/name/label. "
                "Do not add filters that are not explicitly requested by the user, "
                "provided by visible knowledge, or clarified by the user/simulator. "
                "For aggregation, GROUP BY only the requested output grain and "
                "non-aggregated output dimensions; do not group by hidden row-level IDs "
                "or intermediate columns such as scan/archive/count identifiers unless "
                "the user explicitly asks for them. "
                "For vague count/filter words such as usable, valid, high-risk, or "
                "high-quality, use the clarified criterion; if it was not clarified "
                "but a related formula was clarified, use that formula rather than "
                "guessing an unrelated table. For DDL/DML tasks, return valid "
                "PostgreSQL DDL/DML, including RETURNING only when it is needed and "
                "valid for the statement. Before returning, silently check that every "
                "referenced table alias is unique and every referenced column exists "
                "in the schema. "
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
                "Important constraints:\n"
                "- Use exact schema names; do not create snake_case variants.\n"
                "- Join related tables with explicit JOIN ... ON ... clauses, not scalar subqueries.\n"
                "- Use only valid schema-supported join predicates; do not add extra join predicates on columns that do not exist on both sides.\n"
                "- For requested IDs, prefer registry/key/code primary identifiers over tag/name/label columns.\n"
                "- Do not add unrequested filters from plausible-looking tables.\n"
                "- GROUP BY only the requested output grain and visible non-aggregated output fields.\n"
                "- Reuse clarified formulas consistently for SELECT values, filters, counts, and ordering.\n"
                "- If the task modifies schema/data, produce one valid PostgreSQL DDL/DML statement.\n"
                "- Do not include markdown fences or explanatory prose.\n\n"
                "Write the SQL query:"
            ),
        },
    ], step="generate")


def step3_generate_top3(
    query: str,
    schema: str,
    col_text: str,
    kb_text: str,
    exploration: str,
    question: str | None = None,
    answer: str | None = None,
    step_name: str = "generate_top3",
) -> tuple[list[str], str]:
    if question and answer:
        clarification_text = f"Clarification:\n  Q: {question}\n  A: {answer}\n\n"
        clarification_rule = (
            "Use the clarification as authoritative context. If it gives a formula "
            "or threshold, preserve its meaning exactly while adapting table aliases."
        )
    else:
        clarification_text = (
            "Clarification:\n"
            "  No clarification was requested or provided in this pipeline. Resolve "
            "ambiguity only from the user query, schema, visible column descriptions, "
            "and visible knowledge definitions.\n\n"
        )
        clarification_rule = (
            "This pipeline must not assume hidden clarification details. Do not invent "
            "thresholds, filters, formulas, or output semantics that are not present in "
            "the user query, schema, visible column descriptions, or visible knowledge."
        )

    raw_output = call_llm([
        {
            "role": "system",
            "content": (
                "You are an expert PostgreSQL analyst. Generate the top 3 most likely "
                "SQL queries that could answer the user's question, ranked best to "
                "third-best. Each candidate must be a single PostgreSQL statement. "
                "Use only tables and columns that appear in the provided schema, "
                "spelled exactly as they appear there. Do not invent columns, joins, "
                "or aliases. Use explicit JOIN ... ON ... clauses and only join "
                "predicates supported by the schema. For derived metrics, prefer CTEs. "
                "GROUP BY only the requested output grain and non-aggregated output "
                "dimensions. "
                f"{clarification_rule} "
                "Return ONLY valid JSON in this exact shape: "
                "{\"sql_candidates\":[{\"rank\":1,\"sql\":\"...\"},"
                "{\"rank\":2,\"sql\":\"...\"},{\"rank\":3,\"sql\":\"...\"}]}."
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
                f"{clarification_text}"
                "Important constraints:\n"
                "- Use exact schema names; do not create snake_case variants.\n"
                "- Join related tables with explicit JOIN ... ON ... clauses, not scalar subqueries.\n"
                "- Use only valid schema-supported join predicates.\n"
                "- For requested IDs, prefer registry/key/code primary identifiers over tag/name/label columns.\n"
                "- Do not add unrequested filters from plausible-looking tables.\n"
                "- Reuse any explicit formulas consistently for SELECT values, filters, counts, and ordering.\n"
                "- If the task modifies schema/data, each candidate must be one valid PostgreSQL DDL/DML statement.\n\n"
                "Write the top 3 SQL candidates as JSON:"
            ),
        },
    ], step=step_name)
    return parse_sql_candidates(raw_output), raw_output

# ---------------------------------------------------------------------------
# HERE_NEW
# CURRENT 3ND CALL GENERATE
#
# ---------------------------------------------------------------------------


def step3_generate_sample(
    query: str,
    schema: str,
    col_text: str,
    kb_text: str,
    exploration: str,
    question: str | None = None,
    answer: str | None = None,
    step_name: str = "generate_sample",
    temperature: float = 0.7,
    interpretation_text: str | None = None,
) -> tuple[str, list]:
    if interpretation_text:
        clarification_text = f"Interpretation to implement:\n{interpretation_text}\n\n"
        clarification_rule = (
            "Implement this interpretation exactly. It specifies how every ambiguity "
            "in the query is resolved — treat it as the authoritative source of truth."
        )
    elif question and answer:
        clarification_text = f"Clarification:\n  Q: {question}\n  A: {answer}\n\n"
        clarification_rule = (
            "Use the clarification as authoritative context. If it gives a formula "
            "or threshold, preserve its meaning exactly while adapting table aliases."
        )
    else:
        clarification_text = (
            "Clarification:\n"
            "  No clarification was requested or provided in this pipeline. Resolve "
            "ambiguity only from the user query, schema, visible column descriptions, "
            "and visible knowledge definitions.\n\n"
        )
        clarification_rule = (
            "This pipeline must not assume hidden clarification details. Do not invent "
            "thresholds, filters, formulas, or output semantics that are not present in "
            "the user query, schema, visible column descriptions, or visible knowledge."
        )

    _lp_ctx["last_sql_stats"] = []
    content = call_llm([
        {
            "role": "system",
            "content": (
                "You are an expert PostgreSQL analyst. "
                "Write a single SQL query that answers the user's question. "
                "Use only tables and columns that appear in the provided schema, "
                "spelled exactly as they appear there. Do not invent columns, "
                "thresholds, filters, joins, or aliases. If the clarification gives "
                "a formula or threshold, preserve its meaning exactly while adapting "
                "table aliases to your query. For derived metrics, prefer a CTE so "
                "aggregates, medians, filters, and counts reuse the same expression. "
                "Use explicit JOIN ... ON ... clauses for foreign-key relationships. "
                "Use only join predicates whose columns exist on both joined tables. "
                "When the user asks for an ID, use the table's primary identifier. "
                "Do not add filters that are not explicitly requested or clarified. "
                "GROUP BY only the requested output grain and non-aggregated output dimensions. "
                f"{clarification_rule} "
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
                f"{clarification_text}"
                "Write the SQL query:"
            ),
        },
    ], step=step_name, temperature=temperature)
    return content, _lp_ctx["last_sql_stats"]


def parse_interpretations(raw: str, max_n: int) -> tuple[list[str], list[dict]]:
    text = _strip_markdown_fence(raw)
    try:
        parsed = json.loads(text)
        raw_ambiguities = parsed.get("ambiguities", [])
        # ambiguities can be a dict {term: {option_a: ..., option_b: ...}} or legacy list
        if isinstance(raw_ambiguities, dict):
            ambiguities = [
                f"{term}: {' / '.join(opts.values())}"
                for term, opts in raw_ambiguities.items()
            ]
        else:
            ambiguities = raw_ambiguities

        raw_interps = parsed.get("interpretations", [])
        interpretations = []
        for i, item in enumerate(raw_interps[:max_n]):
            interp_text = item.get("interpretation_text", "").strip()
            if interp_text:
                interpretations.append({
                    "id": item.get("id", i + 1),
                    "assignments": item.get("assignments", {}),
                    "interpretation_text": interp_text,
                })
        return ambiguities, interpretations
    except (json.JSONDecodeError, AttributeError):
        return [], [{"id": 1, "assignments": {}, "interpretation_text": text.strip()}]

# ---------------------------------------------------------------------------
# HERE_NEW
# CURRENT 2ND CALL INTERPRET
#
# ---------------------------------------------------------------------------



def step_interpret(
    query: str,
    schema: str,
    col_text: str,
    kb_text: str,
    exploration: str,
    max_n: int = 6,
) -> tuple[list[str], list[dict], str]:
    raw = call_llm([
        {
            "role": "system",
            "content": (
                f"You are an expert SQL analyst. Given a database schema, knowledge "
                f"definitions, and a user query with an earlier analysis, your job is to:\n"
                f"1. Identify all ambiguous terms or phrases in the user query that could "
                f"affect the generated SQL (e.g. vague filters, unclear metrics, ambiguous "
                f"column choices, multiple join paths, different GROUP BY columns).\n"
                f"2. For each ambiguity, list 2-3 concrete resolutions based only on the "
                f"schema, column descriptions, and knowledge definitions provided. Label "
                f"each option clearly (e.g. option_a, option_b).\n"
                f"3. Build an assignment table: for each interpretation, pick exactly one "
                f"option per ambiguity. COVERAGE RULE — every option you listed for every "
                f"ambiguity must appear in at least one interpretation's assignments. This "
                f"is mandatory. Produce between 2 and {max_n} interpretations, only as "
                f"many as the coverage rule requires. Do not pad beyond that.\n"
                f"4. Write a self-contained interpretation_text for each row of the "
                f"assignment table. It must state exactly what the SQL must compute, "
                f"filter, join, group by, and return — derived directly from the assignments.\n"
                f'Return ONLY valid JSON:\n'
                f'{{"ambiguities":{{"term":{{"option_a":"...","option_b":"..."}},...}},'
                f'"interpretations":[{{"id":1,"assignments":{{"term":"option_a",...}},'
                f'"interpretation_text":"..."}},...]}}'
            ),
        },
        {
            "role": "user",
            "content": (
                f"Schema:\n{schema}\n\n"
                f"Column descriptions:\n{col_text}\n\n"
                f"Knowledge definitions:\n{kb_text}\n\n"
                f"User query:\n{query}\n\n"
                f"Earlier analysis:\n{exploration}\n\n"
                f"Produce the assignment table and interpretations as JSON:"
            ),
        },
    ], step="interpret")
    ambiguities, interpretations = parse_interpretations(raw, max_n)
    return ambiguities, interpretations, raw


def repair_sql(
    query: str,
    schema: str,
    col_text: str,
    kb_text: str,
    exploration: str,
    question: str,
    answer: str,
    original_sql: str,
    failed_sql: str,
    error: str,
    attempt: int,
) -> str:
    return call_llm([
        {
            "role": "system",
            "content": (
                "You are an expert PostgreSQL analyst repairing a failed SQL query. "
                "Use only tables and columns from the provided schema, spelled exactly "
                "as they appear there. Preserve the user's intent and the clarification. "
                "Fix the concrete PostgreSQL error without changing the requested output. "
                "Use explicit JOIN ... ON ... clauses for table relationships. "
                "If the error says a column in a JOIN predicate does not exist, remove "
                "that invalid predicate and keep the valid schema-supported join. Do not "
                "rewrite formulas to fix a join-column error. "
                "Do not algebraically rewrite clarified formulas. Preserve formulas exactly "
                "except for necessary table aliases or PostgreSQL casts. If a previous SQL "
                "version preserved the formula better than the failed SQL, restore that formula. "
                "For formula/function errors, do not invent a new formula. Repair only the "
                "minimum syntax, alias, or cast issue while preserving function arity: "
                "POWER(x, y) must remain two-argument POWER(x, y), LOG(base, x) must remain "
                "two-argument LOG(base, x), and exponents such as 1.5 must not be removed. "
                "Return ONLY the repaired SQL with no explanation or markdown fences."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Schema:\n{schema}\n\n"
                f"Column descriptions:\n{col_text}\n\n"
                f"Knowledge definitions:\n{kb_text}\n\n"
                f"User query:\n{query}\n\n"
                f"Earlier analysis:\n{exploration}\n\n"
                f"Clarification:\n"
                f"  Q: {question}\n"
                f"  A: {answer}\n\n"
                "Locked clarification formula/source of truth:\n"
                f"{answer}\n\n"
                f"Original generated SQL:\n{original_sql}\n\n"
                f"Failed SQL:\n{failed_sql}\n\n"
                f"PostgreSQL error:\n{error}\n\n"
                "Repair the SQL with the smallest possible edit. Before returning, silently "
                "check that every alias is unique, every referenced column exists in the schema, "
                "and GROUP BY contains only non-aggregated output dimensions needed for the "
                "requested grain. For joins, every ON predicate must reference columns that exist "
                "on the two tables being joined; remove invalid extra predicates instead of "
                "inventing replacement columns. Remove filters that were not requested or "
                "clarified. For aggregate queries, remove hidden row-level identifiers from "
                "GROUP BY when they are not part of the requested output grain. Preserve "
                "clarified formulas exactly; do not move parentheses, exponents, multiplication "
                "factors, thresholds, or aggregate boundaries unless required to fix the concrete "
                "PostgreSQL error. If the failed SQL has changed the locked formula, restore the "
                "locked formula first, then apply aliases/casts."
            ),
        },
    ], step=f"repair_{attempt}")


def review_sql(
    query: str,
    schema: str,
    col_text: str,
    kb_text: str,
    exploration: str,
    question: str,
    answer: str,
    candidate_sql: str,
) -> str:
    return call_llm([
        {
            "role": "system",
            "content": (
                "You are an expert PostgreSQL SQL reviewer. Review the candidate SQL "
                "against the user query, schema, visible knowledge, and clarification. "
                "If the SQL is correct, return it unchanged. If it has any issue, return "
                "a corrected version. Use only tables and columns from the schema, spelled "
                "exactly as they appear there. Use explicit JOIN ... ON ... clauses for "
                "relationships. Use only join predicates whose columns exist on both joined "
                "tables and are supported by the schema; do not add extra equality predicates "
                "just because names seem related. Preserve clarified formulas, thresholds, "
                "filters, counts, and ordering. Do not algebraically rewrite clarified "
                "formulas; only adapt table aliases or add necessary PostgreSQL casts. Keep "
                "function arity unchanged: POWER(x, y) must remain two-argument POWER(x, y), "
                "LOG(base, x) must remain two-argument LOG(base, x), and exponents such as "
                "1.5 must not be removed. When the user asks for an ID, prefer "
                "registry/key/code primary identifiers over tag/name/label columns unless "
                "specifically requested otherwise. Do not add filters that are not explicitly "
                "requested or clarified. Check the output grain carefully: return one row per "
                "entity the user requested, and do not GROUP BY extra columns that are not "
                "needed as non-aggregated output dimensions. Return ONLY SQL with no "
                "explanation or markdown fences."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Schema:\n{schema}\n\n"
                f"Column descriptions:\n{col_text}\n\n"
                f"Knowledge definitions:\n{kb_text}\n\n"
                f"User query:\n{query}\n\n"
                f"Earlier analysis:\n{exploration}\n\n"
                f"Clarification:\n"
                f"  Q: {question}\n"
                f"  A: {answer}\n\n"
                "Locked clarification formula/source of truth:\n"
                f"{answer}\n\n"
                f"Candidate SQL:\n{candidate_sql}\n\n"
                "Return the reviewed SQL only. If you revise the SQL, preserve the locked "
                "formula exactly except for table aliases or casts."
            ),
        },
    ], step="review")
