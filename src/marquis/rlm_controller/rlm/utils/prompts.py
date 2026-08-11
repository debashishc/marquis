"""Prompt contracts for the MARQUIS RLM controller."""

from __future__ import annotations

DEFAULT_QUERY = "Please read through the context and answer any queries or respond to any instructions contained within it."

DEFAULT_PACING = (
    "PACING:\n"
    "  - Prefer one focused tool call per iteration.\n"
    "  - Gather enough evidence before judging.\n"
    "  - Judge selected facts before writing the final report."
)

REPL_SYSTEM_PROMPT = """You answer queries using an interactive Python REPL, called iteratively until you submit a final answer.

THINK-ACT-OBSERVE LOOP:
  Each iteration: THINK (brief reasoning), ACT (one code block), OBSERVE the output.
  THINK phase: READ the memory snapshot below -- it shows your findings (global knowledge) and per-video facts. Base your next action on what you ALREADY KNOW, not assumptions.

{pacing}

ENVIRONMENT:
  - context['task'], context['video_pool'], context['tools'] are read-only.
  - `memory` is a persistent dict (survives compaction).
  - Tools are pre-loaded as plain Python functions; call them directly.

FORMAT: THINK (2-4 sentences), then ONE ```repl``` code block (1-5 lines, ONE tool call). NO for-loops over videos.

FINAL ANSWER: report = write_report(memory['selected_facts']), then FINAL_VAR(report) outside the code block.
"""

ROOT_THINK_PROMPT = """TASK: {query_text}

CURRENT FINDINGS:
{findings_str}

FACT TABLE SUMMARY:
{fact_summary}

VIDEO STATUS:
{video_status}

You are the analytical brain. Based on all facts collected so far:

1. NEW_FINDINGS: List any new high-level findings (one sentence each) not already in CURRENT FINDINGS. If a new fact CONTRADICTS an existing finding, say CONFLICT: <existing> vs <new>.

2. UPDATED_FINDINGS: Output the complete updated findings list (old + new, deduplicated). One finding per line, prefixed with `- `.

3. NEXT_STEPS: What should the agent do next? Be specific: which video, which tool, which question.

Be concise.
"""

ROOT_JUDGE_PROMPT = """TASK: {query_text}

FINDINGS (root's current understanding):
{findings_str}

FACT TABLE ({n} facts):
{fact_lines}

You are a strict quality judge. Review ALL facts above for the task.

1. ITEM REVIEW: For each fact (F#0, F#1, ...), give a verdict.
   BE CONSERVATIVE -- only REMOVE if clearly irrelevant or duplicate. When in doubt, KEEP.
     KEEP    -- useful, specific, or even mildly relevant (default)
     REMOVE  -- clearly irrelevant or duplicate of another listed fact
     REWRITE -- needs more detail or has a missing timestamp (flag, do NOT drop)
   Format: F#0: KEEP / F#3: REMOVE (dup of F#1) / F#5: REWRITE (missing timestamp)

2. SELECTED: Pick the 10-40 BEST facts for a comprehensive report (prefer MORE coverage). List their IDs: SELECTED: F#0, F#2, F#7, ...

3. MISSING TIMESTAMPS: List facts that are useful but lack timestamps; suggest video_qa queries to resolve them.

4. GAPS: What information is still missing for a thorough report?

5. READY: Can we write a good report now? (yes / no / almost)

Be specific and concise.
"""

BEHAVIOR_JUDGE_PROMPT = """You are evaluating an AI agent's performance on iteration {iteration}/{max_iter}.
TASK:    {query}
MEMORY STATE BEFORE: {mem_before}
THINK:   {think_text}
ACT:     {code}
OBSERVE: {observe}
MEMORY STATE AFTER:  {mem_after}

Rate each dimension 1-5 with ONE sentence justification.

## Core dimensions:
1. Reasoning (1-5):   Did THINK show sound reasoning based on memory?
2. Action (1-5):      Was the chosen action relevant and logical?
3. Granularity (1-5): One focused step, or too much at once?
4. Progress (1-5):    Did this iteration meaningfully advance the task?

## Efficiency breakdown (5 sub-scores):
5a. Eff_Redundancy (1-5)         -- avoided repeating a tool call?
5b. Eff_Think_Conciseness (1-5) -- THINK tight and non-repetitive?
5c. Eff_Code_Minimality (1-5)   -- minimal code for its purpose?
5d. Eff_Output_Waste (1-5)      -- avoided producing useless output?
5e. Eff_Tool_Choice (1-5)       -- most cost-effective tool for this sub-goal?

Format EXACTLY: one line per dimension as `Name: <score> -- <reason>`, then `TOTAL: <sum>/45`.
"""

USER_PROMPT = """Continue the MARQUIS RLM THINK-ACT-OBSERVE loop for the original query: "{query}".

Use the REPL namespace, memory bank, and available MARQUIS tools to gather or select evidence. Your next action:"""


def build_system_prompt(pacing: str | None = None) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": REPL_SYSTEM_PROMPT.format(pacing=pacing or DEFAULT_PACING),
        },
    ]


def build_system_prompt_vlm(pacing: str | None = None) -> list[dict[str, str]]:
    return build_system_prompt(pacing=pacing)


build_system_prompt_v2 = build_system_prompt_vlm


def build_root_think_prompt(
    *,
    query_text: str,
    findings_str: str,
    fact_summary: str,
    video_status: str,
) -> str:
    return ROOT_THINK_PROMPT.format(
        query_text=query_text,
        findings_str=findings_str,
        fact_summary=fact_summary,
        video_status=video_status,
    )


def build_root_judge_prompt(
    *,
    query_text: str,
    findings_str: str,
    fact_lines: str,
    n: int,
) -> str:
    return ROOT_JUDGE_PROMPT.format(
        query_text=query_text,
        findings_str=findings_str,
        fact_lines=fact_lines,
        n=n,
    )


def build_behavior_judge_prompt(
    *,
    iteration: int,
    max_iter: int,
    query: str,
    mem_before: str,
    think_text: str,
    code: str,
    observe: str,
    mem_after: str,
) -> str:
    return BEHAVIOR_JUDGE_PROMPT.format(
        iteration=iteration,
        max_iter=max_iter,
        query=query,
        mem_before=mem_before,
        think_text=think_text,
        code=code,
        observe=observe,
        mem_after=mem_after,
    )


def next_action_prompt(
    query: str, iteration: int = 0, final_answer: bool = False
) -> dict[str, str]:
    if final_answer:
        return {
            "role": "user",
            "content": "Based on memory['selected_facts'], produce the final cited report using write_report and FINAL_VAR.",
        }
    if iteration == 0:
        safeguard = (
            "You have not inspected the REPL context or memory yet. Start by calling memory_summary() "
            "or inspecting context before gathering evidence.\n\n"
        )
        return {"role": "user", "content": safeguard + USER_PROMPT.format(query=query)}
    return {
        "role": "user",
        "content": "The previous messages contain your prior REPL observations. "
        + USER_PROMPT.format(query=query),
    }
