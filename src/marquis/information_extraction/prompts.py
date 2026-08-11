"""Prompts for the information-extraction branch.

Two kinds live here:
  * Step-1 extraction (general notes / query claims) and Step-1.5 scoring use the
    prompt *builders* defined at the bottom of this module.
  * The QA runners use the string templates below (``.format(...)``).

Generic LLM/JSON helpers (``_strict_json_tail``, ``_retry_wrapper`` …) stay in
``marquis.common.prompts``; we import them here.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from marquis.common.prompts import (
    _format_optional_json_block,
    _json_dump,
    _retry_wrapper,
    _strict_json_tail,
)

# qa-decompose: rewrite a query into searchable research questions.
QA_DECOMPOSE_PROMPT = """
You are a research decomposition specialist. Your task is to take a user's query and break it down into an exhaustive set of searchable research questions — complete questions that could be used to retrieve all the information needed to fully answer the original query.

You will receive the following inputs:
- Title: {title}
- Language: {language}
- Persona: {persona_title}
- Background: {background}
- Query: {query}

Decomposition Rules:

1. Coverage: Extract every distinct piece of information the user is asking for. Do not merge separate information needs into one question. If the query asks for multiple related but distinct data points, each one should become its own question.

2. Granularity: Each question should target ONE specific, retrievable piece of information. Prefer atomic questions over compound ones.

3. Implicit needs: Go beyond what is explicitly stated. Based on the background and persona_title, infer what additional information the user would likely need but did not explicitly ask for. Consider what a professional in that role would typically require to produce complete, high-quality work on this topic.

4. Search-friendly format: Each sub-query must be written as a concise, well-formed question that could plausibly be entered into a search engine or research database.

5. Context anchoring: Each question should include enough context (e.g., specific names, dates, locations, technical terms) to be independently searchable without ambiguity.

6. Source-awareness: If the user requests source information or credibility indicators, generate questions specifically targeting official sources, methodologies, and data provenance.

7. Dimensional expansion: For each core information need identified, consider whether the user would benefit from additional perspectives or breakdowns. Ask yourself: can this information be meaningfully decomposed further by time, place, category, cause, mechanism, comparison, or any other axis that is natural and relevant to the topic? Only expand along dimensions that genuinely add value given the query's subject matter and the user's background.

8. No redundancy: Each question must be meaningfully distinct. Do not produce near-duplicates that would return the same search results.

9. Language: Always generate questions in English, regardless of the language field in the input.

10. Quantity: Generate between 10 and 25 questions. Focus on quality and relevance over quantity.

11. Avoid mechanical repetition: Do not mechanically prepend the full topic title to every question. Each question should contain only the context necessary for an effective search.

12. Focus on information needs: Focus on the specific information being sought rather than repeating the topic name unnecessarily.

Return ONLY a JSON array of strings. No explanation, no markdown, and no code blocks.

For example, given a query about the 2025 Canadian federal election asking for seat counts and vote shares, good questions would be:

[
"What was the total number of seats won by each political party in the 2025 Canadian federal election?",
"What percentage of the national popular vote did each major party receive in the 2025 Canadian federal election?",
"How many seats did each party gain or lose compared with the 2021 Canadian federal election?",
"What official datasets published by Elections Canada contain vote totals and seat counts for the 2025 federal election?",
"What demographic voting patterns were observed in the 2025 Canadian federal election?"
]

NOT:

[
"What happened in the 2025 Canadian federal election?",
"What were the results of the 2025 Canadian federal election?",
"What information is available about the 2025 Canadian federal election?"
]

JSON array:
"""

# qa-answer / qa: single-video question answering ({question}).
QA_ASK_PROMPT = """
        Question:
        {question}

        Answer the question using the video and transcript.

        The question may carry background qualifiers — dates, locations, nationality, operator/organization names — that situate where it comes from. Treat these as context, NOT as preconditions you must verify against the video. A single clip usually cannot prove the date, place, or nationality, so do NOT answer "I don't know" merely because the video does not confirm those qualifiers; report what the video actually shows about the *subject* of the question.

        Answer concisely and to the best of your abilities. If you can answer only part of the question, answer that part (Ex. if asked about Liberal and Conservative vote counts but only the Liberal counts are shown, return those). Only return "I don't know" if the video and transcript show nothing relevant to the subject of the question. Never say the event hasnt taken place.

        Return ONLY the final factual answer.
        """

# qa-answer / qa: merge per-video answers into one ({subquery}, {valid_answers}).
QA_COMBINE_PROMPT = """
        You are given extracted answers from videos. These answers are factual and must be used.

        Question: {subquery}

        Extracted Answers (treat as ground truth):
        {valid_answers}

        Combine them into a single answer.

        Do NOT use prior knowledge.
        Do NOT say the event has not taken place.
        Only use the provided answers.

        If you recieve conflicting information, make a best guess NOT on prior knowlege but based on the nature of the question (Ex. for a question about how many seats a party has one, its reasonable to assume the largest number is the most recent seat count)

        Return only the final answer. You might not be able to answer it fully, and that's okay. Answer what you can and then say specifically what information is unknown.
        """

# qa (iterative): propose one follow-up question ({context}).
QA_FOLLOWUP_PROMPT = """
You are refining a research question based on prior answers.

Context:
{context}

Generate ONE new question that:
- extracts new information not yet covered
- is more specific or differently framed

If no meaningful new question can be formed, output:
NONE

Return ONLY the question or NONE.
"""


# ===========================================================================
# Step-1.5 scoring: qwen support-probability prompts
# ===========================================================================

_ANSWER_TAG_RE = re.compile(r"<answer>\s*([0-9]*\.?[0-9]+)\s*</answer>", re.IGNORECASE)


def _format_timestamp(timestamp: Sequence[float] | None) -> str:
    if timestamp is None:
        return "null"
    if len(timestamp) != 2:
        raise ValueError("timestamp must have exactly two values")
    return _json_dump([float(timestamp[0]), float(timestamp[1])])


QWEN_SCORE_INSTRUCTION = """
To help you make more accurate and consistent judgments, here is an expanded explanation of how to interpret and assign support percentages. These examples are designed to cover a range of real-world cases you may encounter in the annotation task.
100% - /Full and unambiguous support:
The video clearly shows the exact event described in the claim. There is no need for guessing or interpretation.
80-100% - Almost complete support:
The main content in the claim is shown, but there may be minor ambiguity in location, identity, or completeness. The overall claims are supported by the video.
60-80% - Strong partial support:
The video strongly suggests the claim is true, but some critical details may be missing, obscured, or ambiguous, limiting the ability to confirm the claim with certainty. The video gives strong but not definitive support.
40-60% - Moderate partial support:
There is some alignment with the claim, but large portions are either missing, unclear, or open to interpretation. While the footage may point in the same general direction as the claim, it lacks the clarity or completeness needed for confident verification.
20-40% - Minimal weak support:
There are small visual or audio cues that could hint at the claim, but they are insufficient to be confident.
0-20% - Very weak or speculative support:
There may be the slightest indirect reference, such as a related object or setting, but nothing concrete happens.
0% - No support or contradiction:
The video does not relate to the claim at all, or it directly shows something opposite.
""".strip()

QWEN_SCORE_PROMPT = """
Based on the provided video and text, evaluate the probability that the text is true.
Your answer must be a decimal number between 0 and 1, and you must strictly follow the format below:
<answer>probability_value</answer>
Where probability_value is the result you calculate.
The text to evaluate is:

{text}
""".strip()


def prompt_qwen_score(text: str) -> str:
    return f"{QWEN_SCORE_INSTRUCTION}\n\n{QWEN_SCORE_PROMPT.format(text=text)}"


def prompt_qwen_score_retry(text: str) -> str:
    return (
        prompt_qwen_score(text)
        + "\n\nYour previous answer was invalid. Reply with only one decimal in the exact form "
        + "<answer>0.73</answer>."
    )


def parse_qwen_score_answer(raw_text: str) -> float | None:
    if raw_text is None:
        return None
    text = str(raw_text).strip()
    match = _ANSWER_TAG_RE.search(text)
    if not match:
        return None
    try:
        value = float(match.group(1))
    except Exception:
        return None
    if 0.0 <= value <= 1.0:
        return value
    return None


# ===========================================================================
# Step-1 extraction: general notes / query claims prompt builders
# ===========================================================================


def prompt_general_notes(
    *,
    topic: str | None = None,
    video_id: str,
    include_topic: bool = True,
    timestamp: Sequence[float] | None = None,
) -> str:
    expected_shape = {
        "notes": [
            {
                "text": "...",
                "modality": "visual",
                "timestamp": [0.0, 1.0],
            }
        ]
    }
    parts = [
        "You are extracting observation notes directly from a raw video.",
        "",
        "Video context:",
        f"- video_id: {video_id}",
        f"- timestamp_span: {_format_timestamp(timestamp)}",
        "",
        "Rules:",
        "- Record only directly observable content.",
        "- No inference, speculation, causality, or cross-video synthesis.",
        "- Capture OCR (on-screen text), events, and visible scene details.",
        "- One note per atomic visible, audible, or textual fact.",
        "- Use modality `visual` for scene content, `ocr` for on-screen text, and `audio` for transcript or speech.",
        "- Use the provided timestamp span for each note when no narrower timestamp is available.",
        "- If there is no usable evidence, return an empty notes list.",
    ]
    if include_topic:
        parts.insert(4, f"- topic: {topic or ''}")
    return "\n".join(parts) + _strict_json_tail(expected_shape)


def prompt_general_notes_retry(
    *,
    topic: str | None = None,
    video_id: str,
    include_topic: bool = True,
    timestamp: Sequence[float] | None = None,
) -> str:
    return _retry_wrapper(
        prompt_general_notes(
            topic=topic,
            video_id=video_id,
            include_topic=include_topic,
            timestamp=timestamp,
        ),
        {
            "notes": [
                {
                    "text": "...",
                    "modality": "visual",
                    "timestamp": [0.0, 1.0],
                }
            ]
        },
    )


def prompt_query_claims(
    *,
    query_id: str,
    query: str,
    persona_title: str,
    background: str,
    topic: str,
    video_id: str,
    per_video_target: int = 5,
) -> str:
    expected_shape = {
        "claims": [
            {
                "claim": "...",
                "confidence": 0.85,
                "evidence": "...",
                "source": "video_visual",
                "timestamp": [0.0, 1.0],
            }
        ]
    }
    parts = [
        "You are extracting query-relevant claims directly from a raw video.",
        "",
        "Query context:",
        f"- query_id: {query_id}",
        f"- topic: {topic}",
        f"- persona_title: {persona_title}",
        f"- background: {background}",
        f"- query: {query}",
        f"- video_id: {video_id}",
        "",
        "Rules (read these BEFORE the query context):",
        f"- HARD FLOOR: If the video shows ANY content related to the topic '{topic}'",
        "you MUST output at least 3 claims",
        f"- Extract up to {per_video_target} claims from this video.",
    ]
    return "\n".join(parts) + _strict_json_tail(expected_shape)


def prompt_query_claims_retry(
    *,
    query_id: str,
    query: str,
    persona_title: str,
    background: str,
    topic: str,
    video_id: str,
    per_video_target: int = 5,
) -> str:
    return _retry_wrapper(
        prompt_query_claims(
            query_id=query_id,
            query=query,
            persona_title=persona_title,
            background=background,
            topic=topic,
            video_id=video_id,
            per_video_target=per_video_target,
        ),
        {
            "claims": [
                {
                    "claim": "...",
                    "confidence": 0.85,
                    "evidence": "...",
                    "source": "video_visual",
                    "timestamp": [0.0, 1.0],
                }
            ]
        },
    )


def prompt_query_claims_expanded(
    *,
    query_id: str,
    query: str,
    persona_title: str,
    background: str,
    topic: str,
    video_id: str,
    sub_queries: Sequence[str],
    per_video_target: int = 5,
) -> str:
    expected_shape = {
        "claims": [
            {
                "claim": "...",
                "confidence": 0.85,
                "evidence": "...",
                "source": "video_visual",
                "timestamp": [0.0, 1.0],
            }
        ]
    }
    parts = [
        "You are extracting query-relevant claims directly from a raw video.",
        "",
        "Query context:",
        f"- query_id: {query_id}",
        f"- topic: {topic}",
        f"- persona_title: {persona_title}",
        f"- background: {background}",
        f"- query: {query}",
        f"- video_id: {video_id}",
        "",
        _format_optional_json_block("Coverage guidance subqueries", list(sub_queries)),
        "",
        "Rules:",
        f"- Extract up to {per_video_target} claims from this video.",
        "- Use subqueries only as coverage guidance, not as evidence.",
        "- Do not mention subqueries in the output.",
        "- Prefer claims that directly answer the query (or its subquery facets); assign these confidence >= 0.6.",
        "- If direct evidence is sparse, also extract claims that are on-topic and plausibly useful context for the persona/background, even if they do not directly answer the query; assign these confidence <= 0.5.",
        "- Every claim, regardless of tier, must be directly supported by observable video content (no inference or speculation).",
        "- Do not emit unsupported claims even if a subquery suggests them.",
        "- Avoid duplicates and paraphrases.",
        "- Return an empty claims list only if the video has no observable content related to the topic at all.",
        "- source must be one of `video_visual`, `video_text`, or `transcript`.",
        "- timestamp must be [start, end].",
        "- confidence must be a float between 0 and 1, encoding the relevance/support tier described above.",
    ]
    return "\n".join(parts) + _strict_json_tail(expected_shape)


def prompt_query_claims_expanded_retry(
    *,
    query_id: str,
    query: str,
    persona_title: str,
    background: str,
    topic: str,
    video_id: str,
    sub_queries: Sequence[str],
    per_video_target: int = 5,
) -> str:
    return _retry_wrapper(
        prompt_query_claims_expanded(
            query_id=query_id,
            query=query,
            persona_title=persona_title,
            background=background,
            topic=topic,
            video_id=video_id,
            sub_queries=sub_queries,
            per_video_target=per_video_target,
        ),
        {
            "claims": [
                {
                    "claim": "...",
                    "confidence": 0.85,
                    "evidence": "...",
                    "source": "video_visual",
                    "timestamp": [0.0, 1.0],
                }
            ]
        },
    )
