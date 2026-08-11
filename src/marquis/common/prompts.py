#!/usr/bin/env python3
"""
Prompt builders and parsing helpers for the v1 note-taking/query-based pipelines.

This module freezes the prompt contracts before they are wired into the
extraction pipeline so later implementation can depend on stable interfaces.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence

_logger = logging.getLogger(__name__)
_SPECULATIVE_RE = re.compile(
    r"\b(maybe|might|possibly|probably|appears|seems|suggests|likely|unclear)\b",
    re.IGNORECASE,
)
_ANSWER_TAG_RE = re.compile(r"<answer>\s*([0-9]*\.?[0-9]+)\s*</answer>", re.IGNORECASE)


def _json_dump(data: object) -> str:
    return json.dumps(data, ensure_ascii=True, indent=2)


def _clean_text(text: str | None) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _format_timestamp(timestamp: Sequence[float] | None) -> str:
    if timestamp is None:
        return "null"
    if len(timestamp) != 2:
        raise ValueError("timestamp must have exactly two values")
    return _json_dump([float(timestamp[0]), float(timestamp[1])])


def _format_optional_json_block(label: str, value: object) -> str:
    return f"{label}:\n{_json_dump(value)}"


def _strict_json_tail(expected_shape: object) -> str:
    return (
        "\nOutput strict JSON only.\n"
        "No markdown, no code fences, no explanation, no extra keys outside the schema.\n"
        f"Expected shape:\n{_json_dump(expected_shape)}"
    )


def _retry_wrapper(base_prompt: str, expected_shape: object) -> str:
    return base_prompt + "\n\nYour previous answer was invalid." + _strict_json_tail(expected_shape)


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


def prompt_observation_notes(
    *,
    topic: str,
    video_id: str,
    caption_text: str = "",
    ocr_text: str = "",
    transcript_text: str = "",
    timestamp: Sequence[float] | None = None,
    chunk_metadata: object | None = None,
) -> str:
    expected_shape = {
        "observations": [
            {
                "text": "...",
                "modality": "visual",
                "timestamp": [0.0, 1.0],
            }
        ]
    }
    parts = [
        "You are extracting observation notes from evidence for a single video.",
        "",
        "Video context:",
        f"- topic: {topic}",
        f"- video_id: {video_id}",
        f"- timestamp_span: {_format_timestamp(timestamp)}",
    ]
    if chunk_metadata is not None:
        parts.extend(["", _format_optional_json_block("Chunk metadata", chunk_metadata)])
    parts.extend(
        [
            "",
            "Evidence:",
            f"Caption text:\n{caption_text or ''}",
            f"OCR text:\n{ocr_text or ''}",
            f"Transcript text:\n{transcript_text or ''}",
            "",
            "Rules:",
            "- Record only directly supported observations.",
            "- No inference, speculation, causality, identity guessing, or cross-video synthesis.",
            "- One observation per atomic visible, audible, or textual fact.",
            "- Preserve uncertainty explicitly when the evidence is ambiguous.",
            "- Use modality `visual` for scene content, `ocr` for on-screen text, and `audio` for transcript or speech.",
            "- Use the provided timestamp span for each observation when no narrower timestamp is available.",
            "- If no evidence is present, return an empty observations list.",
        ]
    )
    return "\n".join(parts) + _strict_json_tail(expected_shape)


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


def prompt_observation_notes_retry(
    *,
    topic: str,
    video_id: str,
    caption_text: str = "",
    ocr_text: str = "",
    transcript_text: str = "",
    timestamp: Sequence[float] | None = None,
    chunk_metadata: object | None = None,
) -> str:
    return _retry_wrapper(
        prompt_observation_notes(
            topic=topic,
            video_id=video_id,
            caption_text=caption_text,
            ocr_text=ocr_text,
            transcript_text=transcript_text,
            timestamp=timestamp,
            chunk_metadata=chunk_metadata,
        ),
        {
            "observations": [
                {
                    "text": "...",
                    "modality": "visual",
                    "timestamp": [0.0, 1.0],
                }
            ]
        },
    )


def prompt_grounded_notes(
    *,
    topic: str,
    video_id: str,
    observation_notes: Sequence[dict],
    video_path: str | None = None,
) -> str:
    expected_shape = {
        "grounded_notes": [
            {
                "claim": "...",
                "source_observation_ids": ["obs-1"],
                "timestamp_union": [0.0, 1.0],
            }
        ]
    }
    parts = [
        "You are deriving grounded notes from observation notes for a single video.",
        "",
        "Video context:",
        f"- topic: {topic}",
        f"- video_id: {video_id}",
        f"- video_path: {video_path or ''}",
        "",
        _format_optional_json_block("Observation notes", list(observation_notes)),
        "",
        "Rules:",
        "- Every grounded note must be fully supported by one or more supplied observation notes.",
        "- No grounded note may reference unstated evidence.",
        "- No cross-video aggregation.",
        "- Claims must be atomic, specific, and non-duplicative.",
        "- If an observation is too weak to support a claim, do not emit a grounded note from it.",
        "- source_observation_ids must reference the supporting observation note ids.",
        "- timestamp_union must be the minimal span covering the linked observations when timestamps exist; otherwise use null.",
    ]
    return "\n".join(parts) + _strict_json_tail(expected_shape)


def prompt_grounded_notes_retry(
    *,
    topic: str,
    video_id: str,
    observation_notes: Sequence[dict],
    video_path: str | None = None,
) -> str:
    return _retry_wrapper(
        prompt_grounded_notes(
            topic=topic,
            video_id=video_id,
            observation_notes=observation_notes,
            video_path=video_path,
        ),
        {
            "grounded_notes": [
                {
                    "claim": "...",
                    "source_observation_ids": ["obs-1"],
                    "timestamp_union": [0.0, 1.0],
                }
            ]
        },
    )


def prompt_query_conditioned_single(
    *,
    query_id: str,
    query: str,
    persona_title: str,
    background: str,
    evidence: object,
    topic: str | None = None,
    per_video_target: int = 5,
) -> str:
    expected_shape = {
        "facts": [
            {
                "fact": "...",
                "confidence": 0.85,
                "evidence": "...",
                "source": "video_visual",
                "timestamp": [0.0, 1.0],
                "video_id": "...",
                "video_path": "...",
                "caption": "...",
                "ocr": "...",
            }
        ]
    }
    parts = [
        "You are extracting candidate facts from ONE video's evidence for a report.",
        "",
        "Query context:",
        f"- query_id: {query_id}",
        f"- topic: {topic or ''}",
        f"- persona_title: {persona_title}",
        f"- background: {background}",
        f"- query: {query}",
        "",
        _format_optional_json_block("Evidence", evidence),
        "",
        "Rules:",
        f"- Extract up to {per_video_target} candidate facts from this video if possible.",
        "- Extract only information relevant to the query.",
        "- Facts must be evidence-grounded and citation-ready.",
        "- Avoid generic scene summary unless it directly serves the query.",
        "- Avoid duplicates and paraphrases.",
        "- Prefer concrete, report-usable facts over broad descriptions.",
        "- If the evidence does not answer the query, return an empty facts list.",
        "- source must be one of `video_visual`, `video_text`, or `transcript`.",
        "- timestamp must be [start, end].",
        "- confidence must be a float between 0 and 1.",
        "- video_id and video_path must match the evidence provided.",
        "- caption field should contain the supporting caption text if source is video_visual.",
        "- ocr field should contain the supporting OCR text if source is video_text.",
    ]
    return "\n".join(parts) + _strict_json_tail(expected_shape)


def prompt_query_conditioned_single_retry(
    *,
    query_id: str,
    query: str,
    persona_title: str,
    background: str,
    evidence: object,
    topic: str | None = None,
    per_video_target: int = 5,
) -> str:
    expected_shape = {
        "facts": [
            {
                "fact": "...",
                "confidence": 0.85,
                "evidence": "...",
                "source": "video_visual",
                "timestamp": [0.0, 1.0],
                "video_id": "...",
                "video_path": "...",
                "caption": "...",
                "ocr": "...",
            }
        ]
    }
    return _retry_wrapper(
        prompt_query_conditioned_single(
            query_id=query_id,
            query=query,
            persona_title=persona_title,
            background=background,
            evidence=evidence,
            topic=topic,
            per_video_target=per_video_target,
        ),
        expected_shape,
    )


def prompt_query_conditioned_expanded(
    *,
    query_id: str,
    query: str,
    persona_title: str,
    background: str,
    sub_queries: Sequence[str],
    evidence: object,
    topic: str | None = None,
    per_video_target: int = 5,
) -> str:
    expected_shape = {
        "facts": [
            {
                "fact": "...",
                "confidence": 0.85,
                "evidence": "...",
                "source": "video_visual",
                "timestamp": [0.0, 1.0],
                "video_id": "...",
                "video_path": "...",
                "caption": "...",
                "ocr": "...",
            }
        ]
    }
    parts = [
        "You are extracting candidate facts from ONE video's evidence for a report.",
        "",
        "Query context:",
        f"- query_id: {query_id}",
        f"- topic: {topic or ''}",
        f"- persona_title: {persona_title}",
        f"- background: {background}",
        f"- query: {query}",
        "",
        _format_optional_json_block("Coverage guidance subqueries", list(sub_queries)),
        "",
        _format_optional_json_block("Evidence", evidence),
        "",
        "Rules:",
        f"- Extract up to {per_video_target} candidate facts from this video if possible.",
        "- Use subqueries only as coverage guidance, not as evidence.",
        "- Do not mention subqueries in the output.",
        "- Extract only information relevant to the official query.",
        "- Facts must be evidence-grounded and citation-ready.",
        "- Do not emit unsupported facts even if a subquery suggests them.",
        "- Avoid duplicates and paraphrases.",
        "- If the evidence does not answer the query, return an empty facts list.",
        "- source must be one of `video_visual`, `video_text`, or `transcript`.",
        "- timestamp must be [start, end].",
        "- confidence must be a float between 0 and 1.",
        "- video_id and video_path must match the evidence provided.",
        "- caption field should contain the supporting caption text if source is video_visual.",
        "- ocr field should contain the supporting OCR text if source is video_text.",
    ]
    return "\n".join(parts) + _strict_json_tail(expected_shape)


def prompt_query_conditioned_expanded_retry(
    *,
    query_id: str,
    query: str,
    persona_title: str,
    background: str,
    sub_queries: Sequence[str],
    evidence: object,
    topic: str | None = None,
    per_video_target: int = 5,
) -> str:
    expected_shape = {
        "facts": [
            {
                "fact": "...",
                "confidence": 0.85,
                "evidence": "...",
                "source": "video_visual",
                "timestamp": [0.0, 1.0],
                "video_id": "...",
                "video_path": "...",
                "caption": "...",
                "ocr": "...",
            }
        ]
    }
    return _retry_wrapper(
        prompt_query_conditioned_expanded(
            query_id=query_id,
            query=query,
            persona_title=persona_title,
            background=background,
            sub_queries=sub_queries,
            evidence=evidence,
            topic=topic,
            per_video_target=per_video_target,
        ),
        expected_shape,
    )


def prompt_observation_video(
    *,
    topic: str,
    video_id: str,
    transcript_text: str = "",
    timestamp: Sequence[float] | None = None,
    perception_query: str = "",
) -> str:
    expected_shape = {
        "observations": [
            {
                "text": "...",
                "modality": "visual",
                "timestamp": [0.0, 1.0],
            }
        ]
    }
    parts = [
        "You are extracting observation notes directly from a raw video or sampled video chunk.",
        "",
        "Video context:",
        f"- topic: {topic}",
        f"- video_id: {video_id}",
        f"- timestamp_span: {_format_timestamp(timestamp)}",
        "",
        f"Transcript text:\n{transcript_text or ''}",
        "",
        "Rules:",
        "- Output observations only, not claims.",
        "- Record only directly supported observations from the sampled video evidence.",
        "- No inference, speculation, causality, identity guessing, or cross-video synthesis.",
        "- If transcript is used, keep audio-derived observations separate from visual observations via modality.",
        "- Use the provided timestamp span when no narrower timestamp is available.",
        "- If there is no usable evidence, return an empty observations list.",
    ]
    if perception_query:
        parts.append(f"- Focus on details relevant to: {perception_query}")
    return "\n".join(parts) + _strict_json_tail(expected_shape)


def prompt_observation_video_retry(
    *,
    topic: str,
    video_id: str,
    transcript_text: str = "",
    timestamp: Sequence[float] | None = None,
    perception_query: str = "",
) -> str:
    return _retry_wrapper(
        prompt_observation_video(
            topic=topic,
            video_id=video_id,
            transcript_text=transcript_text,
            timestamp=timestamp,
            perception_query=perception_query,
        ),
        {
            "observations": [
                {
                    "text": "...",
                    "modality": "visual",
                    "timestamp": [0.0, 1.0],
                }
            ]
        },
    )


def strip_code_fences(text: str) -> str:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def extract_first_json_object(text: str) -> str:
    text = strip_code_fences(text)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object start found")

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    raise ValueError("no complete JSON object found")


def parse_json_with_expected_key(text: str, expected_key: str) -> dict:
    blob = extract_first_json_object(text)
    obj = json.loads(blob)
    if not isinstance(obj, dict):
        raise ValueError("parsed JSON is not an object")
    if expected_key not in obj:
        raise ValueError(f"expected top-level key missing: {expected_key}")
    return obj


def call_llm_with_retry(model, prompt, retry_prompt, expected_key, *, video_path=None):
    """Call model.infer(), parse JSON, retry once on failure. Returns dict."""
    raw = model.infer(video_path, prompt)
    try:
        return parse_json_with_expected_key(raw, expected_key)
    except (ValueError, json.JSONDecodeError):
        pass
    raw = model.infer(video_path, retry_prompt)
    try:
        return parse_json_with_expected_key(raw, expected_key)
    except (ValueError, json.JSONDecodeError) as exc:
        _logger.warning("LLM JSON parse failed after retry: %s", exc)
        return {expected_key: []}


def contains_speculative_language(text: str) -> bool:
    return bool(_SPECULATIVE_RE.search(str(text or "")))


# ---------------------------------------------------------------------------
# PR2: General VLM note extraction prompts
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# PR3: Query-conditioned VLM claim extraction prompts
# ---------------------------------------------------------------------------


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
        "Rules:",
        f"- Extract up to {per_video_target} claims relevant to the query from this video.",
        "- Claims must be directly supported by observable video content.",
        "- Avoid generic scene summary unless it directly serves the query.",
        "- Avoid duplicates and paraphrases.",
        "- If the video does not contain evidence for the query, return an empty claims list.",
        "- source must be one of `video_visual`, `video_text`, or `transcript`.",
        "- timestamp must be [start, end].",
        "- confidence must be a float between 0 and 1.",
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
        f"- Extract up to {per_video_target} claims relevant to the query from this video.",
        "- Use subqueries only as coverage guidance, not as evidence.",
        "- Do not mention subqueries in the output.",
        "- Claims must be directly supported by observable video content.",
        "- Do not emit unsupported claims even if a subquery suggests them.",
        "- Avoid duplicates and paraphrases.",
        "- If the video does not contain evidence for the query, return an empty claims list.",
        "- source must be one of `video_visual`, `video_text`, or `transcript`.",
        "- timestamp must be [start, end].",
        "- confidence must be a float between 0 and 1.",
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


# ---------------------------------------------------------------------------
# PR6: Higher-level inference prompts
# ---------------------------------------------------------------------------


def prompt_higher_level_inference(
    *,
    query_id: str,
    query: str,
    topic: str,
    evidence_items: Sequence[dict],
) -> str:
    expected_shape = {
        "inferences": [
            {
                "claim": "...",
                "source_ids": ["id-1", "id-2"],
            }
        ]
    }
    parts = [
        "You are synthesizing higher-level inferences from evidence items for a query.",
        "",
        "Query context:",
        f"- query_id: {query_id}",
        f"- topic: {topic}",
        f"- query: {query}",
        "",
        _format_optional_json_block("Evidence items", list(evidence_items)),
        "",
        "Rules:",
        "- Synthesize higher-level claims that integrate multiple evidence items.",
        "- Every inference must cite its supporting evidence via source_ids.",
        "- source_ids must reference the IDs of evidence items used.",
        "- Do not fabricate information beyond what the evidence supports.",
        "- If the evidence is insufficient for any inference, return an empty inferences list.",
        "- Avoid duplicating evidence items verbatim as inferences.",
    ]
    return "\n".join(parts) + _strict_json_tail(expected_shape)


def prompt_higher_level_inference_retry(
    *,
    query_id: str,
    query: str,
    topic: str,
    evidence_items: Sequence[dict],
) -> str:
    return _retry_wrapper(
        prompt_higher_level_inference(
            query_id=query_id,
            query=query,
            topic=topic,
            evidence_items=evidence_items,
        ),
        {
            "inferences": [
                {
                    "claim": "...",
                    "source_ids": ["id-1", "id-2"],
                }
            ]
        },
    )
