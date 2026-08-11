"""MARQUIS-RLM tool namespace and memory helpers.

The paper RLM controller runs in a persistent REPL whose namespace contains
task context, a mutable memory bank, and tool functions backed by MARQUIS
pipeline modules. This module provides the public contract plus lightweight
fallback adapters for local smoke runs.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

RLM_TOOL_SPECS: list[dict[str, str]] = [
    {
        "name": "video_caption",
        "signature": "video_caption(vid)",
        "category": "perception",
        "backing_module": "local Qwen3.5-9B captioning backend",
    },
    {
        "name": "general_notes",
        "signature": "general_notes(vid)",
        "category": "perception",
        "backing_module": "marquis.information_extraction.general_notes",
    },
    {
        "name": "query_claims",
        "signature": "query_claims(vid)",
        "category": "perception",
        "backing_module": "marquis.information_extraction.query_claims",
    },
    {
        "name": "video_qa",
        "signature": "video_qa(vid, question)",
        "category": "targeted_query",
        "backing_module": "marquis.information_extraction.qa",
    },
    {
        "name": "transcribe",
        "signature": "transcribe(vid)",
        "category": "targeted_query",
        "backing_module": "local Whisper model",
    },
    {
        "name": "retrieve_chunks",
        "signature": "retrieve_chunks(vid)",
        "category": "targeted_query",
        "backing_module": "marquis.retrieval chunk retriever",
    },
    {
        "name": "write_report",
        "signature": "write_report(facts)",
        "category": "generation",
        "backing_module": "marquis.article_generation",
    },
    {
        "name": "memory_summary",
        "signature": "memory_summary()",
        "category": "memory",
        "backing_module": "marquis.rlm_controller.tool_api",
    },
    {
        "name": "print_memory",
        "signature": "print_memory(slot=None)",
        "category": "memory",
        "backing_module": "marquis.rlm_controller.tool_api",
    },
    {
        "name": "add_keyword",
        "signature": "add_keyword(vid, kw)",
        "category": "memory",
        "backing_module": "marquis.rlm_controller.tool_api",
    },
    {
        "name": "search_by_keyword",
        "signature": "search_by_keyword(kw)",
        "category": "memory",
        "backing_module": "marquis.rlm_controller.tool_api",
    },
    {
        "name": "remove_fact",
        "signature": "remove_fact(vid, idx)",
        "category": "memory",
        "backing_module": "marquis.rlm_controller.tool_api",
    },
    {
        "name": "clear_facts",
        "signature": "clear_facts(vid=None)",
        "category": "memory",
        "backing_module": "marquis.rlm_controller.tool_api",
    },
    {
        "name": "llm_think",
        "signature": "llm_think()",
        "category": "memory",
        "backing_module": "root fact table to findings pass",
    },
    {
        "name": "llm_judge",
        "signature": "llm_judge()",
        "category": "memory",
        "backing_module": "root fact table to selected_facts pass",
    },
]

RLM_TOOL_NAMES = tuple(spec["name"] for spec in RLM_TOOL_SPECS)


def create_memory_bank(context: dict[str, Any] | None = None) -> dict[str, Any]:
    context = context or {}
    videos = {
        str(vid): {
            "status": "unprocessed",
            "tools_used": [],
            "path": (context.get("video_pool", {}).get("video_paths", {}) or {}).get(str(vid), ""),
            "caption": "",
        }
        for vid in context.get("video_pool", {}).get("video_ids", []) or []
    }
    return {
        "findings": [],
        "keywords": {vid: [] for vid in videos},
        "fact_table": {vid: [] for vid in videos},
        "selected_facts": [],
        "videos": videos,
    }


def create_toolkit(
    context: dict[str, Any],
    *,
    note_taking_fn: Callable[..., Any] | None = None,
    video_qa_fn: Callable[..., Any] | None = None,
    general_notes_fn: Callable[..., Any] | None = None,
    query_claims_fn: Callable[..., Any] | None = None,
    write_report_fn: Callable[..., str] | None = None,
    transcribe_fn: Callable[..., str] | None = None,
    caption_fn: Callable[..., str] | None = None,
    retrieve_chunks_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    memory = create_memory_bank(context)
    claims = list(context.get("claims", []) or [])
    notes = list(context.get("general_notes", []) or [])
    video_paths = dict(context.get("video_pool", {}).get("video_paths", {}) or {})
    query_text = str(context.get("task", {}).get("query", ""))

    def _ensure_video(vid: str) -> dict[str, Any]:
        vid = str(vid)
        if vid not in memory["videos"]:
            memory["videos"][vid] = {
                "status": "unprocessed",
                "tools_used": [],
                "path": video_paths.get(vid, ""),
                "caption": "",
            }
        memory["keywords"].setdefault(vid, [])
        memory["fact_table"].setdefault(vid, [])
        return memory["videos"][vid]

    def _mark_used(vid: str, tool_name: str) -> None:
        status = _ensure_video(vid)
        if tool_name not in status["tools_used"]:
            status["tools_used"].append(tool_name)
        status["status"] = "processed"

    def _append_fact(vid: str, fact: dict[str, Any]) -> dict[str, Any]:
        vid = str(vid)
        _ensure_video(vid)
        memory["fact_table"][vid].append(fact)
        return fact

    def _fact_from_claim(
        claim: dict[str, Any], source_tool: str = "query_claims"
    ) -> dict[str, Any]:
        return {
            "fact": claim.get("claim") or claim.get("text") or claim.get("fact") or "",
            "timestamp": claim.get("timestamp"),
            "source_tool": source_tool,
            "confidence": claim.get("confidence"),
            "video_id": claim.get("video_id"),
            "source": claim.get("source"),
            "evidence": claim.get("evidence"),
            "claim_id": claim.get("claim_id"),
        }

    def video_caption(vid: str) -> str:
        vid = str(vid)
        _mark_used(vid, "video_caption")
        if caption_fn is not None:
            caption = str(caption_fn(video_paths.get(vid, vid)))
        else:
            caption = memory["videos"].get(vid, {}).get("caption") or ""
        memory["videos"][vid]["caption"] = caption
        return caption or f"[video_caption] No caption adapter configured for {vid}."

    def general_notes(vid: str) -> list[dict[str, Any]] | str:
        vid = str(vid)
        _mark_used(vid, "general_notes")
        if general_notes_fn is not None:
            result = general_notes_fn(vid)
            for item in result if isinstance(result, list) else []:
                _append_fact(vid, _fact_from_claim(item, source_tool="general_notes"))
            return result

        matching = [item for item in notes if str(item.get("video_id")) == vid]
        if matching:
            facts = [
                _append_fact(vid, _fact_from_claim(item, source_tool="general_notes"))
                for item in matching
            ]
            return facts

        if note_taking_fn is not None:
            path = video_paths.get(vid, vid)
            result = note_taking_fn(
                path, f"Extract directly observable notes relevant to: {query_text}"
            )
            fact = _append_fact(
                vid,
                {
                    "fact": str(result),
                    "timestamp": None,
                    "source_tool": "general_notes",
                    "confidence": None,
                    "video_id": vid,
                },
            )
            return [fact]
        return f"[general_notes] No notes or adapter configured for {vid}."

    def query_claims(vid: str) -> list[dict[str, Any]] | str:
        vid = str(vid)
        _mark_used(vid, "query_claims")
        matching = [claim for claim in claims if str(claim.get("video_id")) == vid]
        if query_claims_fn is not None:
            result = query_claims_fn(vid)
            for item in result if isinstance(result, list) else []:
                _append_fact(vid, _fact_from_claim(item, source_tool="query_claims"))
            return result
        if matching:
            return [
                _append_fact(vid, _fact_from_claim(claim, source_tool="query_claims"))
                for claim in matching
            ]
        return f"[query_claims] No pre-extracted claims or adapter configured for {vid}."

    def video_qa(vid: str, question: str) -> str:
        vid = str(vid)
        _mark_used(vid, "video_qa")
        if video_qa_fn is not None:
            answer = str(video_qa_fn(vid, question))
        elif note_taking_fn is not None:
            answer = str(note_taking_fn(video_paths.get(vid, vid), question))
        else:
            answer = f"[video_qa] No QA adapter configured for {vid}."
        _append_fact(
            vid,
            {
                "fact": answer,
                "timestamp": None,
                "source_tool": "video_qa",
                "confidence": None,
                "video_id": vid,
                "question": question,
            },
        )
        return answer

    def transcribe(vid: str) -> str:
        vid = str(vid)
        _mark_used(vid, "transcribe")
        if transcribe_fn is not None:
            return str(transcribe_fn(video_paths.get(vid, vid)))
        return f"[transcribe] No transcription adapter configured for {vid}."

    def retrieve_chunks(vid: str) -> Any:
        vid = str(vid)
        _mark_used(vid, "retrieve_chunks")
        if retrieve_chunks_fn is not None:
            return retrieve_chunks_fn(vid)
        return context.get("chunks", {}).get(vid, [])

    def write_report(facts: list[dict[str, Any]] | None = None) -> str:
        if write_report_fn is not None:
            return write_report_fn(facts or [])
        facts = list(facts or memory.get("selected_facts") or [])
        if not facts:
            return "No selected facts were available to write a report."
        lines = ["MARQUIS report", ""]
        refs = []
        for idx, fact in enumerate(facts, start=1):
            text = str(fact.get("fact") if isinstance(fact, dict) else fact).strip()
            if not text:
                continue
            lines.append(f"{text} ({idx})")
            if isinstance(fact, dict):
                refs.append(
                    f"({idx}) video_id={fact.get('video_id')}, "
                    f"timestamp={fact.get('timestamp')}, source={fact.get('source_tool')}"
                )
        if refs:
            lines.extend(["", "References:", *refs])
        return "\n".join(lines)

    def memory_summary() -> str:
        fact_counts = {vid: len(items) for vid, items in memory["fact_table"].items()}
        summary = {
            "findings": len(memory["findings"]),
            "fact_counts": fact_counts,
            "selected_facts": len(memory["selected_facts"]),
            "videos": memory["videos"],
            "tools": RLM_TOOL_NAMES,
        }
        return json.dumps(summary, ensure_ascii=True, indent=2)

    def print_memory(slot: str | None = None) -> str:
        if slot is None:
            return json.dumps(memory, ensure_ascii=True, indent=2)
        return json.dumps(memory.get(slot), ensure_ascii=True, indent=2)

    def add_keyword(vid: str, kw: str) -> list[str]:
        vid = str(vid)
        _ensure_video(vid)
        if kw not in memory["keywords"][vid]:
            memory["keywords"][vid].append(str(kw))
        return memory["keywords"][vid]

    def search_by_keyword(kw: str) -> list[str]:
        needle = str(kw).lower()
        return [
            vid
            for vid, keywords in memory["keywords"].items()
            if any(needle in str(item).lower() for item in keywords)
        ]

    def remove_fact(vid: str, idx: int) -> dict[str, Any] | None:
        vid = str(vid)
        facts = memory["fact_table"].setdefault(vid, [])
        if 0 <= int(idx) < len(facts):
            return facts.pop(int(idx))
        return None

    def clear_facts(vid: str | None = None) -> None:
        if vid is None:
            for key in list(memory["fact_table"].keys()):
                memory["fact_table"][key] = []
            return None
        memory["fact_table"][str(vid)] = []
        return None

    def llm_think() -> str:
        flat = [
            fact
            for facts in memory["fact_table"].values()
            for fact in facts
            if str(fact.get("fact", "")).strip()
        ]
        new_findings = [str(fact["fact"]).strip() for fact in flat[:10]]
        for finding in new_findings:
            if finding not in memory["findings"]:
                memory["findings"].append(finding)
        return "\n".join(f"- {finding}" for finding in memory["findings"]) or "No findings yet."

    def llm_judge() -> list[dict[str, Any]]:
        flat = [
            fact
            for facts in memory["fact_table"].values()
            for fact in facts
            if str(fact.get("fact", "")).strip()
        ]
        flat.sort(key=lambda item: float(item.get("confidence") or 0.0), reverse=True)
        memory["selected_facts"] = flat[:40]
        return memory["selected_facts"]

    return {
        "memory": memory,
        "video_caption": video_caption,
        "general_notes": general_notes,
        "query_claims": query_claims,
        "video_qa": video_qa,
        "transcribe": transcribe,
        "retrieve_chunks": retrieve_chunks,
        "write_report": write_report,
        "memory_summary": memory_summary,
        "print_memory": print_memory,
        "add_keyword": add_keyword,
        "search_by_keyword": search_by_keyword,
        "remove_fact": remove_fact,
        "clear_facts": clear_facts,
        "llm_think": llm_think,
        "llm_judge": llm_judge,
    }
