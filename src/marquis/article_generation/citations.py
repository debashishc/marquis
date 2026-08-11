"""Citation formatting and parsing for generated MARQUIS reports."""

from __future__ import annotations

import re
from typing import Any

_CITATION_RE = re.compile(r"\[(?P<video_id>[^\],]+)(?:,\s*(?P<span>[^\]]+))?\]")


def _clean_timestamp_token(value: Any) -> str:
    text = str(value).strip()
    if text.endswith("s") and len(text) > 1:
        text = text[:-1].strip()
    return text or "?"


def _looks_like_time_token(value: Any) -> bool:
    token = _clean_timestamp_token(value)
    return token == "?" or bool(re.fullmatch(r"\d+(?:\.\d+)?", token))


def normalize_timestamp(timestamp: Any) -> list[str] | None:
    if timestamp is None or timestamp == "":
        return None
    if isinstance(timestamp, (list, tuple)):
        if len(timestamp) == 0:
            return None
        if len(timestamp) == 1:
            return [_clean_timestamp_token(timestamp[0])]
        return [_clean_timestamp_token(timestamp[0]), _clean_timestamp_token(timestamp[1])]
    text = str(timestamp).strip()
    if not text:
        return None
    if "-" in text:
        start, end = text.split("-", 1)
        if _looks_like_time_token(start) and _looks_like_time_token(end):
            return [_clean_timestamp_token(start), _clean_timestamp_token(end)]
    return [_clean_timestamp_token(text)]


def format_timestamp(timestamp: Any) -> str:
    normalized = normalize_timestamp(timestamp)
    if not normalized:
        return "?"
    if len(normalized) == 1:
        return normalized[0]
    return f"{normalized[0]}-{normalized[1]}"


def format_citation(video_id: str, timestamp: list[Any] | tuple[Any, ...] | None = None) -> str:
    return f"[{video_id}, {format_timestamp(timestamp)}]"


def parse_citations(text: str) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    for match in _CITATION_RE.finditer(text or ""):
        span = match.group("span")
        citations.append(
            {
                "video_id": match.group("video_id").strip(),
                "timestamp": normalize_timestamp(span),
                "raw": match.group(0),
            }
        )
    return citations
