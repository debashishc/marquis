#!/usr/bin/env python3
"""Convert MARQUIS mv_full reports into the MiRAGE generation prediction format.

The MiRAGE generation eval (microvent-eval/mirage) expects predictions shaped like
``microvent-eval/example_files/generation/demo_day_6-1.json``::

    {"<query_id>": [{"text": "<claim>", "citations": ["<chunk_id>", ...]}, ...], ...}

where ``<chunk_id>`` is a video segment id such as ``J2vYKin5lPXfBgGb_0000``.

This script turns each of the four mv_full report styles into that format:

* ``baseline`` / ``ginger`` -- ``reports_*.json`` keyed by query_id, each a single
  ``report`` string with inline ``[video_id, time-span]`` markers.
* ``qa``                    -- ``reports/reports_qa_iter_q_*.json`` markdown reports
  with inline ``[video_id, video_id, ...]`` markers (comma-separated ids, no spans).
* ``bullet``                -- ``bullet_query/all_reports.json``, a list of per-query
  records whose ``sections`` are claims grounded by ``source_citations[*].video_id``.

For the prose styles the report is split into sentences; each sentence keeps the
chunk ids cited within it and has its ``[...]`` markers stripped. Bullet claims map
one record -> one sentence.

Usage::

    python scripts/reports_to_mirage.py            # convert everything under outputs/mv_full
    python scripts/reports_to_mirage.py --root outputs/mv_full --out-dir outputs/mv_full/mirage_predictions
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Any

# A video chunk id is a 16-ish char base64url token followed by a 4-digit segment,
# e.g. ``hs5PefuGfmCrvh0d_0000``. This matches the id wherever it appears: inside an
# inline ``[...]`` marker, or embedded in a bullet ``source_id`` like
# ``qc-1-hs5PefuGfmCrvh0d_0000-000``. Time spans (``0.0-5.0``) never match.
CHUNK_ID_RE = re.compile(r"[A-Za-z0-9_-]+_\d{4}")
# An inline citation marker: a bracketed group. Markers never contain a sentence
# terminator followed by whitespace, so it is safe to sentence-split with them in
# place and strip them afterwards.
MARKER_RE = re.compile(r"\[[^\]]*\]")
# Split on whitespace that follows sentence-ending punctuation. Deliberately naive
# (mirrors the reference splitter, which also breaks on abbreviations like "a.m.").
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _dedupe(ids: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for x in ids:
        seen.setdefault(x, None)
    return list(seen)


def _strip_markdown(report: str) -> str:
    """Drop markdown header lines and leading list/heading markers, keep prose."""
    lines = []
    for line in report.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        if stripped.startswith("#"):  # markdown header -> drop entirely
            continue
        # strip a leading bullet/number marker ("- ", "* ", "1. ")
        stripped = re.sub(r"^(?:[-*+]\s+|\d+\.\s+)", "", stripped)
        lines.append(stripped)
    return "\n".join(lines)


def prose_to_sentences(report: str) -> list[dict[str, Any]]:
    """Split a report string into ``{text, citations}`` sentences."""
    text = _strip_markdown(report)
    # Treat paragraph breaks as sentence boundaries too, then collapse newlines.
    text = re.sub(r"\n+", " ", text)
    out: list[dict[str, Any]] = []
    for raw in SENT_SPLIT_RE.split(text):
        raw = raw.strip()
        if not raw:
            continue
        citations = _dedupe(CHUNK_ID_RE.findall(raw))
        clean = MARKER_RE.sub("", raw)
        clean = re.sub(r"[ \t]+", " ", clean).strip()
        if not clean:
            continue
        out.append({"text": clean, "citations": citations})
    return out


def _report_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("report") or value.get("text") or ""
    return ""


def convert_prose_reports(path: str) -> dict[str, list[dict[str, Any]]]:
    """baseline/ginger/qa report json -> {query_id: [sentences]}."""
    data = json.load(open(path))
    pred: dict[str, list[dict[str, Any]]] = {}
    for qid, value in data.items():
        pred[str(qid)] = prose_to_sentences(_report_string(value))
    return pred


def convert_bullet(all_reports_path: str) -> dict[str, list[dict[str, Any]]]:
    """bullet_query/all_reports.json -> {query_id: [sentences]}.

    The file is a list of per-query records, each a list of ``sections``; every
    section is one claim (``text``) grounded by ``source_citations`` whose
    ``video_id`` fields are the chunk ids.
    """
    data = json.load(open(all_reports_path))
    pred: dict[str, list[dict[str, Any]]] = {}
    for rec in data:
        qid = str(rec.get("query_id"))
        sents: list[dict[str, Any]] = []
        for section in rec.get("sections") or []:
            text = (section.get("text") or "").strip()
            if not text:
                continue
            sources = section.get("source_citations") or []
            if not sources and section.get("citation"):
                sources = [section["citation"]]
            citations = _dedupe(
                c["video_id"] for c in sources if c.get("video_id")
            )
            sents.append({"text": text, "citations": citations})
        pred[qid] = sents
    return pred


def _sorted_by_qid(pred: dict[str, list]) -> dict[str, list]:
    def key(q: str):
        return (0, int(q)) if q.isdigit() else (1, q)

    return {q: pred[q] for q in sorted(pred, key=key)}


def _write(pred: dict[str, list], out_path: str) -> None:
    pred = _sorted_by_qid(pred)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(pred, open(out_path, "w"), indent=2, ensure_ascii=False)
    n_sent = sum(len(v) for v in pred.values())
    n_cite = sum(len(s["citations"]) for v in pred.values() for s in v)
    print(f"wrote {out_path}: {len(pred)} queries, {n_sent} sentences, {n_cite} citations")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default="outputs/mv_full", help="mv_full directory")
    ap.add_argument(
        "--out-dir",
        default=None,
        help="output directory (default: <root>/mirage_predictions)",
    )
    args = ap.parse_args()

    root = args.root
    out_dir = args.out_dir or os.path.join(root, "mirage_predictions")

    # (label, builder) for each report style that is present.
    baseline = os.path.join(root, "baseline", "reports_baseline.json")
    if os.path.exists(baseline):
        _write(convert_prose_reports(baseline), os.path.join(out_dir, "baseline.json"))

    ginger = os.path.join(root, "ginger", "reports_ginger.json")
    if os.path.exists(ginger):
        _write(convert_prose_reports(ginger), os.path.join(out_dir, "ginger.json"))

    bullet = os.path.join(root, "bullet_query", "all_reports.json")
    if os.path.exists(bullet):
        _write(convert_bullet(bullet), os.path.join(out_dir, "bullet.json"))

    # qa has one report file per styling variant; emit each.
    for qa_path in sorted(glob.glob(os.path.join(root, "qa", "reports", "reports_qa_*.json"))):
        variant = os.path.basename(qa_path)[len("reports_qa_") : -len(".json")]
        _write(convert_prose_reports(qa_path), os.path.join(out_dir, f"qa_{variant}.json"))


if __name__ == "__main__":
    main()
