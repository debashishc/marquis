"""Bullet article generation: note-taking infer→report→annotate citation chain.

Structurally distinct from the single-shot strategies (baseline / ginger / qa):
``bullet`` runs as a three-step pipeline over retrieval packets, using a VLM to
draw higher-level inferences, assemble cited reports, and inline-annotate
citations. The step is a positional sub-command; the model comes from the
``model`` group, the output dir from ``output``, query_ids from ``data``, and the
remaining bullet-specific knobs from ``runtime.bullet`` of the Hydra config tree
under ``configs/article_generation/``. Override on the command line with Hydra
syntax::

    python -m article_generation.cli bullet infer \
        runtime.bullet.stream=query-based runtime.bullet.packets_dir=... \
        runtime.bullet.claims=... output.out_dir=...
    python -m article_generation.cli bullet annotate \
        runtime.bullet.inferences=... runtime.bullet.packets_dir=... \
        runtime.bullet.claims=... output.out_dir=...

Pass ``--api URL`` (shorthand for ``model.api=URL``) to send inference to a
remote OpenAI-compatible server instead of loading the VLM locally::

    python -m article_generation.cli bullet infer --api http://rack3n07:4000/v1 \
        --api-model qwen-27b runtime.bullet.stream=query-based ...

The legacy ``report`` step is kept for completeness; ``annotate`` can build
reports straight from ``infer`` output, so the usual flow is infer → annotate.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import sys

from omegaconf import DictConfig, OmegaConf

from marquis.article_generation._common import build_config
from marquis.article_generation.citations import normalize_timestamp
from marquis.article_generation.prompts import (
    prompt_annotate_citations,
    prompt_annotate_citations_retry,
    prompt_higher_level_inference,
    prompt_higher_level_inference_retry,
)
from marquis.common.contracts import (
    load_queries,
    validate_higher_level_inference,
    validate_report_citation,
)
from marquis.common.prompts import call_llm_with_retry
from marquis.common.run_metadata import build_run_manifest, write_run_manifest

# Shared helpers


def sort_evidence(evidence: list[dict]) -> list[dict]:
    """Sort evidence records by confidence, highest first."""
    return sorted(evidence, key=lambda item: float(item.get("confidence") or 0.0), reverse=True)


def _format_inline_citation(video_id: str, timestamp) -> str:
    ts = normalize_timestamp(timestamp)
    if not ts:
        span = "?"
    elif len(ts) >= 2:
        span = f"{ts[0]}-{ts[1]}"
    else:
        span = ts[0]
    return f"[{video_id}, {span}]"


def render_bullet_report(evidence: list[dict], *, title: str = "MARQUIS Evidence") -> str:
    """Render evidence as a conservative bullet-point report with inline citations."""
    lines = [f"# {title}", ""]
    for idx, item in enumerate(evidence, start=1):
        text = item.get("claim") or item.get("text") or item.get("fact") or ""
        citation = _format_inline_citation(str(item.get("video_id", "?")), item.get("timestamp"))
        lines.append(f"{idx}. {text} {citation}")
    return "\n".join(lines)


def _iter_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _write_jsonl(path: str, records: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _parse_query_ids(value: str | None) -> set[str] | None:
    if not value:
        return None
    query_ids = {qid.strip() for qid in str(value).split(",") if qid.strip()}
    return query_ids or None


def _filter_packets_by_query_ids(packets: list[dict], query_ids: set[str] | None) -> list[dict]:
    if not query_ids:
        return packets
    return [p for p in packets if str(p.get("query_id", "")) in query_ids]


def _load_packets(packets_dir: str) -> list[dict]:
    """Load all packets from a directory (combined file or per-query files)."""
    combined = os.path.join(packets_dir, "all_packets.json")
    if os.path.exists(combined):
        with open(combined, encoding="utf-8") as f:
            return json.load(f)
    packets = []
    for fname in sorted(os.listdir(packets_dir)):
        if fname.startswith("query_") and fname.endswith(".json"):
            with open(os.path.join(packets_dir, fname), encoding="utf-8") as f:
                packets.append(json.load(f))
    return packets


def _build_note_index(notes_jsonl: str) -> dict[str, dict]:
    """Index general notes by note_id."""
    idx = {}
    for note in _iter_jsonl(notes_jsonl):
        nid = note.get("note_id", "")
        if nid:
            idx[nid] = note
    return idx


def _build_claim_index(claims_jsonl: str) -> dict[str, dict]:
    """Index query-conditioned claims by claim_id."""
    idx = {}
    for claim in _iter_jsonl(claims_jsonl):
        cid = claim.get("claim_id", "")
        if cid:
            idx[cid] = claim
    return idx


def _load_model(model_cfg: DictConfig, **kwargs):
    """Lazily build the VLM used across the bullet pipeline.

    Uses the remote API backend when ``model.api`` is set (via the --api flag
    or a ``model.api=URL`` override); otherwise loads ``model.vlm`` locally.
    """
    if model_cfg.get("api"):
        from marquis.common.model_backends import APIVLM

        print(f"\nUsing API backend: {model_cfg.api} (model: {model_cfg.api_model})")
        return APIVLM(
            api_base=model_cfg.api,
            api_model=model_cfg.api_model,
            api_key=model_cfg.api_key,
            **kwargs,
        )

    from marquis.common.model_backends import Qwen3_5_VL

    print(f"\nInitializing model: {model_cfg.vlm}")
    return Qwen3_5_VL(
        model=model_cfg.vlm, download_dir=model_cfg.cache_dir or None, **kwargs
    )


# Step 1: INFER


def _sanitize_source_ids(
    source_ids: list[str], allowed_ids: list[str]
) -> tuple[list[str], list[str]]:
    """Keep only packet-local source ids, preserving order and uniqueness."""
    allowed = set(allowed_ids)
    valid = []
    invalid = []
    seen = set()
    for sid in source_ids or []:
        if sid in allowed:
            if sid not in seen:
                valid.append(sid)
                seen.add(sid)
        else:
            invalid.append(sid)
    return valid, invalid


def infer_over_packet(model, packet, evidence_items, query, stream, run_id):
    """Run LLM inference over a single packet's evidence."""
    qid = query["query_id"]
    topic = packet.get("topic", "")
    allowed_ids = [item.get("id", "") for item in evidence_items if item.get("id")]

    prompt = prompt_higher_level_inference(
        query_id=qid,
        query=query.get("query", ""),
        topic=topic,
        evidence_items=evidence_items,
    )
    retry = prompt_higher_level_inference_retry(
        query_id=qid,
        query=query.get("query", ""),
        topic=topic,
        evidence_items=evidence_items,
    )
    result = call_llm_with_retry(model, prompt, retry, "inferences", video_path=None)

    inferences = []
    warnings = []
    for idx, inf in enumerate(result.get("inferences", [])):
        source_ids, invalid_ids = _sanitize_source_ids(inf.get("source_ids", []), allowed_ids)
        if invalid_ids:
            warnings.append(
                f"query {qid} inference[{idx}]: dropped invalid source_ids {invalid_ids}"
            )
        if not source_ids:
            warnings.append(
                f"query {qid} inference[{idx}]: skipped because no source_ids "
                "matched packet contents"
            )
            continue
        inferences.append(
            {
                "inference_id": f"hli-{stream}-{qid}-{idx:03d}",
                "query_id": qid,
                "topic": topic,
                "claim": inf.get("claim", ""),
                "source_ids": source_ids,
                "confidence": None,
                "is_post_grounded": True,
                "calibration": None,
                "stream": stream,
                "run_id": run_id,
            }
        )

    return inferences, warnings


def run_infer(cfg: DictConfig) -> None:
    """Step 1: higher-level inference over packets."""
    b = cfg.runtime.bullet
    if not b.stream:
        raise SystemExit("infer requires runtime.bullet.stream=general-note|query-based")
    if b.stream not in ("general-note", "query-based"):
        raise SystemExit(f"invalid runtime.bullet.stream: {b.stream}")
    if not b.packets_dir or not cfg.output.out_dir:
        raise SystemExit("infer requires runtime.bullet.packets_dir and output.out_dir")

    manifest = build_run_manifest(
        script_name="article_generation.bullet infer",
        argv=sys.argv,
        args_dict=OmegaConf.to_container(b, resolve=True),
        run_config={
            "stream": b.stream,
            "model": cfg.model.vlm,
            "api": cfg.model.api,
            "api_model": cfg.model.api_model,
            "temperature": b.infer.temperature,
            "max_tokens": b.infer.max_tokens,
            "query_ids": cfg.data.query_ids,
        },
    )
    manifest_path = write_run_manifest(cfg.output.out_dir, manifest)

    queries = load_queries(b.queries_jsonl)
    query_by_id = {q["query_id"]: q for q in queries}

    packets = _load_packets(b.packets_dir)
    print(f"Loaded {len(packets)} packets from {b.packets_dir}")
    query_ids = _parse_query_ids(cfg.data.query_ids)
    if query_ids:
        before = len(packets)
        packets = _filter_packets_by_query_ids(packets, query_ids)
        print(f"Filtered packets by query_ids={sorted(query_ids)}: {before} -> {len(packets)}")

    if b.stream == "general-note":
        stream_key = "general_note"
        if not b.notes:
            raise SystemExit("--notes (bullet.notes) is required for general-note stream")
        evidence_idx = _build_note_index(b.notes)
        id_field, text_field = "note_ids", "text"
        print(f"Indexed {len(evidence_idx)} general notes")
    else:
        stream_key = "query_based"
        if not b.claims:
            raise SystemExit("--claims (bullet.claims) is required for query-based stream")
        evidence_idx = _build_claim_index(b.claims)
        id_field, text_field = "claim_ids", "claim"
        print(f"Indexed {len(evidence_idx)} query-conditioned claims")

    model = _load_model(
        cfg.model,
        temperature=b.infer.temperature,
        presence_penalty=b.infer.presence_penalty,
        max_tokens=b.infer.max_tokens,
    )

    os.makedirs(cfg.output.out_dir, exist_ok=True)
    all_inferences = []

    for packet in packets:
        qid = packet.get("query_id", "")
        query = query_by_id.get(qid, {"query_id": qid, "query": ""})

        item_ids = packet.get(id_field, [])
        evidence_items = []
        for eid in item_ids:
            rec = evidence_idx.get(eid, {})
            if rec:
                evidence_items.append(
                    {
                        "id": eid,
                        "text": rec.get(text_field, ""),
                        "video_id": rec.get("video_id", ""),
                    }
                )

        if not evidence_items:
            if b.verbose:
                print(f"  query {qid}: no evidence items, skipping")
            continue

        inferences, inference_warnings = infer_over_packet(
            model,
            packet,
            evidence_items,
            query,
            stream_key,
            manifest["run_id"],
        )
        if b.verbose:
            for warning in inference_warnings:
                print(f"  WARN {warning}")

        for inf in inferences:
            errs = validate_higher_level_inference(inf)
            if errs and b.verbose:
                print(f"  WARN inference {inf.get('inference_id')}: {errs}")

        all_inferences.extend(inferences)
        if b.verbose:
            print(
                f"  query {qid}: {len(inferences)} inferences from "
                f"{len(evidence_items)} evidence items"
            )

    # Write per-query files
    inferences_by_query: dict[str, list[dict]] = {}
    for inf in all_inferences:
        inferences_by_query.setdefault(inf.get("query_id", ""), []).append(inf)
    for qid, infs in inferences_by_query.items():
        _write_jsonl(os.path.join(cfg.output.out_dir, f"query_{qid}.jsonl"), infs)

    combined_path = os.path.join(cfg.output.out_dir, "inferences.jsonl")
    _write_jsonl(combined_path, all_inferences)

    print(f"\n[ok] {len(all_inferences)} higher-level inferences from {len(packets)} packets")
    print(f"[ok] -> {cfg.output.out_dir}/query_{{N}}.jsonl")
    print(f"[ok] -> {combined_path}")
    print(f"[ok] -> {manifest_path}")


# Step 2: REPORT (legacy) / shared report builders


def _utc_now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).replace(microsecond=0).isoformat()


def _load_grounded_index(path: str | None) -> dict[str, dict]:
    if not path or not os.path.exists(path):
        return {}
    idx = {}
    for note in _iter_jsonl(path):
        nid = note.get("note_id", "")
        if nid:
            idx[nid] = note
    return idx


def _load_inference_index(inferences_dir: str | None) -> dict[str, list[dict]]:
    """Load inferences grouped by query_id."""
    if not inferences_dir or not os.path.isdir(inferences_dir):
        return {}
    combined = os.path.join(inferences_dir, "inferences.jsonl")
    by_query: dict[str, list[dict]] = {}
    if os.path.exists(combined):
        for inf in _iter_jsonl(combined):
            by_query.setdefault(inf.get("query_id", ""), []).append(inf)
    return by_query


def _resolve_citation(note_id: str, grounded_idx: dict[str, dict]) -> dict:
    note = grounded_idx.get(note_id, {})
    return {
        "note_id": note_id,
        "video_id": note.get("video_id", "unknown"),
        "timestamp": note.get("timestamp_union"),
        "claim": note.get("claim", ""),
    }


def _generate_note_taking_report(packet, query, grounded_idx, run_id) -> dict:
    sections, citations = [], []
    note_ids = packet.get("retrieved_note_ids", [])
    scores = packet.get("scores", [])
    for i, nid in enumerate(note_ids):
        note = grounded_idx.get(nid, {})
        claim = note.get("claim", "")
        if not claim:
            continue
        citation = _resolve_citation(nid, grounded_idx)
        citations.append(citation)
        score = scores[i] if i < len(scores) else 0.0
        sections.append({"text": claim, "citation": citation, "score": round(score, 4)})
    return {
        "query_id": query["query_id"],
        "topic": packet.get("topic", ""),
        "pipeline": "note_taking",
        "sections": sections,
        "citations": citations,
        "provenance": {
            "run_id": run_id,
            "packet_source": "note_taking",
            "created_at": _utc_now_iso(),
        },
    }


def _generate_query_based_report_legacy(packet, query, run_id) -> dict:
    sections, citations = [], []
    for i, fact in enumerate(packet.get("facts", [])):
        vid = fact.get("video_id", "unknown")
        citation = {
            "note_id": f"fact-{query['query_id']}-{i:03d}",
            "video_id": vid,
            "timestamp": fact.get("timestamp"),
            "claim": fact.get("fact", ""),
        }
        citations.append(citation)
        sections.append({"text": fact.get("fact", ""), "citation": citation, "score": 1.0})
    return {
        "query_id": query["query_id"],
        "topic": packet.get("topic", ""),
        "pipeline": packet.get("pipeline", "query_based"),
        "sections": sections,
        "citations": citations,
        "provenance": {
            "run_id": run_id,
            "packet_source": packet.get("pipeline", "query_based"),
            "created_at": _utc_now_iso(),
        },
    }


def _generate_stream_report(
    packet, query, evidence_idx, inferences, run_id, *, pipeline, id_field, text_field
) -> dict:
    """Build a report for a general-note or query-based stream.

    ``evidence_idx`` maps source ids to records; ``text_field`` is the field
    holding the evidence text (``text`` for notes, ``claim`` for claims).
    """
    sections, citations = [], []

    if inferences:
        for inf in inferences:
            source_citations = []
            for sid in inf.get("source_ids", []):
                rec = evidence_idx.get(sid, {})
                source_citations.append(
                    {
                        "note_id": sid,
                        "video_id": rec.get("video_id", "unknown"),
                        "timestamp": rec.get("timestamp"),
                        "claim": rec.get(text_field, ""),
                    }
                )
            primary = (
                source_citations[0]
                if source_citations
                else {
                    "note_id": inf.get("inference_id", ""),
                    "video_id": "unknown",
                    "timestamp": None,
                    "claim": inf.get("claim", ""),
                }
            )
            citations.append(primary)
            citations.extend(source_citations[1:])
            sections.append(
                {
                    "text": inf.get("claim", ""),
                    "citation": primary,
                    "score": 1.0,
                    "inference_id": inf.get("inference_id"),
                    "source_citations": source_citations,
                }
            )
    else:
        ids = packet.get(id_field, [])
        scores = packet.get("scores", [])
        for i, sid in enumerate(ids):
            rec = evidence_idx.get(sid, {})
            text = rec.get(text_field, "")
            if not text:
                continue
            citation = {
                "note_id": sid,
                "video_id": rec.get("video_id", "unknown"),
                "timestamp": rec.get("timestamp"),
                "claim": text,
            }
            citations.append(citation)
            score = scores[i] if i < len(scores) else 0.0
            sections.append({"text": text, "citation": citation, "score": round(score, 4)})

    return {
        "query_id": query["query_id"],
        "topic": packet.get("topic", ""),
        "pipeline": pipeline,
        "sections": sections,
        "citations": citations,
        "provenance": {
            "run_id": run_id,
            "packet_source": pipeline,
            "has_inferences": bool(inferences),
            "created_at": _utc_now_iso(),
        },
    }


def _build_report(
    packet, query, stream, grounded_idx, note_idx, claim_idx, inference_idx, run_id
) -> dict:
    """Build a single report dict from a packet + resolved evidence indices."""
    qid = packet.get("query_id", "")
    if stream == "general-note":
        report = _generate_stream_report(
            packet,
            query,
            note_idx,
            inference_idx.get(qid, []),
            run_id,
            pipeline="general_note",
            id_field="note_ids",
            text_field="text",
        )
    elif stream == "query-based":
        report = _generate_stream_report(
            packet,
            query,
            claim_idx,
            inference_idx.get(qid, []),
            run_id,
            pipeline="query_based",
            id_field="claim_ids",
            text_field="claim",
        )
    else:
        pipeline = packet.get("pipeline", "")
        if pipeline == "note_taking" and grounded_idx:
            report = _generate_note_taking_report(packet, query, grounded_idx, run_id)
        else:
            report = _generate_query_based_report_legacy(packet, query, run_id)
    report["query"] = query.get("query", "")
    return report


def _load_evidence_indices(stream, grounded, notes, claims, inferences, verbose=False):
    """Load grounded/note/claim/inference indices depending on the stream mode."""
    grounded_idx: dict[str, dict] = {}
    note_idx: dict[str, dict] = {}
    claim_idx: dict[str, dict] = {}
    inference_idx: dict[str, list[dict]] = {}

    if stream is None:
        grounded_idx = _load_grounded_index(grounded)
        if grounded_idx and verbose:
            print(f"Loaded {len(grounded_idx)} grounded notes for citation resolution")
    elif stream == "general-note":
        note_idx = _build_note_index(notes) if notes else {}
        if verbose:
            print(f"Loaded {len(note_idx)} general notes")
        inference_idx = _load_inference_index(inferences)
    elif stream == "query-based":
        claim_idx = _build_claim_index(claims) if claims else {}
        if verbose:
            print(f"Loaded {len(claim_idx)} query-conditioned claims")
        inference_idx = _load_inference_index(inferences)

    if inference_idx and verbose:
        total = sum(len(v) for v in inference_idx.values())
        print(f"Loaded {total} inferences across {len(inference_idx)} queries")

    return grounded_idx, note_idx, claim_idx, inference_idx


def _render_markdown_report(report: dict, query: dict) -> str:
    lines = [
        f"# Report: Query {report['query_id']}",
        "",
        f"**Topic:** {report['topic']}",
        f"**Pipeline:** {report['pipeline']}",
        f"**Query:** {query.get('query', '')[:200]}...",
        "",
        "## Findings",
        "",
    ]
    for i, section in enumerate(report.get("sections", []), 1):
        cit = section.get("citation", {})
        vid = cit.get("video_id", "?")
        ts = normalize_timestamp(cit.get("timestamp"))
        ts_str = f" [{'-'.join(ts)}s]" if ts else ""
        lines.append(f"{i}. {section['text']} [video:{vid}{ts_str}]")
        lines.append("")
    lines.extend(
        [
            "## Citations",
            "",
            "| # | Video ID | Timestamp | Claim |",
            "|---|----------|-----------|-------|",
        ]
    )
    for i, cit in enumerate(report.get("citations", []), 1):
        ts = normalize_timestamp(cit.get("timestamp"))
        ts_str = f"{'-'.join(ts)}s" if ts else "N/A"
        lines.append(
            f"| {i} | {cit.get('video_id', '?')} | {ts_str} | {(cit.get('claim', ''))[:80]} |"
        )
    return "\n".join(lines)


def _validate_citations(citations: list) -> list:
    errors = []
    for i, cit in enumerate(citations):
        errs = validate_report_citation(cit)
        if errs:
            errors.extend([f"citation[{i}]: {e}" for e in errs])
        if cit.get("video_id") == "unknown":
            errors.append(f"citation[{i}]: unresolved video_id")
    return errors


def run_report(cfg: DictConfig) -> None:
    """Step 2 (legacy): generate reports from query packets."""
    b = cfg.runtime.bullet
    if not b.packets_dir or not cfg.output.out_dir:
        raise SystemExit("report requires runtime.bullet.packets_dir and output.out_dir")

    manifest = build_run_manifest(
        script_name="article_generation.bullet report",
        argv=sys.argv,
        args_dict=OmegaConf.to_container(b, resolve=True),
        run_config={"stream": b.stream, "strict": b.strict, "query_ids": cfg.data.query_ids},
    )
    manifest_path = write_run_manifest(cfg.output.out_dir, manifest)

    queries = load_queries(b.queries_jsonl)
    query_by_id = {q["query_id"]: q for q in queries}

    packets = _load_packets(b.packets_dir)
    print(f"Loaded {len(packets)} query packets")
    query_ids = _parse_query_ids(cfg.data.query_ids)
    if query_ids:
        before = len(packets)
        packets = _filter_packets_by_query_ids(packets, query_ids)
        print(f"Filtered packets by query_ids={sorted(query_ids)}: {before} -> {len(packets)}")

    stream = b.stream or None
    grounded_idx, note_idx, claim_idx, inference_idx = _load_evidence_indices(
        stream,
        b.grounded,
        b.notes,
        b.claims,
        b.inferences,
        verbose=True,
    )

    os.makedirs(cfg.output.out_dir, exist_ok=True)
    all_reports = []
    total_citation_errors = 0

    for packet in packets:
        qid = packet.get("query_id", "")
        query = query_by_id.get(qid, {"query_id": qid, "query": ""})
        report = _build_report(
            packet,
            query,
            stream,
            grounded_idx,
            note_idx,
            claim_idx,
            inference_idx,
            manifest["run_id"],
        )

        cit_errors = _validate_citations(report.get("citations", []))
        if cit_errors and b.verbose:
            for e in cit_errors:
                print(f"  WARN query {qid}: {e}")
        total_citation_errors += len(cit_errors)
        all_reports.append(report)

        with open(
            os.path.join(cfg.output.out_dir, f"report_{qid}.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        with open(os.path.join(cfg.output.out_dir, f"report_{qid}.md"), "w", encoding="utf-8") as f:
            f.write(_render_markdown_report(report, query))
        if b.verbose:
            section_count = len(report.get("sections", []))
            citation_count = len(report.get("citations", []))
            print(f"  query {qid}: {section_count} sections, {citation_count} citations")

    combined_path = os.path.join(cfg.output.out_dir, "all_reports.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2, ensure_ascii=False)

    print(f"\n[ok] {len(all_reports)} reports generated")
    if total_citation_errors:
        print(f"[warn] {total_citation_errors} citation resolution issues")
    print(f"[ok] -> {cfg.output.out_dir}/report_{{N}}.json / .md")
    print(f"[ok] -> {combined_path}")
    print(f"[ok] -> {manifest_path}")

    if b.strict and total_citation_errors > 0:
        raise SystemExit(f"[FAIL] --strict: {total_citation_errors} citation error(s)")


# Step 3: ANNOTATE


_MARKER_RE = re.compile(r"\[(\d+)\]")


def _format_citation_marker(cit: dict) -> str:
    vid = cit.get("video_id", "?")
    ts = normalize_timestamp(cit.get("timestamp"))
    if ts:
        return f"[{vid}, {'-'.join(ts)}s]"
    return f"[{vid}]"


def _remap_numeric_markers(text: str, source_citations: list) -> str:
    """Replace numeric [N] markers (1-based) with [video_id, start-end] markers."""

    def repl(m):
        n = int(m.group(1)) - 1
        if 0 <= n < len(source_citations):
            return _format_citation_marker(source_citations[n])
        return m.group(0)

    return _MARKER_RE.sub(repl, text)


def annotate_section(model, section: dict, verbose: bool = False) -> dict:
    """Insert inline [video_id, start-end] citation markers into section['text']."""
    source_citations = section.get("source_citations", [])
    text = section.get("text", "")
    if not source_citations or not text:
        return section

    if len(source_citations) == 1:
        section = dict(section)
        section["annotated_text"] = f"{text}{_format_citation_marker(source_citations[0])}"
        return section

    prompt = prompt_annotate_citations(text=text, sources=source_citations)
    retry = prompt_annotate_citations_retry(text=text, sources=source_citations)
    result = call_llm_with_retry(model, prompt, retry, "annotated_text", video_path=None)

    raw = result.get("annotated_text", "")
    if not raw:
        if verbose:
            print("    WARN: LLM returned empty annotated_text, falling back to appended markers")
        raw = text + "".join(f"[{i + 1}]" for i in range(len(source_citations)))

    section = dict(section)
    section["annotated_text"] = _remap_numeric_markers(raw, source_citations)
    return section


def _assemble_report_text(sections: list) -> str:
    parts = []
    for section in sections:
        text = (section.get("annotated_text") or section.get("text", "")).strip()
        if text:
            parts.append(text)
    return " ".join(parts)


def _render_markdown_annotate(report: dict, query: str) -> str:
    lines = [
        f"# Report: Query {report['query_id']}",
        "",
        f"**Topic:** {report['topic']}",
        f"**Pipeline:** {report['pipeline']}",
        f"**Query:** {query}",
        "",
        "## Findings",
        "",
    ]
    body = _assemble_report_text(report.get("sections", []))
    if body:
        lines.extend([body, ""])
    return "\n".join(lines)


def _load_reports(reports_dir: str) -> list:
    combined = os.path.join(reports_dir, "all_reports.json")
    if os.path.exists(combined):
        with open(combined, encoding="utf-8") as f:
            return json.load(f)
    reports = []
    for fname in sorted(os.listdir(reports_dir)):
        if fname.startswith("report_") and fname.endswith(".json"):
            with open(os.path.join(reports_dir, fname), encoding="utf-8") as f:
                reports.append(json.load(f))
    return reports


def _reports_from_inferences(b, query_id_filter, run_id) -> list:
    """Build report dicts straight from `infer` output (no separate report step)."""
    queries = load_queries(b.queries_jsonl)
    query_by_id = {q["query_id"]: q for q in queries}

    packets = _load_packets(b.packets_dir)
    print(f"Loaded {len(packets)} query packets from {b.packets_dir}")
    if query_id_filter:
        before = len(packets)
        packets = _filter_packets_by_query_ids(packets, query_id_filter)
        print(
            f"Filtered packets by query_ids={sorted(query_id_filter)}: {before} -> {len(packets)}"
        )

    stream = b.stream or None
    grounded_idx, note_idx, claim_idx, inference_idx = _load_evidence_indices(
        stream,
        b.grounded,
        b.notes,
        b.claims,
        b.inferences,
        verbose=b.verbose,
    )

    reports = []
    for packet in packets:
        qid = packet.get("query_id", "")
        query = query_by_id.get(qid, {"query_id": qid, "query": ""})
        reports.append(
            _build_report(
                packet, query, stream, grounded_idx, note_idx, claim_idx, inference_idx, run_id
            )
        )
    return reports


def run_annotate(cfg: DictConfig) -> None:
    """Step 3: build reports from infer output and inline-annotate citations."""
    b = cfg.runtime.bullet
    if not cfg.output.out_dir:
        raise SystemExit("annotate requires output.out_dir")
    if not b.inferences and not b.reports_dir:
        raise SystemExit(
            "annotate requires runtime.bullet.inferences "
            "(+ runtime.bullet.packets_dir/stream) or runtime.bullet.reports_dir"
        )

    manifest = build_run_manifest(
        script_name="article_generation.bullet annotate",
        argv=sys.argv,
        args_dict=OmegaConf.to_container(b, resolve=True),
        run_config={
            "stream": b.stream,
            "model": cfg.model.vlm,
            "api": cfg.model.api,
            "api_model": cfg.model.api_model,
            "query_ids": cfg.data.query_ids,
        },
    )

    query_id_filter = _parse_query_ids(cfg.data.query_ids)

    if b.inferences:
        if not b.packets_dir:
            raise SystemExit(
                "runtime.bullet.packets_dir is required when using runtime.bullet.inferences"
            )
        reports = _reports_from_inferences(b, query_id_filter, manifest["run_id"])
    else:
        print(f"Loading reports from: {b.reports_dir}")
        reports = _load_reports(b.reports_dir)
        if query_id_filter is not None:
            before = len(reports)
            reports = [r for r in reports if str(r.get("query_id")) in query_id_filter]
            print(
                f"  query_id filter {sorted(query_id_filter)}: kept {len(reports)}/{before} reports"
            )
    print(f"Loaded {len(reports)} reports")

    model = _load_model(
        cfg.model,
        temperature=b.annotate.temperature,
        max_tokens=b.annotate.max_tokens,
    )

    os.makedirs(cfg.output.out_dir, exist_ok=True)
    all_annotated = []

    for report in reports:
        qid = report.get("query_id", "?")
        query = report.get("query", "")
        annotated_sections = []
        for j, section in enumerate(report.get("sections", [])):
            if b.verbose:
                source_count = len(section.get("source_citations", []))
                print(f"  query {qid} section {j + 1}: {source_count} source(s)")
            annotated_sections.append(annotate_section(model, section, verbose=b.verbose))

        annotated_report = dict(report)
        annotated_report["sections"] = annotated_sections
        annotated_report["report"] = _assemble_report_text(annotated_sections)
        all_annotated.append(annotated_report)

        with open(
            os.path.join(cfg.output.out_dir, f"report_{qid}.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(annotated_report, f, indent=2, ensure_ascii=False)
        with open(os.path.join(cfg.output.out_dir, f"report_{qid}.md"), "w", encoding="utf-8") as f:
            f.write(_render_markdown_annotate(annotated_report, query))
        print(f"  query {qid}: {len(annotated_sections)} sections annotated")

    combined_path = os.path.join(cfg.output.out_dir, "all_reports.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_annotated, f, indent=2, ensure_ascii=False)
    manifest_path = write_run_manifest(cfg.output.out_dir, manifest)

    print(f"\n[ok] {len(all_annotated)} reports annotated")
    print(f"[ok] -> {cfg.output.out_dir}/report_{{N}}.json / .md")
    print(f"[ok] -> {combined_path}")
    print(f"[ok] -> {manifest_path}")


# Dispatcher

_STEPS = {"infer": run_infer, "report": run_report, "annotate": run_annotate}

# Convenience flags translated into Hydra overrides before composing the config.
_FLAG_TO_OVERRIDE = {"--api": "model.api", "--api-model": "model.api_model"}


def _translate_flags(argv: list[str]) -> list[str]:
    """Rewrite ``--api URL`` / ``--api=URL`` style flags as Hydra overrides."""
    overrides = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        flag, eq, inline_value = tok.partition("=")
        if flag in _FLAG_TO_OVERRIDE:
            if eq:
                value = inline_value
            else:
                if i + 1 >= len(argv):
                    raise SystemExit(f"{flag} requires a value")
                i += 1
                value = argv[i]
            overrides.append(f"{_FLAG_TO_OVERRIDE[flag]}={value}")
        else:
            overrides.append(tok)
        i += 1
    return overrides


def main(argv: list[str] | None = None) -> int:
    """Entry point: ``<step> [--api URL] [hydra overrides...]``.

    ``step`` ∈ {infer,report,annotate}. ``--api URL`` (shorthand for
    ``model.api=URL``) routes inference to a remote OpenAI-compatible server
    instead of loading the VLM locally; ``--api-model NAME`` selects the
    model ID on that server.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in _STEPS:
        print(
            "usage: marquis-generate bullet {infer|report|annotate} "
            "[--api URL] [--api-model NAME] [key=value ...]",
            file=sys.stderr,
        )
        return 2
    step, overrides = argv[0], _translate_flags(argv[1:])
    cfg = build_config(overrides)
    _STEPS[step](cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
