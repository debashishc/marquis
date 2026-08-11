"""
Step 1.5+: Packet assembly for both streams.

Assembles per-query packets from general notes or query-conditioned claims.

Two modes via --stream flag:
  - general-note: reads general notes, assembles one note_packet per query
  - query-based: reads query-conditioned claims, assembles one claim_packet per query

Usage:
    python src/information_extraction/assemble_packets.py \
        --stream general-note \
        --notes outputs/general_notes/general_notes.jsonl \
        --out-dir outputs/note_packets

    python src/information_extraction/assemble_packets.py \
        --stream query-based \
        --claims outputs/query_claims_single/query_conditioned_claims.jsonl \
        --out-dir outputs/claim_packets
"""

import datetime as _dt
import json
import os
import re
import sys
from collections import defaultdict

from omegaconf import DictConfig

from marquis.common.contracts import (
    load_queries,
    validate_claim_packet,
    validate_note_packet,
)
from marquis.common.run_metadata import build_run_manifest, write_run_manifest
from marquis.information_extraction._common import build_config, query_id_filter

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _utc_now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).replace(microsecond=0).isoformat()


def _iter_jsonl(path: str):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _tokenize(text: str) -> set:
    return set(_TOKEN_RE.findall(str(text or "").lower()))


def _get_unli_prob(rec: dict) -> float | None:
    """Extract UNLI probability from a calibrated record, or None if absent."""
    cal = rec.get("calibration")
    if isinstance(cal, dict):
        unli = cal.get("unli")
        if isinstance(unli, dict):
            p = unli.get("prob")
            if isinstance(p, (int, float)):
                return float(p)
    return None


def _support_score(rec: dict) -> float:
    prob = _get_unli_prob(rec)
    if prob is not None:
        return prob
    return float(rec.get("confidence") or 0.0)


def _score_note_against_query(note: dict, query: dict) -> float:
    """Token-overlap score between a note and a query."""
    note_tokens = _tokenize(note.get("text", ""))
    if not note_tokens:
        return 0.0

    query_text = " ".join(
        [
            query.get("query", ""),
            query.get("title", ""),
            query.get("persona_title", ""),
        ]
    )
    query_tokens = _tokenize(query_text)
    if not query_tokens:
        return 0.0

    overlap = len(note_tokens & query_tokens)
    return overlap / len(note_tokens)


def assemble_note_packets(
    notes_jsonl: str,
    queries: list[dict],
    query_topic: dict[str, str],
    run_id: str,
    top_k: int = 50,
    unli_threshold: float | None = None,
    verbose: bool = False,
) -> list[dict]:
    """Assemble note packets from general notes."""
    # Load all notes, applying optional UNLI threshold filter
    raw_notes: list[dict] = []
    total_loaded = 0
    filtered_out = 0
    missing_prob = 0
    for note in _iter_jsonl(notes_jsonl):
        total_loaded += 1
        if unli_threshold is not None:
            prob = _get_unli_prob(note)
            if prob is None:
                missing_prob += 1
                filtered_out += 1
                continue
            if prob < unli_threshold:
                filtered_out += 1
                continue
        raw_notes.append(note)

    if unli_threshold is not None:
        kept_notes = total_loaded - filtered_out
        print(
            f"  UNLI threshold={unli_threshold}: kept {kept_notes}/{total_loaded} notes "
            f"(filtered {filtered_out})"
        )
        if missing_prob:
            print(f"  WARN: filtered {missing_prob} notes without calibration.unli.prob")

    # Deduplicate notes with identical (video_id, text), keeping the first
    # occurrence and merging timestamp ranges.
    seen: dict[tuple, dict] = {}
    dedup_count = 0
    for note in raw_notes:
        dedup_key = (note.get("video_id", ""), note.get("text", ""))
        if dedup_key in seen:
            dedup_count += 1
            existing = seen[dedup_key]
            et = existing.get("timestamp")
            nt = note.get("timestamp")
            if isinstance(et, list) and isinstance(nt, list) and len(et) == 2 and len(nt) == 2:
                existing["timestamp"] = [min(et[0], nt[0]), max(et[1], nt[1])]
        else:
            seen[dedup_key] = note

    deduped_notes = list(seen.values())
    if dedup_count > 0:
        print(
            f"  Dedup: {len(raw_notes)} -> {len(deduped_notes)} notes "
            f"({dedup_count} duplicates merged)"
        )

    notes_by_topic: dict[str, list[dict]] = defaultdict(list)
    for note in deduped_notes:
        notes_by_topic[note.get("topic", "")].append(note)

    packets = []
    for q in sorted(queries, key=lambda q: str(q.get("query_id"))):
        topic = query_topic.get(str(q.get("query_id")), "")
        topic_notes = notes_by_topic.get(topic, [])
        scored = []
        for note in topic_notes:
            overlap = _score_note_against_query(note, q)
            if overlap > 0:
                unli_weight = _get_unli_prob(note) or 0.0
                score = overlap * unli_weight
                scored.append((score, note))
        scored.sort(key=lambda x: (-x[0], x[1].get("note_id", "")))
        top = scored[:top_k]

        packet = {
            "query_id": q["query_id"],
            "topic": topic,
            "stream": "general_note",
            "note_ids": [n.get("note_id", "") for _, n in top],
            "scores": [round(s, 4) for s, _ in top],
            "provenance": {
                "run_id": run_id,
                "retrieval_method": "token_overlap",
                "top_k": top_k,
                "unli_threshold": unli_threshold,
                "candidate_count": len(topic_notes),
                "created_at": _utc_now_iso(),
            },
        }
        packets.append(packet)

        if verbose:
            print(f"  query {q['query_id']} ({topic}): {len(top)} notes in packet")

    return packets


def assemble_claim_packets(
    claims_jsonl: str,
    queries: list[dict],
    query_topic: dict[str, str],
    run_id: str,
    unli_threshold: float | None = None,
    verbose: bool = False,
) -> list[dict]:
    """Assemble claim packets from query-conditioned claims."""
    # Group claims by query_id
    claims_by_query: dict[str, list[dict]] = defaultdict(list)
    total_loaded = 0
    filtered_out = 0
    missing_prob = 0
    for claim in _iter_jsonl(claims_jsonl):
        total_loaded += 1
        if unli_threshold is not None:
            prob = _get_unli_prob(claim)
            if prob is None:
                missing_prob += 1
                filtered_out += 1
                continue
            if prob < unli_threshold:
                filtered_out += 1
                continue
        claims_by_query[claim.get("query_id", "")].append(claim)

    if unli_threshold is not None:
        kept_claims = total_loaded - filtered_out
        print(
            f"  UNLI threshold={unli_threshold}: kept {kept_claims}/{total_loaded} "
            f"claims (filtered {filtered_out})"
        )
        if missing_prob:
            print(f"  WARN: filtered {missing_prob} claims without calibration.unli.prob")

    packets = []
    for q in sorted(queries, key=lambda q: str(q.get("query_id"))):
        qid = q["query_id"]
        topic = query_topic.get(str(qid), "")
        query_claims = claims_by_query.get(qid, [])

        # Rank by calibrated support probability when present, then confidence fallback.
        ranked = sorted(
            query_claims,
            key=lambda c: (-_support_score(c), c.get("claim_id", "")),
        )

        packet = {
            "query_id": qid,
            "topic": topic,
            "stream": "query_based",
            "claim_ids": [c.get("claim_id", "") for c in ranked],
            "scores": [round(_support_score(c), 4) for c in ranked],
            "provenance": {
                "run_id": run_id,
                "retrieval_method": "support_ranked",
                "score_field": "calibration.unli.prob_or_confidence",
                "unli_threshold": unli_threshold,
                "candidate_count": len(query_claims),
                "created_at": _utc_now_iso(),
            },
        }
        packets.append(packet)

        if verbose:
            print(f"  query {qid} ({topic}): {len(ranked)} claims in packet")

    return packets


# Hydra entry point


def run(cfg: DictConfig) -> None:
    """Assemble per-query packets for the general-note or query-based stream."""
    p = cfg.runtime.packets
    data = cfg.data
    stream = p.stream
    out_dir = cfg.output.out_dir
    if out_dir is None:
        raise SystemExit("packets requires output.out_dir")
    qidf = query_id_filter(data.query_ids)

    manifest = build_run_manifest(
        script_name="information_extraction.assemble_packets",
        argv=sys.argv,
        args_dict={"stream": stream, "top_k": p.top_k, "unli_threshold": p.unli_threshold},
        run_config={"stream": stream, "top_k": p.top_k, "unli_threshold": p.unli_threshold},
    )
    manifest_path = write_run_manifest(out_dir, manifest)

    queries = load_queries(data.queries_jsonl)
    # query_id -> topic label (notes/claims are organized per query, topic is a tag).
    query_topic = {str(q["query_id"]): str(q["topic_id"]) for q in queries}

    if qidf is not None:
        queries = [q for q in queries if str(q.get("query_id")) in qidf]
        print(f"  query_id filter {sorted(qidf)}: kept {len(queries)} queries")

    if stream == "general-note":
        if not data.notes:
            raise SystemExit("general-note stream requires data.notes")
        print(f"Assembling note packets from: {data.notes}")
        packets = assemble_note_packets(
            data.notes,
            queries,
            query_topic,
            manifest["run_id"],
            top_k=p.top_k,
            unli_threshold=p.unli_threshold,
            verbose=cfg.runtime.verbose,
        )
        validator = validate_note_packet
    elif stream == "query-based":
        if not data.claims:
            raise SystemExit("query-based stream requires data.claims")
        print(f"Assembling claim packets from: {data.claims}")
        packets = assemble_claim_packets(
            data.claims,
            queries,
            query_topic,
            manifest["run_id"],
            unli_threshold=p.unli_threshold,
            verbose=cfg.runtime.verbose,
        )
        validator = validate_claim_packet
    else:
        raise SystemExit(f"invalid runtime.packets.stream: {stream}")

    errors = []
    for packet in packets:
        errors.extend(validator(packet) or [])
    if errors:
        print(f"\nValidation issues ({len(errors)}):")
        for e in errors[:10]:
            print(f"  {e}")

    os.makedirs(out_dir, exist_ok=True)
    for packet in packets:
        with open(os.path.join(out_dir, f"query_{packet['query_id']}.json"), "w") as f:
            json.dump(packet, f, indent=2, ensure_ascii=False)
    combined_path = os.path.join(out_dir, "all_packets.json")
    with open(combined_path, "w") as f:
        json.dump(packets, f, indent=2, ensure_ascii=False)

    print(f"\n[ok] {len(packets)} {stream} packets assembled")
    print(f"[ok] -> {out_dir}/query_{{N}}.json")
    print(f"[ok] -> {combined_path}")
    print(f"[ok] -> {manifest_path}")


def main(argv: list | None = None) -> int:
    """Entry point: remaining argv are Hydra ``key=value`` overrides."""
    overrides = list(sys.argv[1:] if argv is None else argv)
    run(build_config(overrides))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
