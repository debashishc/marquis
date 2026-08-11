"""
Step 1: VLM extraction from raw video — general notes or query-conditioned claims.

A single entry point whose behavior is selected by --mode:
  - general         : Step 1a. Shared general prompt per video, no query
                      conditioning. Produces general_notes.jsonl
                      (per-video, modality-tagged, evidence-first notes).
  - query-single    : Step 1b. Per-query claim extraction using the original
                      query text.
  - query-expanded  : Step 1b. Per-query claim extraction using expanded
                      sub-queries.

The query-* modes write per-query query_{id}.jsonl plus a combined
query_conditioned_claims.jsonl.

Usage:
    # general notes
    python src/information_extraction/extract.py --mode general \
        --model Qwen/Qwen3.5-9B --out-dir outputs/general_notes

    # query-conditioned claims (single / expanded)
    python src/information_extraction/extract.py --mode query-single \
        --model Qwen/Qwen3.5-9B --out-dir outputs/query_claims_single
    python src/information_extraction/extract.py --mode query-expanded \
        --model Qwen/Qwen3.5-9B --out-dir outputs/query_claims_expanded
"""

import json
import os
import sys

from omegaconf import DictConfig

from marquis.common.contracts import (
    build_query_video_map,
    load_expanded_queries,
    load_queries,
    load_query_video_mapping,
    load_reference,
    resolve_video_path,
    validate_general_note,
    validate_query_conditioned_claim,
)
from marquis.common.prompts import call_llm_with_retry
from marquis.common.run_metadata import build_run_manifest, write_run_manifest
from marquis.information_extraction._common import build_config, query_id_filter
from marquis.information_extraction.prompts import (
    prompt_general_notes,
    prompt_general_notes_retry,
    prompt_query_claims,
    prompt_query_claims_expanded,
    prompt_query_claims_expanded_retry,
    prompt_query_claims_retry,
)

# Per-mode defaults that historically differed between the two scripts.
# query_mode is the internal label passed to the prompt builders.
MODE_DEFAULTS = {
    "general": {
        "query_mode": None,
        "max_tokens": 2048,
        "seed": 42,
        "out_dir": "outputs/general_notes",
    },
    "query-single": {
        "query_mode": "single-query",
        "max_tokens": 4096,
        "seed": 40,
        "out_dir": "outputs/query_claims_single_query",
    },
    "query-expanded": {
        "query_mode": "expanded-query",
        "max_tokens": 4096,
        "seed": 40,
        "out_dir": "outputs/query_claims_expanded_query",
    },
}


def _make_note_id(video_id: str, idx: int) -> str:
    return f"gn1a-{video_id}-{idx:03d}"


def _write_jsonl(path: str, records: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _read_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _read_query_claim_files(out_dir: str) -> list[dict]:
    records = []
    if not os.path.isdir(out_dir):
        return records
    for fname in sorted(os.listdir(out_dir)):
        if fname.startswith("query_") and fname.endswith(".jsonl"):
            records.extend(_read_jsonl(os.path.join(out_dir, fname)))
    return records


# ─────────────────────────────────────────────
# general mode
# ─────────────────────────────────────────────
def extract_general_notes(
    model,
    topic_map,
    video_root,
    run_id: str,
    *,
    include_topic_in_prompt: bool = True,
    verbose=False,
):
    """Extract general notes from raw video via VLM."""
    all_notes = []
    missing = []

    for topic, video_ids in topic_map.items():
        for vid in video_ids:
            video_id = str(vid)
            video_path = resolve_video_path(video_root, video_id)

            if video_path is None:
                missing.append(
                    {
                        "topic": topic,
                        "video_id": video_id,
                        "missing": ["video_file"],
                    }
                )
                if verbose:
                    print(f"  {topic}/{video_id}: SKIPPED (no video file)")
                continue

            prompt = prompt_general_notes(
                topic=topic,
                video_id=video_id,
                include_topic=include_topic_in_prompt,
            )
            retry = prompt_general_notes_retry(
                topic=topic,
                video_id=video_id,
                include_topic=include_topic_in_prompt,
            )
            result = call_llm_with_retry(
                model,
                prompt,
                retry,
                "notes",
                video_path=video_path,
            )

            for idx, note in enumerate(result.get("notes", [])):
                record = {
                    "note_id": _make_note_id(video_id, idx),
                    "video_id": video_id,
                    "topic": topic,
                    "text": note.get("text", ""),
                    "modality": note.get("modality", "visual"),
                    "timestamp": note.get("timestamp"),
                    "extractor": "vlm_general",
                    "confidence": None,
                    "source_path": video_path,
                    "run_id": run_id,
                    "is_post_grounded": False,
                }
                all_notes.append(record)

            if verbose:
                n = len(result.get("notes", []))
                print(f"  {topic}/{video_id}: {n} notes")

    return all_notes, missing


def validate_notes(notes: list, verbose: bool = False) -> list:
    """Run schema validation on all notes. Returns errors."""
    errors = []
    for i, note in enumerate(notes):
        errs = validate_general_note(note)
        if errs:
            errors.extend([f"note[{i}] {note.get('note_id', '?')}: {e}" for e in errs])
    if verbose and not errors:
        print(f"  Schema validation: {len(notes)} notes, all valid")
    return errors


def run_general(args, manifest, query_id_filter=None) -> None:
    queries = load_queries(args.queries_jsonl)
    # Video set comes straight from query_video_mapping (query_id -> videos):
    # general notes are query-agnostic, so we take the union of the selected
    # queries' videos with no intersection against the topic map. `topic` is only
    # a per-video label, resolved exactly as for query claims so notes and claims
    # carry identical labels.
    query_video_map = load_query_video_mapping(args.query_video_mapping)
    # Topic is only a per-video label here, taken straight from each query's topic_id.
    query_topic = {str(q["query_id"]): str(q["topic_id"]) for q in queries}

    topic_map: dict[str, list[str]] = {}
    for q in queries:
        qid = str(q.get("query_id"))
        if query_id_filter is not None and qid not in query_id_filter:
            continue
        bucket = topic_map.setdefault(query_topic.get(qid, ""), [])
        for v in query_video_map.get(qid, []):
            if v not in bucket:
                bucket.append(v)
    topic_map = {t: vids for t, vids in topic_map.items() if vids}

    kept = sum(len(v) for v in topic_map.values())
    scope = f"query_id filter {sorted(query_id_filter)}: " if query_id_filter is not None else ""
    print(f"  {scope}{kept} videos across {len(topic_map)} topics (from query_video_mapping)")

    model = _build_model(args)

    print("\nExtracting general notes from video...")
    notes, missing = extract_general_notes(
        model,
        topic_map,
        args.video_root,
        manifest["run_id"],
        include_topic_in_prompt=args.include_topic_in_prompt,
        verbose=args.verbose,
    )

    errors = validate_notes(notes, verbose=args.verbose)
    if errors:
        print(f"\nSchema validation FAILED ({len(errors)} errors):")
        for e in errors[:10]:
            print(f"  {e}")
        sys.exit(1)

    out_notes = os.path.join(args.out_dir, "general_notes.jsonl")
    out_missing = os.path.join(args.out_dir, "missing.jsonl")
    _write_jsonl(out_notes, notes)
    _write_jsonl(out_missing, missing)

    topics_with_notes = len(set(n["topic"] for n in notes))
    videos_with_notes = len(set(n["video_id"] for n in notes))
    print(
        f"\n[ok] {len(notes)} general notes from {videos_with_notes} videos "
        f"across {topics_with_notes} topics"
    )
    print(f"[ok] {len(missing)} videos with missing data")
    print(f"[ok] -> {out_notes}")
    print(f"[ok] -> {out_missing}")


# ─────────────────────────────────────────────
# query-* modes
# ─────────────────────────────────────────────
def extract_claims_vlm(
    model,
    query: dict,
    video_ids: list[str],
    video_root: str,
    query_mode: str,
    run_id: str,
    *,
    sub_queries: list[str] | None = None,
    topic: str = "",
) -> list[dict]:
    """Extract query-relevant claims from raw video via VLM, one video at a time."""
    all_claims = []

    qid = query["query_id"]
    query_text = query.get("query", "")
    persona_title = query.get("persona_title", "")
    background = query.get("background", "")

    for vid in video_ids:
        video_id = str(vid)
        video_path = resolve_video_path(video_root, video_id)

        if video_path is None:
            continue

        if query_mode == "single-query":
            prompt = prompt_query_claims(
                query_id=qid,
                query=query_text,
                persona_title=persona_title,
                background=background,
                topic=topic,
                video_id=video_id,
            )
            retry = prompt_query_claims_retry(
                query_id=qid,
                query=query_text,
                persona_title=persona_title,
                background=background,
                topic=topic,
                video_id=video_id,
            )
        else:
            prompt = prompt_query_claims_expanded(
                query_id=qid,
                query=query_text,
                persona_title=persona_title,
                background=background,
                topic=topic,
                video_id=video_id,
                sub_queries=sub_queries or [],
            )
            retry = prompt_query_claims_expanded_retry(
                query_id=qid,
                query=query_text,
                persona_title=persona_title,
                background=background,
                topic=topic,
                video_id=video_id,
                sub_queries=sub_queries or [],
            )

        result = call_llm_with_retry(
            model,
            prompt,
            retry,
            "claims",
            video_path=video_path,
        )

        for idx, claim_data in enumerate(result.get("claims", [])):
            claim_id = f"qc-{qid}-{video_id}-{idx:03d}"
            record = {
                "claim_id": claim_id,
                "query_id": qid,
                "video_id": video_id,
                "topic": topic,
                "claim": claim_data.get("claim", ""),
                "confidence": claim_data.get("confidence"),
                "evidence": claim_data.get("evidence"),
                "source": claim_data.get("source"),
                "timestamp": claim_data.get("timestamp"),
                "source_path": video_path,
                "run_id": run_id,
                "is_post_grounded": False,
            }
            all_claims.append(record)

    return all_claims


def run_query(args, manifest, query_id_filter=None) -> None:
    queries = load_queries(args.queries_jsonl)
    # Video selection: query_id -> [video_id, ...], read directly from the file.
    query_video_map = load_query_video_mapping(args.query_video_mapping)
    # Topic is kept only as a per-query label, taken straight from the query's topic_id.
    query_topic = {str(q["query_id"]): str(q["topic_id"]) for q in queries}

    expanded = {}
    if args.query_mode == "expanded-query":
        expanded = load_expanded_queries(args.expanded_queries)

    print(f"Query mode: {args.query_mode}")
    if query_id_filter is not None:
        print(f"Query id filter: {sorted(query_id_filter)}")
    print(f"Queries: {len(queries)}")

    model = _build_model(args)

    os.makedirs(args.out_dir, exist_ok=True)
    all_claims = []

    for q in sorted(queries, key=lambda q: str(q["query_id"])):
        qid = q["query_id"]
        if query_id_filter is not None and str(qid) not in query_id_filter:
            continue
        topic = query_topic.get(str(qid), "")
        video_ids = query_video_map.get(str(qid), [])

        query_path = os.path.join(args.out_dir, f"query_{qid}.jsonl")
        # Resume: skip queries already extracted in a prior run.
        if os.path.exists(query_path):
            with open(query_path) as f:
                if len(f.readlines()) > 0:
                    all_claims.extend(_read_jsonl(query_path))
                    continue

        sub_queries = None
        if args.query_mode == "expanded-query":
            exp = expanded.get(qid, {})
            sub_queries = exp.get("sub_queries", [])

        claims = extract_claims_vlm(
            model,
            q,
            video_ids,
            args.video_root,
            args.query_mode,
            manifest["run_id"],
            sub_queries=sub_queries,
            topic=topic,
        )

        for claim in claims:
            errs = validate_query_conditioned_claim(claim)
            if errs and args.verbose:
                print(f"  WARN claim {claim.get('claim_id')}: {errs}")

        all_claims.extend(claims)
        _write_jsonl(query_path, claims)

        if args.verbose:
            sq_info = f" ({len(sub_queries)} subqueries)" if sub_queries else ""
            print(f"  query {qid} ({topic}): {len(claims)} claims{sq_info}")

    combined_path = os.path.join(args.out_dir, "query_conditioned_claims.jsonl")
    all_claims = _read_query_claim_files(args.out_dir)
    _write_jsonl(combined_path, all_claims)

    print(f"\n[ok] {len(all_claims)} query-conditioned claims across {len(queries)} queries")
    print(f"[ok] -> {args.out_dir}/query_{{N}}.jsonl")
    print(f"[ok] -> {combined_path}")


# ─────────────────────────────────────────────
# shared
# ─────────────────────────────────────────────
def _build_model(args):
    from marquis.common.model_backends import Qwen3_5_VL  # lazy

    print(f"\nInitializing model: {args.model}")
    return Qwen3_5_VL(
        model=args.model,
        download_dir=args.download_dir,
        fps=args.fps,
        enable_thinking=args.enable_thinking,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        seed=args.seed,
        repetition_penalty=args.repetition_penalty,
        presence_penalty=args.presence_penalty,
        max_frames=args.max_frames,
        allowed_local_media_path=args.video_root,
    )


# Hydra entry points


def _args_from_cfg(cfg: DictConfig, mode: str):
    """Build an argparse-style namespace the run_* workers expect, from cfg."""
    from types import SimpleNamespace

    md = MODE_DEFAULTS[mode]
    r = cfg.runtime
    data = cfg.data
    return SimpleNamespace(
        mode=mode,
        query_mode=md["query_mode"],
        query_video_mapping=data.query_video_mapping,
        video_root=data.video_root,
        model=cfg.model.model,
        download_dir=cfg.model.download_dir or "",
        fps=r.fps,
        max_frames=r.max_frames,
        enable_thinking=r.enable_thinking,
        temperature=r.temperature,
        top_p=r.top_p,
        top_k=r.top_k,
        max_tokens=r.max_tokens if r.max_tokens is not None else md["max_tokens"],
        seed=r.seed if r.seed is not None else md["seed"],
        repetition_penalty=r.repetition_penalty,
        presence_penalty=r.presence_penalty,
        out_dir=cfg.output.out_dir if cfg.output.out_dir is not None else md["out_dir"],
        include_topic_in_prompt=r.include_topic_in_prompt,
        queries_jsonl=data.queries_jsonl,
        expanded_queries=data.expanded_queries,
        verbose=r.verbose,
        resolved_config_out="",
    )


def run(cfg: DictConfig, mode: str) -> None:
    """Run Step-1 extraction for ``mode`` (general / query-single / query-expanded)."""
    args = _args_from_cfg(cfg, mode)
    qid_filter = query_id_filter(cfg.data.query_ids)

    run_config = {
        "mode": mode,
        "query_ids": sorted(qid_filter) if qid_filter else None,
        "model": args.model,
        "download_dir": args.download_dir,
        "fps": args.fps,
        "max_frames": args.max_frames,
        "enable_thinking": args.enable_thinking,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "repetition_penalty": args.repetition_penalty,
        "presence_penalty": args.presence_penalty,
        "allowed_local_media_path": args.video_root,
    }
    if mode == "general":
        run_config["include_topic_in_prompt"] = args.include_topic_in_prompt
    else:
        run_config["query_mode"] = args.query_mode

    manifest = build_run_manifest(
        script_name="information_extraction.extract",
        argv=sys.argv,
        args_dict=vars(args),
        run_config=run_config,
    )
    manifest_path = write_run_manifest(args.out_dir, manifest)

    if mode == "general":
        run_general(args, manifest, qid_filter)
    else:
        run_query(args, manifest, qid_filter)
    print(f"[ok] -> {manifest_path}")


def general_notes_main(argv: list | None = None) -> int:
    """`general-notes` command: Step-1a general notes."""
    overrides = list(sys.argv[1:] if argv is None else argv)
    run(build_config(overrides), mode="general")
    return 0


def query_claims_main(argv: list | None = None) -> int:
    """`query-claims` command: Step-1b query-conditioned claims (single|expanded)."""
    overrides = list(sys.argv[1:] if argv is None else argv)
    cfg = build_config(overrides)
    mode = f"query-{cfg.runtime.query_mode}"  # query-single | query-expanded
    if mode not in MODE_DEFAULTS:
        raise SystemExit(
            f"invalid runtime.query_mode: {cfg.runtime.query_mode} (use single|expanded)"
        )
    run(cfg, mode=mode)
    return 0


def build_query_mapping_main(argv: list | None = None) -> int:
    """`build-query-mapping` command: write query_video_mapping.json from
    reference.json + queries (joined by topic_id; the file extract/qa read directly)."""
    overrides = list(sys.argv[1:] if argv is None else argv)
    cfg = build_config(overrides)
    out_path = cfg.data.query_video_mapping

    queries = load_queries(cfg.data.queries_jsonl)
    reference = load_reference(cfg.data.reference)
    query_video_map = build_query_video_map(queries, reference)
    query_video_map = {k: query_video_map[k] for k in sorted(query_video_map, key=lambda x: int(x))}

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(query_video_map, f, indent=2, ensure_ascii=False)
        f.write("\n")
    total = sum(len(v) for v in query_video_map.values())
    print(f"[ok] {len(query_video_map)} queries, {total} video assignments -> {out_path}")
    return 0


def prepare_videos_main(argv: list | None = None) -> int:
    """`prepare-videos` command: materialize the loose ``{video_id}.mp4`` files
    that extract/generate read (``data.video_root``) out of the MicroVENT
    WebDataset tar shards (``data.video_shards_root``).

    Video selection reuses the same ``query_video_mapping`` + ``query_ids``
    filter as extraction, so a run prepares exactly the videos its queries need
    (``data.query_ids`` null = every query in the mapping)."""
    from marquis.common.video import materialize_videos

    overrides = list(sys.argv[1:] if argv is None else argv)
    cfg = build_config(overrides)
    query_video_map = load_query_video_mapping(cfg.data.query_video_mapping)
    qid_filter = query_id_filter(cfg.data.query_ids)

    video_ids = [
        v
        for qid, vids in query_video_map.items()
        if qid_filter is None or qid in qid_filter
        for v in vids
    ]
    scope = f"query_id filter {sorted(qid_filter)}" if qid_filter is not None else "all queries"
    print(f"prepare-videos ({scope}): {cfg.data.video_shards_root} -> {cfg.data.video_root}")
    summary = materialize_videos(
        video_ids,
        cfg.data.video_root,
        dataset_root=cfg.data.video_shards_root,
        force=cfg.runtime.prepare_videos.force,
    )
    return 0 if not summary["missing"] else 1


def prepare_audio_main(argv: list | None = None) -> int:
    """`prepare-audio` command: materialize the loose ``{chunk_id}{audio_ext}``
    files that QA's Whisper transcribes (``data.audio_root``) out of the MicroVENT
    audio WebDataset shards (``<data.video_shards_root>/audio``).

    Audio selection reuses the same ``query_video_mapping`` + ``query_ids`` filter
    as ``prepare-videos``, so a run prepares exactly the audio its queries need."""
    from marquis.common.video import materialize_audio

    overrides = list(sys.argv[1:] if argv is None else argv)
    cfg = build_config(overrides)
    query_video_map = load_query_video_mapping(cfg.data.query_video_mapping)
    qid_filter = query_id_filter(cfg.data.query_ids)

    video_ids = [
        v
        for qid, vids in query_video_map.items()
        if qid_filter is None or qid in qid_filter
        for v in vids
    ]
    scope = f"query_id filter {sorted(qid_filter)}" if qid_filter is not None else "all queries"
    print(f"prepare-audio ({scope}): {cfg.data.video_shards_root}/audio -> {cfg.data.audio_root}")
    summary = materialize_audio(
        video_ids,
        cfg.data.audio_root,
        dataset_root=cfg.data.video_shards_root,
        ext=cfg.data.get("audio_ext", ".m4a"),
        force=cfg.runtime.prepare_videos.force,
        silent_for_missing=cfg.runtime.prepare_videos.get("silent_for_missing", True),
    )
    return 0 if not summary["missing"] else 1


if __name__ == "__main__":
    raise SystemExit(general_notes_main())
