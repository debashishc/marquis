"""
Step 1.5: Video-grounded support scoring + calibration (unified entry point).

Behavior is selected by --stage:
  - predict   : Score each artifact (general note / query-conditioned claim)
                against its source video, producing unli_predictions.jsonl.
  - calibrate : Merge support probabilities back into the artifact records
                (writes calibration.unli + confidence), producing
                <artifact>_calibrated.jsonl.
  - all       : predict -> calibrate in one command (default).

Artifact types:
  - general-notes : general_notes.jsonl from extract.py --mode general
  - query-claims  : query_conditioned_claims.jsonl from extract.py --mode query-*
  - claims        : legacy per-video claims.jsonl (calibrate-only)
  - inferences    : higher-level inference JSONL (calibrate-only)

Usage:
    # full pipeline (score then calibrate)
    python src/information_extraction/calibrate.py --stage all \
        --artifact-type general-notes --scorer-backend unli \
        --artifacts-jsonl outputs/general_notes/general_notes.jsonl \
        --out-dir outputs/unli_general_notes

    # only score
    python src/information_extraction/calibrate.py --stage predict \
        --artifact-type query-claims \
        --artifacts-jsonl outputs/query_claims_single/query_conditioned_claims.jsonl \
        --out-dir outputs/unli_query_claims

    # only merge (predictions produced elsewhere)
    python src/information_extraction/calibrate.py --stage calibrate \
        --artifact-type general-notes \
        --artifacts-jsonl outputs/general_notes/general_notes.jsonl \
        --unli-jsonl outputs/unli_general_notes/unli_predictions.jsonl \
        --out-dir outputs/unli_general_notes
"""

import json
import os
import re
import sys
from collections import defaultdict
from collections.abc import Iterable

from omegaconf import DictConfig

from marquis.common.contracts import (
    DEFAULT_UNLI_LORA_PATH,
    DEFAULT_UNLI_MODEL,
    DEFAULT_VLM_MODEL,
    resolve_video_path,
)
from marquis.common.run_metadata import build_run_manifest, write_run_manifest
from marquis.information_extraction._common import build_config, query_id_filter
from marquis.information_extraction.prompts import (
    parse_qwen_score_answer,
    prompt_qwen_score,
    prompt_qwen_score_retry,
)

# NOTE: marquis.common.model_backends (torch / transformers) is imported lazily inside the
# predict stage so that --stage calibrate stays dependency-light.

PREDICTIONS_NAME = "unli_predictions.jsonl"

_ANSWER_RE = re.compile(r"<answer>\s*([0-9]*\.?[0-9]+)\s*</answer>", re.IGNORECASE)


def _iter_jsonl(path: str) -> Iterable[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _write_jsonl(path: str, records: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _calibrated_name(artifact_type: str) -> str:
    return {
        "general-notes": "general_notes_calibrated.jsonl",
        "query-claims": "query_conditioned_claims_calibrated.jsonl",
        "inferences": "inferences_calibrated.jsonl",
    }.get(artifact_type, "calibrated.jsonl")


# ─────────────────────────────────────────────
# predict stage
# ─────────────────────────────────────────────
def _build_unli(args):
    from marquis.common.model_backends import UNLI  # lazy

    unli_kwargs = dict(
        model=args.model,
        download_dir=args.download_dir,
        fps=args.fps,
        resized_height=args.resized_height,
        resized_width=args.resized_width,
        new_token_num=args.new_token_num,
        new_token_prefix=args.new_token_prefix,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
    )
    if args.base_model is not None:
        unli_kwargs["base_model"] = args.base_model
    if args.lora_path is not None:
        unli_kwargs["lora_path"] = args.lora_path
    return UNLI(**unli_kwargs)


def _build_qwen_scorer(args):
    from marquis.common.model_backends import Qwen3_5_VL  # lazy

    return Qwen3_5_VL(
        model=args.model,
        download_dir=args.download_dir,
        fps=args.qwen_fps,
        max_frames=args.max_frames,
        enable_thinking=args.enable_thinking,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        seed=args.seed,
        repetition_penalty=args.repetition_penalty,
        presence_penalty=args.presence_penalty,
        allowed_local_media_path=args.video_root,
    )


def _score_with_qwen(model, video_path: str, text: str) -> tuple[float | None, str | None]:
    raw_output = model.infer(video_path=video_path, query=prompt_qwen_score(text))
    prob = parse_qwen_score_answer(raw_output)
    if prob is not None:
        return prob, raw_output

    retry_output = model.infer(video_path=video_path, query=prompt_qwen_score_retry(text))
    prob = parse_qwen_score_answer(retry_output)
    return prob, retry_output


def run_predict(args, query_id_filter: set | None) -> str:
    """Score artifacts against source video. Returns the predictions JSONL path."""
    out_path = os.path.join(args.out_dir, PREDICTIONS_NAME)
    os.makedirs(args.out_dir, exist_ok=True)

    manifest = build_run_manifest(
        script_name="src/information_extraction/calibrate.py (predict)",
        argv=sys.argv,
        args_dict=vars(args),
        run_config={
            "stage": "predict",
            "artifact_type": args.artifact_type,
            "scorer_backend": args.scorer_backend,
            "model": args.model,
            "base_model": args.base_model,
            "lora_path": args.lora_path,
            "download_dir": args.download_dir,
            "fps": args.fps,
            "resized_height": args.resized_height,
            "resized_width": args.resized_width,
            "new_token_num": args.new_token_num,
            "new_token_prefix": args.new_token_prefix,
            "device_map": args.device_map,
            "torch_dtype": args.torch_dtype,
            "attn_implementation": args.attn_implementation,
            "qwen_fps": args.qwen_fps,
            "max_frames": args.max_frames,
            "enable_thinking": args.enable_thinking,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "repetition_penalty": args.repetition_penalty,
            "presence_penalty": args.presence_penalty,
            "prompt_family": "qwen_step1_5_score_v1"
            if args.scorer_backend == "qwen_score"
            else "unli_conf_tokens_v1",
        },
    )
    manifest_path = write_run_manifest(args.out_dir, manifest, filename="predict_run_manifest.json")

    # Load and group artifacts by video_id
    print(f"[predict] Loading artifacts: {args.artifacts_jsonl}")
    groups = defaultdict(list)
    filtered_by_qid = 0
    for rec in _iter_jsonl(args.artifacts_jsonl):
        if query_id_filter is not None:
            if str(rec.get("query_id") or "") not in query_id_filter:
                filtered_by_qid += 1
                continue
        vid = str(rec.get("video_id") or "").strip()
        if vid:
            groups[vid].append(rec)
    print(f"  {sum(len(v) for v in groups.values())} items across {len(groups)} videos")
    if query_id_filter is not None:
        print(f"  query_id filter {sorted(query_id_filter)}: dropped {filtered_by_qid} records")

    # Initialize scorer backend
    if args.scorer_backend == "unli" and args.base_model:
        print(f"\n[predict] Initializing UNLI model: {args.base_model} + LoRA {args.lora_path}")
    elif args.scorer_backend == "unli":
        print(f"\n[predict] Initializing UNLI model: {args.model}")
    else:
        print(f"\n[predict] Initializing Qwen scorer model: {args.model}")
    scorer = _build_unli(args) if args.scorer_backend == "unli" else _build_qwen_scorer(args)

    # Score each item
    print("\n[predict] Scoring items...")
    scored = 0
    skipped_videos = 0
    skipped_items = 0

    with open(out_path, "w") as outf:
        for video_id in sorted(groups.keys()):
            video_path = resolve_video_path(args.video_root, video_id)
            if video_path is None:
                skipped_videos += 1
                skipped_items += len(groups[video_id])
                group_size = len(groups[video_id])
                print(f"  WARN: video not found for {video_id}, skipping {group_size} items")
                continue

            for rec in groups[video_id]:
                if args.artifact_type == "general-notes":
                    text = str(rec.get("text") or "").strip()
                    stable_id = rec.get("note_id")
                else:
                    text = str(rec.get("claim") or "").strip()
                    stable_id = rec.get("claim_id")

                if not text:
                    skipped_items += 1
                    continue

                raw_output = None
                try:
                    if args.scorer_backend == "unli":
                        prob = scorer.score(video_path, text)
                    else:
                        prob, raw_output = _score_with_qwen(scorer, video_path, text)
                        if prob is None:
                            raise ValueError(
                                "Qwen scorer did not return a parseable <answer> score"
                            )
                except Exception as e:
                    skipped_items += 1
                    if args.verbose:
                        print(f"  WARN: scoring failed for {stable_id or video_id}: {e}")
                    continue

                pred = {"video_id": video_id, "prob": float(prob)}
                if args.artifact_type == "general-notes":
                    pred["note_id"] = stable_id
                    pred["text"] = text
                else:
                    pred["claim_id"] = stable_id
                    pred["claim"] = text
                if raw_output is not None:
                    pred["raw_output"] = raw_output

                outf.write(json.dumps(pred, ensure_ascii=False) + "\n")
                outf.flush()
                scored += 1

            if args.verbose:
                print(f"  {video_id}: scored {len(groups[video_id])} items")

    print(
        f"\n[predict] Scored: {scored}, skipped videos: {skipped_videos}, "
        f"skipped items: {skipped_items}"
    )
    print(f"[predict] -> {out_path}")
    print(f"[predict] -> {manifest_path}")
    return out_path


# ─────────────────────────────────────────────
# calibrate stage
# ─────────────────────────────────────────────
def _parse_prob(text: str) -> float | None:
    if text is None:
        return None
    s = str(text).strip()
    m = _ANSWER_RE.search(s)
    if m:
        try:
            val = float(m.group(1))
        except Exception:
            return None
        if 0.0 <= val <= 1.0:
            return val
    try:
        val = float(s)
    except Exception:
        return None
    if 0.0 <= val <= 1.0:
        return val
    return None


def _unli_key(id_a: str, text: str) -> tuple[str, str]:
    return (str(id_a or "").strip(), str(text or "").strip())


def _extract_unli_record(rec: dict) -> tuple[tuple[str, str], str | None, dict] | None:
    """Parse a single UNLI prediction record.

    Returns (tuple_key, stable_id_or_None, payload) on success, None on failure.
    """
    if not isinstance(rec, dict):
        return None

    video_id = rec.get("video_id")
    query_id = rec.get("query_id")
    claim = rec.get("claim")
    prob = rec.get("prob")

    meta = rec.get("meta") if isinstance(rec.get("meta"), dict) else {}
    if video_id is None:
        video_id = meta.get("video_id")
    if query_id is None:
        query_id = meta.get("query_id")
    if claim is None:
        claim = meta.get("claim")

    if claim is None:
        claim = (
            rec.get("text")
            if isinstance(rec.get("text"), str) and not _ANSWER_RE.search(str(rec.get("text", "")))
            else None
        )
    if claim is None:
        claim = meta.get("text")

    prob_val = None
    if prob is not None:
        try:
            prob_val = float(prob)
        except Exception:
            prob_val = None
    if prob_val is None:
        if isinstance(rec.get("text"), str):
            prob_val = _parse_prob(rec.get("text"))
        elif isinstance(rec.get("outputs"), list) and rec.get("outputs"):
            prob_val = _parse_prob(rec.get("outputs")[-1])

    if claim is None or prob_val is None:
        return None

    if not (0.0 <= prob_val <= 1.0):
        return None

    primary_id = video_id if video_id is not None else query_id
    if primary_id is None:
        return None

    stable_id = (
        rec.get("note_id")
        or rec.get("claim_id")
        or rec.get("inference_id")
        or meta.get("note_id")
        or meta.get("claim_id")
        or meta.get("inference_id")
    )

    key = _unli_key(primary_id, claim)
    payload = {
        "prob": prob_val,
        "raw": rec,
    }
    if isinstance(meta.get("label"), (int, float, str)):
        payload["label"] = meta.get("label")
    return key, stable_id, payload


def _build_unli_index(unli_jsonl: str) -> tuple[dict[tuple[str, str], dict], dict[str, dict]]:
    """Build UNLI lookup indices (by (primary_id, text) tuple and by stable ID)."""
    key_index: dict[tuple[str, str], dict] = {}
    id_index: dict[str, dict] = {}
    for rec in _iter_jsonl(unli_jsonl):
        parsed = _extract_unli_record(rec)
        if parsed is None:
            continue
        key, stable_id, payload = parsed
        if key not in key_index:
            key_index[key] = payload
        if stable_id is not None and stable_id not in id_index:
            id_index[stable_id] = payload
    return key_index, id_index


def _calibrate_claims(
    claims_jsonl: str, key_index: dict, id_index: dict, out: str
) -> tuple[int, int]:
    """Calibrate per-video claims.jsonl (legacy format — no stable IDs)."""
    total_claims = 0
    matched = 0
    with open(out, "w") as outf:
        for video_rec in _iter_jsonl(claims_jsonl):
            if not isinstance(video_rec, dict):
                continue
            video_id = str(video_rec.get("video_id") or "").strip()
            claims = video_rec.get("claims") if isinstance(video_rec.get("claims"), list) else []
            new_claims = []
            for c in claims:
                total_claims += 1
                if not isinstance(c, dict):
                    new_claims.append(c)
                    continue
                claim_text = str(c.get("claim") or "").strip()
                key = _unli_key(video_id, claim_text)
                item = dict(c)
                if key in key_index:
                    item["calibration"] = {"unli": key_index[key]}
                    item["confidence"] = key_index[key].get("prob")
                    matched += 1
                new_claims.append(item)
            new_video_rec = dict(video_rec)
            new_video_rec["claims"] = new_claims
            outf.write(json.dumps(new_video_rec, ensure_ascii=False) + "\n")
    return total_claims, matched


def _calibrate_general_notes(
    notes_jsonl: str, key_index: dict, id_index: dict, out: str
) -> tuple[int, int]:
    """Calibrate general_notes.jsonl — prefer note_id match, fall back to (video_id, text)."""
    total = 0
    matched = 0
    records = []
    for rec in _iter_jsonl(notes_jsonl):
        total += 1
        item = dict(rec)
        note_id = rec.get("note_id")
        payload = id_index.get(note_id) if note_id else None
        if payload is None:
            video_id = str(rec.get("video_id") or "").strip()
            text = str(rec.get("text") or "").strip()
            key = _unli_key(video_id, text)
            payload = key_index.get(key)
        if payload is not None:
            item["calibration"] = {"unli": payload}
            item["confidence"] = payload.get("prob")
            matched += 1
        records.append(item)
    _write_jsonl(out, records)
    return total, matched


def _calibrate_query_claims(
    claims_jsonl: str, key_index: dict, id_index: dict, out: str, query_id_filter: set | None = None
) -> tuple[int, int]:
    """Calibrate query claims by claim_id, falling back to (video_id, claim)."""
    total = 0
    matched = 0
    records = []
    for rec in _iter_jsonl(claims_jsonl):
        if query_id_filter is not None and str(rec.get("query_id") or "") not in query_id_filter:
            continue
        total += 1
        item = dict(rec)
        claim_id = rec.get("claim_id")
        payload = id_index.get(claim_id) if claim_id else None
        if payload is None:
            video_id = str(rec.get("video_id") or "").strip()
            claim_text = str(rec.get("claim") or "").strip()
            key = _unli_key(video_id, claim_text)
            payload = key_index.get(key)
        if payload is not None:
            item["calibration"] = {"unli": payload}
            item["confidence"] = payload.get("prob")
            matched += 1
        records.append(item)
    _write_jsonl(out, records)
    return total, matched


def _calibrate_inferences(
    inferences_jsonl: str,
    key_index: dict,
    id_index: dict,
    out: str,
    query_id_filter: set | None = None,
) -> tuple[int, int]:
    """Calibrate inference JSONL — prefer inference_id match, fall back to (query_id, claim)."""
    total = 0
    matched = 0
    records = []
    for rec in _iter_jsonl(inferences_jsonl):
        if query_id_filter is not None and str(rec.get("query_id") or "") not in query_id_filter:
            continue
        total += 1
        item = dict(rec)
        inference_id = rec.get("inference_id")
        payload = id_index.get(inference_id) if inference_id else None
        if payload is None:
            query_id = str(rec.get("query_id") or "").strip()
            claim_text = str(rec.get("claim") or "").strip()
            key = _unli_key(query_id, claim_text)
            payload = key_index.get(key)
        if payload is not None:
            item["calibration"] = {"unli": payload}
            item["confidence"] = payload.get("prob")
            matched += 1
        records.append(item)
    _write_jsonl(out, records)
    return total, matched


def run_calibrate(args, unli_jsonl: str, query_id_filter: set | None) -> str:
    """Merge support probabilities back into artifacts. Returns calibrated JSONL path."""
    out_path = os.path.join(args.out_dir, _calibrated_name(args.artifact_type))
    os.makedirs(args.out_dir, exist_ok=True)

    manifest = build_run_manifest(
        script_name="src/information_extraction/calibrate.py (calibrate)",
        argv=sys.argv,
        args_dict=vars(args),
        run_config={
            "stage": "calibrate",
            "artifact_type": args.artifact_type,
            "claims_jsonl": args.artifacts_jsonl,
            "unli_jsonl": unli_jsonl,
            "out": out_path,
        },
        extra={"prediction_manifest": "predict_run_manifest.json"},
    )
    manifest_path = write_run_manifest(
        args.out_dir, manifest, filename="calibrate_run_manifest.json"
    )

    print(f"[calibrate] Indexing predictions: {unli_jsonl}")
    key_index, id_index = _build_unli_index(unli_jsonl)

    if args.artifact_type == "claims":
        total, matched = _calibrate_claims(args.artifacts_jsonl, key_index, id_index, out_path)
    elif args.artifact_type == "general-notes":
        total, matched = _calibrate_general_notes(
            args.artifacts_jsonl, key_index, id_index, out_path
        )
    elif args.artifact_type == "query-claims":
        total, matched = _calibrate_query_claims(
            args.artifacts_jsonl, key_index, id_index, out_path, query_id_filter
        )
    elif args.artifact_type == "inferences":
        total, matched = _calibrate_inferences(
            args.artifacts_jsonl, key_index, id_index, out_path, query_id_filter
        )

    print(f"[calibrate] UNLI records indexed: {len(key_index)} by key, {len(id_index)} by ID")
    print(f"[calibrate] Artifacts processed: {total}, matched: {matched}")
    print(f"[calibrate] -> {out_path}")
    print(f"[calibrate] -> {manifest_path}")
    return out_path


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────


# Hydra entry points


def _args_from_cfg(cfg: DictConfig):
    """Build the argparse-style namespace run_predict/run_calibrate expect."""
    from types import SimpleNamespace

    c = cfg.runtime.calibrate
    r = cfg.runtime
    data = cfg.data

    if cfg.output.out_dir is None:
        raise SystemExit("calibrate requires output.out_dir")
    if data.artifacts_jsonl is None:
        raise SystemExit("calibrate requires data.artifacts_jsonl")

    # Resolve scorer backend + model from the selected model group, falling back
    # to calibrate.scorer_backend defaults (mirrors the original auto-selection).
    mb = cfg.model.get("backend")
    if mb == "unli":
        sb, model, base_model, lora_path = (
            "unli",
            cfg.model.model,
            cfg.model.base_model,
            cfg.model.lora_path,
        )
    elif mb == "qwen_score":
        sb, model, base_model, lora_path = "qwen_score", cfg.model.model, None, None
    else:
        sb = c.scorer_backend
        if sb == "qwen_score":
            model, base_model, lora_path = DEFAULT_VLM_MODEL, None, None
        else:
            model, base_model, lora_path = None, DEFAULT_UNLI_MODEL, DEFAULT_UNLI_LORA_PATH

    return SimpleNamespace(
        stage=c.stage,
        artifact_type=c.artifact_type,
        artifacts_jsonl=data.artifacts_jsonl,
        unli_jsonl=data.unli_jsonl,
        out_dir=cfg.output.out_dir,
        scorer_backend=sb,
        video_root=data.video_root,
        model=model,
        base_model=base_model,
        lora_path=lora_path,
        download_dir=cfg.model.download_dir or "",
        fps=c.fps,
        resized_height=c.resized_height,
        resized_width=c.resized_width,
        new_token_num=c.new_token_num,
        new_token_prefix=c.new_token_prefix,
        device_map=c.device_map,
        torch_dtype=c.torch_dtype,
        attn_implementation=c.attn_implementation,
        max_frames=r.max_frames,
        enable_thinking=r.enable_thinking,
        temperature=r.temperature,
        top_p=r.top_p,
        top_k=r.top_k,
        max_tokens=c.max_tokens,
        seed=c.seed,
        repetition_penalty=r.repetition_penalty,
        presence_penalty=r.presence_penalty,
        qwen_fps=c.qwen_fps,
        resolved_config_out="",
        verbose=r.verbose,
    )


def run(cfg: DictConfig) -> None:
    """Step-1.5 scoring + calibration; stage = predict | calibrate | all."""
    args = _args_from_cfg(cfg)

    if args.stage in ("predict", "all") and args.artifact_type not in (
        "general-notes",
        "query-claims",
    ):
        raise SystemExit(
            f"stage {args.stage} does not support artifact_type {args.artifact_type} "
            "(predict supports general-notes|query-claims only)"
        )

    qidf = query_id_filter(cfg.data.query_ids)
    if qidf and args.artifact_type not in ("query-claims", "inferences"):
        print(f"  WARN: query_ids ignored for artifact-type={args.artifact_type}")
        qidf = None

    os.makedirs(args.out_dir, exist_ok=True)

    if args.stage in ("predict", "all"):
        predictions_path = run_predict(args, qidf)
    else:
        predictions_path = args.unli_jsonl or os.path.join(args.out_dir, PREDICTIONS_NAME)

    if args.stage in ("calibrate", "all"):
        unli_jsonl = (
            predictions_path if args.stage == "all" else (args.unli_jsonl or predictions_path)
        )
        if not os.path.exists(unli_jsonl):
            raise SystemExit(
                f"predictions JSONL not found: {unli_jsonl} "
                "(run stage=predict first or set data.unli_jsonl)"
            )
        run_calibrate(args, unli_jsonl, qidf)


def main(argv: list | None = None) -> int:
    """Entry point: remaining argv are Hydra ``key=value`` overrides."""
    overrides = list(sys.argv[1:] if argv is None else argv)
    run(build_config(overrides))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
