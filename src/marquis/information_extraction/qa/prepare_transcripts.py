"""Precompute Whisper transcripts for QA, in a standalone process.

QA (``qa`` / ``qa-answer``) needs both a Whisper transcript and the Qwen3.5-VL
(vLLM) engine. Loading both in one process opens two CUDA contexts on the same
card; on a GPU in EXCLUSIVE_PROCESS compute mode the vLLM engine subprocess holds
the only allowed context and Whisper's load dies with
``cudaErrorDevicesUnavailable``.

This command runs Whisper *alone* over every video in ``data.query_video_mapping``
(optionally filtered by ``data.query_ids``), writes ``{video_id: transcript}`` to
``data.transcripts``, and exits — fully releasing the GPU before the QA step loads
the VLM. The QA runners then read that cache and never load Whisper::

    python -m information_extraction.cli prepare-transcripts \
        data.query_video_mapping=... data.audio_root=... data.transcripts=transcripts.json
    python -m information_extraction.cli qa data.transcripts=transcripts.json ...
"""

from __future__ import annotations

import json
import os
import sys

from omegaconf import DictConfig

from marquis.information_extraction._common import build_config, query_id_filter
from marquis.information_extraction.qa.query_helper_functions import _load_whisper


def run(cfg: DictConfig) -> None:
    """Transcribe every mapped video with Whisper and cache the results to disk."""
    from marquis.common.contracts import load_query_video_mapping

    mapping_path = cfg.data.get("query_video_mapping")
    if not (mapping_path and os.path.exists(mapping_path)):
        raise SystemExit(
            f"prepare-transcripts requires data.query_video_mapping (not found: {mapping_path!r})"
        )
    out_path = cfg.data.get("transcripts")
    if not out_path:
        raise SystemExit("prepare-transcripts requires data.transcripts (output path)")

    mapping = load_query_video_mapping(mapping_path)
    qidf = query_id_filter(cfg.data.query_ids)
    if qidf:
        mapping = {qid: vids for qid, vids in mapping.items() if str(qid) in qidf}
        print(f"[filter] {len(mapping)} queries after query_ids={sorted(qidf)}")

    video_ids = sorted({vid for vids in mapping.values() for vid in vids})
    audio_dir = cfg.data.audio_root
    audio_ext = cfg.data.get("audio_ext", ".m4a")

    # Resume: keep transcripts already cached so reruns only fill the gaps.
    cache: dict[str, str] = {}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            cache = json.load(f)
        print(f"[transcripts] resuming from {len(cache)} existing transcripts in {out_path}")

    todo = [vid for vid in video_ids if vid not in cache]
    print(f"[transcripts] {len(video_ids)} mapped videos, {len(todo)} to transcribe")
    if not todo:
        print("[transcripts] nothing to do")
        return

    whisper_m = _load_whisper(cfg.runtime.qa.whisper_model)
    for i, vid in enumerate(todo, 1):
        audio_path = os.path.join(audio_dir, vid + audio_ext)
        if not os.path.exists(audio_path):
            print(f"[warn] missing audio file, skipping: {audio_path}")
            continue
        cache[vid] = whisper_m.transcribe(audio_path)["text"]
        # Write incrementally so a crash mid-run doesn't discard finished work.
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        print(f"[transcripts] ({i}/{len(todo)}) {vid}")

    print(f"[ok] wrote {len(cache)} transcripts -> {out_path}")


def main(argv: list | None = None) -> int:
    """Entry point: remaining argv are Hydra ``key=value`` overrides."""
    overrides = list(sys.argv[1:] if argv is None else argv)
    run(build_config(overrides))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
