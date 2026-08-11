"""QA answering (single-shot): answer each sub-query over its videos.

For every query's sub_queries, gather the relevant videos — the query's videos
when a ``query_video_mapping`` covers it, otherwise dense-retrieved once per
sub-query — ask the VLM per video, then combine into one answer. Config comes
from ``configs/information_extraction/``::

    python -m information_extraction.cli qa-answer \
        data.qa_queries=subqueries.jsonl data.qa_output=answers.jsonl
"""

from __future__ import annotations

import json
import sys

from omegaconf import DictConfig

from marquis.information_extraction._common import build_config, query_id_filter
from marquis.information_extraction.qa.query_helper_functions import (
    ask_qwen_batch,
    combine_answers,
    load_models,
    prepare_video_selection,
    run_qa,
)


def answer_one(sub, get_videos, vlm):
    """Single-shot: retrieve videos once, answer per video, combine."""
    vids = get_videos(sub)
    if not vids:
        print(f"[warn] no videos for sub-query: {sub!r}; skipping")
        return []
    answers = list(zip(vids, ask_qwen_batch(vids, sub, vlm), strict=False))
    print("Q:", sub)
    print("ANSWERS ALL:", [ans for _, ans in answers])
    final, sources = combine_answers(sub, answers, vlm)
    return [{"subquery": sub, "num_videos": len(vids), "sources": sources, "answer": final}]


def run(cfg: DictConfig) -> None:
    """Single-shot QA over each query's videos (topic mapping or dense retrieval)."""
    if not cfg.data.qa_queries:
        raise SystemExit("qa-answer requires data.qa_queries (queries-with-subqueries JSONL)")

    with open(cfg.data.qa_queries, encoding="utf-8") as f:
        queries = [json.loads(line) for line in f if line.strip()]

    qidf = query_id_filter(cfg.data.query_ids)
    if qidf:
        queries = [q for q in queries if str(q.get("query_id")) in qidf]
        print(f"[filter] {len(queries)} queries after query_ids={sorted(qidf)}")

    mapping, mode, embedder, video_embeddings, top_k, sim_threshold = prepare_video_selection(
        cfg, queries
    )
    models = load_models(
        cfg.runtime.qa.qa_model,
        cfg.model.download_dir or None,
        cfg.runtime.qa.whisper_model,
        video_root=cfg.data.video_root,
        transcripts=cfg.data.get("transcripts"),
        fps=cfg.runtime.qa.get("fps", 1.0),
        max_frames=cfg.runtime.qa.get("max_frames", 128),
    )
    results = run_qa(
        queries,
        models,
        answer_one,
        query_video_mapping=mapping,
        video_dir=cfg.data.video_root,
        audio_dir=cfg.data.audio_root,
        audio_ext=cfg.data.get("audio_ext", ".m4a"),
        mode=mode,
        embedder=embedder,
        video_embeddings=video_embeddings,
        top_k=top_k,
        sim_threshold=sim_threshold,
        output_path=cfg.data.qa_output,
    )

    print(f"[ok] wrote {len(results)} results -> {cfg.data.qa_output}")


def main(argv: list | None = None) -> int:
    """Entry point: remaining argv are Hydra ``key=value`` overrides."""
    overrides = list(sys.argv[1:] if argv is None else argv)
    run(build_config(overrides))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
