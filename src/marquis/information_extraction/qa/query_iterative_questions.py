"""QA answering (iterative): answer + generate follow-up questions per sub-query.

Like qa-answer, but after each answer it asks the LLM for one new, more-specific
follow-up question and repeats up to ``runtime.qa.max_steps`` times. When videos
come from dense retrieval (no topic mapping), **each step re-retrieves** for the
current follow-up question. Config comes from ``configs/information_extraction/``::

    python -m information_extraction.cli qa data.qa_queries=subqueries.jsonl runtime.qa.max_steps=5
"""

from __future__ import annotations

import json
import sys

from omegaconf import DictConfig

from marquis.information_extraction._common import build_config, query_id_filter
from marquis.information_extraction.prompts import QA_FOLLOWUP_PROMPT
from marquis.information_extraction.qa.query_helper_functions import (
    ask_qwen_batch,
    combine_answers,
    load_models,
    prepare_video_selection,
    run_qa,
)


def generate_followup(context, vlm):
    prompt = QA_FOLLOWUP_PROMPT.format(context=context)
    response = vlm.infer(video_path=None, query=prompt).strip()
    return response.split("\n")[0].strip()


def iterative_answer(subquery, get_videos, vlm, max_steps=5):
    """Answer, then follow up — re-fetching videos for the current question each step."""
    history = []
    current_q = subquery
    for _step in range(max_steps):
        vids = get_videos(current_q)  # topic mode: same videos; retrieval mode: fresh per question
        if not vids:
            break
        responses = ask_qwen_batch(vids, current_q, vlm)
        answers = list(zip(vids, responses, strict=False))
        print("Q:", current_q)
        print("ANSWERS ALL:", [ans for _, ans in answers])

        final, sources = combine_answers(current_q, answers, vlm)
        history.append({"question": current_q, "answer": final, "sources": sources})

        if final.strip().lower() in ["i don't know", "unknown", ""]:
            break
        context = "\n\n".join(f"Q: {h['question']}\nA: {h['answer']}" for h in history)
        next_q = generate_followup(context, vlm)
        if next_q == "NONE":
            break
        if next_q.lower() in [h["question"].lower() for h in history]:
            break
        current_q = next_q
    return history


def run(cfg: DictConfig) -> None:
    """Iterative QA with follow-ups over each query's videos (mapping or retrieval)."""
    if not cfg.data.qa_queries:
        raise SystemExit("qa requires data.qa_queries (queries-with-subqueries JSONL)")

    with open(cfg.data.qa_queries, encoding="utf-8") as f:
        queries = [json.loads(line) for line in f if line.strip()]

    qidf = query_id_filter(cfg.data.query_ids)
    if qidf:
        queries = [q for q in queries if str(q.get("query_id")) in qidf]
        print(f"[filter] {len(queries)} queries after query_ids={sorted(qidf)}")

    mapping, mode, embedder, video_embeddings, top_k, sim_threshold = prepare_video_selection(
        cfg, queries
    )
    max_steps = cfg.runtime.qa.max_steps

    def answer_one(sub, get_videos, vlm):
        history = iterative_answer(sub, get_videos, vlm, max_steps=max_steps)
        return [
            {"subquery": h["question"], "sources": h["sources"], "final_answer": h["answer"]}
            for h in history
        ]

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
