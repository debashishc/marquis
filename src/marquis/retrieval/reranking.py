"""Rerank first-stage retrieval runs with external RankVideo-style scores."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from marquis.retrieval.fusion import RankedResults, load_ranked_results

PairScores = dict[str, dict[str, float]]


def load_pair_scores(path: str) -> PairScores:
    scores: dict[str, dict[str, float]] = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                rec = json.loads(line)
                qid = str(rec.get("query_id") or rec.get("qid") or "")
                doc_id = str(
                    rec.get("video_id") or rec.get("doc_id") or rec.get("document_id") or ""
                )
                score: Any = rec.get("score")
            else:
                tab_parts = line.split("\t")
                if len(tab_parts) == 3:
                    qid, doc_id, score = tab_parts
                else:
                    parts = line.split()
                    if len(parts) < 5:
                        raise ValueError(
                            f"Unsupported reranker score format at {path}:{line_no}: {line}"
                        )
                    qid, doc_id, score = parts[0], parts[2], parts[4]
            if not qid or not doc_id:
                raise ValueError(f"Missing query/video id at {path}:{line_no}: {line}")
            scores[qid][doc_id] = float(score)
    return {qid: dict(doc_scores) for qid, doc_scores in scores.items()}


def rerank_candidates(
    first_stage: Mapping[str, list[tuple[str, float]]],
    rerank_scores: Mapping[str, Mapping[str, float]],
    *,
    depth: int = 100,
) -> RankedResults:
    reranked: RankedResults = {}
    for qid, candidates in first_stage.items():
        limited = candidates[:depth]
        query_scores = rerank_scores.get(qid, {})
        decorated = []
        for original_rank, (doc_id, first_score) in enumerate(limited, start=1):
            has_score = doc_id in query_scores
            score = float(query_scores[doc_id]) if has_score else float(first_score)
            decorated.append((has_score, score, original_rank, doc_id))
        decorated.sort(key=lambda item: (not item[0], -item[1], item[2]))
        reranked[qid] = [(doc_id, score) for has_score, score, _, doc_id in decorated]
    return reranked


def write_reranked_trec(
    reranked: Mapping[str, list[tuple[str, float]]],
    output_path: str,
    *,
    run_name: str,
    depth: int = 100,
) -> None:
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for qid in sorted(reranked.keys()):
            for rank, (doc_id, score) in enumerate(reranked[qid][:depth], start=1):
                f.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} {run_name}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rerank a first-stage run with RankVideo scores")
    parser.add_argument("--run", required=True, help="First-stage TREC/TSV run")
    parser.add_argument("--rankvideo-scores", required=True, help="RankVideo query-video scores")
    parser.add_argument("--output", required=True, help="Output reranked TREC run")
    parser.add_argument("--depth", type=int, default=100)
    parser.add_argument("--run-name", default="rankvideo-rerank")
    args = parser.parse_args(argv)

    first_stage = load_ranked_results(args.run)
    scores = load_pair_scores(args.rankvideo_scores)
    reranked = rerank_candidates(first_stage, scores, depth=args.depth)
    write_reranked_trec(reranked, args.output, run_name=args.run_name, depth=args.depth)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
