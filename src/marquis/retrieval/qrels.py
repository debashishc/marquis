"""Build TREC qrels from reference.json, joined to queries by ``topic_id``.

Reads ``reference.json`` (``topics[].topic_id`` -> ``chunks``) and the query set,
joins each query to its topic by ``topic_id``, and writes ``qrels.txt``. Paths
come from ``configs/retrieval/``::

    python -m retrieval.cli qrels data.reference=... data.qrels=...
"""

from __future__ import annotations

import json
import os
import sys

from omegaconf import DictConfig

from marquis.common.contracts import build_query_video_map, load_reference
from marquis.retrieval._common import build_config


def load_queries(path: str) -> list[dict]:
    queries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line.strip()))
    return queries


def run(cfg: DictConfig) -> None:
    """Join queries to reference topics by topic_id and write a TREC qrels file."""
    reference_file = cfg.data.reference
    queries_file = cfg.data.queries_file
    output_qrels = cfg.data.qrels

    print("=" * 60)
    print("Convert reference.json → TREC qrels")
    print("=" * 60)

    if not os.path.exists(reference_file):
        raise SystemExit(
            f"\nERROR: {reference_file} not found! "
            "Point data.reference at the dataset's reference.json first."
        )
    if not os.path.exists(queries_file):
        raise SystemExit(f"\nERROR: {queries_file} not found!")

    print(f"\nLoading reference: {reference_file}")
    reference = load_reference(reference_file)
    print(f"  {len(reference)} topics, {sum(len(v) for v in reference.values())} relevant chunks")

    print(f"\nLoading queries: {queries_file}")
    queries = load_queries(queries_file)
    print(f"  {len(queries)} queries")

    # query_id -> [chunk_id, ...], joined by topic_id.
    query_video_map = build_query_video_map(queries, reference)
    empty = [qid for qid, vids in query_video_map.items() if not vids]
    if empty:
        print(f"\n  WARNING: {len(empty)} queries with no relevant chunks: {sorted(empty)}")

    print(f"\nGenerating qrels file: {output_qrels}")
    os.makedirs(os.path.dirname(output_qrels) or ".", exist_ok=True)
    total_judgments = 0
    with open(output_qrels, "w", encoding="utf-8") as f:
        for qid in sorted(query_video_map, key=lambda x: int(x) if str(x).isdigit() else x):
            for vid in query_video_map[qid]:
                f.write(f"{qid} 0 {vid} 1\n")
                total_judgments += 1
    print(f"  Written {total_judgments} judgments for {len(query_video_map)} queries")
    print(f"  Output: {output_qrels}")


def main(argv: list[str] | None = None) -> int:
    """Entry point: remaining argv are Hydra ``key=value`` overrides."""
    overrides = list(sys.argv[1:] if argv is None else argv)
    cfg = build_config(overrides)
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
