"""Rank fusion: combine sub-query rank lists into one list per original query.

Holds the fusion primitives (RRF / weighted-RRF / sum / max / mean), TREC I/O and
the ``fusion`` command that writes the standard suite of fusion runs. The single
source of truth for paths and knobs is ``configs/retrieval/``; override on the
command line, e.g.::

    python -m retrieval.cli fusion runtime.fusion_depth=50
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from typing import TypeAlias

from omegaconf import DictConfig

from marquis.retrieval._common import build_config

RankedResults: TypeAlias = dict[str, list[tuple[str, float]]]


# I/O


def load_subquery_mapping(path: str):
    """Load subquery_mapping.json: {orig_qid: [sub_qid, ...]} → (mapping, reverse)."""
    with open(path, encoding="utf-8") as f:
        mapping = json.load(f)
    reverse = {}
    for orig_qid, sub_qids in mapping.items():
        for sq_id in sub_qids:
            reverse[sq_id] = orig_qid
    return mapping, reverse


def load_ranked_results(path: str):
    """Load sub-query rankings (tevatron 3-col TSV or 6-col TREC), score-desc sorted."""
    results = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        first_line = f.readline().strip()
        f.seek(0)
        parts = first_line.split("\t") if "\t" in first_line else first_line.split()

        if len(parts) == 3:
            print("  Detected tevatron format (3 columns)")
            for line in f:
                cols = line.strip().split("\t")
                if len(cols) == 3:
                    sq_id, doc_id, score = cols
                    results[sq_id].append((doc_id, float(score)))
        elif len(parts) >= 6:
            print("  Detected TREC format (6 columns)")
            for line in f:
                cols = line.strip().split()
                if len(cols) >= 6:
                    results[cols[0]].append((cols[2], float(cols[4])))
        else:
            raise SystemExit(f"Unknown rank-list format ({len(parts)} columns): {first_line}")

    for sq_id in results:
        results[sq_id].sort(key=lambda x: x[1], reverse=True)
    print(f"  Loaded results for {len(results)} sub-queries")
    return results


def write_trec(fused, output_path: str, run_name: str, depth: int = 100) -> None:
    """Write fused results as TREC: ``query_id Q0 doc_id rank score run_name``."""
    with open(output_path, "w", encoding="utf-8") as f:
        for orig_qid in sorted(fused.keys()):
            sorted_docs = sorted(fused[orig_qid].items(), key=lambda x: x[1], reverse=True)[:depth]
            for rank, (doc_id, score) in enumerate(sorted_docs, start=1):
                f.write(f"{orig_qid} Q0 {doc_id} {rank} {score:.6f} {run_name}\n")
    print(f"  Written: {output_path} ({len(fused)} queries)")


# Fusion strategies


def fusion_rrf(subquery_results, reverse_map, k: int = 60):
    """Reciprocal Rank Fusion: score(doc) = Σ 1/(k + rank_i)."""
    fused = defaultdict(lambda: defaultdict(float))
    for sq_id, doc_list in subquery_results.items():
        orig_qid = reverse_map.get(sq_id)
        if orig_qid is None:
            continue
        for rank, (doc_id, _) in enumerate(doc_list, start=1):
            fused[orig_qid][doc_id] += 1.0 / (k + rank)
    return fused


def fusion_sum(subquery_results, reverse_map):
    """Sum of similarity scores across sub-queries."""
    fused = defaultdict(lambda: defaultdict(float))
    for sq_id, doc_list in subquery_results.items():
        orig_qid = reverse_map.get(sq_id)
        if orig_qid is None:
            continue
        for doc_id, score in doc_list:
            fused[orig_qid][doc_id] += score
    return fused


def fusion_max(subquery_results, reverse_map):
    """Max similarity score across sub-queries."""
    fused = defaultdict(dict)
    for sq_id, doc_list in subquery_results.items():
        orig_qid = reverse_map.get(sq_id)
        if orig_qid is None:
            continue
        for doc_id, score in doc_list:
            if doc_id not in fused[orig_qid] or score > fused[orig_qid][doc_id]:
                fused[orig_qid][doc_id] = score
    return fused


def fusion_mean(subquery_results, reverse_map):
    """Mean similarity over the sub-queries in which a doc appears."""
    fused_sum = defaultdict(lambda: defaultdict(float))
    fused_count = defaultdict(lambda: defaultdict(int))
    for sq_id, doc_list in subquery_results.items():
        orig_qid = reverse_map.get(sq_id)
        if orig_qid is None:
            continue
        for doc_id, score in doc_list:
            fused_sum[orig_qid][doc_id] += score
            fused_count[orig_qid][doc_id] += 1
    fused = defaultdict(lambda: defaultdict(float))
    for orig_qid in fused_sum:
        for doc_id in fused_sum[orig_qid]:
            fused[orig_qid][doc_id] = fused_sum[orig_qid][doc_id] / fused_count[orig_qid][doc_id]
    return fused


def fusion_weighted_rrf(subquery_results, reverse_map, k: int = 60):
    """RRF weighted by each sub-query's top (max) score."""
    sq_weights = {
        sq_id: (doc_list[0][1] if doc_list else 0.0) for sq_id, doc_list in subquery_results.items()
    }
    fused = defaultdict(lambda: defaultdict(float))
    for sq_id, doc_list in subquery_results.items():
        orig_qid = reverse_map.get(sq_id)
        if orig_qid is None:
            continue
        weight = sq_weights.get(sq_id, 1.0)
        for rank, (doc_id, _) in enumerate(doc_list, start=1):
            fused[orig_qid][doc_id] += weight * 1.0 / (k + rank)
    return fused


reciprocal_rank_fusion = fusion_rrf
weighted_reciprocal_rank_fusion = fusion_weighted_rrf
score_sum = fusion_sum
score_max = fusion_max
score_mean = fusion_mean


def fuse(subquery_results, reverse_map, *, method: str = "rrf", k: int = 60):
    """Dispatch to a single fusion method by name."""
    if method == "rrf":
        return fusion_rrf(subquery_results, reverse_map, k=k)
    if method == "weighted-rrf":
        return fusion_weighted_rrf(subquery_results, reverse_map, k=k)
    if method == "sum":
        return fusion_sum(subquery_results, reverse_map)
    if method == "max":
        return fusion_max(subquery_results, reverse_map)
    if method == "mean":
        return fusion_mean(subquery_results, reverse_map)
    raise ValueError(f"unknown fusion method: {method}")


# `fusion` command (standard suite)


def run(cfg: DictConfig) -> None:
    """Generate the standard suite of fusion run files (.trec)."""
    subquery_results_path = cfg.data.subquery_results
    subquery_mapping_path = cfg.data.subquery_mapping
    output_dir = cfg.output.out_dir
    depth = cfg.runtime.fusion_depth

    print("=" * 60)
    print("Multi-Fusion: Generating different rank lists")
    print("=" * 60)

    for path, name in [
        (subquery_results_path, "sub-query results"),
        (subquery_mapping_path, "sub-query mapping"),
    ]:
        if not os.path.exists(path):
            raise SystemExit(
                f"ERROR: {name} not found at {path}. Run the sub-query search step first."
            )

    os.makedirs(output_dir, exist_ok=True)

    print("\nLoading sub-query mapping...")
    mapping, reverse_map = load_subquery_mapping(subquery_mapping_path)
    print(f"  {len(mapping)} original queries, {len(reverse_map)} sub-queries")

    print("\nLoading sub-query search results...")
    subquery_results = load_ranked_results(subquery_results_path)

    known_sqs = set(reverse_map.keys())
    matched = known_sqs & set(subquery_results.keys())
    print(f"  Matched {len(matched)} / {len(known_sqs)} sub-queries from mapping")
    if len(matched) < len(known_sqs):
        missing = known_sqs - set(subquery_results.keys())
        print(f"  WARNING: {len(missing)} sub-queries from mapping not found in results")
        print(f"  Examples: {list(missing)[:5]}")

    k = cfg.runtime.rrf_k
    strategies = [
        ("rank-expansion-rrf-k60", lambda: fusion_rrf(subquery_results, reverse_map, k=60)),
        ("rank-expansion-rrf-k10", lambda: fusion_rrf(subquery_results, reverse_map, k=10)),
        ("rank-expansion-rrf-k100", lambda: fusion_rrf(subquery_results, reverse_map, k=100)),
        ("rank-expansion-sum", lambda: fusion_sum(subquery_results, reverse_map)),
        ("rank-expansion-max", lambda: fusion_max(subquery_results, reverse_map)),
        ("rank-expansion-mean", lambda: fusion_mean(subquery_results, reverse_map)),
        (
            "rank-expansion-weighted-rrf",
            lambda: fusion_weighted_rrf(subquery_results, reverse_map, k=k),
        ),
    ]

    print(f"\nGenerating {len(strategies)} fusion rank lists...\n")
    for run_name, fusion_fn in strategies:
        print(f"Strategy: {run_name}")
        write_trec(fusion_fn(), os.path.join(output_dir, f"{run_name}.trec"), run_name, depth=depth)

    print("\nDone! All .trec files generated.")
    print(
        "Next: python -m retrieval.cli evaluate data.qrels=<qrels> "
        f"runtime.eval.run_dir={output_dir}"
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point: remaining argv are Hydra ``key=value`` overrides."""
    overrides = list(sys.argv[1:] if argv is None else argv)
    cfg = build_config(overrides)
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
