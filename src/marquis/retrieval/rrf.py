"""Single-method fusion of sub-query rank lists into a submission-format JSON.

Combines each original query's sub-query rank lists with one fusion method and
writes ``rank-expanded-<method>.json`` ({qid: {doc_id: score}}), then prints a
small baseline-vs-fused comparison. Knobs come from ``configs/retrieval/``::

    python -m retrieval.cli rrf runtime.rrf_method=weighted-rrf runtime.rrf_k=10
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

from omegaconf import DictConfig

from marquis.retrieval._common import build_config
from marquis.retrieval.fusion import load_subquery_mapping


def run(cfg: DictConfig) -> None:
    """Fuse sub-query rankings for each original query with a single method."""
    mapping_path = cfg.data.subquery_mapping
    results_path = cfg.data.subquery_results
    baseline_path = cfg.data.baseline_results
    output_dir = cfg.output.out_dir
    k = cfg.runtime.rrf_k
    method = cfg.runtime.rrf_method
    depth = cfg.runtime.fusion_depth

    os.makedirs(output_dir, exist_ok=True)

    mapping, _ = load_subquery_mapping(mapping_path)

    # Load sub-query search results: subq_id -> [(doc_id, score), ...]
    subq_results = defaultdict(list)
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                qid, docid, score = parts
                subq_results[qid].append((docid, float(score)))
    print(f"Loaded results for {len(subq_results)} sub-queries")

    # Fusion per original query
    final_results = {}
    for orig_qid, subq_ids in mapping.items():
        scores = defaultdict(float)
        for subq_id in subq_ids:
            ranked_docs = subq_results.get(subq_id)
            if not ranked_docs:
                print(f"  Warning: no results for {subq_id}")
                continue
            ranked_docs = sorted(ranked_docs, key=lambda x: x[1], reverse=True)
            max_score = max(score for _, score in ranked_docs)
            for rank, (docid, score) in enumerate(ranked_docs, start=1):
                if method == "rrf":
                    scores[docid] += 1.0 / (k + rank)
                elif method == "weighted-rrf":
                    scores[docid] += max_score * 1.0 / (k + rank)
                elif method == "sum":
                    scores[docid] += score
                elif method == "max":
                    scores[docid] = max(scores[docid], score)
                else:
                    raise SystemExit(f"unsupported rrf_method for JSON output: {method}")
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:depth]
        final_results[orig_qid] = {docid: score for docid, score in sorted_docs}

    output_path = os.path.join(output_dir, f"rank-expanded-{method}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nSaved {method} results to {output_path}")
    print(f"Queries: {len(final_results)}")

    # Compare with baseline (best-effort; skip if missing)
    if os.path.exists(baseline_path):
        with open(baseline_path, encoding="utf-8") as f:
            baseline = json.load(f)
        print(f"\n=== Comparison: Baseline vs Expanded+{method} ===")
        for qid in sorted(final_results.keys(), key=int)[:5]:
            if qid not in baseline:
                continue
            baseline_top = list(baseline[qid].items())[:3]
            fused_top = list(final_results[qid].items())[:3]
            print(f"\nQuery {qid}:")
            print(f"  Baseline top 3: {[d for d, _ in baseline_top]}")
            print(f"  {method} top 3:  {[d for d, _ in fused_top]}")
    else:
        print(f"\n(no baseline at {baseline_path}, skipping comparison)")


def main(argv: list[str] | None = None) -> int:
    """Entry point: remaining argv are Hydra ``key=value`` overrides."""
    overrides = list(sys.argv[1:] if argv is None else argv)
    cfg = build_config(overrides)
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
