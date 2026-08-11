"""Flatten expanded queries into a sub-query JSONL + parent mapping for retrieval.

Reads ``expanded_queries.json`` and writes ``subqueries.jsonl`` (for tevatron
encoding) plus ``subquery_mapping.json`` (sub-query → parent query). Paths come
from ``configs/retrieval/``::

    python -m retrieval.cli prepare-subqueries
"""

from __future__ import annotations

import json
import os
import sys

from omegaconf import DictConfig

from marquis.retrieval._common import build_config


def run(cfg: DictConfig) -> None:
    """Flatten expanded queries and emit the sub-query JSONL + mapping."""
    expanded_path = cfg.data.expanded_queries
    subq_path = cfg.data.subqueries_jsonl
    mapping_path = cfg.data.subquery_mapping

    print("=== Step 3: Preparing sub-queries ===")
    with open(expanded_path, encoding="utf-8") as f:
        expanded = json.load(f)

    # Flatten all sub-queries into a single list for encoding
    all_subqueries = []
    for qid, data in expanded.items():
        subs = data["sub_queries"] or [data["original_query"]]  # fallback for empty
        for i, sq in enumerate(subs):
            all_subqueries.append(
                {
                    "query_id": f"{qid}_sub{i}",
                    "query": sq,
                    "parent_qid": qid,
                }
            )
    print(f"Total sub-queries: {len(all_subqueries)}")

    os.makedirs(os.path.dirname(subq_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(mapping_path) or ".", exist_ok=True)

    # Save as JSONL for tevatron encoding
    with open(subq_path, "w", encoding="utf-8") as f:
        for sq in all_subqueries:
            f.write(json.dumps({"query_id": sq["query_id"], "query": sq["query"]}) + "\n")
    print(f"Saved sub-queries to {subq_path}")

    # Save the parent mapping for later fusion
    mapping = {}
    for sq in all_subqueries:
        mapping.setdefault(sq["parent_qid"], []).append(sq["query_id"])
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    print(f"Saved mapping to {mapping_path}")
    for qid in sorted(mapping.keys(), key=int):
        print(f"  Query {qid}: {len(mapping[qid])} sub-queries")


def main(argv: list[str] | None = None) -> int:
    """Entry point: remaining argv are Hydra ``key=value`` overrides."""
    overrides = list(sys.argv[1:] if argv is None else argv)
    cfg = build_config(overrides)
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
