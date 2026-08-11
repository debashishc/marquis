#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if command -v marquis-retrieve >/dev/null 2>&1; then
  MARQUIS_RETRIEVE=(marquis-retrieve)
else
  MARQUIS_RETRIEVE=(python -m marquis.retrieval.cli)
fi

magmar_root="${MAGMAR_ROOT:?Set MAGMAR_ROOT to the MAGMaR data directory}"

"${MARQUIS_RETRIEVE[@]}" prepare-subqueries \
  "data.expanded_queries=${MARQUIS_EXPANDED_QUERIES:-$magmar_root/expanded_queries.json}" \
  "data.subqueries_jsonl=${MARQUIS_SUBQUERIES_JSONL:-outputs/retrieval/subqueries.jsonl}" \
  "data.subquery_mapping=${MARQUIS_SUBQUERY_MAPPING:-outputs/retrieval/subquery_mapping.json}"

"${MARQUIS_RETRIEVE[@]}" fusion \
  "data.subquery_results=${MARQUIS_SUBQUERY_RESULTS:-$magmar_root/rank-subqueries.txt}" \
  "data.subquery_mapping=${MARQUIS_SUBQUERY_MAPPING:-outputs/retrieval/subquery_mapping.json}" \
  "output.out_dir=${MARQUIS_RETRIEVAL_DIR:-outputs/retrieval}"

if [[ -n "${MARQUIS_RANKVIDEO_SCORES:-}" ]]; then
  "${MARQUIS_RETRIEVE[@]}" rerank \
    --run "${MARQUIS_FIRST_STAGE_RUN:-${MARQUIS_RETRIEVAL_DIR:-outputs/retrieval}/rank-expansion-rrf-k10.trec}" \
    --rankvideo-scores "$MARQUIS_RANKVIDEO_SCORES" \
    --output "${MARQUIS_RERANKED_RUN:-${MARQUIS_RETRIEVAL_DIR:-outputs/retrieval}/rank-expansion-rrf-k10-rankvideo.trec}" \
    --depth "${MARQUIS_RERANK_DEPTH:-100}" \
    --run-name "${MARQUIS_RERANK_RUN_NAME:-rank-expansion-rrf-k10-rankvideo}"
fi
