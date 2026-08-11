#!/usr/bin/env bash
set -euo pipefail

magmar_root="${MAGMAR_ROOT:?Set MAGMAR_ROOT to the MAGMaR data directory}"

marquis-evaluate retrieval \
  --qrels "${MARQUIS_QRELS:-$magmar_root/qrels.txt}" \
  --run-dir "${MARQUIS_RETRIEVAL_DIR:-outputs/retrieval}" \
  --output "${MARQUIS_EVAL_CSV:-outputs/evaluation/retrieval_results.csv}"
