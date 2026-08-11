#!/usr/bin/env bash
set -euo pipefail

claims_path="${MARQUIS_CLAIMS_PATH:-outputs/query_claims_single/query_conditioned_claims.jsonl}"

marquis-generate baseline \
  "data.claims_path=$claims_path" \
  "output.out_dir=${MARQUIS_REPORT_OUTPUT_DIR:-outputs/reports_baseline}"

marquis-generate ginger \
  "data.claims_path=$claims_path" \
  "output.out_dir=${MARQUIS_GINGER_OUTPUT_DIR:-outputs/reports_ginger}"
