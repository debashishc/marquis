#!/usr/bin/env bash
set -euo pipefail

command="${MARQUIS_RLM_COMMAND:-magmar-notes}"
magmar_root="${MAGMAR_ROOT:?Set MAGMAR_ROOT to the MAGMaR data directory}"
queries_jsonl="${MARQUIS_RLM_QUERIES_JSONL:-${MAGMAR_QUERIES_JSONL:-$magmar_root/MAGMaR2026_queries.jsonl}}"
topic_mapping="${MARQUIS_TOPIC_MAPPING:-${MAGMAR_TOPIC_MAPPING:-$magmar_root/topic_video_mapping.json}}"

args=(
  "$command"
  --query_id "${MARQUIS_QUERY_ID:-1}"
  --model "${MARQUIS_RLM_ROOT_MODEL:-gpt-5}"
  --sub_model "${MARQUIS_RLM_SUB_MODEL:-gpt-5-mini}"
  --max_iterations "${MARQUIS_RLM_MAX_ITERATIONS:-60}"
  --out-dir "${MARQUIS_RLM_OUTPUT_DIR:-outputs/rlm}"
  --queries-jsonl "$queries_jsonl"
  --topic-mapping "$topic_mapping"
  --local-magmar-dir "${MARQUIS_LOCAL_MAGMAR_DIR:-$magmar_root}"
  --video-root "${MAGMAR_VIDEO_ROOT:-$magmar_root}"
)

if [[ "$command" == "magmar-notes" ]]; then
  args+=(--claims-jsonl "${MARQUIS_CLAIMS_PATH:-$magmar_root/features/claims}")
else
  args+=(--vlm_backend "${MARQUIS_RLM_VLM_BACKEND:-openai_vision}")
fi

if [[ "$command" == "magmar" && -n "${MARQUIS_RLM_VLM_MODEL:-}" ]]; then
  args+=(--vlm_model "$MARQUIS_RLM_VLM_MODEL")
fi

marquis-rlm "${args[@]}" "$@"
