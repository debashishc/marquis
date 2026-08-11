#!/usr/bin/env bash
set -euo pipefail

magmar_root="${MAGMAR_ROOT:?Set MAGMAR_ROOT to the MAGMaR data directory}"
topic_mapping="${MARQUIS_TOPIC_MAPPING:-$magmar_root/topic_video_mapping.json}"
query_video_mapping="${MARQUIS_QUERY_VIDEO_MAPPING:-$magmar_root/query_video_mapping.json}"
queries_jsonl="${MAGMAR_QUERIES_JSONL:-$magmar_root/MAGMaR2026_queries.jsonl}"

# Build the query_id->video selection table that extraction reads.
marquis-extract build-query-mapping \
  "data.mapping=$topic_mapping" \
  "data.queries_jsonl=$queries_jsonl" \
  "data.query_video_mapping=$query_video_mapping"

marquis-extract general-notes \
  "data.mapping=$topic_mapping" \
  "data.query_video_mapping=$query_video_mapping" \
  "data.video_root=${MAGMAR_VIDEO_ROOT:-$magmar_root}" \
  "output.out_dir=${MARQUIS_GENERAL_NOTES_DIR:-outputs/general_notes}"

marquis-extract query-claims \
  "data.mapping=$topic_mapping" \
  "data.query_video_mapping=$query_video_mapping" \
  "data.queries_jsonl=$queries_jsonl" \
  "data.video_root=${MAGMAR_VIDEO_ROOT:-$magmar_root}" \
  "output.out_dir=${MARQUIS_QUERY_CLAIMS_DIR:-outputs/query_claims_single}"
