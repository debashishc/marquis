# Examples

This directory contains tiny public inputs for smoke tests and documentation.
They are not the paper-scale MAGMaR data.

- `data/MAGMaR2026_queries.jsonl`: query records matching the public query
  schema.
- `data/MAGMaR2026_queries_with_subqueries.jsonl`: the same queries paired with
  decomposed sub-questions, used by the QA runners.
- `data/reference.json`: `{topics: [{topic_id, chunks}, ...]}` — the curated
  source of truth for which videos are relevant to each topic. Queries join to a
  topic by `topic_id`; used by validation and qrels.
- `data/query_video_mapping.json`: `{query_id: [video_id, ...]}` — the video
  selection table information extraction reads (each query reads its own videos).
  Built from `reference.json` by joining queries on `topic_id`.
- `data/expanded_queries.json`: sample expanded subqueries used by the contract
  validator.

Paper-scale videos, retrieval runs, extraction outputs, generated reports, and
evaluation summaries should be downloaded from the release artifacts once the
Hugging Face artifact URL is available.
