# Release artifacts

Large artifacts do not belong in GitHub. Put them in the public Hugging Face
release or link to the MAGMaR data release.

- MAGMaR query JSONL and topic-video mapping, or pointers to the MAGMaR release.
- Retrieval subqueries, first-stage run files, RankVideo reranker scores,
  reranked run files, qrels, and retrieval evaluation summaries.
- Extracted general notes and query-conditioned claims.
- Support-scoring and calibration outputs.
- Note packets, claim packets, higher-level inferences, and generated reports
  for Bullet, CAG, GINGER, QA, and RLM variants where available.
- Article-generation and RLM evaluation summaries.
- Optional RLM trajectories for representative runs.

Configs and scripts accept path overrides through environment variables or CLI
arguments. See `docs/reproduction.md` for the required environment variables.

The paper source is not bundled here. Use `docs/prompt_map.md` and
`docs/paper_map.md` to check prompt and component coverage.

MAGMaR 2026 shared-task resource: https://github.com/rekriz11/MAGMAR_2026
Hugging Face paper page: https://huggingface.co/papers/2605.17640
