# Paper map

Use this map to check each paper component against the code.

| Paper component | Code | Config | Main command |
| --- | --- | --- | --- |
| Retrieval: query expansion, fusion, RankVideo reranking adapter | `marquis.retrieval` | `configs/retrieval` | `marquis-retrieve` |
| Information extraction: general notes | `marquis.information_extraction.extract` | `configs/information_extraction` | `marquis-extract general-notes` |
| Information extraction: query-conditioned claims | `marquis.information_extraction.extract` | `configs/information_extraction` | `marquis-extract query-claims` |
| QA extraction | `marquis.information_extraction.qa` | `configs/information_extraction` | `marquis-extract qa-decompose`, `marquis-extract qa-answer` |
| Support scoring and calibration | `marquis.information_extraction.calibrate` | `configs/information_extraction` | `marquis-extract predict-unli`, `marquis-extract calibrate-unli` |
| Evidence packets | `marquis.information_extraction.assemble_packets` | `configs/information_extraction` | `marquis-extract packets` |
| Article generation: Bullet, CAG baseline, GINGER | `marquis.article_generation` | `configs/article_generation` | `marquis-generate` |
| RLM controller | `marquis.rlm_controller` | `configs/rlm_controller` | `marquis-rlm` |
| Evaluation | `marquis.evaluation` | `configs/evaluation` | `marquis-evaluate` |

Prompt coverage is tracked in `docs/prompt_map.md`. The paper source is not
bundled here. The MAGMaR 2026 repository is the shared-task and data resource;
this repository is the MARQUIS implementation.

Configs require explicit paths via environment variables.
`examples/data` and `examples/quicktest` provide small fixtures for
no-model checks.
