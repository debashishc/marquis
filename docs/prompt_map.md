# Prompt map

Use this table to find the prompt families described in the paper. The paper
source is not bundled in this repo.

| Appendix prompt | Code |
| --- | --- |
| Retrieval query expansion | `marquis.retrieval.query_decomposition.PROMPT_TEMPLATE` |
| General note extraction | `marquis.common.prompts.prompt_general_notes` |
| Query-conditioned claim extraction | `marquis.common.prompts.prompt_query_claims` |
| QA decomposition into questions | `marquis.information_extraction.qa.prompts.QA_DECOMPOSITION_PROMPT_TEMPLATE` |
| QA video answering | `marquis.information_extraction.qa.prompts.QA_ANSWER_PROMPT_TEMPLATE` |
| QA answer aggregation | `marquis.information_extraction.qa.prompts.COMBINE_ANSWERS_PROMPT_TEMPLATE` |
| QA follow-up question generation | `marquis.information_extraction.qa.prompts.FOLLOWUP_PROMPT_TEMPLATE` |
| Qwen 3.5 support scoring | `marquis.common.prompts.QWEN_SCORE_INSTRUCTION` and `QWEN_SCORE_PROMPT` |
| Bullet evidence rendering | `marquis.article_generation.bullet` |
| Baseline article generation | `marquis.article_generation.baseline.REPORT_PROMPT` |
| GINGER clustering, ranking, summarization, fluency | `marquis.article_generation.ginger` prompt constants |
| MARQUIS-RLM REPL system prompt | `marquis.rlm_controller.rlm.utils.prompts.REPL_SYSTEM_PROMPT` |
| MARQUIS-RLM Root Think prompt | `marquis.rlm_controller.rlm.utils.prompts.ROOT_THINK_PROMPT` |
| MARQUIS-RLM Root Judge prompt | `marquis.rlm_controller.rlm.utils.prompts.ROOT_JUDGE_PROMPT` |
| MARQUIS-RLM behavior-level judge prompt | `marquis.rlm_controller.rlm.utils.prompts.BEHAVIOR_JUDGE_PROMPT` |

The RLM appendix also defines the REPL tool namespace and memory schema. Those
contracts live in `marquis.rlm_controller.tool_api`.
