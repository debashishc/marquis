from __future__ import annotations

from marquis.article_generation import baseline, ginger
from marquis.common import prompts as common_prompts
from marquis.information_extraction.qa import prompts as qa_prompts
from marquis.information_extraction.qa import settings as qa_settings
from marquis.retrieval.query_decomposition import PROMPT_TEMPLATE as RETRIEVAL_PROMPT
from marquis.rlm_controller.rlm.utils import prompts as rlm_prompts
from marquis.rlm_controller.tool_api import RLM_TOOL_NAMES, create_toolkit


def test_appendix_prompt_families_are_exposed() -> None:
    assert "searchable sub-queries" in RETRIEVAL_PROMPT
    assert "Generate between 10 and 25 sub-queries" in RETRIEVAL_PROMPT

    general = common_prompts.prompt_general_notes(topic="topic", video_id="vid")
    assert "You are extracting observation notes directly from a raw video." in general
    assert '"notes"' in general

    claims = common_prompts.prompt_query_claims(
        query_id="1",
        query="What happened?",
        persona_title="Analyst",
        background="Background",
        topic="topic",
        video_id="vid",
    )
    assert "You are extracting query-relevant claims directly from a raw video." in claims
    assert '"claims"' in claims

    assert "complete questions" in qa_prompts.QA_DECOMPOSITION_PROMPT_TEMPLATE
    assert "Return ONLY the final factual answer" in qa_prompts.QA_ANSWER_PROMPT_TEMPLATE
    assert "Extracted Answers (treat as ground truth)" in qa_prompts.COMBINE_ANSWERS_PROMPT_TEMPLATE
    assert "Return ONLY the question or NONE" in qa_prompts.FOLLOWUP_PROMPT_TEMPLATE

    assert "support percentages" in common_prompts.QWEN_SCORE_INSTRUCTION
    assert "<answer>probability_value</answer>" in common_prompts.QWEN_SCORE_PROMPT

    assert "report writing assistant" in baseline.REPORT_PROMPT
    assert "facet clusters" in ginger.CLUSTER_PROMPT
    assert "rank the clusters" in ginger.RANK_PROMPT
    assert "SINGLE sentence" in ginger.SUMMARIZE_PROMPT
    assert "Final polished report" in ginger.FLUENCY_PROMPT

    assert "THINK-ACT-OBSERVE LOOP" in rlm_prompts.REPL_SYSTEM_PROMPT
    assert "NEW_FINDINGS" in rlm_prompts.ROOT_THINK_PROMPT
    assert "SELECTED" in rlm_prompts.ROOT_JUDGE_PROMPT
    assert "Eff_Redundancy" in rlm_prompts.BEHAVIOR_JUDGE_PROMPT


def test_rlm_tool_namespace_matches_appendix_surface() -> None:
    expected_tools = {
        "video_caption",
        "general_notes",
        "query_claims",
        "video_qa",
        "transcribe",
        "retrieve_chunks",
        "write_report",
        "memory_summary",
        "print_memory",
        "add_keyword",
        "search_by_keyword",
        "remove_fact",
        "clear_facts",
        "llm_think",
        "llm_judge",
    }
    assert expected_tools.issubset(set(RLM_TOOL_NAMES))


def test_rlm_toolkit_can_select_preextracted_claims() -> None:
    context = {
        "task": {"query": "Describe flood impacts."},
        "video_pool": {"video_ids": ["vid1"], "video_paths": {"vid1": "vid1.mp4"}},
        "claims": [
            {
                "claim_id": "qc-1",
                "video_id": "vid1",
                "claim": "Flood water covered a road.",
                "confidence": 0.9,
                "timestamp": [1.0, 2.0],
                "source": "video_visual",
                "evidence": "Water is visible on the road.",
            }
        ],
    }
    toolkit = create_toolkit(context)
    facts = toolkit["query_claims"]("vid1")
    assert facts[0]["fact"] == "Flood water covered a road."

    selected = toolkit["llm_judge"]()
    assert selected[0]["claim_id"] == "qc-1"
    report = toolkit["write_report"](selected)
    assert "Flood water covered a road." in report
    assert "References:" in report


def test_qa_retrieval_hyperparameters_match_paper() -> None:
    assert qa_settings.SIM_THRESHOLD == 0.10
    assert qa_settings.TOP_K == 4
    assert qa_settings.MAX_FRAMES == 32
