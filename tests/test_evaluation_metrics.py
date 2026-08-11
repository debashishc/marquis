from __future__ import annotations

from marquis.evaluation.extraction import (
    compute_comparison,
    compute_extraction_metrics,
    compute_report_metrics,
    validate_structure,
)


def sample_queries() -> list[dict]:
    return [
        {
            "query_id": "1",
            "topic_id": "flood_damage",
            "query": "flood damage in city",
            "query_type": "fact",
            "language": "en",
        },
        {
            "query_id": "2",
            "topic_id": "election_rally",
            "query": "election rally turnout",
            "query_type": "comparison",
            "language": "fr",
        },
    ]


def test_validate_structure_reports_contract_and_reference_failures() -> None:
    errors = validate_structure(
        queries=sample_queries(),
        observations=[{"note_id": "obs-1", "video_id": "vid-1"}],
        grounded=[
            {
                "note_id": "grounded-1",
                "video_id": "vid-1",
                "topic": "flood_damage",
                "claim": "Flooding is visible.",
                "source_observation_ids": ["missing-obs"],
            }
        ],
        reports_by_pipeline={
            "report_pipe": [
                {
                    "query_id": "1",
                    "topic": "flood_damage",
                    "citations": [
                        {"note_id": "note-1", "video_id": "unknown"},
                        {"note_id": 123, "video_id": "vid-1"},
                    ],
                }
            ]
        },
        expanded={"not-official": {"subqueries": []}},
        general_notes=[
            {
                "note_id": "general-1",
                "video_id": "vid-1",
                "topic": "flood_damage",
                "text": "Flooding is visible.",
                "modality": "visual",
                "is_post_grounded": True,
            }
        ],
    )

    assert any("references missing observation missing-obs" in error for error in errors)
    assert any("unresolved video_id" in error for error in errors)
    assert any("report_citation: field 'note_id' expected str" in error for error in errors)
    assert any("expanded query ID 'not-official'" in error for error in errors)
    assert any("general_note[0]: is_post_grounded should be false" in error for error in errors)


def test_compute_extraction_metrics_counts_legacy_and_new_artifacts() -> None:
    metrics = compute_extraction_metrics(
        observations=[
            {"note_id": "obs-1", "video_id": "vid-1"},
            {"note_id": "obs-2", "video_id": "vid-2"},
        ],
        grounded=[
            {
                "note_id": "grounded-1",
                "video_id": "vid-1",
                "topic": "flood_damage",
                "claim": "Water covers a street.",
                "source_observation_ids": ["obs-1"],
            },
            {
                "note_id": "grounded-2",
                "video_id": "vid-1",
                "topic": "flood_damage",
                "claim": "Water covers the street today.",
                "source_observation_ids": ["missing-obs"],
            },
        ],
        topic_map={"flood_damage": ["vid-1", "vid-2"], "election_rally": ["vid-3"]},
        general_notes=[
            {"note_id": "general-1", "video_id": "vid-1"},
            {"note_id": "general-2", "video_id": "vid-1"},
            {"note_id": "general-3", "video_id": "vid-2"},
        ],
        query_claims=[
            {"claim_id": "claim-1", "query_id": "1"},
            {"claim_id": "claim-2", "query_id": "1"},
            {"claim_id": "claim-3", "query_id": "2"},
        ],
    )

    assert metrics["total_observations"] == 2
    assert metrics["total_grounded_notes"] == 2
    assert metrics["videos_with_notes"] == 1
    assert metrics["avg_notes_per_video"] == 2.0
    assert metrics["groundedness"] == 0.5
    assert metrics["per_topic"]["flood_damage"]["total_notes"] == 2
    assert metrics["per_topic"]["flood_damage"]["avg_notes_per_video"] == 2.0
    assert metrics["total_general_notes"] == 3
    assert metrics["general_notes_videos"] == 2
    assert metrics["general_notes_avg_per_video"] == 1.5
    assert metrics["total_query_claims"] == 3
    assert metrics["query_claims_queries"] == 2


def test_report_metrics_count_valid_citations_and_per_query_details() -> None:
    metrics = compute_report_metrics(
        reports=[
            {
                "query_id": "1",
                "topic": "flood_damage",
                "sections": [
                    {"text": "flood damage in city"},
                    {"text": "response details"},
                ],
                "citations": [
                    {"note_id": "note-1", "video_id": "vid-1"},
                    {"note_id": "note-2", "video_id": "unknown"},
                ],
            },
            {
                "query_id": "2",
                "topic": "election_rally",
                "sections": [{"text": "election rally turnout"}],
                "citations": [{"note_id": "note-3", "video_id": "vid-2"}],
            },
        ],
        pipeline="pipe_a",
        queries=sample_queries(),
    )

    assert metrics["pipeline"] == "pipe_a"
    assert metrics["total_reports"] == 2
    assert metrics["total_sections"] == 3
    assert metrics["total_citations"] == 3
    assert metrics["citation_validity"] == 0.6667
    assert metrics["unique_videos_cited"] == 2
    assert metrics["per_query"]["1"]["n_valid_citations"] == 1
    assert metrics["per_query"]["2"]["query_type"] == "comparison"
    assert metrics["per_query"]["2"]["language"] == "fr"


def test_comparison_groups_by_query_type_language_and_topic() -> None:
    queries = sample_queries()
    pipe_a = compute_report_metrics(
        reports=[
            {
                "query_id": "1",
                "topic": "flood_damage",
                "sections": [{"text": "flood damage in city"}],
                "citations": [{"note_id": "note-1", "video_id": "vid-1"}],
            },
            {
                "query_id": "2",
                "topic": "election_rally",
                "sections": [{"text": "election rally turnout"}],
                "citations": [{"note_id": "note-2", "video_id": "vid-2"}],
            },
        ],
        pipeline="pipe_a",
        queries=queries,
    )
    pipe_b = compute_report_metrics(
        reports=[
            {
                "query_id": "1",
                "topic": "flood_damage",
                "sections": [
                    {"text": "flood damage"},
                    {"text": "city response"},
                ],
                "citations": [{"note_id": "note-3", "video_id": "vid-3"}],
            }
        ],
        pipeline="pipe_b",
        queries=queries,
    )

    comparison = compute_comparison({"pipe_a": pipe_a, "pipe_b": pipe_b}, queries)

    assert comparison["pipelines"]["pipe_a"]["total_sections"] == 2
    assert comparison["pipelines"]["pipe_b"]["avg_sections_per_query"] == 2.0
    assert comparison["by_query_type"]["fact"]["pipe_a"]["n_queries"] == 1
    assert comparison["by_query_type"]["comparison"]["pipe_a"]["total_sections"] == 1
    assert comparison["by_language"]["en"]["pipe_b"]["total_sections"] == 2
    assert comparison["by_language"]["fr"]["pipe_a"]["n_queries"] == 1
    assert comparison["by_topic"]["flood_damage"]["pipe_b"]["n_queries"] == 1
    assert comparison["by_topic"]["election_rally"]["pipe_a"]["total_sections"] == 1
