from __future__ import annotations

from marquis.common.contracts import build_query_topic_map
from marquis.common.validation import (
    validate_claim_packet,
    validate_general_note,
    validate_query_conditioned_claim,
    validate_report_citation,
)
from marquis.common.video import normalize_video_id, resolve_video_path


def test_query_topic_map_groups_by_topic_id() -> None:
    queries = [
        {"query_id": "1", "topic_id": "TQEmEycTI0RQ"},
        {"query_id": "2", "topic_id": "TQEmEycTI0RQ"},
        {"query_id": "3", "topic_id": "TjHepwuhkYFA"},
    ]
    result = build_query_topic_map(queries)
    assert [q["query_id"] for q in result["TQEmEycTI0RQ"]] == ["1", "2"]
    assert [q["query_id"] for q in result["TjHepwuhkYFA"]] == ["3"]


def test_video_id_normalization_and_prefix_fallback(tmp_path) -> None:
    video = tmp_path / "S0C0crjCBvM-extra.mp4"
    video.write_text("not a real video")

    assert normalize_video_id("youtube-S0C0crjCBvM3") == "S0C0crjCBvM3"
    assert resolve_video_path(str(tmp_path), "S0C0crjCBvM3") == str(video)


def test_schema_validation_accepts_fixture_shapes() -> None:
    cases = [
        (
            {
                "note_id": "gn-1",
                "video_id": "vid1",
                "topic": "flooding",
                "text": "Water covers a street.",
                "modality": "visual",
            },
            validate_general_note,
        ),
        (
            {
                "claim_id": "qc-1",
                "query_id": "1",
                "video_id": "vid1",
                "topic": "flooding",
                "claim": "The video shows street flooding.",
            },
            validate_query_conditioned_claim,
        ),
        (
            {
                "query_id": "1",
                "topic": "flooding",
                "stream": "query_based",
                "claim_ids": ["qc-1"],
            },
            validate_claim_packet,
        ),
        ({"note_id": "gn-1", "video_id": "vid1"}, validate_report_citation),
    ]
    for record, validator in cases:
        assert validator(record) == []


def test_schema_validation_reports_missing_required_fields() -> None:
    errors = validate_general_note({"video_id": "v1"})
    assert any("missing required field 'note_id'" in error for error in errors)


def test_schema_validation_rejects_bad_enums_unknown_fields_and_bool_scores() -> None:
    note_errors = validate_general_note(
        {
            "note_id": "gn-1",
            "video_id": "vid1",
            "topic": "flooding",
            "text": "Water covers a street.",
            "modality": "not-a-modality",
            "extra": "not part of the contract",
        }
    )
    assert any("modality" in error and "expected one of" in error for error in note_errors)
    assert any("unknown field 'extra'" in error for error in note_errors)

    packet_errors = validate_claim_packet(
        {
            "query_id": "1",
            "topic": "flooding",
            "stream": "query_based",
            "scores": [True],
        }
    )
    assert any("scores[0]" in error and "bool" in error for error in packet_errors)
