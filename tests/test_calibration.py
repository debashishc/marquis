from __future__ import annotations

import json
from pathlib import Path

from marquis.information_extraction.calibrate import (
    _build_unli_index,
    _calibrate_general_notes,
    _calibrate_query_claims,
    _parse_prob,
)


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_parse_prob_accepts_answer_tags_plain_scores_and_bounds() -> None:
    assert _parse_prob("<answer>0.42</answer>") == 0.42
    assert _parse_prob(" 0.7 ") == 0.7
    assert _parse_prob("<answer>1.1</answer>") is None
    assert _parse_prob("-0.1") is None
    assert _parse_prob("not a score") is None
    assert _parse_prob(None) is None


def test_unli_index_builds_key_and_id_indexes_and_skips_malformed(tmp_path: Path) -> None:
    predictions = tmp_path / "unli_predictions.jsonl"
    write_jsonl(
        predictions,
        [
            {
                "video_id": "vid-1",
                "note_id": "note-1",
                "text": "Water covers the street.",
                "prob": "0.8",
                "meta": {"label": "supported"},
            },
            {
                "meta": {
                    "video_id": "vid-2",
                    "claim_id": "claim-2",
                    "claim": "A rally is shown.",
                },
                "outputs": ["unparseable", "<answer>0.33</answer>"],
            },
            {"video_id": "vid-3", "text": "<answer>0.5</answer>"},
            {"video_id": "vid-4", "claim": "Out of range.", "prob": 1.5},
            {"video_id": "vid-5", "claim": "Missing probability."},
        ],
    )

    key_index, id_index = _build_unli_index(str(predictions))

    assert set(key_index) == {
        ("vid-1", "Water covers the street."),
        ("vid-2", "A rally is shown."),
    }
    assert set(id_index) == {"note-1", "claim-2"}
    assert key_index[("vid-1", "Water covers the street.")]["prob"] == 0.8
    assert id_index["claim-2"]["prob"] == 0.33


def test_calibrate_general_notes_uses_stable_id_then_fallback_match(tmp_path: Path) -> None:
    predictions = tmp_path / "unli_predictions.jsonl"
    notes = tmp_path / "general_notes.jsonl"
    out = tmp_path / "general_notes_calibrated.jsonl"
    write_jsonl(
        predictions,
        [
            {"video_id": "vid-1", "note_id": "note-1", "text": "Old text.", "prob": 0.91},
            {"video_id": "vid-2", "text": "Fallback note.", "prob": "0.44"},
        ],
    )
    write_jsonl(
        notes,
        [
            {
                "note_id": "note-1",
                "video_id": "vid-1",
                "topic": "topic",
                "text": "Edited text still matches by ID.",
                "modality": "visual",
            },
            {
                "note_id": "note-2",
                "video_id": "vid-2",
                "topic": "topic",
                "text": "Fallback note.",
                "modality": "ocr",
            },
            {
                "note_id": "note-3",
                "video_id": "vid-3",
                "topic": "topic",
                "text": "No prediction.",
                "modality": "audio",
            },
        ],
    )
    key_index, id_index = _build_unli_index(str(predictions))

    total, matched = _calibrate_general_notes(str(notes), key_index, id_index, str(out))

    calibrated = read_jsonl(out)
    assert (total, matched) == (3, 2)
    assert calibrated[0]["confidence"] == 0.91
    assert calibrated[0]["calibration"]["unli"]["prob"] == 0.91
    assert calibrated[1]["confidence"] == 0.44
    assert "confidence" not in calibrated[2]


def test_calibrate_query_claims_filters_query_ids_and_falls_back(tmp_path: Path) -> None:
    predictions = tmp_path / "unli_predictions.jsonl"
    claims = tmp_path / "query_conditioned_claims.jsonl"
    out = tmp_path / "query_conditioned_claims_calibrated.jsonl"
    write_jsonl(
        predictions,
        [
            {"video_id": "vid-1", "claim_id": "claim-1", "claim": "Old claim.", "prob": 0.82},
            {"video_id": "vid-2", "claim": "Fallback claim.", "prob": 0.56},
            {"video_id": "vid-3", "claim_id": "claim-3", "claim": "Filtered out.", "prob": 0.99},
        ],
    )
    write_jsonl(
        claims,
        [
            {
                "claim_id": "claim-1",
                "query_id": "1",
                "video_id": "vid-1",
                "topic": "topic",
                "claim": "Edited claim still matches by ID.",
            },
            {
                "claim_id": "claim-2",
                "query_id": "1",
                "video_id": "vid-2",
                "topic": "topic",
                "claim": "Fallback claim.",
            },
            {
                "claim_id": "claim-3",
                "query_id": "2",
                "video_id": "vid-3",
                "topic": "topic",
                "claim": "Filtered out.",
            },
        ],
    )
    key_index, id_index = _build_unli_index(str(predictions))

    total, matched = _calibrate_query_claims(
        str(claims), key_index, id_index, str(out), query_id_filter={"1"}
    )

    calibrated = read_jsonl(out)
    assert (total, matched) == (2, 2)
    assert [record["claim_id"] for record in calibrated] == ["claim-1", "claim-2"]
    assert calibrated[0]["confidence"] == 0.82
    assert calibrated[1]["calibration"]["unli"]["prob"] == 0.56
