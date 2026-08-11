from __future__ import annotations

from marquis.article_generation.citations import format_citation, parse_citations


def test_format_citation() -> None:
    assert format_citation("abc123", [1.5, 4.0]) == "[abc123, 1.5-4.0]"
    assert format_citation("abc123", None) == "[abc123, ?]"


def test_parse_citations() -> None:
    citations = parse_citations("Flooding is visible [vid1, 1.0-2.5] and later [vid2, ?].")
    assert citations[0]["video_id"] == "vid1"
    assert citations[0]["timestamp"] == ["1.0", "2.5"]
    assert citations[1]["video_id"] == "vid2"
    assert citations[1]["timestamp"] == ["?"]


def test_parse_citations_accepts_bullet_second_markers_and_malformed_spans() -> None:
    citations = parse_citations("A claim [vid1, 1s-2s] and another [vid2, not-a-time].")
    assert citations[0]["video_id"] == "vid1"
    assert citations[0]["timestamp"] == ["1", "2"]
    assert citations[1]["video_id"] == "vid2"
    assert citations[1]["timestamp"] == ["not-a-time"]
