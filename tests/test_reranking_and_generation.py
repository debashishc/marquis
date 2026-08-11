from __future__ import annotations

from marquis.article_generation import baseline, ginger
from marquis.article_generation.bullet import (
    _format_citation_marker,
    render_bullet_report,
    sort_evidence,
)
from marquis.retrieval.reranking import load_pair_scores, rerank_candidates, write_reranked_trec


def test_rankvideo_reranking_prefers_external_scores(tmp_path) -> None:
    first_stage = {
        "1": [("A", 10.0), ("B", 9.0), ("C", 8.0)],
    }
    scores_path = tmp_path / "rankvideo.tsv"
    scores_path.write_text("1\tB\t0.95\n1\tA\t0.20\n", encoding="utf-8")

    scores = load_pair_scores(str(scores_path))
    reranked = rerank_candidates(first_stage, scores, depth=3)

    assert reranked["1"] == [("B", 0.95), ("A", 0.20), ("C", 8.0)]

    out = tmp_path / "reranked.trec"
    write_reranked_trec(reranked, str(out), run_name="rankvideo", depth=2)
    assert out.read_text(encoding="utf-8").splitlines() == [
        "1 Q0 B 1 0.950000 rankvideo",
        "1 Q0 A 2 0.200000 rankvideo",
    ]


def test_bullet_baseline_renders_findings_with_citations() -> None:
    evidence = [
        {"claim": "Second claim.", "video_id": "vid2", "timestamp": [3, 4], "confidence": 0.2},
        {"claim": "First claim.", "video_id": "vid1", "timestamp": [1, 2], "confidence": 0.9},
    ]
    report = render_bullet_report(sort_evidence(evidence), title="Bullet")

    assert report.splitlines()[0] == "# Bullet"
    assert "1. First claim. [vid1, 1-2]" in report
    assert "2. Second claim. [vid2, 3-4]" in report


def test_article_generation_formatters_tolerate_optional_claim_fields() -> None:
    claim = {"claim_id": "c1", "claim": "A claim.", "video_id": "vid1"}
    assert "Timestamp: ?" in baseline.format_claims_for_prompt([claim])
    assert "source: unknown" in ginger.format_claims_for_prompt([claim])


def test_bullet_citation_marker_handles_short_and_string_timestamps() -> None:
    assert _format_citation_marker({"video_id": "vid1", "timestamp": [1]}) == "[vid1, 1s]"
    assert _format_citation_marker({"video_id": "vid1", "timestamp": "1-2"}) == "[vid1, 1-2s]"
