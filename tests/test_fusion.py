from __future__ import annotations

import pytest

from marquis.retrieval.fusion import (
    fuse,
    load_ranked_results,
    reciprocal_rank_fusion,
    score_max,
    score_mean,
    score_sum,
    weighted_reciprocal_rank_fusion,
    write_trec,
)

SUBQUERY_RESULTS = {
    "1_sub0": [("A", 2.0), ("B", 1.0)],
    "1_sub1": [("B", 4.0), ("C", 3.0)],
    "2_sub0": [("D", 5.0)],
}
REVERSE = {"1_sub0": "1", "1_sub1": "1", "2_sub0": "2"}


def test_reciprocal_rank_fusion() -> None:
    fused = reciprocal_rank_fusion(SUBQUERY_RESULTS, REVERSE, k=60)
    assert fused["1"]["A"] == pytest.approx(1 / 61)
    assert fused["1"]["B"] == pytest.approx((1 / 62) + (1 / 61))
    assert fused["2"]["D"] == pytest.approx(1 / 61)


def test_weighted_reciprocal_rank_fusion() -> None:
    fused = weighted_reciprocal_rank_fusion(SUBQUERY_RESULTS, REVERSE, k=60)
    assert fused["1"]["A"] == pytest.approx(2.0 / 61)
    assert fused["1"]["B"] == pytest.approx((2.0 / 62) + (4.0 / 61))


def test_score_fusion_methods() -> None:
    assert score_sum(SUBQUERY_RESULTS, REVERSE)["1"]["B"] == pytest.approx(5.0)
    assert score_max(SUBQUERY_RESULTS, REVERSE)["1"]["B"] == pytest.approx(4.0)
    assert score_mean(SUBQUERY_RESULTS, REVERSE)["1"]["B"] == pytest.approx(2.5)
    assert fuse(SUBQUERY_RESULTS, REVERSE, method="mean")["1"]["B"] == pytest.approx(2.5)


def test_score_max_preserves_negative_scores() -> None:
    fused = score_max({"1_sub0": [("A", -2.0), ("A", -1.0)]}, {"1_sub0": "1"})
    assert fused["1"]["A"] == pytest.approx(-1.0)


def test_load_ranked_results_supports_tsv_and_trec(tmp_path) -> None:
    tsv = tmp_path / "rank.tsv"
    tsv.write_text("q1\td1\t1.0\nq1\td2\t2.0\n")
    assert load_ranked_results(str(tsv))["q1"] == [("d2", 2.0), ("d1", 1.0)]

    trec = tmp_path / "rank.trec"
    trec.write_text("q1 Q0 d1 1 3.0 run\nq1 Q0 d2 2 1.5 run\n")
    assert load_ranked_results(str(trec))["q1"] == [("d1", 3.0), ("d2", 1.5)]


def test_write_trec_depth(tmp_path) -> None:
    out = tmp_path / "run.trec"
    write_trec({"1": {"A": 0.2, "B": 0.3}}, str(out), run_name="test", depth=1)
    assert out.read_text() == "1 Q0 B 1 0.300000 test\n"
