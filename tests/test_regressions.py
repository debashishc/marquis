from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from marquis.information_extraction import assemble_packets, extract
from marquis.retrieval import rrf
from marquis.rlm_controller.rlm import repl as repl_module
from marquis.rlm_controller.rlm import rlm_repl
from marquis.rlm_controller.rlm.repl import REPLEnv


def test_extract_resume_rebuilds_combined_claims_from_existing_query_files(
    tmp_path, monkeypatch
) -> None:
    query_file = tmp_path / "query_1.jsonl"
    query_file.write_text('{"claim_id":"old","claim":"old"}\n', encoding="utf-8")

    monkeypatch.setattr(
        extract, "load_queries", lambda _path: [{"query_id": "1", "topic_id": "topic"}]
    )
    monkeypatch.setattr(extract, "load_query_video_mapping", lambda _path: {"1": ["v1"]})
    monkeypatch.setattr(extract, "_build_model", lambda _args: object())
    monkeypatch.setattr(
        extract,
        "extract_claims_vlm",
        lambda *_args, **_kwargs: [{"claim_id": "new", "claim": "new"}],
    )

    args = SimpleNamespace(
        queries_jsonl="queries.jsonl",
        query_video_mapping="query_video_mapping.json",
        query_mode="single-query",
        out_dir=str(tmp_path),
        verbose=False,
    )
    extract.run_query(args, {"run_id": "run"})

    combined = (tmp_path / "query_conditioned_claims.jsonl").read_text(encoding="utf-8")
    assert '"claim_id": "old"' in combined
    assert '"claim_id": "new"' not in combined


def test_rrf_cli_sorts_subquery_results_before_rank_scoring(tmp_path) -> None:
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps({"1": ["1_sub0"]}), encoding="utf-8")
    results = tmp_path / "results.tsv"
    results.write_text("1_sub0\tlow\t1.0\n1_sub0\thigh\t5.0\n", encoding="utf-8")
    out_dir = tmp_path / "out"
    cfg = SimpleNamespace(
        data=SimpleNamespace(
            subquery_mapping=str(mapping),
            subquery_results=str(results),
            baseline_results=str(tmp_path / "missing.json"),
        ),
        output=SimpleNamespace(out_dir=str(out_dir)),
        runtime=SimpleNamespace(rrf_k=0, rrf_method="rrf", fusion_depth=2),
    )

    rrf.run(cfg)
    fused = json.loads((out_dir / "rank-expanded-rrf.json").read_text(encoding="utf-8"))
    assert list(fused["1"]) == ["high", "low"]


def test_packet_assembly_filters_missing_unli_prob_when_thresholded(tmp_path) -> None:
    notes = tmp_path / "notes.jsonl"
    notes.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "note_id": "cal",
                        "video_id": "v",
                        "topic": "topic",
                        "text": "flood road",
                        "calibration": {"unli": {"prob": 0.9}},
                    }
                ),
                json.dumps(
                    {
                        "note_id": "raw",
                        "video_id": "v",
                        "topic": "topic",
                        "text": "flood road",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    packets = assemble_packets.assemble_note_packets(
        str(notes),
        queries=[{"query_id": "1", "query": "flood road", "title": "", "persona_title": ""}],
        query_topic={"1": "topic"},
        run_id="run",
        unli_threshold=0.5,
    )
    assert packets[0]["note_ids"] == ["cal"]


def test_marquis_extract_validate_uses_common_fixtures_off_cluster() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{Path.cwd() / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["MAGMAR_VIDEO_ROOT"] = "/tmp/marquis-no-such-root"
    env["MAGMAR_QUERIES_JSONL"] = "examples/data/MAGMaR2026_queries.jsonl"
    env["MARQUIS_REFERENCE"] = "examples/data/reference.json"
    env["MAGMAR_EXPANDED_QUERIES"] = "examples/data/expanded_queries.json"

    completed = subprocess.run(
        [sys.executable, "-m", "marquis.information_extraction.cli", "validate"],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "ALL CHECKS PASSED" in completed.stdout


class _NullLogger:
    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


class _FakeLLM:
    def completion(self, _messages):
        return "FINAL(done)"


class _FakeREPLEnv:
    locals = {}


def test_rlm_no_code_response_is_kept_as_assistant_turn(monkeypatch) -> None:
    monkeypatch.setattr(rlm_repl, "REPLEnv", lambda **_kwargs: _FakeREPLEnv())
    rlm = rlm_repl.RLM_REPL.__new__(rlm_repl.RLM_REPL)
    rlm.llm = _FakeLLM()
    rlm._extra_tools = {}
    rlm._system_prompt_builder = lambda: [{"role": "system", "content": "system"}]
    rlm._max_iterations = 1
    rlm.recursive_model = "gpt-5-mini"
    rlm.logger = _NullLogger()
    rlm.repl_env_logger = _NullLogger()
    rlm.trajectory = []

    assert rlm.completion("context", query="question") == "done"
    assert {"role": "assistant", "content": "FINAL(done)"} in rlm.messages
    assert rlm.trajectory[0]["executions"] == []


def test_repl_does_not_double_execute_prefix_side_effects(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setattr(repl_module, "Sub_RLM", lambda model: object())
    repl = REPLEnv(execution_timeout_seconds=5)
    result = repl.code_execution("open('side.txt','a').write('x')\n1/0")
    assert (Path(repl.temp_dir) / "side.txt").read_text(encoding="utf-8") == "x"
    assert "division by zero" in result.stderr


def test_repl_execution_timeout(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setattr(repl_module, "Sub_RLM", lambda model: object())
    repl = REPLEnv(execution_timeout_seconds=0.1)
    result = repl.code_execution("import time\ntime.sleep(1)")
    assert "REPL execution exceeded" in result.stderr
