from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def run_help(module: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    src = str(Path.cwd() / "src")
    env["PYTHONPATH"] = f"{src}{os.pathsep}{env.get('PYTHONPATH', '')}"
    return subprocess.run(
        [sys.executable, "-m", module, "--help"],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


def run_module(args: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    src = str(Path.cwd() / "src")
    env["PYTHONPATH"] = f"{src}{os.pathsep}{env.get('PYTHONPATH', '')}"
    return subprocess.run(
        [sys.executable, "-m", *args],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


def test_console_module_help_smoke() -> None:
    modules = [
        "marquis.retrieval.cli",
        "marquis.information_extraction.cli",
        "marquis.article_generation.cli",
        "marquis.rlm_controller.cli",
        "marquis.evaluation.cli",
    ]
    for module in modules:
        completed = run_help(module)
        assert completed.returncode == 0, completed.stderr
        assert "usage:" in completed.stdout


def test_retrieval_subcommand_help_smoke() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{Path.cwd() / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    commands = ["fusion", "rrf", "expand", "prepare-subqueries", "qrels", "rerank"]
    for command in commands:
        completed = subprocess.run(
            [sys.executable, "-m", "marquis.retrieval.cli", command, "--help"],
            text=True,
            capture_output=True,
            env=env,
            check=False,
        )
        assert completed.returncode == 0, f"{command}\nSTDERR:\n{completed.stderr}"
        assert "usage:" in completed.stdout
    fusion_help = subprocess.run(
        [sys.executable, "-m", "marquis.retrieval.cli", "fusion", "--help"],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    assert "data.subquery_results" in fusion_help.stdout


def test_generation_subcommand_help_smoke() -> None:
    commands = [
        (["marquis.article_generation.cli", "baseline", "--help"], "data.claims_path"),
        (["marquis.article_generation.cli", "bullet", "--help"], "runtime.bullet"),
        (["marquis.article_generation.cli", "ginger", "--help"], "data.claims_path"),
    ]
    for args, expected in commands:
        completed = run_module(args)
        assert completed.returncode == 0, completed.stderr
        assert expected in completed.stdout


def test_contract_validator_help_smoke() -> None:
    completed = run_help("marquis.common.validate_contracts")
    assert completed.returncode == 0, completed.stderr
    assert "--verbose" in completed.stdout


def test_qa_subcommand_help_smoke() -> None:
    commands = [
        (["marquis.information_extraction.cli", "qa-decompose", "--help"], "data.queries_jsonl"),
        (
            ["marquis.information_extraction.cli", "prepare-transcripts", "--help"],
            "data.transcripts",
        ),
        (["marquis.information_extraction.cli", "qa-answer", "--help"], "data.qa_queries"),
        (["marquis.information_extraction.cli", "qa", "--help"], "QA-based"),
    ]
    for args, expected in commands:
        completed = run_module(args)
        assert completed.returncode == 0, completed.stderr
        assert expected in completed.stdout


def test_rlm_subcommand_help_smoke() -> None:
    commands = [
        (["marquis.rlm_controller.cli", "magmar", "--help"], "Self-extraction RLM"),
        (["marquis.rlm_controller.cli", "magmar-notes", "--help"], "pre-extracted claims"),
    ]
    for args, expected in commands:
        completed = run_module(args)
        assert completed.returncode == 0, completed.stderr
        assert expected in completed.stdout


def test_evaluation_subcommand_help_smoke() -> None:
    commands = [
        (["marquis.evaluation.cli", "retrieval", "--help"], "--qrels"),
        (["marquis.evaluation.cli", "extraction", "--help"], "--general-notes"),
    ]
    for args, expected in commands:
        completed = run_module(args)
        assert completed.returncode == 0, completed.stderr
        assert expected in completed.stdout
