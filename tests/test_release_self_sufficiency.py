from __future__ import annotations

import subprocess
from pathlib import Path

RELEASE_PATHS = [
    Path("README.md"),
    Path("docs"),
    Path("configs"),
    Path("scripts"),
    Path("src"),
    Path("tests"),
    Path("examples"),
    Path("slurm/templates"),
]

STALE_PATTERNS = [
    "note_taking/",
    "QA/",
    "RLM-Magmar",
    "report_writing",
    "old-figures-to-delete",
    "/srv/",
    "/home/",
    "MARS-MAGMaR26-RAG-RLM",
]

PAPER_COMPONENTS = [
    ("Retrieval", "marquis-retrieve", "configs/retrieval"),
    ("Information extraction", "marquis-extract", "configs/information_extraction"),
    ("QA extraction", "qa-decompose", "configs/information_extraction"),
    ("Article generation", "marquis-generate", "configs/article_generation"),
    ("RLM controller", "marquis-rlm", "configs/rlm_controller"),
    ("Evaluation", "marquis-evaluate", "configs/evaluation"),
]

SAMPLE_PATHS = [
    Path("examples/data/MAGMaR2026_queries.jsonl"),
    Path("examples/data/reference.json"),
    Path("examples/data/expanded_queries.json"),
    Path("examples/quicktest/general_notes.jsonl"),
    Path("examples/quicktest/query_conditioned_claims.jsonl"),
]


def _iter_text_files(root: Path):
    completed = subprocess.run(
        ["git", "ls-files", "--", str(root)],
        check=True,
        capture_output=True,
        text=True,
    )
    for file_name in completed.stdout.splitlines():
        path = Path(file_name)
        if path.name == "test_release_self_sufficiency.py":
            continue
        if path.is_file() and path.suffix.lower() not in {".pyc", ".png", ".jpg", ".jpeg", ".pdf"}:
            yield path


def test_release_paths_do_not_reference_deleted_legacy_locations() -> None:
    offenders = []
    for root in RELEASE_PATHS:
        if not root.exists():
            continue
        for path in _iter_text_files(root):
            text = path.read_text(encoding="utf-8")
            for pattern in STALE_PATTERNS:
                if pattern in text:
                    offenders.append(f"{path}: {pattern}")
    assert offenders == []


def test_paper_map_mentions_core_components() -> None:
    text = Path("docs/paper_map.md").read_text(encoding="utf-8")
    for label, command, config_dir in PAPER_COMPONENTS:
        assert label in text
        assert command in text
        assert config_dir in text
        assert Path(config_dir).is_dir()


def test_sample_artifacts_exist() -> None:
    for path in SAMPLE_PATHS:
        assert path.exists(), path
        assert path.stat().st_size > 0, path


def test_configs_use_env_vars_not_hardcoded_paths() -> None:
    paths = [
        Path("configs/retrieval/data/default.yaml"),
        Path("configs/information_extraction/data/default.yaml"),
        Path("configs/rlm_controller/data/default.yaml"),
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "/exp/" not in text, f"{path} contains hardcoded /exp/ path"
        assert "oc.env:" in text, f"{path} should use env var references"


def test_internal_slurm_directory_is_ignored() -> None:
    completed = subprocess.run(
        ["git", "check-ignore", "-q", "slurm/internal/example.sbatch"],
        check=False,
    )
    assert completed.returncode == 0


def test_slurm_templates_follow_conventions() -> None:
    templates = sorted(Path("slurm/templates").glob("*.sbatch"))
    assert templates
    for path in templates:
        text = path.read_text(encoding="utf-8")
        assert text.startswith("#!/bin/bash -l")
        assert "#SBATCH --account=CHANGE_ME_ACCOUNT" in text
        assert "#SBATCH --partition=" in text
        assert "#SBATCH --time=" in text
        assert "#SBATCH --mem=" in text
        assert "#SBATCH --cpus-per-task=" in text
        assert 'source "${MARQUIS_SLURM_ENV:-slurm/templates/cluster_env.sh}"' in text
        assert "\nsrun scripts/run_" in text
        assert "#SBATCH --partition=CHANGE_ME_PARTITION" not in text
        assert "/home/" not in text


def test_slurm_env_keeps_caches_configurable() -> None:
    text = Path("slurm/templates/cluster_env.sh").read_text(encoding="utf-8")
    assert "MARQUIS_CACHE_BASE" in text
    assert "XDG_CACHE_HOME" in text
    assert "HF_HOME" in text
    assert "UV_PYTHON_INSTALL_DIR" in text
    assert "APPTAINER_CACHEDIR" in text
    assert "MARQUIS_VENV" in text
    assert "/home/" not in text


def test_release_docs_have_no_todo_placeholders() -> None:
    offenders = []
    for path in [
        Path("README.md"),
        Path("docs/reproduction.md"),
        Path("docs/release_assets.md"),
        Path("docs/paper_map.md"),
        Path("docs/prompt_map.md"),
    ]:
        text = path.read_text(encoding="utf-8")
        if "TODO" in text:
            offenders.append(str(path))
    assert offenders == []
