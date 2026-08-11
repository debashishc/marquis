from __future__ import annotations

from pathlib import Path

import yaml

MODULES = [
    "retrieval",
    "information_extraction",
    "article_generation",
    "rlm_controller",
    "evaluation",
]
GROUPS = ["data", "model", "runtime", "output", "launcher"]


def test_config_trees_have_expected_groups() -> None:
    root = Path("configs")
    for module in MODULES:
        module_root = root / module
        assert (module_root / "config.yaml").exists()
        for group in GROUPS:
            assert (module_root / group).is_dir(), f"{module} missing {group}"
            assert list((module_root / group).glob("*.yaml")), f"{module}/{group} has no options"


def test_yaml_files_parse() -> None:
    for path in Path("configs").glob("**/*.yaml"):
        with open(path, encoding="utf-8") as f:
            assert yaml.safe_load(f) is not None


def test_configs_include_paper_hyperparameters() -> None:
    retrieval_model = yaml.safe_load(Path("configs/retrieval/model/default.yaml").read_text())
    retrieval_runtime = yaml.safe_load(Path("configs/retrieval/runtime/default.yaml").read_text())
    extraction_runtime = yaml.safe_load(
        Path("configs/information_extraction/runtime/default.yaml").read_text()
    )
    generation_runtime = yaml.safe_load(
        Path("configs/article_generation/runtime/default.yaml").read_text()
    )
    evaluation_runtime = yaml.safe_load(Path("configs/evaluation/runtime/default.yaml").read_text())
    rlm_runtime = yaml.safe_load(Path("configs/rlm_controller/runtime/default.yaml").read_text())

    assert retrieval_model["embedder"]["model"] == "Qwen/Qwen2.5-Omni-7B"
    assert retrieval_model["embedder"]["adapter"] == "Tevatron/OmniEmbed-v0.1"
    assert retrieval_runtime["fusion_depth"] == 100
    assert retrieval_runtime["retrieve"]["top_k"] == 4

    assert extraction_runtime["calibrate"]["scorer_backend"] == "unli"
    assert extraction_runtime["qa"]["retrieval"]["sim_threshold"] == 0.1
    assert extraction_runtime["qa"]["retrieval"]["top_k"] == 4
    assert extraction_runtime["qa"]["max_steps"] == 5

    assert generation_runtime["max_claims_per_query"] == 25
    assert generation_runtime["top_n_clusters"] == 5
    assert "bullet" in generation_runtime
    assert generation_runtime["ginger"]["cluster_temperature"] == 0.3
    assert generation_runtime["ginger"]["rank_temperature"] == 0.3
    assert generation_runtime["ginger"]["summarize_temperature"] == 0.5
    assert generation_runtime["ginger"]["fluency_temperature"] == 0.7

    assert evaluation_runtime["metrics"] == [
        "nDCG@10",
        "nDCG@20",
        "nDCG@100",
        "Recall@10",
        "Recall@20",
        "Recall@100",
    ]
    assert "max_steps" in rlm_runtime
