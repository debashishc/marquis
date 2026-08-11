# MARQUIS

<p align="center">
  <img src="assets/marquis-overview.png" alt="MARQUIS system overview" width="900">
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.17640"><img src="https://img.shields.io/badge/Paper-arXiv%202605.17640-b31b1b.svg" alt="arXiv paper"></a>
  <a href="https://huggingface.co/papers/2605.17640"><img src="https://img.shields.io/badge/Hugging%20Face-Paper-yellow.svg" alt="Hugging Face paper page"></a>
  <a href="https://github.com/rekriz11/MAGMAR_2026"><img src="https://img.shields.io/badge/Task-MAGMaR%202026-green.svg" alt="MAGMaR 2026 shared task"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="MIT license"></a>
</p>

MARQUIS (**M**ultimodal **A**rticle generation via **R**etrieval, **QU**ery
decomposition, **U**ncertainty calibration, and **I**terative evidence
**S**ynthesis) is a three-stage system for video retrieval-augmented article
generation. It retrieves relevant videos, turns them into calibrated evidence,
and writes cited reports over large video collections.

On MAGMaR2026, query expansion, rank fusion, and RankVideo reranking improve
retrieval nDCG@10 from `0.195` to `0.759`. The QA-based article generator
improves average human score from `3.09` to `3.83` over the CAG baseline.

---

## Guide

| Need | Start here |
|:--|:--|
| Understand the repo contents | [What is in this repo](#what-is-in-this-repo) |
| Match the code to the paper | [How MARQUIS works](#how-marquis-works), then [paper map](docs/paper_map.md) |
| Find prompt implementations | [Prompt map](docs/prompt_map.md) |
| Run or reproduce the pipeline | [Quick setup](#quick-setup), then [Run the pipeline](#run-the-pipeline) |
| Work with MAGMaR data | [MAGMaR 2026 shared task](https://github.com/rekriz11/MAGMAR_2026), then [release artifacts](docs/release_assets.md) |
| Run on a cluster | [SLURM notes](docs/slurm.md), then [`slurm/templates`](slurm/templates) |
| Inspect package layout | [Repository layout](#repository-layout) |
| Check the release | [Tests](#tests) |

---

## What is in this repo

The Python package lives under `src/marquis`. Its modules match the paper:
`retrieval`, `information_extraction`, `article_generation`, `rlm_controller`,
`evaluation`, and `common`.

The repo tracks code, configs, prompts, tests, small fixtures, SLURM templates,
and the README figure. It leaves out videos, checkpoints, generated runs, model
caches, paper-scale outputs, and the paper source.

Prompt implementations are mapped in [`docs/prompt_map.md`](docs/prompt_map.md).
The component map is in [`docs/paper_map.md`](docs/paper_map.md).
The paper source is not bundled here.

[Back to top](#marquis)

---

## How MARQUIS works

MARQUIS answers five design questions:

| Question | MARQUIS answer |
|:--|:--|
| How is a complex information need represented? | The query is decomposed into atomic sub-queries before retrieval. |
| How are candidate videos selected? | Dense retrieval runs over each sub-query, then rank fusion and RankVideo reranking produce the final run. |
| How does video become usable evidence? | The extraction layer emits notes, claims, QA answers, support scores, calibration metadata, and packets. |
| How are citations preserved? | Generators consume structured evidence records with source video IDs and timestamps. |
| How does the RLM variant differ? | A root LM calls MARQUIS tools from a persistent REPL with structured memory. |


[Back to top](#marquis)

---

## Retrieval

The retrieval stage implements the paper's first system section:

1. Expand and decompose the original MAGMaR query.
2. Run dense retrieval over each sub-query.
3. Fuse ranked lists with RRF or weighted variants.
4. Apply RankVideo reranking scores when available.
5. Convert runs and qrels into evaluation-ready formats.

Primary code: [`src/marquis/retrieval`](src/marquis/retrieval)

Related docs: [`docs/paper_map.md`](docs/paper_map.md),
[`docs/reproduction.md`](docs/reproduction.md)

[Back to top](#marquis)

---

## Information extraction

The extraction stage converts retrieved videos into source-linked evidence:

- Query-agnostic notes
- Query-conditioned claims
- Iterative QA answers
- Video-grounded support scores and calibration metadata
- Evidence packets for downstream generation

Primary code: [`src/marquis/information_extraction`](src/marquis/information_extraction)

Prompt coverage: [`src/marquis/information_extraction/qa/prompts.py`](src/marquis/information_extraction/qa/prompts.py),
[`docs/prompt_map.md`](docs/prompt_map.md)

[Back to top](#marquis)

---

## Article generation

Generation writes cited reports from extracted evidence, not raw videos.
The repo includes these strategies:

| Strategy | Role |
|:--|:--|
| Bullet | Conservative evidence rendering with inline citations. |
| CAG | Single-pass cited article generation over extracted evidence. |
| GINGER | Cluster, rank, summarize, and rewrite evidence into a fluent article. |
| QA | Build reports from query-decomposed video QA evidence. |
| RLM | Let a root LM gather, revise, and select evidence before writing. |

Primary code: [`src/marquis/article_generation`](src/marquis/article_generation)

[Back to top](#marquis)

---

## MARQUIS-RLM

MARQUIS-RLM exposes retrieval, extraction, QA, calibration, and generation as
tools in a persistent Python REPL. The root LM searches structured memory,
inspects evidence, resolves conflicts, and decides what to include.

Primary code: [`src/marquis/rlm_controller`](src/marquis/rlm_controller)

RLM notes: [`docs/rlm.md`](docs/rlm.md)

Model-backed RLM runs require configured model credentials or local model
backends. Default tests only validate import, prompt, schema, and CLI behavior.

[Back to top](#marquis)

---

## Quick setup

```bash
make install-dev
make quickstart
```

The Makefile uses `uv` by default. The base install avoids heavyweight model
dependencies. Install only the extras needed for the stages you plan to run:

```bash
uv sync --extra retrieval --extra eval
uv sync --extra vlm --extra qa --extra rlm
```

`uv.lock` is intentionally not committed because the VLM, QA, and RLM extras
depend on platform-specific model backends. For CI and local development, use
`uv sync --extra dev`. The Makefile sets `UV_CACHE_DIR=.uv-cache` so uv can run
in restricted cluster/home-directory environments; `.uv-cache/` is ignored.

[Back to top](#marquis)

---

## Run the pipeline

Videos, predictions, checkpoints, and paper-scale outputs live outside
the repo. Set these environment variables before running:

```bash
export MAGMAR_ROOT=/path/to/magmar26
export MAGMAR_VIDEO_ROOT="$MAGMAR_ROOT"
export MAGMAR_QUERIES_JSONL="$MAGMAR_ROOT/MAGMaR2026_queries.jsonl"
export MARQUIS_REFERENCE="$MAGMAR_ROOT/reference.json"
export MARQUIS_CLAIMS_PATH="$MAGMAR_ROOT/features/claims"
```

The MAGMaR 2026 shared-task repository has the official queries, `reference.json`
(topic_id to relevant videos), task definitions, and data download instructions.

The main entrypoints are:

```bash
marquis-retrieve --help
marquis-extract --help
marquis-generate --help
marquis-rlm --help
marquis-evaluate --help
```

For full-stage runs, use the script wrappers:

```bash
scripts/run_retrieval.sh
scripts/run_extraction.sh
scripts/run_generation.sh
scripts/run_rlm.sh
scripts/run_evaluation.sh
```

The no-model smoke check validates imports, CLI help, and sample contracts:

```bash
make quickstart
```

[Back to top](#marquis)

---

## Repository layout

```text
src/marquis/      Installable MARQUIS package
configs/          Hydra-style configuration groups
scripts/          Reproduction entrypoint scripts
examples/         Tiny non-paper-scale example inputs
tests/            Unit and smoke tests
docs/             Release notes and reproduction docs
assets/           Small README/project assets only
slurm/templates/  Generic templates for SLURM runs
```

This repo does not store generated outputs, model caches, videos, checkpoints,
or local workbench directories.

[Back to top](#marquis)

---

## Release artifacts

Put large artifacts outside GitHub, typically through the project's Hugging Face
page or a linked benchmark/data release. Expected artifacts include:

- MAGMaR query JSONL and topic-video mapping, or pointers to the MAGMaR release.
- Retrieval subqueries, first-stage runs, RankVideo scores, reranked runs, qrels,
  and retrieval summaries.
- Extracted notes, claims, QA evidence, calibration outputs, packets, and
  generated reports.
- Evaluation summaries and optional representative RLM trajectories.

See [`docs/release_assets.md`](docs/release_assets.md) for the current artifact
checklist.

[Back to top](#marquis)

---

## Tests

```bash
make check
```

Default checks do not download models or require paper-scale data.

[Back to top](#marquis)

---

## Citation

```bibtex
@misc{marquis2026,
  title = {MARQUIS: A Three-Stage Pipeline for Video Retrieval-Augmented Generation},
  author = {Chakraborty, Debashish and Zhang, Dengjia and Jin, Jialiang and Liu, Hanting and Guerrerio, Katherine and Qin, Hanxiang and Skow, Tyler and Martin, Alexander and Kriz, Reno and Van Durme, Benjamin},
  year = {2026},
  eprint = {2605.17640},
  archivePrefix = {arXiv},
  primaryClass = {cs.IR},
  doi = {10.48550/arXiv.2605.17640},
  url = {https://arxiv.org/abs/2605.17640}
}
```

[Back to top](#marquis)

---

## License

Released under the [MIT License](LICENSE).
