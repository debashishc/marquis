# Reproduction

This repository contains code and tiny fixtures only. Configs and scripts
require explicit user-provided paths via environment variables.

Links:

- Code: this repository is the MARQUIS implementation repo
- MAGMaR 2026 shared task: https://github.com/rekriz11/MAGMAR_2026
- Hugging Face paper page: https://huggingface.co/papers/2605.17640
- arXiv: https://arxiv.org/abs/2605.17640
- Expected artifact inventory: `docs/release_assets.md`

## Paper-scale workflow

1. Read the MAGMaR 2026 shared-task repository for official queries,
   topic-video mapping, task definitions, and data download instructions.
2. Download or mount the paper-scale artifacts described in
   `docs/release_assets.md`.
3. Set the required environment variables pointing to your data:

```bash
export MAGMAR_ROOT=/path/to/your/magmar26/data
export MAGMAR_VIDEO_ROOT="$MAGMAR_ROOT"
export MAGMAR_QUERIES_JSONL="$MAGMAR_ROOT/MAGMaR2026_queries.jsonl"
export MARQUIS_REFERENCE="$MAGMAR_ROOT/reference.json"
export MARQUIS_CLAIMS_PATH="$MAGMAR_ROOT/features/claims"
```

4. Run the stage scripts:

```bash
scripts/run_retrieval.sh
scripts/run_extraction.sh
scripts/run_generation.sh
scripts/run_rlm.sh
scripts/run_evaluation.sh
```

Heavy VLM/LLM stages need the matching optional extras and model access.
