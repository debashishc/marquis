# MARQUIS

Three-stage video retrieval-augmented article generation (MAGMaR 2026 / MicroVENT):
**retrieve** videos → **extract** calibrated evidence (notes / claims / QA answers / support
scores) → **generate** cited reports. Plus an RLM variant where a root LM drives MARQUIS
tools from a persistent Python REPL. Paper: arXiv 2605.17640. Repo is a paper-release repo
(code + tiny fixtures only); videos, checkpoints, and runs live outside Git.

## Environment

Install with `uv`:

```bash
make install-dev   # uv sync --extra dev
```

Extras (pyproject): `retrieval`, `eval`, `vlm`, `qa`, `api`, `rlm`, `dev`, `all`.

All data paths are configured via environment variables. Set them before running
any pipeline stage — see `docs/reproduction.md` for the required variables.

## Commands

```bash
make check          # lint + shell -n + pytest + compileall + validate
make quickstart     # no-model smoke: --help on all 5 CLIs + contract validation
make lint / format  # ruff check / ruff format (line-length 100, select E,F,I,UP,B)
python -m pytest -q # tests/ — CPU only, no model downloads, no paper-scale data
```

Entry points (`pyproject.toml` `[project.scripts]` → `marquis.<pkg>.cli:main`):

| Command | Module | Subcommands |
|---|---|---|
| `marquis-retrieve` | `retrieval` | expand, prepare-subqueries, retrieve, rrf, fusion, rerank, qrels, evaluate |
| `marquis-extract` | `information_extraction` | build-query-mapping, prepare-videos, prepare-audio, general-notes, query-claims, predict-unli, calibrate-unli, calibrate, packets, qa-decompose, prepare-transcripts, qa-answer, qa, validate |
| `marquis-generate` | `article_generation` | baseline, bullet {infer\|report\|annotate}, ginger, qa |
| `marquis-rlm` | `rlm_controller` | magmar, magmar-notes |
| `marquis-evaluate` | `evaluation` | retrieval, extraction |

Every subcommand takes **Hydra `key=value` overrides** (`data.*`, `model.*`, `runtime.*`,
`output.*`), composed from `configs/<stage>/`. Group swap: `model=unli_lora`.

Stage wrappers: `scripts/run_{retrieval,extraction,generation,rlm,evaluation}.sh`.
All require `MAGMAR_ROOT` to be set.

## Layout

- `src/marquis/common/` — **`contracts.py` is the single source of truth** for query/reference
  loading, query→topic join, and all artifact schemas. Also `prompts.py` (extraction/scoring
  prompt builders), `model_backends.py` (vLLM `Qwen3_5_VL`, `APIVLM` OpenAI-compatible, UNLI),
  `video.py` (WebDataset tar shard streaming), `validate_contracts.py`.
- `src/marquis/retrieval/` — two *separate* paths: (A) tevatron first-stage → `fusion`/`rrf`/
  `qrels`/`evaluate` (`.trec` runs); (B) `retrieve` = OmniEmbed dense search feeding the QA
  system. Do not conflate them.
- `src/marquis/information_extraction/` — `extract.py` (Step 1 notes/claims),
  `calibrate.py` (Step 1.5 UNLI/Qwen support scoring), `assemble_packets.py`, `qa/` (decompose
  → transcripts → answer/iterative).
- `src/marquis/article_generation/` — `baseline.py` (CAG), `ginger.py`, `qa.py`, `bullet.py`
  (3-step infer→report→annotate with citations).
- `src/marquis/rlm_controller/` — `magmar.py` / `magmar_with_notes.py`, `tool_api.py` (REPL tool
  namespace), `rlm/` (repl, loggers, prompts, vlm backends).
- `configs/<stage>/{data,model,runtime,output,launcher}/` — Hydra tree, the source of truth.
- `docs/` — `paper_map.md`, `prompt_map.md`, `reproduction.md`, `rlm.md`, `slurm.md`.
- `examples/data|quicktest|microvent/` — tiny fixtures. `examples/quicktest/README.md` and
  `README_microvent.md` are the best step-by-step runnable walkthroughs in the repo.
- Each of `retrieval/`, `information_extraction/`, `article_generation/` has a detailed
  module `README.md` — read it before changing that stage.

## Gotchas

- **`reference.json` replaced `topic_video_mapping.json`**. Flow is now
  `reference.json` (topic_id → chunks) + queries (carry `topic_id`) → `marquis-extract
  build-query-mapping` → `query_video_mapping.json` (query_id → video ids), which extraction
  and QA read. `rlm_controller` still genuinely uses a topic mapping.
- **`validate` / `make validate`** needs env vars pointing to sample data. Run it the way CI does:
  `MAGMAR_QUERIES_JSONL=examples/data/MAGMaR2026_queries.jsonl MARQUIS_REFERENCE=examples/data/reference.json MAGMAR_EXPANDED_QUERIES=examples/data/expanded_queries.json python -m marquis.common.validate_contracts`
- **`data.query_ids` defaults to `'1'`** (not null) in IE and generation configs — pass
  `data.query_ids=null` to run all queries.
- **MicroVENT videos** are stored as WebDataset `shard_NNNNNN.tar` shards (ids are `chunk_id`).
  They must be materialized first via `marquis-extract prepare-videos` / `prepare-audio`.
  Validate your dataset paths with `python scripts/check_microvent_starter_pack.py`.
- **Whisper vs vLLM CUDA clash** on EXCLUSIVE_PROCESS GPUs: run `prepare-transcripts` first
  (caches `{video_id: transcript}` to `data.transcripts`, exits, frees the GPU), then `qa`.
- Heavy deps (`torch`, `transformers`, `vllm`, `whisper`, `ir_measures`) are imported
  **lazily** — keep `--help` and config composition working without them.
- `--api URL` / `model.api=` routes bullet/VLM inference to an OpenAI-compatible endpoint
  (`MARQUIS_API_BASE`, `MARQUIS_API_MODEL`, `MARQUIS_API_KEY`) instead of local GPUs.
- The RLM REPL executes model-generated Python — **not a sandbox**. Timeout override:
  `MARQUIS_REPL_TIMEOUT_SECONDS` (default 300).
- `uv.lock` is gitignored on purpose.

## Conventions

- Keep changes inside `src/marquis`; prefer config defaults / env vars / CLI flags over
  hard-coded paths. Don't commit outputs, caches, videos, or checkpoints.
- Preserve documented JSON artifact shapes; changing one needs a migration note + tests.
- Add tests for schema, citation, fusion, config, or CLI surfaces you touch. Default tests
  must never download models or need paper-scale data.
