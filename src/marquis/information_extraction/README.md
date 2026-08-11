# Information Extraction

Step-1 / Step-1.5 of the note-taking pipeline plus the video-QA stage: extract
evidence from raw video, score/calibrate it, assemble per-query packets, and run
question answering. One CLI (`marquis-extract`) and one Hydra config tree
(`configs/information_extraction/`).

## Layout

```
src/marquis/information_extraction/
  cli.py                 # subcommand dispatcher
  _common.py             # Hydra glue (build_config, query_id_filter)
  prompts.py             # QA prompt templates (extraction/scoring use marquis.common.prompts builders)
  extract.py             # general-notes | query-claims (VLM, Step 1)
  calibrate.py           # predict-unli | calibrate-unli | calibrate (Step 1.5)
  assemble_packets.py    # packets
  validate_contracts.py  # validate (schema/contract checks)
  qa/
    query_helper_functions.py  # shared: model loading, transcription, VLM QA
    query_direct_QA.py         # qa-decompose : query -> sub-questions
    query_answering.py         # qa-answer    : single-shot QA over a topic's videos
    query_iterative_questions.py # qa         : iterative QA with follow-ups
```

## Commands

| Command | Module | What it does |
|---|---|---|
| `general-notes` | `extract.py` | Step 1a: VLM general notes per video (query-agnostic). |
| `query-claims` | `extract.py` | Step 1b: query-conditioned claims (`runtime.query_mode=single\|expanded`). |
| `predict-unli` | `calibrate.py` | Step 1.5 predict: score artifacts against source video. |
| `calibrate-unli` | `calibrate.py` | Step 1.5 calibrate: merge support probs back into artifacts. |
| `calibrate` | `calibrate.py` | predict → calibrate in one go (`stage=all`). |
| `packets` | `assemble_packets.py` | Assemble per-query packets (`general-note`/`query-based` stream). |
| `qa-decompose` | `qa/query_direct_QA.py` | Rewrite each query into 10-25 sub-questions. |
| `qa-answer` | `qa/query_answering.py` | Single-shot QA over each (sub-)query's videos. |
| `qa` | `qa/query_iterative_questions.py` | Iterative QA: answer + follow-up questions. |
| `validate` | `validate_contracts.py` | Run the PR1 contract/schema checks. |

The `predict-unli` / `calibrate-unli` commands are thin shims that set
`runtime.calibrate.stage` to `predict` / `calibrate`.

**QA video selection.** `qa-answer` / `qa` pick which videos to answer over via
`runtime.qa.retrieval.mode`:

- `auto` (default) — per query, use `data.topic_mapping` when it covers the
  topic; otherwise dense-retrieve videos. No mapping file at all ⇒ everything is
  retrieved.
- `topic` — always use the mapping (errors if it's missing).
- `retrieval` — always dense-retrieve.

Retrieval uses **precomputed video embeddings** loaded from
`runtime.qa.retrieval.video_embeddings` (`.pt` / `.npz` / `.jsonl` / `.json`,
keyed by video id); only the query is embedded at runtime (OmniEmbed query
encoder reused from `marquis.retrieval.video_retrieval`). Transcripts are
generated lazily for the retrieved videos. Single-shot QA retrieves once per
sub-query; **iterative QA re-retrieves on every follow-up step**.

## Usage

```bash
export PYTHONPATH="$PWD/src"

# Step 1: extraction (VLM, GPU)
python -m marquis.information_extraction.cli general-notes output.out_dir=outputs/general_notes
python -m marquis.information_extraction.cli query-claims  runtime.query_mode=single output.out_dir=outputs/query_claims_expanded data.query

# Step 1.5: scoring + calibration (select the scorer via the model group)
python -m marquis.information_extraction.cli calibrate model=unli_lora \
    runtime.calibrate.artifact_type=query-claims \
    data.artifacts_jsonl=outputs/query_claims_single/query_conditioned_claims.jsonl \
    output.out_dir=outputs/unli_query_claims

# packets
python -m marquis.information_extraction.cli packets runtime.packets.stream=query-based \
    data.claims=outputs/unli_query_claims/query_conditioned_claims_calibrated.jsonl output.out_dir=outputs/packets

# QA
python -m marquis.information_extraction.cli qa-decompose data.subqueries_output=subqueries.jsonl
python -m marquis.information_extraction.cli qa-answer data.qa_output=answers.jsonl
python -m marquis.information_extraction.cli qa        data.qa_queries=subqueries.jsonl runtime.qa.max_steps=5

# QA without a topic mapping: force dense retrieval over precomputed video embeddings
python -m marquis.information_extraction.cli qa runtime.qa.retrieval.mode=retrieval \
    runtime.qa.retrieval.video_embeddings=examples/data/video_embeddings.pt

# contracts
python -m marquis.information_extraction.cli validate
```

## Configuration (`configs/information_extraction/`)

| Group | Key fields (defaults) |
|---|---|
| `data/default` | `video_root`, `queries_jsonl`, `expanded_queries`, `reference`, `query_video_mapping`, `artifacts_jsonl`, `notes`, `claims`, `unli_jsonl`, `qa_queries`, `qa_output`, `subqueries_output`, `query_ids: null` |
| `model/qwen3_5_vl` *(default)* | `backend: qwen_vl`, `model` (`MARQUIS_VLM_MODEL`), `download_dir` |
| `model/unli_lora` | `backend: unli`, `model`, `base_model`, `lora_path` — select with `model=unli_lora` |
| `model/qwen_score_9b` | `backend: qwen_score`, `model` — select with `model=qwen_score_9b` |
| `runtime/default` | VLM decoding knobs (`fps`, `max_frames`, `temperature`, `top_p`, `top_k`, `max_tokens: null`, `seed: null`); `include_topic_in_prompt`, `query_mode: single`; `calibrate.*` (stage / artifact_type / scorer_backend / scorer knobs); `packets.*` (stream / top_k / unli_threshold); `qa.*` (qa_model / whisper_model / max_new_tokens / max_steps; `qa.retrieval.*`: mode / video_embeddings / embed_model / embed_adapter / top_k / sim_threshold) |
| `output/default` | `out_dir: null` (→ per-command/per-mode default) |
| `launcher/local` | `kind: local` |

Notes on resolution:

- `runtime.max_tokens`, `runtime.seed` and `output.out_dir` are `null` so the
  historical per-mode defaults (`extract.MODE_DEFAULTS`) fill them in; set any of
  them explicitly to override.
- `calibrate` picks its scorer from the selected **model group** (`model=unli_lora`
  → UNLI, `model=qwen_score_9b` → Qwen scorer); with the default VLM group it
  falls back to `runtime.calibrate.scorer_backend`.

### Environment variables

| Variable | Used for |
|---|---|
| `MAGMAR_VIDEO_ROOT` | raw video root |
| `MAGMAR_QUERIES_JSONL` | query set |
| `MARQUIS_VLM_MODEL` | extraction VLM |
| `MARQUIS_UNLI_MODEL` / `MARQUIS_UNLI_BASE` / `MARQUIS_UNLI_LORA_PATH` | UNLI scorer |
| `MARQUIS_QA_MODEL` / `MARQUIS_QA_WHISPER_MODEL` | QA text LLM + Whisper |
| `MARQUIS_REFERENCE` | reference.json (topic_id → relevant videos) |
| `HF_HOME` | HuggingFace cache dir |

## Notes

- All extraction/scoring/QA commands need a GPU; the QA stage also needs the
  video files and `whisper`. `validate` runs on CPU.
- Heavy deps are imported lazily; `--help`, config composition and input
  validation all run without loading `torch`.
- Step-1 extraction and Step-1.5 scoring reuse the prompt *builders* in
  `marquis.common.prompts`; only the QA string templates live in `prompts.py`.
