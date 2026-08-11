# Retrieval

Query-expansion + rank-fusion retrieval utilities. One CLI (`marquis-retrieve`)
and one Hydra config tree (`configs/retrieval/`).

There are **two distinct retrieval paths** here; they share `expand` /
`prepare-subqueries` but diverge after that and must not be conflated:

- **A. IR / fusion path (tevatron).** First-stage dense search is run by an
  external retriever (tevatron) producing `rank-subqueries.txt`; the `fusion` /
  `rrf` / `qrels` / `evaluate` commands prepare its inputs and fuse/evaluate its
  outputs. This is the ranking-evaluation pipeline.
- **B. QA-system path (OmniEmbed).** The `retrieve` command (`video_retrieval.py`)
  embeds videos + sub-queries with OmniEmbed and writes `retrieved_videos.json`
  — the top videos per sub-query that the QA / generation system answers over.
  It does **not** feed fusion and does **not** touch `rank-subqueries.txt`.

## Layout

```
src/marquis/retrieval/
  cli.py                # subcommand dispatcher
  _common.py            # Hydra glue (build_config, resolve_query_ids)
  prompts.py            # QUERY_EXPANSION_PROMPT
  query_expansion.py    # expand    : query -> sub-queries (LLM)
  prepare_subqueries.py # prepare-subqueries : flatten expansion -> JSONL + mapping
  fusion.py             # fusion    : standard suite of .trec runs + fusion primitives
  rrf.py                # rrf       : single-method fusion -> submission JSON
  qrels.py              # qrels     : topic mapping -> TREC qrels
  evaluate.py           # evaluate  : ir-measures over run files
  video_retrieval.py    # retrieve  : OmniEmbed video search for QA (GPU)
```

## Pipeline / commands

Shared front end:

| Step | Command | Module | In → Out |
|---|---|---|---|
| 1 | `expand` | `query_expansion.py` | queries → `expanded_queries.json` (GPU/LLM) |
| 2 | `prepare-subqueries` | `prepare_subqueries.py` | `expanded_queries.json` → `subqueries.jsonl` + `subquery_mapping.json` |

Path A — IR / fusion (tevatron embeddings):

| Step | Command | Module | In → Out |
|---|---|---|---|
| 3 | *(external)* | tevatron | `subqueries.jsonl` → `rank-subqueries.txt` |
| 4a | `rrf` | `rrf.py` | sub-query runs → `rank-expanded-<method>.json` (+ baseline comparison) |
| 4b | `fusion` | `fusion.py` | sub-query runs → 7 `rank-expansion-*.trec` files |
| 5 | `qrels` | `qrels.py` | `reference.json` (joined to queries by `topic_id`) → `qrels.txt` |
| 6 | `evaluate` | `evaluate.py` | `*.trec` + `qrels.txt` → metric tables (+ CSV) |

Path B — QA system (OmniEmbed embeddings):

| Step | Command | Module | In → Out |
|---|---|---|---|
| 3' | `retrieve` | `video_retrieval.py` | `subqueries.jsonl` + `video_links.json` → `retrieved_videos.json` (GPU) |
| 4' | *(external generator)* | — | `retrieved_videos.json` → answers |

## QA retriever (`video_retrieval.py`, path B)

The OmniEmbed video retriever for the QA system — **separate** from tevatron /
fusion. It transcribes each video with Whisper, embeds videos and sub-queries
with OmniEmbed (`Tevatron/OmniEmbed-v0.1` on `Qwen/Qwen2.5-Omni-7B`), ranks
videos by cosine similarity, and writes the top videos per sub-query. It does
retrieval only (no answer generation). Run it as a command:

```bash
python -m marquis.retrieval.cli retrieve            # GPU; reads subqueries.jsonl + video_links.json
python -m marquis.retrieval.cli retrieve runtime.retrieve.top_k=4 runtime.retrieve.download=true
```

The same functions are also importable as an engine (e.g. for an in-process QA
pipeline that needs the transcripts the index already holds):

```python
from marquis.retrieval.video_retrieval import load_retrieval_models, build_index, retrieve

processor, embed_model, whisper_model = load_retrieval_models()
index = build_index(video_links, processor, embed_model, whisper_model)  # GPU
hits = retrieve(subquery, index, processor, embed_model, top_k=4)        # list of index entries
```

`torch` / `transformers` / `whisper` / `yt_dlp` are imported lazily, so the
module itself stays cheap to import. `build_index` expects pre-downloaded,
downsampled assets by default; pass `download=True` / `downsample=True` (or the
`runtime.retrieve.{download,downsample}` config flags) to fetch and preprocess
on the fly.

## Usage

```bash
export PYTHONPATH="$PWD/src"

python -m marquis.retrieval.cli expand                       # GPU
python -m marquis.retrieval.cli prepare-subqueries

# Path A (tevatron → fusion → evaluate)
python -m marquis.retrieval.cli fusion runtime.fusion_depth=100
python -m marquis.retrieval.cli rrf    runtime.rrf_method=weighted-rrf runtime.rrf_k=10
python -m marquis.retrieval.cli qrels
python -m marquis.retrieval.cli evaluate data.qrels=QRELS runtime.eval.run_dir=DIR runtime.eval.output_csv=results.csv

# Path B (OmniEmbed retrieval for the QA system)
python -m marquis.retrieval.cli retrieve                     # GPU

# everything reads/writes under data.base_dir; point it elsewhere for a scratch run:
python -m marquis.retrieval.cli fusion data.base_dir=/tmp/run
```

## Configuration (`configs/retrieval/`)

| Group | Key fields (defaults) |
|---|---|
| `data/default` | `base_dir` (`MARQUIS_RETRIEVAL_BASE_DIR`), `queries_file`, and artifact paths interpolated from `base_dir`. Path A: `expanded_queries`, `subqueries_jsonl`, `subquery_mapping`, `subquery_results`, `baseline_results`, `reference`, `qrels`. Path B: `video_links`, `video_dir` (`VIDEO_DIR`), `audio_dir` (`AUDIO_DIR`), `retrieved_videos`. `query_ids: null` |
| `model/default` | `model` (`MARQUIS_RETRIEVAL_MODEL`, default `Qwen/Qwen3.5-2B`), `cache_dir` (`HF_HOME`); `embedder.{model,adapter,whisper,cache_dir}` for the `retrieve` command (path B) |
| `runtime/default` | `rrf_k: 60`, `fusion_depth: 100`, `rrf_method: rrf`, `expansion.{max_new_tokens,temperature,top_p}`, `retrieve.{top_k,sim_threshold,download,downsample}`, `eval.{run_dir,run_file,output_csv}` |
| `output/default` | `out_dir` (`MARQUIS_RETRIEVAL_OUTPUT_DIR`, default = `data.base_dir`) |
| `launcher/local` | `kind: local` |

All artifact paths derive from `data.base_dir` via interpolation, so a single
`data.base_dir=...` override redirects every input and output. `rrf_method` is
one of `rrf | weighted-rrf | sum | max | mean`.

### Environment variables

| Variable | Used for |
|---|---|
| `MARQUIS_RETRIEVAL_BASE_DIR` | working dir for rank lists / mappings / qrels |
| `MARQUIS_RETRIEVAL_QUERIES_FILE` / `MAGMAR_QUERIES_JSONL` | query set |
| `MARQUIS_RETRIEVAL_MODEL` | query-expansion LLM |
| `MARQUIS_RETRIEVAL_OUTPUT_DIR` | output dir (defaults to base_dir) |
| `VIDEO_DIR` / `AUDIO_DIR` | video / audio asset dirs for the `retrieve` command |
| `HF_HOME` | HuggingFace cache dir |

## Notes

- Steps 2, 4a, 4b, 5, 6 run on CPU; steps 1 (`expand`) and 3 (`retrieve`) need a GPU.
- `torch`/`transformers`/`whisper`/`ir_measures` are imported lazily, so `--help` and
  config composition don't load them.
