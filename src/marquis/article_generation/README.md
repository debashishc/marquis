# Article Generation

Synthesize per-query evidence (claims / QA answers / note-taking packets) into
cited reports. Four strategies share one CLI (`marquis-generate`) and one Hydra
config tree (`configs/article_generation/`).

## Layout

```
src/marquis/article_generation/
  cli.py        # subcommand dispatcher (baseline | bullet | ginger | qa)
  _common.py    # Hydra glue + shared loaders (claims, model, llm_generate, json parse)
  prompts.py    # all prompt templates + bullet's note-taking prompt builders
  baseline.py   # one-shot: claims -> cited report
  ginger.py     # cluster -> rank -> summarize -> fluency over claims
  qa.py         # QA-system answers -> report (baseline and/or ginger method)
  bullet.py     # note-taking infer -> report -> annotate pipeline (VLM)
```

## Commands

| Command | Module | What it does |
|---|---|---|
| `baseline` | `baseline.py` | Synthesize a query's claims into one fluent, cited report. |
| `ginger` | `ginger.py` | Facet-cluster the claims, rank clusters, summarize top-N, polish. |
| `qa` | `qa.py` | Turn a QA system's answer dump into a report (`baseline`/`ginger`/`both`). |
| `bullet` | `bullet.py` | Three sub-steps (`infer` →  `annotate`) over retrieval packets, using a VLM. Structurally distinct from the single-shot strategies. |

## Usage

```bash
export PYTHONPATH="$PWD/src"

# baseline (defaults from the YAML tree)
python -m marquis.article_generation.cli baseline

# override on the CLI with Hydra syntax
python -m marquis.article_generation.cli baseline data.query_ids=3 runtime.max_claims_per_query=10 model.model=Qwen/Qwen3.5-2B
python -m marquis.article_generation.cli ginger  data.query_ids='3,9,16' runtime.top_n_clusters=3
python -m marquis.article_generation.cli qa      data.qa_file=ss runtime.qa_method=baseline

# bullet: positional sub-step, then Hydra overrides
python -m marquis.article_generation.cli bullet infer    runtime.bullet.stream=query-based runtime.bullet.packets_dir=outputs/packets runtime.bullet.claims=outputs/unli_query_claims/query_conditioned_claims_calibrated.jsonl output.out_dir=outputs/reports

python -m marquis.article_generation.cli bullet annotate \
  runtime.bullet.stream=query-based \
  runtime.bullet.inferences=outputs/reports \
  runtime.bullet.packets_dir=outputs/packets \
  runtime.bullet.claims=outputs/unli_query_claims/query_conditioned_claims_calibrated.jsonl \
  output.out_dir=outputs/reports
```

`query_ids` accepts a scalar (`3`), a quoted comma list (`'3,9'`), a list
(`[3,9]`), or `null` for all queries.

## Configuration (`configs/article_generation/`)

The YAML tree is the single source of truth; every value is overridable on the
CLI and most mirror an environment variable via `${oc.env:VAR,default}`.

| Group | Key fields (defaults) |
|---|---|
| `data/default` | `claims_path`, `query_ids: null`, `qa_dir`, `qa_file: iter_q`, `queries_with_subqueries` |
| `model/default` | `model` (`MARQUIS_GENERATION_MODEL`, default `Qwen/Qwen3.5-27B`), `cache_dir` (`HF_HOME`), `vlm` (`MARQUIS_VLM_MODEL`, default `Qwen/Qwen3.5-2B`, used by `bullet`) |
| `runtime/default` | `max_claims_per_query: 25`, `top_n_clusters: 5`, `temperature: 0.7`, `top_p: 0.9`, `max_qa_per_query: 20`, `qa_method: both`, `ginger.*` (per-stage tokens/temps), `bullet.*` (stream, packet/claims paths, per-step gen knobs) |
| `output/default` | `out_dir` (`MARQUIS_REPORT_OUTPUT_DIR`, default `outputs/reports`) |
| `launcher/local` | `kind: local` |

`bullet` reads its model from `model.vlm`, output dir from `output.out_dir`, and
query subset from `data.query_ids`; everything else bullet-specific lives under
`runtime.bullet`.

### Common environment variables

| Variable | Used for |
|---|---|
| `MARQUIS_CLAIMS_PATH` | claims JSONL (baseline / ginger) |
| `MARQUIS_GENERATION_MODEL` | report-writing causal LM |
| `MARQUIS_VLM_MODEL` | bullet VLM |
| `MARQUIS_QA_DIR` | QA answer dumps (`qa`) |
| `MARQUIS_REPORT_OUTPUT_DIR` | output directory |
| `HF_HOME` | HuggingFace cache dir |
| `MARQUIS_CONFIG_DIR` | override the config dir (advanced) |

## Outputs

- `baseline` → `reports_baseline.json` + `report_q{qid}_{topic}.txt`
- `ginger` → `reports_ginger.json` + `report_ginger_q{qid}_{topic}.txt`
- `qa` → `reports_qa_{qa_file}_baseline.json` / `..._ginger.json`
- `bullet` → per-query `report_{qid}.json` / `.md` + `all_reports.json` + run manifest

## Notes

- Heavy deps (`torch`, `transformers`) are imported lazily inside the worker
  functions, so `--help` and config composition stay fast.
- Model runs require a GPU and a real claims, QA, or packet input.
