# Quick-test fixtures

Tiny inputs (one query — *2025 Canadian Federal Election*) for smoke-testing the
**retrieval**, **information_extraction**, and **article_generation** branches
without the full dataset. Point the relevant `data.*` config keys at these files
via Hydra overrides.

> **MicroVENT?** The fixtures below are MAGMaR-shaped (loose `.mp4` under
> `$MAGMAR_VIDEO_ROOT`). MicroVENT ships videos inside `videos/shard_NNNNNN.tar`
> WebDataset shards instead, with `microvent_`-prefixed fixtures of its own — see
> **[README_microvent.md](README_microvent.md)** for the full information-extraction
> and generation walkthrough on MicroVENT.

## Files

| File | What | Source |
|---|---|---|
| `queries.jsonl` | 1 query (id `1`, carries `topic_id`) | subset of the real query set |
| `reference.json` | 1 topic → 6 chunk ids (`topics[].chunks`) | from the repo sample |
| `query_video_mapping.json` | 1 query id → 6 video ids (IE video selection) | built from `reference.json` (joined by `topic_id`) |
| `expanded_queries.json` | expanded sub-queries for query 1 | subset of real expansion |
| `MAGMaR2026_queries_with_subqueries.jsonl` | query 1 + its 25 sub-queries | subset of real data |
| `qa_outputs/answers_iter_q.json` | 8 QA answers for query 1 (`iter_q`) | subset of real QA output |
| `video_links.json` | 6 YouTube URLs for query 1's topic | derived from the query→video mapping |
| `query_conditioned_claims.jsonl` | 8 claims for query 1 | **synthetic sample** ⚠️ |
| `general_notes.jsonl` | 6 general notes for query 1's topic | **synthetic sample** ⚠️ |
| `rank-subqueries.txt` | per-sub-query video rank lists (tevatron 3-col TSV) | **synthetic sample** ⚠️ |


**MicroVENT fixtures** (see [README_microvent.md](README_microvent.md)):

| File | What | Source |
|---|---|---|
| `microvent_prepare_videos.py` | preprocess: materialize a query's videos from the tar shards | — |
| `microvent_queries.jsonl` | 1 MicroVENT query (id `1`) | subset of MicroVENT annotations |
| `microvent_query_video_mapping.json` | query 1 → its 6 relevant chunk ids | from MicroVENT `annotations/reference.json` (the topic's `chunks`, joined by `topic_id`) |
| `microvent_expanded_queries.json` | expanded sub-queries for query 1 | **synthetic sample** ⚠️ |
| `microvent_query_conditioned_claims.jsonl` | 6 claims for query 1 (1 per chunk) | **synthetic sample** ⚠️ |
| `microvent_general_notes.jsonl` | 6 general notes for query 1 (1 per chunk) | **synthetic sample** ⚠️ |

⚠️ `query_conditioned_claims.jsonl`, `general_notes.jsonl`, and
`rank-subqueries.txt` are hand-written / generated sample content (no real video
was processed to make them) — fine for exercising the data flow, not for
measuring quality.

## What runs where

| Tier | Commands | Needs |
|---|---|---|
| CPU only | retrieval `prepare-subqueries`, `rrf`, `fusion`, `qrels`, `evaluate`; IE `packets`, `validate` | nothing — runs on the fixtures as-is |
| GPU, no video | retrieval `expand`; IE `calibrate-unli`, `qa-decompose`; generation `baseline` / `ginger` / `qa` / `bullet` | a small model (e.g. `Qwen/Qwen3.5-2B`); inputs are tiny |
| GPU **and** real video | retrieval `retrieve`; IE `general-notes`, `query-claims`, `predict-unli`, `calibrate`, `qa-answer`, `qa` | the `.mp4` files for query 1's topic under `MAGMAR_VIDEO_ROOT` (these fixtures can't shrink video) |

```bash
export PYTHONPATH="$PWD/src"
    QT=examples/quicktest
```

---

## retrieval (`marquis-retrieve`)

Two distinct paths share the early steps:

- **Fusion / eval path** (CPU): `prepare-subqueries` → external dense search →
  `fusion` / `rrf` → `evaluate` (against `qrels`). The external dense search
  (tevatron, GPU) is *not* a `marquis` command and can't shrink to a fixture, so
  `rank-subqueries.txt` stands in for its output and lets the CPU steps run.
- **QA-system path** (GPU + video): `prepare-subqueries` → `retrieve`
  (OmniEmbed) → `retrieved_videos.json`, consumed by the QA generator. Separate
  artifact, separate from the fusion pipeline.

Most `data.*` keys default to `${data.base_dir}/<file>`, so the blocks below
override inputs to `$QT/...` and write everything under one out dir:

```bash
QTR=outputs/qt_retrieval
```

### CPU path — runs on the fixtures as-is

```bash
# prepare-subqueries: flatten expanded_queries.json -> subqueries.jsonl + subquery_mapping.json
python -m marquis.retrieval.cli prepare-subqueries \
    data.expanded_queries=$QT/expanded_queries.json \
    data.subqueries_jsonl=$QTR/subqueries.jsonl \
    data.subquery_mapping=$QTR/subquery_mapping.json

# fusion: sub-query rank lists -> the standard suite of .trec runs in out_dir
#   data.subquery_results is the synthetic rank-subqueries.txt (stands in for the
#   external tevatron dense search); data.subquery_mapping is from the step above.
python -m marquis.retrieval.cli fusion \
    data.subquery_results=$QT/rank-subqueries.txt \
    data.subquery_mapping=$QTR/subquery_mapping.json \
    output.out_dir=$QTR

# rrf: single-method fusion -> rank-expanded-<method>.json (method via runtime.rrf_method)
python -m marquis.retrieval.cli rrf runtime.rrf_method=rrf \
    data.subquery_results=$QT/rank-subqueries.txt \
    data.subquery_mapping=$QTR/subquery_mapping.json \
    output.out_dir=$QTR

# qrels: reference.json + queries.jsonl -> TREC qrels.txt
#   (joins each query to its reference topic by topic_id)
python -m marquis.retrieval.cli qrels \
    data.reference=$QT/reference.json \
    data.queries_file=$QT/queries.jsonl \
    data.qrels=$QTR/qrels.txt

# evaluate: score every *.trec in run_dir against the qrels with ir-measures
python -m marquis.retrieval.cli evaluate \
    data.qrels=$QTR/qrels.txt runtime.eval.run_dir=$QTR
```

### expand (GPU, no video)

```bash
# query -> sub-queries via the expansion LLM -> expanded_queries.json
# (the fixture expanded_queries.json above is a saved copy of this output)
python -m marquis.retrieval.cli expand \
    data.queries_file=$QT/queries.jsonl \
    data.expanded_queries=$QTR/expanded_queries.json \
    model.model=Qwen/Qwen3.5-2B
```

### retrieve (GPU + real video)

```bash
# OmniEmbed dense retrieval over the topic's videos -> retrieved_videos.json
#   video_links.json lists the YouTube URLs; the .mp4/.wav must exist under
#   VIDEO_DIR/AUDIO_DIR (or set runtime.retrieve.download=true to fetch them).
python -m marquis.retrieval.cli retrieve runtime.retrieve.top_k=4 \
    data.subqueries_jsonl=$QTR/subqueries.jsonl \
    data.video_links=$QT/video_links.json \
    data.retrieved_videos=$QTR/retrieved_videos.json
```

---

## information_extraction (`marquis-extract`)

Pipeline order: **extract → score/calibrate → packets → QA → validate**. Outputs
chain via `out_dir`; each block notes its inputs and what it writes.

**Filters & mappings — why some commands carry an arg and others don't:**

- **query→video mapping** (`query_video_mapping.json`, a `{query_id: [video_id, ...]}`
  table) is the video-selection source: each query reads its own video list
  (biased/unbiased queries on one event share that event's videos). Passed via
  `data.query_video_mapping` to `general-notes` / `query-claims` and `qa-answer` /
  `qa`.
- **topic label** comes straight from each query's `topic_id` — no separate
  topic-mapping file. `general-notes` / `query-claims` / `packets` tag their
  records with it. Downstream commands (`predict-unli` / `calibrate`, generation)
  read `video_id` straight off each record, so they need no mapping;
  `calibrate-unli`, `qa-decompose` and `validate` don't touch video at all.
- **`data.query_ids`** restricts which queries/records are processed. It defaults to
  `'1'` in the config, so on these fixtures (query 1 only) it is effectively a no-op,
  but it is set explicitly below where the command honours it. Exceptions:
  `qa-decompose` ignores it (processes every query in `queries_jsonl`) and `validate`
  checks all queries in the mapping.

### Step 1 — extraction (GPU + real video)

```bash
# general notes per video (query-agnostic) -> outputs/qt_general_notes/general_notes.jsonl
python -m marquis.information_extraction.cli general-notes \
    data.queries_jsonl=$QT/queries.jsonl \
    data.query_video_mapping=$QT/query_video_mapping.json \
    data.query_ids='1' model.model=Qwen/Qwen3.5-2B \
    output.out_dir=outputs/qt_general_notes

# query-conditioned claims, single-query mode -> outputs/qt_query_claims_single/query_conditioned_claims.jsonl
python -m marquis.information_extraction.cli query-claims runtime.query_mode=single \
    data.queries_jsonl=$QT/queries.jsonl \
    data.query_video_mapping=$QT/query_video_mapping.json \
    data.query_ids='1' model.model=Qwen/Qwen3.5-2B \
    output.out_dir=outputs/qt_query_claims_single

# query-conditioned claims, expanded-query mode (uses expanded_queries.json)
python -m marquis.information_extraction.cli query-claims runtime.query_mode=expanded \
    data.queries_jsonl=$QT/queries.jsonl \
    data.query_video_mapping=$QT/query_video_mapping.json \
    data.expanded_queries=$QT/expanded_queries.json \
    data.query_ids='1' model.model=Qwen/Qwen3.5-2B \
    output.out_dir=outputs/qt_query_claims_expanded
```

### Step 1.5 — scoring / calibration

```bash
# predict (GPU + video): score claims against their source video
#   -> outputs/qt_unli_query_claims/unli_predictions.jsonl
python -m marquis.information_extraction.cli predict-unli model=unli_lora \
    runtime.calibrate.artifact_type=query-claims \
    data.artifacts_jsonl=$QT/query_conditioned_claims.jsonl \
    data.query_ids='1' data.video_root=$MAGMAR_VIDEO_ROOT \
    output.out_dir=outputs/qt_unli_query_claims

# calibrate (GPU, no video): merge support probs back into the artifacts
#   -> outputs/qt_unli_query_claims/query_conditioned_claims_calibrated.jsonl
python -m marquis.information_extraction.cli calibrate-unli \
    runtime.calibrate.artifact_type=query-claims \
    data.artifacts_jsonl=$QT/query_conditioned_claims.jsonl \
    data.unli_jsonl=outputs/qt_unli_query_claims/unli_predictions.jsonl \
    output.out_dir=outputs/qt_unli_query_claims

# calibrate (GPU + video): predict + calibrate in one go (stage=all)
python -m marquis.information_extraction.cli calibrate model=unli_lora \
    runtime.calibrate.artifact_type=query-claims \
    data.artifacts_jsonl=$QT/query_conditioned_claims.jsonl \
    data.query_ids='1' data.video_root=$MAGMAR_VIDEO_ROOT \
    output.out_dir=outputs/qt_unli_query_claims
```

### packets (CPU only)

```bash
# query-based packets from the sample claims -> outputs/qt_packets_query
python -m marquis.information_extraction.cli packets runtime.packets.stream=query-based \
    data.claims=$QT/query_conditioned_claims.jsonl \
    data.queries_jsonl=$QT/queries.jsonl \
    data.query_ids='1' output.out_dir=outputs/qt_packets_query

# general-note packets from the sample notes -> outputs/qt_packets_note
python -m marquis.information_extraction.cli packets runtime.packets.stream=general-note \
    data.notes=$QT/general_notes.jsonl \
    data.queries_jsonl=$QT/queries.jsonl \
    data.query_ids='1' output.out_dir=outputs/qt_packets_note
```

### QA

```bash
# qa-decompose (GPU, no video): query -> sub-questions -> outputs/qt_qa/subqueries.jsonl
# NOTE: qa-decompose ignores data.query_ids — it decomposes EVERY query in queries_jsonl
#       (here that is just query 1).
python -m marquis.information_extraction.cli qa-decompose \
    data.queries_jsonl=$QT/queries.jsonl \
    data.subqueries_output=outputs/qt_qa/subqueries.jsonl

# qa-answer (GPU + video): single-shot QA over the query's videos -> outputs/qt_qa/answers.jsonl
python -m marquis.information_extraction.cli qa-answer \
    data.qa_queries=$QT/MAGMaR2026_queries_with_subqueries.jsonl \
    data.query_video_mapping=$QT/query_video_mapping.json data.query_ids='1' \
    data.qa_output=outputs/qt_qa/answers.jsonl

# qa (GPU + video): iterative QA with follow-ups -> outputs/qt_qa/answers_iter.jsonl
python -m marquis.information_extraction.cli qa \
    data.qa_queries=$QT/MAGMaR2026_queries_with_subqueries.jsonl \
    data.query_video_mapping=$QT/query_video_mapping.json data.query_ids='1' \
    runtime.qa.max_steps=3 data.qa_output=outputs/qt_qa/answers_iter.jsonl
```


---

## article_generation (`marquis-generate`)

All generation commands are GPU (small model, no video). They consume the
fixtures (or packets produced by the IE block above) and write reports.

### baseline / ginger (from the sample claims)

```bash
python -m marquis.article_generation.cli baseline \
    data.claims_path=$QT/query_conditioned_claims.jsonl \
    model.model=Qwen/Qwen3.5-2B output.out_dir=outputs/qt_baseline

python -m marquis.article_generation.cli ginger \
    data.claims_path=$QT/query_conditioned_claims.jsonl \
    model.model=Qwen/Qwen3.5-2B output.out_dir=outputs/qt_ginger
```

### qa report (from the sample QA answers)

```bash
python -m marquis.article_generation.cli qa \
    data.qa_dir=$QT data.qa_file=iter_q \
    data.queries_with_subqueries=$QT/MAGMaR2026_queries_with_subqueries.jsonl \
    model.model=Qwen/Qwen3.5-2B output.out_dir=outputs/qt_qa_report
```

### bullet (query-based): two steps over the packets produced above

```bash
# Step 1 — infer: VLM draws higher-level inferences; writes inferences.jsonl to out_dir.
python -m marquis.article_generation.cli bullet infer runtime.bullet.stream=query-based \
    runtime.bullet.packets_dir=outputs/qt_packets_query \
    runtime.bullet.claims=$QT/query_conditioned_claims.jsonl \
    runtime.bullet.queries_jsonl=$QT/queries.jsonl \
    model.vlm=Qwen/Qwen3.5-2B output.out_dir=outputs/qt_bullet_query

# Step 2 — annotate: build cited reports from the infer output and inline the citations.
# NOTE: runtime.bullet.inferences is the *directory* holding inferences.jsonl, NOT the file.
#       It must match step 1's out_dir, and stream must be set, or annotate falls back to
#       claim-only sections with no inline citations.
python -m marquis.article_generation.cli bullet annotate runtime.bullet.stream=query-based \
    runtime.bullet.inferences=outputs/qt_bullet_query \
    runtime.bullet.packets_dir=outputs/qt_packets_query \
    runtime.bullet.claims=$QT/query_conditioned_claims.jsonl \
    runtime.bullet.queries_jsonl=$QT/queries.jsonl \
    model.vlm=Qwen/Qwen3.5-2B output.out_dir=outputs/qt_bullet_query
```

### bullet (general-note): same two steps, fed by the general-note packets

```bash
python -m marquis.article_generation.cli bullet infer runtime.bullet.stream=general-note \
    runtime.bullet.packets_dir=outputs/qt_packets_note \
    runtime.bullet.notes=$QT/general_notes.jsonl \
    runtime.bullet.queries_jsonl=$QT/queries.jsonl \
    model.vlm=Qwen/Qwen3.5-2B output.out_dir=outputs/qt_bullet_note

python -m marquis.article_generation.cli bullet annotate runtime.bullet.stream=general-note \
    runtime.bullet.inferences=outputs/qt_bullet_note \
    runtime.bullet.packets_dir=outputs/qt_packets_note \
    runtime.bullet.notes=$QT/general_notes.jsonl \
    runtime.bullet.queries_jsonl=$QT/queries.jsonl \
    model.vlm=Qwen/Qwen3.5-2B output.out_dir=outputs/qt_bullet_note
```

---

## MicroVENT quicktest

MicroVENT uses the same fixture layout but ships videos as
`videos/shard_NNNNNN.tar` WebDataset shards, so it adds one **preprocess** step
that materializes a query's videos into a `video_root` before the standard
`marquis-extract` / `marquis-generate` commands run. The convenience wrapper
`microvent_prepare_videos.py` does this for the quicktest; for full-scale runs
the same logic is packaged as `marquis-extract prepare-videos` (it reads
`data.video_shards_root` / `data.query_video_mapping` / `data.query_ids` and
streams each video shard-by-shard into `data.video_root`). The full walkthrough —
information extraction and generation — lives in
**[README_microvent.md](README_microvent.md)**.

To target a different query, regenerate the two fixtures from the annotations
(`relevance==1` chunks) — see the snippet in the script's module docstring.
