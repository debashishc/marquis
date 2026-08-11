# RLM controller

The RLM controller implementation lives in `marquis.rlm_controller`.

The RLM path uses `topic_video_mapping.json` (the legacy per-topic video list)
rather than `reference.json`. The rest of the pipeline has migrated to
`reference.json`; the RLM CLI and loader have not yet been updated. Both files
ship in `examples/data/` for the quicktest fixtures.

Set the required environment variables pointing to your data:

```bash
export MAGMAR_ROOT=/path/to/your/magmar26/data
```

Use pre-extracted mode to test the REPL/controller with existing claim
artifacts:

```bash
marquis-rlm magmar-notes \
  --query_id 1 \
  --queries-jsonl "$MAGMAR_ROOT/MAGMaR2026_queries.jsonl" \
  --topic-mapping "$MAGMAR_ROOT/topic_video_mapping.json" \
  --claims-jsonl "$MAGMAR_ROOT/features/claims" \
  --video-root "$MAGMAR_ROOT" \
  --out-dir outputs/rlm
```

Use self-extraction mode when the controller calls a VLM-backed note tool over
videos:

```bash
marquis-rlm magmar \
  --query_id 1 \
  --queries-jsonl "$MAGMAR_ROOT/MAGMaR2026_queries.jsonl" \
  --topic-mapping "$MAGMAR_ROOT/topic_video_mapping.json" \
  --video-root "$MAGMAR_ROOT" \
  --vlm_backend openai_vision \
  --out-dir outputs/rlm
```

The script wrapper defaults to `magmar-notes`:

```bash
scripts/run_rlm.sh
MARQUIS_RLM_COMMAND=magmar scripts/run_rlm.sh
```

The RLM controller executes generated Python actions inside a research REPL. It
is not a security sandbox: generated code can import modules, open files inside
the job environment, and call configured tools. Run it only with trusted model
prompts, data, credentials, and filesystem access. The REPL has a default
execution timeout of 300 seconds; override it with
`MARQUIS_REPL_TIMEOUT_SECONDS` when profiling long tool calls.

`marquis.rlm_controller.tool_api` declares the paper-facing REPL namespace. It
includes the appendix tools
`general_notes(vid)`, `query_claims(vid)`, `video_qa(vid, question)`,
`write_report(facts)`, and the memory operators used by the Root LM. The
`magmar-notes` path accepts either a query-conditioned claims JSONL or the
internal topic-keyed claims directory under `features/claims`.
