# reference.json migration

`reference.json` replaced `topic_video_mapping.json` as the canonical
topic→video mapping.

## New flow

```
reference.json (topic_id → chunks)
  + queries (carry topic_id)
  → marquis-extract build-query-mapping
  → query_video_mapping.json (query_id → video ids)
  → extraction / QA read this
```

## Stale references

- `scripts/run_extraction.sh` still passes `data.mapping=` pointing to the
  removed file. Needs updating.
- `rlm_controller` still genuinely uses a topic mapping via
  `MARQUIS_TOPIC_MAPPING` / `MAGMAR_TOPIC_MAPPING` env vars. This is
  intentional, not stale.

## Validation

```bash
MAGMAR_QUERIES_JSONL=examples/data/MAGMaR2026_queries.jsonl \
MARQUIS_REFERENCE=examples/data/reference.json \
MAGMAR_EXPANDED_QUERIES=examples/data/expanded_queries.json \
python -m marquis.common.validate_contracts
```

`make validate` uses these same env vars via the Makefile.
