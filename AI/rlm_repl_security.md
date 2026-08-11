# RLM REPL security model

The RLM controller (`marquis-rlm`) runs a persistent Python REPL that
executes model-generated code. This is **not sandboxed**.

## Risk surface

- The root LM generates Python code that runs with the user's full
  permissions.
- The REPL has access to the full MARQUIS tool namespace
  (`marquis.rlm_controller.tool_api`), the filesystem, and any
  environment variables.
- Model hallucinations can produce destructive file operations.

## Mitigations

- `MARQUIS_REPL_TIMEOUT_SECONDS` (default 300) kills long-running
  generated code.
- Run RLM jobs in SLURM with limited filesystem access (dedicated
  output directory, read-only data mounts).
- The tool namespace (`RLM_TOOL_SPECS`) is frozen at import time.
  The model cannot register new tools at runtime.

## When modifying

- Never add filesystem-write tools to `RLM_TOOL_SPECS` without explicit
  path restrictions.
- Never expose network-calling tools (HTTP, subprocess) to the REPL
  namespace.
- Test RLM changes with `--dry-run` or on small fixture data before
  paper-scale runs.
