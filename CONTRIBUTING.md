# Contributing

MARQUIS is organized as a paper-release repository. Keep changes focused on
reproducible code, small fixtures, and documentation needed to run or evaluate
the system.

## Development

```bash
make install-dev
make check
```

Default tests must not download models, load checkpoints, or require paper-scale
video data. Put heavy VLM/LLM checks behind explicit integration workflows.

## Repository Boundaries

- Package code belongs under `src/marquis`.
- Generated outputs, model caches, videos, checkpoints, and paper source do not
  belong in Git.
- Public examples should stay tiny and non-paper-scale.
- Prefer config defaults, environment variables, or CLI flags over hard-coded
  local paths.

## Style

- Keep command-line entrypoints importable and usable with `--help` without
  requiring model downloads.
- Preserve documented JSON artifact shapes unless a migration note and tests are
  added.
- Add tests for schema, citation, fusion, config, or CLI behavior when those
  surfaces change.
- Write public docs in a direct, specific voice.
