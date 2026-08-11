# Lazy import pattern

Heavy dependencies (`torch`, `transformers`, `vllm`, `whisper`,
`ir_measures`) are imported lazily, only when a code path that needs
them is actually executed.

## Why

- `--help`, config composition, and contract validation must work without
  GPU libraries installed.
- CI runs (`make check`, `make quickstart`) do not install torch or vllm.
- The `dev` extra is sufficient for linting, testing, and validation.

## How to maintain

When adding a new dependency on a heavy library:

1. Import it inside the function that uses it, not at module top-level.
2. If the import is needed in multiple functions in the same module,
   use a module-level helper:
   ```python
   def _get_vllm():
       import vllm
       return vllm
   ```
3. Verify `make quickstart` still passes without the library installed.
4. Add the library to the appropriate optional extra in `pyproject.toml`
   (`vlm`, `qa`, `retrieval`, etc.), not to the base dependencies.
