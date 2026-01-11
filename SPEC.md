# ContextTune Project Specification

> **Keep this file updated** as the project evolves.

## Directory Conventions

```
artifacts/preds/<run_id>/preds.jsonl   # Prediction files
results/<run_id>/                       # Evaluation outputs
  ├── results.json
  ├── instance_results.jsonl
  ├── cmd.txt
  ├── stdout.log
  └── stderr.log
```

## Run ID Format

- Pattern: `<prefix>_<YYYYmmdd_HHMMSS>_<4-char-hex>`
- Example: `sanity_django__django-16379_20260111_143022_a1b2`
- Random suffix prevents collision in parallel runs.

## Code Conventions

- **Python version**: 3.10+
- **Dependencies**: Standard library only for core utils (no external deps).
- **Type hints**: Required for all function signatures.
- **Imports**: `from __future__ import annotations` at top of each module.
- **JSON output**: Single-line, UTF-8, `sort_keys=True`, newline-terminated.

## Shell Scripts

- Shebang: `#!/usr/bin/env bash`
- Use `set -euo pipefail` for safety.
- Compatible with Linux and WSL/Git Bash on Windows.
- Use `$(dirname "$0")` for script-relative paths.

## SWE-bench Integration

- Do NOT modify SWE-bench harness code.
- Call via `python -m swebench.harness.run_evaluation`.
- Default dataset: `princeton-nlp/SWE-bench_Verified`.

## Current Status

- [x] Harness sanity check + dummy prediction writer
- [ ] Model inference integration
- [ ] Context policy implementation
