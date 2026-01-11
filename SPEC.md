# ContextTune Project Specification

> **Keep this file updated** as the project evolves.

## Execution Environment

- **Primary**: WSL2 / Linux (Ubuntu recommended)
- **Why**: SWE-bench harness uses Docker, git worktrees, bash scripts
- **Windows**: Use WSL2. Native Windows adds complexity (path handling, subprocess).
- **Project path in WSL**: `/mnt/c/code/contexttune` or clone to `~/code/contexttune`

## Directory Conventions

```
artifacts/preds/<run_id>/preds.jsonl   # Prediction files
artifacts/logs/<run_id>/<instance_id>.log  # Per-instance debug logs
artifacts/repos_cache/<repo>.git       # Bare git mirrors
artifacts/worktrees/<repo>/<commit>/   # Git worktrees (checked out repos)
artifacts/contexts/<repo>/<commit>/context.md  # Pre-generated context (Step 5)
results/<run_id>/                       # Evaluation outputs
  ├── results.json
  ├── instance_results.jsonl
  ├── cmd.txt
  ├── stdout.log
  └── stderr.log
```

### Cleanup

To reset git cache and worktrees:
```bash
rm -rf artifacts/repos_cache artifacts/worktrees
```

Worktrees are not auto-pruned. Delete manually when disk space is needed.

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

## Model Inference Interface (Thin Waist)

All model inference uses **OpenAI-compatible HTTP API**:

```bash
export OPENAI_BASE_URL="http://localhost:8000/v1"  # or https://api.openai.com/v1
export OPENAI_API_KEY="sk-..."                      # dummy ok for local
```

- Standardizes against any serving stack (vLLM, Ollama, OpenAI, Anthropic-via-proxy).
- Use `requests` library, no vendor SDKs needed.
- Endpoint: `POST /chat/completions` with standard OpenAI schema.

## Dependencies

Minimal, pinned in `requirements.txt`:
- `swebench` — harness
- `requests` — HTTP client
- `datasets` — HuggingFace dataset loading

## Current Status

- [x] Harness sanity check + dummy prediction writer
- [x] Minimal single-shot inference runner (Step 3)
- [ ] Context policy implementation (Step 5)
