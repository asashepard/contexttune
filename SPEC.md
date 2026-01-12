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
artifacts/repos_cache/<repo_dirname>.git   # Bare git mirrors
artifacts/worktrees/<repo_dirname>/<commit>/  # Git worktrees (checked out repos)
artifacts/signals/<repo_dirname>/<commit>/signals.json  # Repo signals (Step 4)
artifacts/contexts/<repo_dirname>/<commit>/context.json  # Structured context
artifacts/contexts/<repo_dirname>/<commit>/context.md    # Rendered context (Step 5)
results/<run_id>/                       # Evaluation outputs
  ├── results.json
  ├── instance_results.jsonl
  ├── cmd.txt
  ├── stdout.log
  └── stderr.log
```

**repo_dirname**: Always `repo.replace("/", "__")` (e.g., `astropy/astropy` → `astropy__astropy`).

### Cleanup

To reset git cache and worktrees:
```bash
rm -rf artifacts/repos_cache artifacts/worktrees
```

Worktrees are not auto-pruned. Delete manually when disk space is needed.

To reset signals:
```bash
rm -rf artifacts/signals
```

## Determinism Rules

- **Sorted lists**: All list outputs are sorted alphabetically.
- **Normalized paths**: Use `/` separator, relative to repo root.
- **No timestamps**: `signals.json` and `context.md` exclude timestamps for byte-identical reruns.
- **No randomness**: No random suffixes or UUIDs in deterministic outputs.
- **Repo dirname**: Always use `repo.replace("/", "__")` for filesystem paths (e.g., `astropy/astropy` → `astropy__astropy`).

## Context Budget

- **Unit**: Characters (approx tokens via `chars // 4`).
- **Total context body**: 3200 chars max (~800 tokens).
- **Per-card budget**: 900 chars max.
- **No tokenizer dependency**: Simpler, deterministic.

### Context Cards (Fixed Order)

1. `repo_identity` — Repository metadata, primary packages
2. `tests_howto` — Test commands, directories
3. `editing_norms` — Global editing guidelines (constant)
4. `routing_guidance` — Hot paths, module prefixes, entrypoints
5. `pitfalls` — Warnings about layout quirks

## Run ID Format

- Pattern: `<prefix>_<YYYYmmdd_HHMMSS>_<4-char-hex>`
- Example: `sanity_django__django-16379_20260111_143022_a1b2`
- Random suffix prevents collision in parallel runs.

## Code Conventions

- **Python version**: 3.11+ (required for `tomllib`)
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
- [x] Repo signals extraction (Step 4)
- [x] Baseline context generation (Step 5)
- [x] Pipeline orchestration (Step 6)

## Step 6: Verified Mini Baseline Experiment

### Usage

```bash
# Dry-run (skip model calls, test plumbing)
python scripts/run_verified_mini_baseline.py --model local/placeholder --dry_run

# Full run with local model
export OPENAI_BASE_URL="http://localhost:8000/v1"
python scripts/run_verified_mini_baseline.py --model my-model --max_workers_eval 4
```

### Output Layout

```
artifacts/preds/<group_id>/no_context/preds.jsonl
artifacts/preds/<group_id>/baseline_context/preds.jsonl
results/<group_id>__no_context/           # harness outputs
results/<group_id>__baseline_context/     # harness outputs
results/<group_id>/
  ├── logs/                               # pipeline step logs
  ├── summary.json                        # comparison summary
```

### summary.json Schema

```json
{
  "group_id": "verified_mini_20260111_175000_a1b2",
  "dataset": "princeton-nlp/SWE-bench_Verified",
  "split": "test",
  "model": "local/placeholder",
  "instance_count": 50,
  "conditions": {
    "no_context": {"resolved": 0, "total": 50, "rate": 0.0, ...},
    "baseline_context": {"resolved": 0, "total": 50, "rate": 0.0, ...}
  },
  "delta": {"resolved": 0, "rate": 0.0}
}
```

### Prerequisites

- Docker must be running for harness evaluation
- Signals and contexts are built automatically (cached unless --force)
