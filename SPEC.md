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

Direct hard trim (retains summary snapshots only):
```bash
python scripts/hard_trim_repo.py --keep_summaries
```

Aggressive default (delete all generated artifacts/results):
```bash
python scripts/hard_trim_repo.py
bash scripts/trim_delete.sh
```

PowerShell wrapper:
```powershell
./scripts/trim_delete.ps1
```

## Determinism Rules

- **Sorted lists**: All list outputs are sorted alphabetically.
- **Normalized paths**: Use `/` separator, relative to repo root.
- **No timestamps**: `context.md` excludes timestamps for byte-identical reruns.
- **No randomness**: No random suffixes or UUIDs in deterministic outputs.
- **Repo dirname**: Always use `repo.replace("/", "__")` for filesystem paths (e.g., `astropy/astropy` → `astropy__astropy`).

## Context Budget

- **Unit**: Characters (approx tokens via `chars // 4`).
- **Total context body**: 3200 chars max (~800 tokens).
- **Per-card budget**: 900 chars max.
- **No tokenizer dependency**: Simpler, deterministic.

### Context Cards (Policy-Driven)

Baseline/tuned policies currently use task-centric cards:

1. `issue_focus` — Condensed problem statement focus
2. `fix_plan` — Minimal-change implementation guidance
3. `validation` — Testing/verification checklist
4. `editing_norms` — Global editing guidelines

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
- [x] Baseline/tuned context generation (task metadata + policy)
- [x] Pipeline orchestration (Step 6)

## Step 6: Verified Mini Experiment (Baseline vs Tuned)

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
artifacts/preds/<group_id>/baseline/preds.jsonl
artifacts/preds/<group_id>/tuned/preds.jsonl
results/<group_id>__baseline/             # harness outputs
results/<group_id>__tuned/                # harness outputs
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
    "baseline": {"resolved": 0, "total": 50, "rate": 0.0, ...},
    "tuned": {"resolved": 0, "total": 50, "rate": 0.0, ...}
  },
  "delta": {"resolved": 0, "rate": 0.0}
}
```

### Prerequisites

- Docker must be running for harness evaluation
- Contexts are built automatically (cached unless --force)

## Runner Backends

The inference runner supports multiple backends via `--runner`:

| Runner | Description | Requirements |
|--------|-------------|--------------|
| `single_shot` | Single LLM call, extract diff from response | OpenAI-compatible API |
| `mini_swe_agent` | Agentic loop via mini-swe-agent CLI (local env) | `pip install mini-swe-agent`, OpenAI/Anthropic API |
| `mini_swe_agent_swebench` | Agentic loop in SWE-bench Docker environment | Docker daemon + Linux, `pip install mini-swe-agent docker` |

### Recommended Runner for Non-Zero Baseline

For achieving a non-zero baseline on SWE-bench Verified Mini, use `mini_swe_agent_swebench`:

```bash
# Requires: Docker daemon running, Linux/WSL2 environment
python scripts/run_inference.py \
    --model openai/gpt-4 \
    --runner mini_swe_agent_swebench \
  --ablation baseline \
    --limit 50
```

This runner executes the agent inside the same Docker container environment that SWE-bench
uses for evaluation, ensuring environment parity (correct Python version, dependencies, etc.).

### Usage

```bash
# Default: single-shot runner
python scripts/run_inference.py --model gpt-4 --limit 1

# Agentic runner - local environment (requires mini-swe-agent installed)
python scripts/run_inference.py --model openai/gpt-4 --runner mini_swe_agent --limit 1

# Agentic runner - SWE-bench Docker environment (requires Docker + Linux)
python scripts/run_inference.py --model openai/gpt-4 --runner mini_swe_agent_swebench --limit 1
```

### Context Block Delimiters

When context is injected, it uses these delimiters (identical across all runners):

```
======BEGIN_REPO_CONTEXT=====
<context_md content>
======END_REPO_CONTEXT=====
```

This ensures the experimental variable is the **context artifact**, not the runner wiring.

## Adaptive Loop (SWE-Smith v1)

- `scripts/import_swesmith_tasks.py` converts SWE-Smith outputs to normalized task JSONL.
- `scripts/run_adaptive_context_loop.py` runs multi-round adaptive experiments:
  - build baseline/tuned contexts
  - run conditions (`baseline`, `tuned`)
  - evaluate
  - update policy via OpenAI-compatible LLM call

### Invariant

The SWE-bench harness evaluation is **unchanged** regardless of runner. All runners output the same JSONL format:

```json
{"instance_id": "...", "model_name_or_path": "...", "model_patch": "..."}
```
