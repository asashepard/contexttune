# ContextTune Project Specification

> **Keep this file updated** as the project evolves.

## Execution Environment

- **Primary**: WSL2 / Linux (Ubuntu recommended)
- **Why**: SWE-bench harness uses Docker, git worktrees, bash scripts
- **Windows**: Use WSL2. Native Windows is for development only.

## Directory Conventions

```
artifacts/preds/<experiment_id>/<condition>/preds.jsonl
artifacts/tasks/<repo_dirname>/train.jsonl
artifacts/tasks/<repo_dirname>/holdout.jsonl
artifacts/guidance/<repo_dirname>/best_guidance.json
artifacts/guidance/<repo_dirname>/versions/v0.json ... vN.json
artifacts/guidance/<repo_dirname>/tuning_state.json
results/<experiment_id>/
  ├── experiment_config.json
  ├── experiment_state.json
  ├── experiment_summary.json
  ├── guidance/<repo_dirname>/best_guidance.json
  └── logs/
```

**repo_dirname**: Always `repo.replace("/", "__")` (e.g., `django/django` → `django__django`).

## Guidance Design

### Tunable Object: RepoGuidance

A single bounded text block per repository, defined in `context_policy/guidance/schema.py`:

```python
@dataclass
class RepoGuidance:
    repo: str           # e.g. "django/django"
    commit: str         # base commit SHA
    lines: list[str]    # guidance bullet points
    version: int = 0    # monotonic version counter
    char_budget: int = 3200  # hard character limit
```

### Guidance Budget

- **Unit**: Characters (approx tokens via `chars // 4`).
- **Total guidance text**: 3200 chars max (~800 tokens).
- **No tokenizer dependency**: Simpler, deterministic.

### Context Injection Format

```
# REPO GUIDANCE (AUTO-TUNED)
- line 1
- line 2
...
# END REPO GUIDANCE
```

This block is prepended to the problem statement for agent-based runners,
or appended to the user message for single-shot prompting.

## Hill-Climbing Tuner

1. **Init G₀**: LLM generates initial guidance from repo tree + conventions.
2. **Score G₀**: Run N SWE-smith tasks through Docker agent, measure resolve rate.
3. **Iterate T times**:
   - Propose K candidate edits via LLM.
   - Score each candidate on N tasks.
   - If any candidate beats current best, adopt it.
4. **Save best guidance** for final evaluation.

### Default Budget (Final)

| Parameter | Dev | Final |
|-----------|-----|-------|
| T (iterations) | 3 | 10 |
| K (candidates) | 4 | 6 |
| N (tasks/score) | 10 | 20 |

### 12 Experiment Repos

django, astropy, sympy, scikit-learn, matplotlib, flask,
sphinx, pylint, pytest, requests, xarray, seaborn.

## Run ID Format

- Pattern: `<prefix>_<YYYYmmdd_HHMMSS>_<4-char-hex>`
- Example: `guidance_tune_20260115_143022_a1b2`
- Random suffix prevents collision in parallel runs.

## Code Conventions

- **Python version**: 3.11+
- **Type hints**: Required for all function signatures.
- **Imports**: `from __future__ import annotations` at top of each module.
- **JSON output**: `indent=2`, UTF-8, `sort_keys=True`, newline-terminated.

## Shell Scripts

- Shebang: `#!/usr/bin/env bash`
- Use `set -euo pipefail` for safety.

## SWE-bench Integration

- Do NOT modify SWE-bench harness code.
- Call via `python -m swebench.harness.run_evaluation`.
- Default dataset: `princeton-nlp/SWE-bench_Verified`.

## Model Inference Interface

All model inference uses **OpenAI-compatible HTTP API**:

```bash
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="sk-..."
```

- Endpoint: `POST /chat/completions` with standard OpenAI schema.
- Single `--model` flag for all purposes (init, propose, score, eval).
- `requests` library, no vendor SDKs.

## Runner Backends

| Runner | Description | Requirements |
|--------|-------------|--------------|
| `single_shot` | Single LLM call, extract diff | OpenAI-compatible API |
| `mini_swe_agent` | Agentic loop via mini-swe-agent CLI | `pip install mini-swe-agent` |
| `mini_swe_agent_swebench` | Agentic loop in SWE-bench Docker env | Docker + `pip install mini-swe-agent` |

**Primary runner**: `mini_swe_agent_swebench` for both tuning and evaluation.

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_experiment.py` | Full experiment: tune all repos + Verified eval |
| `scripts/tune_single_repo.py` | Tune one repo (for array jobs) |
| `scripts/run_inference.py` | Standalone inference (any runner) |
| `scripts/generate_swesmith_tasks.py` | Generate SWE-smith tasks for one repo |
| `scripts/generate_all_swesmith.sh` | Generate tasks for all 12 repos |
| `scripts/run_swebench_eval.sh` | Run SWE-bench harness evaluation |
| `scripts/build_docker_images.py` | Build Docker images for SWE-bench instances |

## Slurm Jobs

| Script | Purpose |
|--------|---------|
| `slurm/smoke_experiment.sh` | Quick smoke test (1 repo, minimal budget) |
| `slurm/tune_array.sh` | Array job: tune 12 repos in parallel |
| `slurm/eval_verified.sh` | Evaluate after tuning completes |

## Experiment Workflow

```
1. Generate SWE-smith tasks:
   bash scripts/generate_all_swesmith.sh commit_map.txt

2a. Sequential (single node):
    python scripts/run_experiment.py --model <model> --repo-config repos.json

2b. Parallel (Slurm):
    sbatch --array=0-11 slurm/tune_array.sh   # tune 12 repos
    sbatch slurm/eval_verified.sh               # eval after tuning

3. Results:
   cat results/<exp_id>/experiment_summary.json
```

## Prediction Format (Invariant)

All runners output the same JSONL format:

```json
{"instance_id": "...", "model_name_or_path": "...", "model_patch": "..."}
```

The SWE-bench harness evaluation is **unchanged** regardless of runner or guidance.
