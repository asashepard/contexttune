# ContextTune Project Specification

> **Keep this file updated** as the project evolves.

## Execution Environment

- **Primary**: WSL2 / Linux (Ubuntu recommended)
- **Why**: SWE-bench harness uses Docker, git worktrees, bash scripts
- **Windows**: Use WSL2. Native Windows is for development only.

## Directory Conventions

```
artifacts/preds/<experiment_id>/<condition>/preds.jsonl
artifacts/guidance/<repo_dirname>/kb/kb.json
artifacts/guidance/<repo_dirname>/kb/agents_md_v0.md
artifacts/guidance/<repo_dirname>/kb/probes_summary.json
artifacts/guidance/<repo_dirname>/versions/v0.json ... vN.json
artifacts/guidance/<repo_dirname>/best_guidance.json
artifacts/guidance/<repo_dirname>/tuning_state.json
results/<experiment_id>/
  ├── experiment_config.json
  ├── experiment_state.json
  ├── experiment_summary.json
  ├── guidance/<repo_dirname>/...
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

## Tree-Sitter Probe Layer

Static analysis of the repo using **py-tree-sitter** (Python only):

1. **Import graph**: Build `importedBy` map, identify hub modules (≤12).
2. **Symbol index**: Extract functions, classes, constants with signatures + callers (≤10/file, ≤30 files).
3. **Entry points**: Content-based detection (if-main, decorators, CLI, ASGI/WSGI, ≤10).
4. **Clustering**: Co-import scoring (shared importers ≥2), greedy agglomeration (≤6 clusters, ≤8 chains).
5. **Conventions**: Docstring style, type hints, import patterns, linter configs.
6. **Tests**: Test directories, conftest, pytest config.

Probe results are deterministic (no LLM calls).

## Knowledge Base (KB)

Built from probe results as a structured markdown document with sections:

| Section | Line Budget |
|---------|-------------|
| Architecture (hubs + entry points) | 200 |
| Symbol Map | 300 |
| Context (clusters + tests) | 200 |
| Conventions | unbounded (small) |

KB is saved as JSON (`kb.json`) and rendered into an AGENTS.md file.

## AGENTS.md

A ≤3,200 character instruction document rendered deterministically from the KB.
This is the `static_kb` experimental condition.

## Oracle Evaluator Loop (LLM-as-Judge)

Replaces the old SWE-Smith hill-climbing tuner.

1. **Build KB**: Checkout repo → run probes → build KB (deterministic).
2. **Render static AGENTS.md** from KB (`static_kb` condition).
3. **Generate micro-test probes** from KB (5 categories, ≤10 total):
   - hub-safety, entry-point, naming, architecture, harvested
4. **For T iterations** (default 5):
   a. **Simulate**: Send AGENTS.md as system prompt + probe task → AI response.
   b. **Judge**: LLM evaluates response against expected behaviors → PASS/FAIL.
   c. **Diagnose**: Collect failures → LLM proposes structured edits.
   d. **Apply**: LLM merges edits into AGENTS.md (re-enforce 3,200-char budget).
   e. If new version improves pass rate, adopt it.
5. **Save best AGENTS.md** → `oracle_tuned` condition.

### Experimental Conditions

| Condition | Agent sees | Tuning |
|-----------|-----------|--------|
| `no_context` | Issue + tree only | None |
| `static_kb` | Issue + tree + KB-derived AGENTS.md | None (deterministic) |
| `oracle_tuned` | Issue + tree + oracle-refined AGENTS.md | LLM-as-judge loop |

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
| `scripts/tune_single_repo.py` | Tune one repo via oracle loop (for array jobs) |
| `scripts/run_inference.py` | Standalone inference (any runner) |
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
1. Configure repos (repos.json with repo + commit):
   [{"repo": "django/django", "commit": "<sha>"}, ...]

2a. Sequential (single node):
    python scripts/run_experiment.py --model <model> --repo-config repos.json \
        --conditions no_context static_kb oracle_tuned --oracle-iterations 5

2b. Parallel (Slurm):
    sbatch --array=0-11 slurm/tune_array.sh   # oracle tune 12 repos
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
