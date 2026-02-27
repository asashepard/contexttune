# Quick Start: SWE-bench Sanity Check

## Required vs Generated

- **Required (keep)**: `context_policy/`, `scripts/`, `slurm/`, `schema/`, `requirements.txt`, `pyproject.toml`, docs.
- **Generated (disposable)**: `artifacts/` and `results/` run outputs.
- **Single-command hard delete**:
	- PowerShell: `./scripts/trim_delete.ps1`
	- Bash: `bash scripts/trim_delete.sh`
- Optional summary retention: `python scripts/hard_trim_repo.py --keep_summaries`

## Install Dependencies

```bash
pip install swebench
```

## Run Sanity Check

```bash
./scripts/sanity_one_instance.sh django__django-16379
```

## Output Locations

- **Predictions**: `artifacts/preds/<run_id>/preds.jsonl`
- **Results**: `results/<run_id>/` (stdout.log, stderr.log, cmd.txt)

## Custom Dataset

```bash
./scripts/sanity_one_instance.sh django__django-16379 princeton-nlp/SWE-bench_Lite
```

## GPMoo / Slurm Workflow

For cluster runs on GPMoo, use the runbook and Slurm templates:

- Runbook: `docs/GPMOO_RUNBOOK.md`
- Slurm scripts: `slurm/run_preflight.sh`, `slurm/run_smoke.sh`, `slurm/run_mini_condition.sh`, `slurm/run_eval_condition.sh`, `slurm/run_full_condition_array.sh`, `slurm/run_full_eval_array.sh`
- Submission/sharding helpers: `scripts/split_instance_ids.py`, `scripts/submit_full_verified_array.sh`
