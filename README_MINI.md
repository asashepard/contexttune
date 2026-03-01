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
pip install -r requirements.txt
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
- Slurm scripts: `slurm/smoke_experiment.sh`, `slurm/tune_array.sh`, `slurm/eval_verified.sh`

## EC2 4-Instance Smoke Run + Eval

Run a standardized 4-instance sanity pack on EC2 or gpmoo shell:

```bash
bash scripts/ec2_smoke.sh --model openai/gpt-5.2
```

Run all three conditions in one command:

```bash
bash scripts/ec2_smoke.sh --model openai/gpt-5.2 --conditions no_context,static_kb,oracle_tuned
```

Default IDs file: `scripts/easy_4_ids.txt`.

## Tree-Sitter Probe + Oracle Tuning

Build a knowledge base and tune AGENTS.md for a single repo:

```bash
python scripts/tune_single_repo.py \
  --repo django/django \
  --commit HEAD \
  --model openai/gpt-5.2 \
  --output-dir results/exp1/guidance/django__django \
  --iterations 5
```

Run a full experiment across all repos with three conditions:

```bash
python scripts/run_experiment.py \
  --model openai/gpt-5.2 \
  --repo-config artifacts/configs/repos_12.json \
  --conditions no_context static_kb oracle_tuned \
  --oracle-iterations 5
```
