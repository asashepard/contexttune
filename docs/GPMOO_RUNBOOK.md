# GPMoo Runbook for ContextTune

This runbook standardizes how to run ContextTune SWE-bench experiments on GPMoo with Slurm.

## Scope

- Keep experiment semantics unchanged.
- Run intensive work only via Slurm jobs.
- Use `gpmoo-b` by default, and reserve `gpmoo-a` for heavier model-serving workloads.

## Cluster Notes

- Head/login node: `gpmoo.cs.williams.edu`
- Compute nodes:
  - `gpmoo-a` partition (A100)
  - `gpmoo-b` partition (A6000)
- Use `change_password` for password updates on GPMoo.

## 1) Login and Repository Setup

```bash
ssh <username>@gpmoo.cs.williams.edu
cd ~
git clone <your-contexttune-repo-url> contexttune
cd contexttune
```

If off-campus, use VPN or an SSH `ProxyJump` through an approved access host.

## 2) Create Conda Environment

```bash
conda create --name contexttune-py311 python=3.11 -y
conda activate contexttune-py311
pip install -r requirements.txt
```

Optional for agentic runners:

```bash
pip install "mini-swe-agent~=1.17"
```

## 3) Export Model Endpoint Variables

Set these in your shell before submitting jobs:

```bash
export OPENAI_BASE_URL="http://<model-host>:<port>/v1"
export OPENAI_API_KEY="<token-or-dummy>"
```

## 4) Slurm Operations Quick Reference

```bash
sinfo
squeue -u $USER
sbatch <job-script.sh>
scancel <jobid>
```

Tail logs while running:

```bash
tail -f slurm-<jobid>.out
```

## 5) First Run Sequence

Run this exact order:

1. Smoke experiment (single repo, minimal budget)
2. Tune all repos with Slurm array
3. Run Verified eval over selected conditions
4. Summarize and compare

### 5.1 Smoke Experiment

```bash
MODEL_NAME="openai/<model>" sbatch slurm/smoke_experiment.sh
```

### 5.2 Tune All Repos (Array)

```bash
MODEL_NAME="openai/<model>" EXP_ID="gpmoo_exp_001" \
  sbatch --array=0-11 slurm/tune_array.sh
```

### 5.3 Verified Evaluation

```bash
MODEL_NAME="openai/<model>" EXP_ID="gpmoo_exp_001" \
  sbatch slurm/eval_verified.sh
```

To restrict to specific conditions or instance IDs, run directly:

```bash
python scripts/run_experiment.py \
  --model openai/<model> \
  --repo-config artifacts/configs/repos_12.json \
  --experiment-id gpmoo_exp_001 \
  --conditions no_context static_kb oracle_tuned \
  --oracle-iterations 5 \
  --eval-instance-ids scripts/verified_mini_ids.txt
```

## 5.5 One-Command 4-Instance Sanity Run

For a quick, reusable run+eval command (EC2 or gpmoo shell):

```bash
bash scripts/ec2_smoke.sh --model openai/gpt-5.2
```

To run both conditions:

```bash
bash scripts/ec2_smoke.sh --model openai/gpt-5.2 --conditions no_context,static_kb,oracle_tuned
```

IDs are read from `scripts/easy_4_ids.txt` by default.

## 6) Output Paths

- Predictions:
  - `artifacts/preds/<RUN_ID>/<CONDITION>/preds.jsonl`
- Inference logs:
  - `artifacts/logs/<RUN_ID>__<CONDITION>/`
- Eval results:
  - `results/<RUN_ID>__<CONDITION>/`

## 7) Resume and Recovery

- `run_inference.py` is append/resume-safe by `instance_id` in existing `preds.jsonl`.
- If a condition job fails mid-run, rerun the same `sbatch` command with same `RUN_ID` and `CONDITION`.
- To force full rerun of a condition, remove that condition’s `preds.jsonl`.

## 8) Full Verified Scaling

- Use `scripts/run_experiment.py` with a full instance ID list and all conditions.
- Keep model, timeout, runner, and step limits identical across conditions.
- Prefer one shared `EXP_ID` so tuning/eval artifacts stay in one results folder.

Example full run:

```bash
python scripts/run_experiment.py \
  --model openai/<model> \
  --repo-config artifacts/configs/repos_12.json \
  --experiment-id verified_full_gpmoo_001 \
  --conditions no_context static_kb oracle_tuned \
  --oracle-iterations 5 \
  --eval-instance-ids /path/to/full_verified_ids.txt
```
