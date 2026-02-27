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

1. Preflight checks (short job)
2. Smoke run on 3 IDs
3. Mini baseline condition jobs (`no_context` and `baseline_context`)
4. Eval jobs for each condition
5. Summarize and compare

### 5.1 Preflight

```bash
sbatch slurm/run_preflight.sh
```

### 5.2 Smoke Run (3 IDs)

```bash
sbatch --export=ALL,MODEL_NAME="openai/<model>",RUN_TAG="gpmoo_smoke" slurm/run_smoke.sh
```

### 5.3 Mini Condition Jobs

Submit one job per condition:

```bash
sbatch --export=ALL,MODEL_NAME="openai/<model>",RUN_ID="verified_mini_gpmoo_001",CONDITION="no_context" slurm/run_mini_condition.sh
sbatch --export=ALL,MODEL_NAME="openai/<model>",RUN_ID="verified_mini_gpmoo_001",CONDITION="baseline_context" slurm/run_mini_condition.sh
```

### 5.4 Eval Jobs

Submit one eval per condition after inference files exist:

```bash
sbatch --export=ALL,RUN_ID="verified_mini_gpmoo_001",CONDITION="no_context",DATASET_NAME="princeton-nlp/SWE-bench_Verified" slurm/run_eval_condition.sh
sbatch --export=ALL,RUN_ID="verified_mini_gpmoo_001",CONDITION="baseline_context",DATASET_NAME="princeton-nlp/SWE-bench_Verified" slurm/run_eval_condition.sh
```

## 5.5 One-Command 4-Instance Sanity Run

For a quick, reusable run+eval command (EC2 or gpmoo shell):

```bash
bash scripts/run_smoke4_eval.sh --model openai/gpt-5.2
```

To run both conditions:

```bash
bash scripts/run_smoke4_eval.sh --model openai/gpt-5.2 --conditions baseline_context,no_context
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

- Split instance IDs into shards (see `scripts/split_instance_ids.py`).
- Submit one array per condition for inference, with eval arrays chained by dependency.
- Keep model, timeout, runner, and step limits identical across conditions.

### 8.1 Build Shards

```bash
python scripts/split_instance_ids.py \
  --input_file scripts/verified_mini_ids.txt \
  --shards 4 \
  --out_dir artifacts/shards/verified_mini \
  --prefix verified_mini
```

For full Verified, provide your full instance ID list file instead of `scripts/verified_mini_ids.txt`.

### 8.2 Submit Full Array Jobs (End-to-End)

```bash
export MODEL_NAME="openai/<model>"
export RUN_ID="verified_full_gpmoo_001"
export SHARD_DIR="artifacts/shards/verified_mini"

bash scripts/submit_full_verified_array.sh
```

This submits:

- Inference array: `slurm/run_full_condition_array.sh` (for each condition)
- Eval array: `slurm/run_full_eval_array.sh` with `afterok` dependency on inference

Job IDs are recorded at:

- `results/<RUN_ID>/slurm_jobs.tsv`
