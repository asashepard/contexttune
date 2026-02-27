#!/usr/bin/env bash
#SBATCH -J ct_smoke
#SBATCH -p gpmoo-b
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -t 0-02:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
ENV_NAME="${ENV_NAME:-contexttune-py311}"
MODEL_NAME="${MODEL_NAME:?set MODEL_NAME, e.g. openai/<model>}"
RUN_TAG="${RUN_TAG:-gpmoo_smoke}"
DATASET_NAME="${DATASET_NAME:-princeton-nlp/SWE-bench_Verified}"
SPLIT="${SPLIT:-test}"
RUNNER="${RUNNER:-mini_swe_agent_swebench}"
TIMEOUT_S="${TIMEOUT_S:-600}"
STEP_LIMIT="${STEP_LIMIT:-30}"
MAX_WORKERS_EVAL="${MAX_WORKERS_EVAL:-1}"

cd "$REPO_ROOT"
source ~/.bashrc || true
conda activate "$ENV_NAME"

python scripts/build_context.py \
  --dataset_name "$DATASET_NAME" \
  --split "$SPLIT" \
  --instance_ids_file scripts/smoke_3_ids.txt \
  --mode baseline

RUN_ID="${RUN_TAG}__baseline"
PREDS_PATH="artifacts/preds/${RUN_TAG}/baseline/preds.jsonl"

python scripts/run_inference.py \
  --dataset_name "$DATASET_NAME" \
  --split "$SPLIT" \
  --instance_ids_file scripts/smoke_3_ids.txt \
  --model "$MODEL_NAME" \
  --ablation baseline \
  --runner "$RUNNER" \
  --timeout_s "$TIMEOUT_S" \
  --step_limit "$STEP_LIMIT" \
  --run_id "$RUN_ID" \
  --out "$PREDS_PATH"

bash scripts/run_swebench_eval.sh \
  "$DATASET_NAME" \
  "$PREDS_PATH" \
  "$RUN_ID" \
  "$MAX_WORKERS_EVAL"
