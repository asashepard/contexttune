#!/usr/bin/env bash
#SBATCH -J ct_mini_cond
#SBATCH -p gpmoo-b
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 0-08:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
ENV_NAME="${ENV_NAME:-contexttune-py311}"
MODEL_NAME="${MODEL_NAME:?set MODEL_NAME, e.g. openai/<model>}"
RUN_ID="${RUN_ID:?set RUN_ID, e.g. verified_mini_gpmoo_001}"
CONDITION="${CONDITION:?set CONDITION=baseline|tuned}"
DATASET_NAME="${DATASET_NAME:-princeton-nlp/SWE-bench_Verified}"
SPLIT="${SPLIT:-test}"
INSTANCE_IDS_FILE="${INSTANCE_IDS_FILE:-scripts/verified_mini_ids.txt}"
RUNNER="${RUNNER:-mini_swe_agent_swebench}"
TIMEOUT_S="${TIMEOUT_S:-600}"
STEP_LIMIT="${STEP_LIMIT:-30}"

if [[ "$CONDITION" != "baseline" && "$CONDITION" != "tuned" ]]; then
  echo "Invalid CONDITION: $CONDITION"
  exit 2
fi

cd "$REPO_ROOT"
source ~/.bashrc || true
conda activate "$ENV_NAME"

python scripts/build_context.py \
  --dataset_name "$DATASET_NAME" \
  --split "$SPLIT" \
  --instance_ids_file "$INSTANCE_IDS_FILE" \
  --mode "$CONDITION"

COND_RUN_ID="${RUN_ID}__${CONDITION}"
PREDS_PATH="artifacts/preds/${RUN_ID}/${CONDITION}/preds.jsonl"

python scripts/run_inference.py \
  --dataset_name "$DATASET_NAME" \
  --split "$SPLIT" \
  --instance_ids_file "$INSTANCE_IDS_FILE" \
  --model "$MODEL_NAME" \
  --ablation "$CONDITION" \
  --runner "$RUNNER" \
  --timeout_s "$TIMEOUT_S" \
  --step_limit "$STEP_LIMIT" \
  --run_id "$COND_RUN_ID" \
  --out "$PREDS_PATH"

echo "inference complete: $PREDS_PATH"
