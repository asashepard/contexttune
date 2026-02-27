#!/usr/bin/env bash
#SBATCH -J ct_full_eval
#SBATCH -p gpmoo-b
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 0-08:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
ENV_NAME="${ENV_NAME:-contexttune-py311}"
RUN_ID="${RUN_ID:?set RUN_ID, e.g. verified_full_gpmoo_001}"
CONDITION="${CONDITION:?set CONDITION=baseline|tuned}"
SHARD_DIR="${SHARD_DIR:?set SHARD_DIR, e.g. artifacts/shards/full_verified}"
DATASET_NAME="${DATASET_NAME:-princeton-nlp/SWE-bench_Verified}"
MAX_WORKERS_EVAL="${MAX_WORKERS_EVAL:-1}"

if [[ "$CONDITION" != "baseline" && "$CONDITION" != "tuned" ]]; then
  echo "Invalid CONDITION: $CONDITION"
  exit 2
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "SLURM_ARRAY_TASK_ID not set. Submit with --array."
  exit 3
fi

cd "$REPO_ROOT"
source ~/.bashrc || true
conda activate "$ENV_NAME"

mapfile -t SHARD_FILES < <(find "$SHARD_DIR" -maxdepth 1 -type f -name '*.txt' | sort)
if [[ "${#SHARD_FILES[@]}" -eq 0 ]]; then
  echo "No shard files found in $SHARD_DIR"
  exit 4
fi

TASK_ID="$SLURM_ARRAY_TASK_ID"
if (( TASK_ID < 0 || TASK_ID >= ${#SHARD_FILES[@]} )); then
  echo "Array index $TASK_ID out of bounds for ${#SHARD_FILES[@]} shards"
  exit 5
fi

INSTANCE_IDS_FILE="${SHARD_FILES[$TASK_ID]}"
SHARD_NAME="$(basename "$INSTANCE_IDS_FILE" .txt)"
COND_RUN_ID="${RUN_ID}__${CONDITION}__${SHARD_NAME}"
PREDS_PATH="artifacts/preds/${RUN_ID}/${CONDITION}/${SHARD_NAME}.jsonl"

if [[ ! -f "$PREDS_PATH" ]]; then
  echo "Missing predictions file: $PREDS_PATH"
  exit 6
fi

echo "task_id=$TASK_ID shard=$INSTANCE_IDS_FILE condition=$CONDITION"

bash scripts/run_swebench_eval.sh \
  "$DATASET_NAME" \
  "$PREDS_PATH" \
  "$COND_RUN_ID" \
  "$MAX_WORKERS_EVAL"
