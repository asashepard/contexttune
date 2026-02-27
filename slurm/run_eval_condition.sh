#!/usr/bin/env bash
#SBATCH -J ct_eval_cond
#SBATCH -p gpmoo-b
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 0-06:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
ENV_NAME="${ENV_NAME:-contexttune-py311}"
RUN_ID="${RUN_ID:?set RUN_ID, e.g. verified_mini_gpmoo_001}"
CONDITION="${CONDITION:?set CONDITION=no_context|baseline_context}"
DATASET_NAME="${DATASET_NAME:-princeton-nlp/SWE-bench_Verified}"
MAX_WORKERS_EVAL="${MAX_WORKERS_EVAL:-1}"

if [[ "$CONDITION" != "no_context" && "$CONDITION" != "baseline_context" ]]; then
  echo "Invalid CONDITION: $CONDITION"
  exit 2
fi

cd "$REPO_ROOT"
source ~/.bashrc || true
conda activate "$ENV_NAME"

COND_RUN_ID="${RUN_ID}__${CONDITION}"
PREDS_PATH="artifacts/preds/${RUN_ID}/${CONDITION}/preds.jsonl"

if [[ ! -f "$PREDS_PATH" ]]; then
  echo "Missing predictions file: $PREDS_PATH"
  exit 3
fi

bash scripts/run_swebench_eval.sh \
  "$DATASET_NAME" \
  "$PREDS_PATH" \
  "$COND_RUN_ID" \
  "$MAX_WORKERS_EVAL"
