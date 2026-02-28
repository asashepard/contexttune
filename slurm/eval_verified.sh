#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# SWE-bench Verified evaluation — run both conditions (no_context,
# tuned_context) after tuning is complete.
#
# Submit:
#   MODEL_NAME=openai/gpt-4o EXP_ID=exp001 \
#     sbatch slurm/eval_verified.sh
#
# Required: MODEL_NAME, EXP_ID
# Optional: DATASET_NAME, IDS_FILE, MAX_WORKERS_EVAL, TIMEOUT_S, STEP_LIMIT
# ──────────────────────────────────────────────────────────────────────
#SBATCH -J ct_eval
#SBATCH -p gpmoo-b
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 2-00:00
#SBATCH -o slurm-eval-%j.out
#SBATCH -e slurm-eval-%j.err
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
ENV_NAME="${ENV_NAME:-contexttune-py311}"
MODEL_NAME="${MODEL_NAME:?set MODEL_NAME}"
EXP_ID="${EXP_ID:?set EXP_ID}"

DATASET_NAME="${DATASET_NAME:-princeton-nlp/SWE-bench_Verified}"
IDS_FILE="${IDS_FILE:-}"
MAX_WORKERS_EVAL="${MAX_WORKERS_EVAL:-4}"
TIMEOUT_S="${TIMEOUT_S:-600}"
STEP_LIMIT="${STEP_LIMIT:-30}"

cd "$REPO_ROOT"
source ~/.bashrc || true
conda activate "$ENV_NAME" 2>/dev/null || true

# Build repo config from tuning output
REPO_CONFIG="results/${EXP_ID}/repo_config.json"

echo "============================================================"
echo "SWE-bench Verified Evaluation"
echo "Experiment: $EXP_ID"
echo "Model: $MODEL_NAME"
echo "============================================================"

IDS_FLAG=""
if [[ -n "$IDS_FILE" ]]; then
    IDS_FLAG="--eval-instance-ids $IDS_FILE"
fi

python scripts/run_experiment.py \
    --model "$MODEL_NAME" \
    --repo-config "$REPO_CONFIG" \
    --experiment-id "$EXP_ID" \
    --eval-dataset "$DATASET_NAME" \
    --max-workers-eval "$MAX_WORKERS_EVAL" \
    --timeout-s "$TIMEOUT_S" \
    --step-limit "$STEP_LIMIT" \
    $IDS_FLAG

echo "Evaluation complete. Check results/${EXP_ID}/experiment_summary.json"
