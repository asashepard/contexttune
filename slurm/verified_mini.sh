#!/usr/bin/env bash
#SBATCH -J ct_vmini
#SBATCH -p gpmoo-b
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 0-10:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
# ──────────────────────────────────────────────────────────────────────
# GPMoo SWE-bench Verified Mini (50 instances) — single condition.
#
# Submit one condition at a time, then evaluate:
#   MODEL_NAME=openai/gpt-4o RUN_ID=vmini_001 CONDITION=baseline \
#     sbatch slurm/verified_mini.sh
#
#   MODEL_NAME=openai/gpt-4o RUN_ID=vmini_001 CONDITION=tuned \
#     sbatch slurm/verified_mini.sh
#
# After both complete, evaluate each:
#   RUN_ID=vmini_001 CONDITION=baseline sbatch slurm/verified_mini.sh --eval-only
#
# Or run inference + eval together (default).
#
# Required env vars: MODEL_NAME, RUN_ID, CONDITION
# Optional: RUNNER, TIMEOUT_S, STEP_LIMIT, MAX_WORKERS_EVAL, EVAL_ONLY
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
ENV_NAME="${ENV_NAME:-contexttune-py311}"
MODEL_NAME="${MODEL_NAME:?set MODEL_NAME}"
RUN_ID="${RUN_ID:?set RUN_ID}"
CONDITION="${CONDITION:?set CONDITION=baseline|tuned}"
DATASET_NAME="${DATASET_NAME:-princeton-nlp/SWE-bench_Verified}"
SPLIT="${SPLIT:-test}"
IDS_FILE="${IDS_FILE:-scripts/verified_mini_ids.txt}"
RUNNER="${RUNNER:-mini_swe_agent_swebench}"
TIMEOUT_S="${TIMEOUT_S:-600}"
STEP_LIMIT="${STEP_LIMIT:-30}"
MAX_WORKERS_EVAL="${MAX_WORKERS_EVAL:-1}"
EVAL_ONLY="${EVAL_ONLY:-0}"

if [[ "$CONDITION" != "baseline" && "$CONDITION" != "tuned" ]]; then
  echo "Invalid CONDITION: $CONDITION (use baseline or tuned)"
  exit 2
fi

cd "$REPO_ROOT"
source ~/.bashrc || true
conda activate "$ENV_NAME"

COND_RUN_ID="${RUN_ID}__${CONDITION}"
CTX_ROOT="artifacts/contexts/${RUN_ID}/${CONDITION}"
PREDS_PATH="artifacts/preds/${RUN_ID}/${CONDITION}/preds.jsonl"

echo "============================================================"
echo "Verified Mini — $CONDITION"
echo "============================================================"
echo "Model: $MODEL_NAME   Runner: $RUNNER"
echo "Run ID: $COND_RUN_ID"
echo "IDs: $IDS_FILE"
echo

if [[ "$EVAL_ONLY" != "1" ]]; then
  # Build context
  echo "[1/3] Building $CONDITION contexts..."
  python scripts/build_context.py \
    --dataset_name "$DATASET_NAME" --split "$SPLIT" \
    --instance_ids_file "$IDS_FILE" \
    --mode "$CONDITION" --contexts_root "$CTX_ROOT"

  # Inference
  echo "[2/3] Running inference..."
  python scripts/run_inference.py \
    --dataset_name "$DATASET_NAME" --split "$SPLIT" \
    --instance_ids_file "$IDS_FILE" \
    --model "$MODEL_NAME" --ablation "$CONDITION" \
    --runner "$RUNNER" \
    --timeout_s "$TIMEOUT_S" --step_limit "$STEP_LIMIT" \
    --contexts_root "$CTX_ROOT" \
    --run_id "$COND_RUN_ID" --out "$PREDS_PATH"
fi

# Evaluation
if [[ ! -f "$PREDS_PATH" ]]; then
  echo "Missing predictions: $PREDS_PATH"
  exit 3
fi

echo "[3/3] Evaluating..."
bash scripts/run_swebench_eval.sh \
  "$DATASET_NAME" "$PREDS_PATH" "$COND_RUN_ID" "$MAX_WORKERS_EVAL"

echo
echo "Done. Results: results/$COND_RUN_ID/"
