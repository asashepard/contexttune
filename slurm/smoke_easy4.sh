#!/usr/bin/env bash
#SBATCH -J ct_easy4
#SBATCH -p gpmoo-b
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -t 0-03:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
# ──────────────────────────────────────────────────────────────────────
# GPMoo easy-4 smoke test — quick validation of the full pipeline.
#
# Submit:
#   MODEL_NAME=openai/gpt-4o sbatch slurm/smoke_easy4.sh
#
# Optional env vars:
#   RUNNER, TIMEOUT_S, STEP_LIMIT, CONDITIONS, MAX_WORKERS_EVAL
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
ENV_NAME="${ENV_NAME:-contexttune-py311}"
MODEL_NAME="${MODEL_NAME:?set MODEL_NAME, e.g. openai/gpt-4o}"
DATASET_NAME="${DATASET_NAME:-princeton-nlp/SWE-bench_Verified}"
SPLIT="${SPLIT:-test}"
IDS_FILE="${IDS_FILE:-scripts/easy_4_ids.txt}"
RUNNER="${RUNNER:-mini_swe_agent_swebench}"
TIMEOUT_S="${TIMEOUT_S:-600}"
STEP_LIMIT="${STEP_LIMIT:-30}"
MAX_WORKERS_EVAL="${MAX_WORKERS_EVAL:-1}"
CONDITIONS="${CONDITIONS:-baseline,tuned}"
RUN_TAG="${RUN_TAG:-gpmoo_easy4}"

cd "$REPO_ROOT"
source ~/.bashrc || true
conda activate "$ENV_NAME"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${RUN_TAG}_${RUN_STAMP}"

echo "============================================================"
echo "GPMoo Easy-4 Smoke Test"
echo "============================================================"
echo "Model: $MODEL_NAME   Runner: $RUNNER"
echo "Run ID: $RUN_ID"
echo "Conditions: $CONDITIONS"
echo

IFS=',' read -r -a COND_LIST <<< "$CONDITIONS"

# Build contexts
for COND in "${COND_LIST[@]}"; do
  COND="$(echo "$COND" | xargs)"
  CTX_ROOT="artifacts/contexts/${RUN_ID}/${COND}"
  echo "[ctx] Building $COND contexts..."
  python scripts/build_context.py \
    --dataset_name "$DATASET_NAME" --split "$SPLIT" \
    --instance_ids_file "$IDS_FILE" \
    --mode "$COND" --contexts_root "$CTX_ROOT"
done

# Inference + evaluation per condition
for COND in "${COND_LIST[@]}"; do
  COND="$(echo "$COND" | xargs)"
  COND_RUN_ID="${RUN_ID}__${COND}"
  CTX_ROOT="artifacts/contexts/${RUN_ID}/${COND}"
  PREDS_PATH="artifacts/preds/${RUN_ID}/${COND}/preds.jsonl"

  echo "[infer] $COND ..."
  python scripts/run_inference.py \
    --dataset_name "$DATASET_NAME" --split "$SPLIT" \
    --instance_ids_file "$IDS_FILE" \
    --model "$MODEL_NAME" --ablation "$COND" \
    --runner "$RUNNER" \
    --timeout_s "$TIMEOUT_S" --step_limit "$STEP_LIMIT" \
    --contexts_root "$CTX_ROOT" \
    --run_id "$COND_RUN_ID" --out "$PREDS_PATH"

  echo "[eval] $COND ..."
  bash scripts/run_swebench_eval.sh \
    "$DATASET_NAME" "$PREDS_PATH" "$COND_RUN_ID" "$MAX_WORKERS_EVAL"
done

echo
echo "Done. Run ID: $RUN_ID"
echo "Results: results/${RUN_ID}__baseline  results/${RUN_ID}__tuned"
