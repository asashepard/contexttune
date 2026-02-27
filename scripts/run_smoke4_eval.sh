#!/usr/bin/env bash
set -euo pipefail

# One-command standardized smoke run for EC2 or gpmoo login/compute shells.
# Runs: build_signals -> build_context -> inference -> evaluation
#
# Example:
#   bash scripts/run_smoke4_eval.sh --model openai/gpt-5.2
#   bash scripts/run_smoke4_eval.sh --model openai/gpt-5.2 --conditions baseline_context,no_context

MODEL=""
IDS_FILE="scripts/easy_4_ids.txt"
DATASET_NAME="princeton-nlp/SWE-bench_Verified"
SPLIT="test"
RUNNER="mini_swe_agent_swebench"
TIMEOUT_S=1200
STEP_LIMIT=50
MAX_WORKERS_EVAL=1
CONDITIONS="baseline_context"
RUN_TAG="smoke4"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --ids) IDS_FILE="$2"; shift 2 ;;
    --dataset) DATASET_NAME="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --runner) RUNNER="$2"; shift 2 ;;
    --timeout) TIMEOUT_S="$2"; shift 2 ;;
    --steps) STEP_LIMIT="$2"; shift 2 ;;
    --workers) MAX_WORKERS_EVAL="$2"; shift 2 ;;
    --conditions) CONDITIONS="$2"; shift 2 ;;
    --run-tag) RUN_TAG="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "Usage: bash scripts/run_smoke4_eval.sh --model openai/<model> [--conditions baseline_context,no_context]"
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ ! -f "$IDS_FILE" ]]; then
  echo "IDs file not found: $IDS_FILE"
  exit 3
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set"
  exit 4
fi

docker info >/dev/null 2>&1 || { echo "Docker daemon not reachable"; exit 5; }

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID_BASE="${RUN_TAG}_${RUN_STAMP}"

echo "============================================================"
echo "Smoke4 standardized run"
echo "============================================================"
echo "Model: $MODEL"
echo "IDs: $IDS_FILE"
echo "Conditions: $CONDITIONS"
echo "Runner: $RUNNER"
echo "Run base: $RUN_ID_BASE"
echo

echo "[1/4] Building signals"
python scripts/build_signals.py \
  --dataset_name "$DATASET_NAME" \
  --split "$SPLIT" \
  --instance_ids_file "$IDS_FILE"

echo "[2/4] Building context"
python scripts/build_context.py \
  --dataset_name "$DATASET_NAME" \
  --split "$SPLIT" \
  --instance_ids_file "$IDS_FILE"

IFS=',' read -r -a CONDITION_LIST <<< "$CONDITIONS"

for CONDITION in "${CONDITION_LIST[@]}"; do
  CONDITION="$(echo "$CONDITION" | xargs)"
  if [[ "$CONDITION" != "baseline_context" && "$CONDITION" != "no_context" ]]; then
    echo "Invalid condition: $CONDITION"
    exit 6
  fi

  COND_RUN_ID="${RUN_ID_BASE}__${CONDITION}"
  PREDS_PATH="artifacts/preds/${RUN_ID_BASE}/${CONDITION}/preds.jsonl"

  echo "[3/4][$CONDITION] Inference"
  python scripts/run_inference.py \
    --dataset_name "$DATASET_NAME" \
    --split "$SPLIT" \
    --instance_ids_file "$IDS_FILE" \
    --model "$MODEL" \
    --ablation "$CONDITION" \
    --runner "$RUNNER" \
    --timeout_s "$TIMEOUT_S" \
    --step_limit "$STEP_LIMIT" \
    --run_id "$COND_RUN_ID" \
    --out "$PREDS_PATH"

  echo "[4/4][$CONDITION] Evaluation"
  bash scripts/run_swebench_eval.sh \
    "$DATASET_NAME" \
    "$PREDS_PATH" \
    "$COND_RUN_ID" \
    "$MAX_WORKERS_EVAL"
done

echo
echo "Done."
echo "Run base: $RUN_ID_BASE"
echo "Predictions root: artifacts/preds/$RUN_ID_BASE"
echo "Results roots: results/${RUN_ID_BASE}__baseline_context and/or results/${RUN_ID_BASE}__no_context"
