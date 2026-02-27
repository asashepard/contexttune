#!/usr/bin/env bash
set -euo pipefail
# ──────────────────────────────────────────────────────────────────────
# ec2_smoke.sh — End-to-end EC2 smoke test (easy-4 instances)
#
# Prerequisites:
#   - EC2 bootstrapped (scripts/ec2_bootstrap.sh)
#   - Docker running, user in docker group
#   - .env file at repo root OR OPENAI_* env vars exported
#
# Usage:
#   bash scripts/ec2_smoke.sh --model openai/gpt-4o
#   bash scripts/ec2_smoke.sh --model openai/gpt-4o --ids scripts/easy_4_ids.txt
#   bash scripts/ec2_smoke.sh --model openai/gpt-4o --conditions baseline,tuned
# ──────────────────────────────────────────────────────────────────────

MODEL=""
IDS_FILE="scripts/easy_4_ids.txt"
DATASET="princeton-nlp/SWE-bench_Verified"
SPLIT="test"
RUNNER="mini_swe_agent_swebench"
TIMEOUT_S=600
STEP_LIMIT=30
MAX_WORKERS_EVAL=1
CONDITIONS="baseline,tuned"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)      MODEL="$2"; shift 2 ;;
    --ids)        IDS_FILE="$2"; shift 2 ;;
    --dataset)    DATASET="$2"; shift 2 ;;
    --runner)     RUNNER="$2"; shift 2 ;;
    --timeout)    TIMEOUT_S="$2"; shift 2 ;;
    --steps)      STEP_LIMIT="$2"; shift 2 ;;
    --workers)    MAX_WORKERS_EVAL="$2"; shift 2 ;;
    --conditions) CONDITIONS="$2"; shift 2 ;;
    *)            echo "Unknown arg: $1"; exit 2 ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "Usage: bash scripts/ec2_smoke.sh --model openai/<model>"
  exit 2
fi

# ── Resolve project root ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# ── Activate venv / conda ──
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# ── Load .env if present ──
if [[ -f .env ]]; then
  set -a; source .env; set +a
fi

export LITELLM_LOG="${LITELLM_LOG:-ERROR}"

# ── Validate prerequisites ──
echo "============================================================"
echo "EC2 Smoke Test"
echo "============================================================"
echo "  Model:      $MODEL"
echo "  IDs file:   $IDS_FILE"
echo "  Conditions: $CONDITIONS"
echo "  Runner:     $RUNNER"
echo "  Timeout:    ${TIMEOUT_S}s"
echo "  Steps:      $STEP_LIMIT"
echo

echo "[preflight] Checking prerequisites..."
python -c "from context_policy.utils.paths import PROJECT_ROOT; print('  project:', PROJECT_ROOT)"
docker info >/dev/null 2>&1 && echo "  docker: ok" || { echo "  docker: FAIL"; exit 1; }
[[ -n "${OPENAI_API_KEY:-}" ]] && echo "  api key: set" || { echo "  api key: NOT SET"; exit 1; }
echo

# ── Build Docker images ──
echo "[1/4] Building Docker images (skip if exist)..."
python scripts/build_docker_images.py --instance_ids_file "$IDS_FILE"
echo

# ── Prepare run ID ──
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="ec2_smoke_${RUN_STAMP}"

# ── Build contexts for each condition ──
IFS=',' read -r -a COND_LIST <<< "$CONDITIONS"

echo "[2/4] Building contexts..."
for COND in "${COND_LIST[@]}"; do
  COND="$(echo "$COND" | xargs)"
  CTX_ROOT="artifacts/contexts/${RUN_ID}/${COND}"
  python scripts/build_context.py \
    --dataset_name "$DATASET" --split "$SPLIT" \
    --instance_ids_file "$IDS_FILE" \
    --mode "$COND" --contexts_root "$CTX_ROOT"
done
echo

# ── Inference + evaluation per condition ──
for COND in "${COND_LIST[@]}"; do
  COND="$(echo "$COND" | xargs)"
  COND_RUN_ID="${RUN_ID}__${COND}"
  CTX_ROOT="artifacts/contexts/${RUN_ID}/${COND}"
  PREDS_PATH="artifacts/preds/${RUN_ID}/${COND}/preds.jsonl"

  echo "[3/4][$COND] Inference..."
  python scripts/run_inference.py \
    --dataset_name "$DATASET" --split "$SPLIT" \
    --instance_ids_file "$IDS_FILE" \
    --model "$MODEL" --ablation "$COND" \
    --runner "$RUNNER" \
    --timeout_s "$TIMEOUT_S" --step_limit "$STEP_LIMIT" \
    --contexts_root "$CTX_ROOT" \
    --run_id "$COND_RUN_ID" --out "$PREDS_PATH"

  echo "[4/4][$COND] Evaluation..."
  bash scripts/run_swebench_eval.sh "$DATASET" "$PREDS_PATH" "$COND_RUN_ID" "$MAX_WORKERS_EVAL"
done

# ── Summary ──
echo
echo "============================================================"
echo "DONE — $RUN_ID"
echo "============================================================"
echo "Predictions: artifacts/preds/$RUN_ID/"

for COND in "${COND_LIST[@]}"; do
  COND="$(echo "$COND" | xargs)"
  RESULTS_FILE="results/${RUN_ID}__${COND}/results.json"
  if [[ -f "$RESULTS_FILE" ]]; then
    python -c "
import json
with open('$RESULTS_FILE') as f:
    d = json.load(f)
r = d.get('resolved', [])
a = d.get('applied', [])
print(f'  $COND: {len(r)} resolved, {len(a)} applied')
for x in r:
    print(f'    ✓ {x}')
"
  else
    echo "  $COND: results not found (check logs)"
  fi
done
