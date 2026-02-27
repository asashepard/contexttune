#!/usr/bin/env bash
set -euo pipefail
# ──────────────────────────────────────────────────────────────────────
# run_ec2_smoke.sh — One-command EC2 smoke test for SWE-bench agent
#
# Prerequisites:
#   - EC2 bootstrapped (scripts/ec2_bootstrap.sh)
#   - Docker running, user in docker group
#   - .env file at repo root OR OPENAI_* env vars exported
#
# Usage:
#   bash scripts/run_ec2_smoke.sh --model openai/gpt-5.2-codex
#   bash scripts/run_ec2_smoke.sh --model openai/gpt-4o --ids scripts/smoke_3_ids.txt
# ──────────────────────────────────────────────────────────────────────

# Defaults
MODEL=""
IDS_FILE="scripts/smoke_3_ids.txt"
TIMEOUT_S=180
STEP_LIMIT=30
DATASET="princeton-nlp/SWE-bench_Verified"

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)    MODEL="$2"; shift 2 ;;
    --ids)      IDS_FILE="$2"; shift 2 ;;
    --timeout)  TIMEOUT_S="$2"; shift 2 ;;
    --steps)    STEP_LIMIT="$2"; shift 2 ;;
    *)          echo "Unknown arg: $1"; exit 2 ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "Usage: bash scripts/run_ec2_smoke.sh --model openai/<model_name>"
  exit 2
fi

# ── Resolve project root ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# ── Activate venv ──
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# ── Load .env if present ──
if [[ -f .env ]]; then
  echo "[env] Loading .env"
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# Suppress litellm info spam
export LITELLM_LOG="${LITELLM_LOG:-ERROR}"

# ── Validate prerequisites ──
echo "============================================================"
echo "EC2 Smoke Test"
echo "============================================================"
echo "  Model:     $MODEL"
echo "  IDs file:  $IDS_FILE"
echo "  Timeout:   ${TIMEOUT_S}s per instance"
echo "  Max steps: $STEP_LIMIT"
echo ""

echo "[1/5] Checking prerequisites..."
python -c "from minisweagent.agents.default import DefaultAgent" 2>/dev/null \
  && echo "  ✓ mini-swe-agent" \
  || { echo "  ✗ mini-swe-agent not installed"; exit 1; }
python -c "import docker" 2>/dev/null \
  && echo "  ✓ docker SDK" \
  || { echo "  ✗ docker SDK not installed"; exit 1; }
docker info >/dev/null 2>&1 \
  && echo "  ✓ Docker daemon" \
  || { echo "  ✗ Docker daemon not reachable (re-login for group?)"; exit 1; }
[[ -n "${OPENAI_API_KEY:-}" ]] \
  && echo "  ✓ OPENAI_API_KEY set" \
  || { echo "  ✗ OPENAI_API_KEY not set"; exit 1; }
echo ""

# ── Build context ──
echo "[2/5] Building baseline context (skip if exist)..."
python scripts/build_context.py --instance_ids_file "$IDS_FILE" --mode baseline
echo ""

# ── Build Docker images ──
echo "[3/5] Building Docker images (skip if exist)..."
python scripts/build_docker_images.py --instance_ids_file "$IDS_FILE"
echo ""

# ── Run inference ──
echo "[4/5] Running inference..."
python scripts/run_inference.py \
  --model "$MODEL" \
  --runner mini_swe_agent_swebench \
  --ablation baseline \
  --instance_ids_file "$IDS_FILE" \
  --timeout_s "$TIMEOUT_S" \
  --step_limit "$STEP_LIMIT"
echo ""

# ── Find run ID and evaluate ──
RUN_ID=$(ls -t artifacts/preds/ | head -1)
PREDS_PATH="artifacts/preds/$RUN_ID/preds.jsonl"

echo "[5/5] Running SWE-bench evaluation..."
echo "  Run ID: $RUN_ID"
echo "  Preds:  $PREDS_PATH"

PRED_COUNT=$(wc -l < "$PREDS_PATH")
echo "  Records: $PRED_COUNT"

NON_EMPTY=$(python -c "
import json, sys
count = 0
for line in open('$PREDS_PATH'):
    r = json.loads(line)
    if r.get('model_patch', '').strip():
        count += 1
print(count)
")
echo "  Non-empty patches: $NON_EMPTY / $PRED_COUNT"

bash scripts/run_swebench_eval.sh \
  "$DATASET" \
  "$PREDS_PATH" \
  "$RUN_ID" \
  1

# ── Summary ──
echo ""
echo "============================================================"
echo "DONE"
echo "============================================================"
echo "  Predictions: $PREDS_PATH"
RESULTS_FILE="results/$RUN_ID/results.json"
if [[ -f "$RESULTS_FILE" ]]; then
  echo "  Results: $RESULTS_FILE"
  python -c "
import json
with open('$RESULTS_FILE') as f:
    data = json.load(f)
resolved = data.get('resolved', [])
applied = data.get('applied', [])
print(f'  Resolved: {len(resolved)} / $PRED_COUNT')
print(f'  Applied:  {len(applied)} / $PRED_COUNT')
if resolved:
    for r in resolved:
        print(f'    ✓ {r}')
"
else
  echo "  Results file not found at $RESULTS_FILE"
  echo "  Check logs in results/$RUN_ID/"
fi
