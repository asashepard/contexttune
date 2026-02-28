#!/usr/bin/env bash
# Run SWE-bench evaluation with a local OpenAI-compatible model.
#
# One-command flow:
#   1) Build required Docker images for the requested instance IDs
#   2) Run inference with mini_swe_agent_swebench (baseline)
#   3) Run SWE-bench harness evaluation
#   4) Print patch + resolve summary
#
# Optional:
#   - If --guidance-dir is provided, also runs tuned condition.
#
# Example (4-instance smoke):
#   MODEL_NAME=openai/Qwen2.5-Coder-32B-Instruct \
#   OPENAI_BASE_URL=http://127.0.0.1:8000/v1 OPENAI_API_KEY=dummy \
#   bash scripts/run_local_model_eval.sh \
#     --ids-file scripts/verified_smoke_4_ids.txt --exp-id smoke_local
#
# Example (50-instance verified mini):
#   MODEL_NAME=openai/Qwen2.5-Coder-32B-Instruct \
#   OPENAI_BASE_URL=http://127.0.0.1:8000/v1 OPENAI_API_KEY=dummy \
#   bash scripts/run_local_model_eval.sh \
#     --ids-file scripts/verified_mini_ids.txt --exp-id verified_mini_local

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

DATASET_NAME="princeton-nlp/SWE-bench_Verified"
SPLIT="test"
IDS_FILE=""
EXP_ID=""
MAX_WORKERS="4"
TIMEOUT_S="900"
STEP_LIMIT="30"
GUIDANCE_DIR=""

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_local_model_eval.sh --ids-file <path> [options]

Required:
  --ids-file <path>         Instance ID list file.

Optional:
  --exp-id <id>             Experiment ID (default: local_eval_<timestamp>)
  --max-workers <int>       Workers for image build/eval (default: 4)
  --timeout-s <int>         Per-instance timeout for agent runner (default: 900)
  --step-limit <int>        Agent step limit (default: 30)
  --guidance-dir <path>     If set, also run tuned condition with this guidance dir
  --dataset-name <name>     Dataset name (default: princeton-nlp/SWE-bench_Verified)
  --split <name>            Dataset split (default: test)

Required env vars:
  MODEL_NAME
  OPENAI_BASE_URL
  OPENAI_API_KEY
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ids-file)
      IDS_FILE="$2"
      shift 2
      ;;
    --exp-id)
      EXP_ID="$2"
      shift 2
      ;;
    --max-workers)
      MAX_WORKERS="$2"
      shift 2
      ;;
    --timeout-s)
      TIMEOUT_S="$2"
      shift 2
      ;;
    --step-limit)
      STEP_LIMIT="$2"
      shift 2
      ;;
    --guidance-dir)
      GUIDANCE_DIR="$2"
      shift 2
      ;;
    --dataset-name)
      DATASET_NAME="$2"
      shift 2
      ;;
    --split)
      SPLIT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$IDS_FILE" ]]; then
  echo "ERROR: --ids-file is required"
  usage
  exit 1
fi

if [[ ! -f "$IDS_FILE" ]]; then
  echo "ERROR: ids file not found: $IDS_FILE"
  exit 1
fi

if [[ -z "${MODEL_NAME:-}" ]]; then
  echo "ERROR: MODEL_NAME is not set"
  exit 1
fi
if [[ -z "${OPENAI_BASE_URL:-}" ]]; then
  echo "ERROR: OPENAI_BASE_URL is not set"
  exit 1
fi
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY is not set"
  exit 1
fi

if [[ -z "$EXP_ID" ]]; then
  EXP_ID="local_eval_$(date +%Y%m%d_%H%M%S)"
fi

echo "============================================================"
echo "Local-model SWE-bench eval"
echo "Project:     $PROJECT_ROOT"
echo "Experiment:  $EXP_ID"
echo "Model:       $MODEL_NAME"
echo "Endpoint:    $OPENAI_BASE_URL"
echo "IDs file:    $IDS_FILE"
echo "Dataset:     $DATASET_NAME ($SPLIT)"
echo "Workers:     $MAX_WORKERS"
echo "Timeout/step: ${TIMEOUT_S}s / ${STEP_LIMIT}"
echo "============================================================"

echo
echo "[1/4] Building Docker images for requested instances..."
python scripts/build_docker_images.py \
  --instance_ids_file "$IDS_FILE" \
  --dataset_name "$DATASET_NAME" \
  --split "$SPLIT" \
  --max_workers "$MAX_WORKERS"

mkdir -p artifacts/contexts/_empty

run_condition () {
  local condition="$1"
  local run_id="${EXP_ID}__${condition}"
  local out_path="artifacts/preds/${EXP_ID}/${condition}/preds.jsonl"

  echo
  echo "[2/4] Inference ($condition)..."
  if [[ "$condition" == "baseline" ]]; then
    python scripts/run_inference.py \
      --dataset_name "$DATASET_NAME" \
      --split "$SPLIT" \
      --instance_ids_file "$IDS_FILE" \
      --model "$MODEL_NAME" \
      --runner mini_swe_agent_swebench \
      --ablation baseline \
      --contexts_root artifacts/contexts/_empty \
      --timeout_s "$TIMEOUT_S" \
      --step_limit "$STEP_LIMIT" \
      --run_id "$run_id" \
      --out "$out_path"
  else
    python scripts/run_inference.py \
      --dataset_name "$DATASET_NAME" \
      --split "$SPLIT" \
      --instance_ids_file "$IDS_FILE" \
      --model "$MODEL_NAME" \
      --runner mini_swe_agent_swebench \
      --ablation tuned \
      --guidance_dir "$GUIDANCE_DIR" \
      --timeout_s "$TIMEOUT_S" \
      --step_limit "$STEP_LIMIT" \
      --run_id "$run_id" \
      --out "$out_path"
  fi

  echo
  echo "[3/4] SWE-bench harness eval ($condition)..."
  bash scripts/run_swebench_eval.sh "$DATASET_NAME" "$out_path" "$run_id" "$MAX_WORKERS"
}

run_condition baseline

if [[ -n "$GUIDANCE_DIR" ]]; then
  if [[ ! -d "$GUIDANCE_DIR" ]]; then
    echo "ERROR: guidance dir not found: $GUIDANCE_DIR"
    exit 1
  fi
  run_condition tuned
fi

echo
echo "[4/4] Final summary"
GUIDANCE_ENABLED="0"
if [[ -n "$GUIDANCE_DIR" ]]; then
  GUIDANCE_ENABLED="1"
fi

python - <<PY
import json
from pathlib import Path
from context_policy.report.summarize import load_results, compute_rate

exp_id = ${EXP_ID@Q}
conditions = ["baseline"]
if ${GUIDANCE_ENABLED} == 1:
    conditions.append("tuned")

for cond in conditions:
    preds = Path(f"artifacts/preds/{exp_id}/{cond}/preds.jsonl")
    rows = [json.loads(x) for x in preds.read_text(encoding="utf-8").splitlines() if x.strip()] if preds.exists() else []
    nonempty = sum(1 for r in rows if (r.get("model_patch") or "").strip())
    resolved, total = load_results(Path(f"results/{exp_id}__{cond}"))
    rate = compute_rate(resolved, total)
    print(f"{cond}: rows={len(rows)} nonempty_patches={nonempty} resolved={resolved} total={total} rate={rate:.4f}")

print(f"\nExperiment complete: {exp_id}")
PY
