#!/usr/bin/env bash
set -euo pipefail
# ──────────────────────────────────────────────────────────────────────
# GPMoo SWE-bench Verified Full — sharded array jobs.
#
# This script has two modes:
#   1. SUBMIT mode (run from head node): splits IDs into shards, submits
#      Slurm array jobs for inference + eval per condition.
#   2. WORKER mode (run inside Slurm): picks its shard and runs inference
#      + eval for one condition.
#
# ── Submit ────────────────────────────────────────────────────────────
#   MODEL_NAME=openai/gpt-4o RUN_ID=vfull_001 \
#     bash slurm/verified_full.sh submit [--shards 10]
#
# ── Worker (called by Slurm, not manually) ────────────────────────────
#   Automatically invoked via sbatch array.
#
# Required env vars: MODEL_NAME, RUN_ID
# Optional: DATASET_NAME, RUNNER, TIMEOUT_S, STEP_LIMIT, PARTITION,
#           N_SHARDS, MAX_WORKERS_EVAL
# ──────────────────────────────────────────────────────────────────────

MODEL_NAME="${MODEL_NAME:?set MODEL_NAME}"
RUN_ID="${RUN_ID:?set RUN_ID}"
DATASET_NAME="${DATASET_NAME:-princeton-nlp/SWE-bench_Verified}"
SPLIT="${SPLIT:-test}"
RUNNER="${RUNNER:-mini_swe_agent_swebench}"
TIMEOUT_S="${TIMEOUT_S:-600}"
STEP_LIMIT="${STEP_LIMIT:-30}"
MAX_WORKERS_EVAL="${MAX_WORKERS_EVAL:-1}"
PARTITION="${PARTITION:-gpmoo-b}"
N_SHARDS="${N_SHARDS:-10}"
ENV_NAME="${ENV_NAME:-contexttune-py311}"

# ======================================================================
# Submit mode
# ======================================================================
if [[ "${1:-}" == "submit" ]]; then
  shift
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --shards) N_SHARDS="$2"; shift 2 ;;
      *) echo "Unknown submit arg: $1"; exit 2 ;;
    esac
  done

  SHARD_DIR="artifacts/shards/${RUN_ID}"
  mkdir -p "$SHARD_DIR"

  # ── Download all instance IDs from the dataset ──
  ALL_IDS="$SHARD_DIR/all_ids.txt"
  if [[ ! -f "$ALL_IDS" ]]; then
    echo "Fetching instance IDs from $DATASET_NAME..."
    python -c "
from context_policy.datasets.swebench import load_instances
instances = load_instances(dataset_name='$DATASET_NAME', split='$SPLIT')
for inst in instances:
    print(inst['instance_id'])
" > "$ALL_IDS"
  fi

  TOTAL=$(wc -l < "$ALL_IDS")
  echo "Total instances: $TOTAL, Shards: $N_SHARDS"

  # ── Round-robin split into shards ──
  for i in $(seq 0 $((N_SHARDS - 1))); do
    awk "NR % $N_SHARDS == $i" "$ALL_IDS" > "$SHARD_DIR/shard_$(printf '%03d' $i).txt"
  done
  echo "Shards written to $SHARD_DIR/"

  ARRAY_MAX=$((N_SHARDS - 1))
  RESULTS_DIR="results/${RUN_ID}"
  mkdir -p "$RESULTS_DIR"
  JOB_MANIFEST="$RESULTS_DIR/slurm_jobs.tsv"
  echo -e "condition\tinfer_eval_job_id\tarray\tshard_dir" > "$JOB_MANIFEST"

  # ── Submit array job per condition ──
  for CONDITION in baseline tuned; do
    JOB_ID="$(sbatch --parsable \
      --array "0-${ARRAY_MAX}" \
      --partition "$PARTITION" \
      --job-name "ct_vfull_${CONDITION}" \
      --cpus-per-task 16 --mem 64G --gres gpu:1 \
      --time 0-14:00 \
      --output "slurm-%A_%a.out" \
      --error "slurm-%A_%a.err" \
      --export=ALL,MODEL_NAME="$MODEL_NAME",RUN_ID="$RUN_ID",CONDITION="$CONDITION",SHARD_DIR="$SHARD_DIR",DATASET_NAME="$DATASET_NAME",SPLIT="$SPLIT",RUNNER="$RUNNER",TIMEOUT_S="$TIMEOUT_S",STEP_LIMIT="$STEP_LIMIT",MAX_WORKERS_EVAL="$MAX_WORKERS_EVAL" \
      slurm/verified_full.sh)"

    echo "$CONDITION: job $JOB_ID (array 0-${ARRAY_MAX})"
    echo -e "${CONDITION}\t${JOB_ID}\t0-${ARRAY_MAX}\t${SHARD_DIR}" >> "$JOB_MANIFEST"
  done

  echo "Job manifest: $JOB_MANIFEST"
  echo "Track with: squeue -u \$USER"
  exit 0
fi

# ======================================================================
# Worker mode (inside Slurm array job)
# ======================================================================
CONDITION="${CONDITION:?set CONDITION=baseline|tuned}"
SHARD_DIR="${SHARD_DIR:?set SHARD_DIR}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "Not in a Slurm array job. Use 'submit' mode from head node."
  exit 2
fi

if [[ "$CONDITION" != "baseline" && "$CONDITION" != "tuned" ]]; then
  echo "Invalid CONDITION: $CONDITION"
  exit 2
fi

cd "${REPO_ROOT:-$PWD}"
source ~/.bashrc || true
conda activate "$ENV_NAME"

# Pick shard file
mapfile -t SHARD_FILES < <(find "$SHARD_DIR" -maxdepth 1 -name 'shard_*.txt' -type f | sort)
TASK_ID="$SLURM_ARRAY_TASK_ID"

if (( TASK_ID >= ${#SHARD_FILES[@]} )); then
  echo "Array index $TASK_ID out of bounds (${#SHARD_FILES[@]} shards)"
  exit 3
fi

IDS_FILE="${SHARD_FILES[$TASK_ID]}"
SHARD_NAME="$(basename "$IDS_FILE" .txt)"
COND_RUN_ID="${RUN_ID}__${CONDITION}__${SHARD_NAME}"
CTX_ROOT="artifacts/contexts/${RUN_ID}/${CONDITION}"
PREDS_PATH="artifacts/preds/${RUN_ID}/${CONDITION}/${SHARD_NAME}.jsonl"

echo "task=$TASK_ID shard=$IDS_FILE condition=$CONDITION"

# Context
python scripts/build_context.py \
  --dataset_name "$DATASET_NAME" --split "$SPLIT" \
  --instance_ids_file "$IDS_FILE" \
  --mode "$CONDITION" --contexts_root "$CTX_ROOT"

# Inference
python scripts/run_inference.py \
  --dataset_name "$DATASET_NAME" --split "$SPLIT" \
  --instance_ids_file "$IDS_FILE" \
  --model "$MODEL_NAME" --ablation "$CONDITION" \
  --runner "$RUNNER" \
  --timeout_s "$TIMEOUT_S" --step_limit "$STEP_LIMIT" \
  --contexts_root "$CTX_ROOT" \
  --run_id "$COND_RUN_ID" --out "$PREDS_PATH"

# Evaluation
bash scripts/run_swebench_eval.sh \
  "$DATASET_NAME" "$PREDS_PATH" "$COND_RUN_ID" "$MAX_WORKERS_EVAL"

echo "Done: $COND_RUN_ID"
