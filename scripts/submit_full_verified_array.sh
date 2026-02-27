#!/usr/bin/env bash
set -euo pipefail

# Submit full-Verified shard array jobs for both conditions.
#
# Required env vars:
#   MODEL_NAME, RUN_ID, SHARD_DIR
#
# Optional env vars:
#   DATASET_NAME, SPLIT, RUNNER, TIMEOUT_S, STEP_LIMIT,
#   MAX_WORKERS_EVAL, INFER_PARTITION, EVAL_PARTITION,
#   INFER_TIME, EVAL_TIME

MODEL_NAME="${MODEL_NAME:?set MODEL_NAME, e.g. openai/<model>}"
RUN_ID="${RUN_ID:?set RUN_ID, e.g. verified_full_gpmoo_001}"
SHARD_DIR="${SHARD_DIR:?set SHARD_DIR, e.g. artifacts/shards/full_verified}"

DATASET_NAME="${DATASET_NAME:-princeton-nlp/SWE-bench_Verified}"
SPLIT="${SPLIT:-test}"
RUNNER="${RUNNER:-mini_swe_agent_swebench}"
TIMEOUT_S="${TIMEOUT_S:-600}"
STEP_LIMIT="${STEP_LIMIT:-30}"
MAX_WORKERS_EVAL="${MAX_WORKERS_EVAL:-1}"

INFER_PARTITION="${INFER_PARTITION:-gpmoo-b}"
EVAL_PARTITION="${EVAL_PARTITION:-gpmoo-b}"
INFER_TIME="${INFER_TIME:-0-12:00}"
EVAL_TIME="${EVAL_TIME:-0-08:00}"

if [[ ! -d "$SHARD_DIR" ]]; then
  echo "Shard directory not found: $SHARD_DIR"
  exit 2
fi

mapfile -t SHARD_FILES < <(find "$SHARD_DIR" -maxdepth 1 -type f -name '*.txt' | sort)
SHARD_COUNT="${#SHARD_FILES[@]}"
if [[ "$SHARD_COUNT" -eq 0 ]]; then
  echo "No shard files found in $SHARD_DIR"
  exit 3
fi

ARRAY_SPEC="0-$((SHARD_COUNT - 1))"
RESULTS_DIR="results/${RUN_ID}"
mkdir -p "$RESULTS_DIR"
JOB_MANIFEST="$RESULTS_DIR/slurm_jobs.tsv"

echo -e "condition\tinfer_job_id\teval_job_id\tarray\tshard_dir" > "$JOB_MANIFEST"

echo "Submitting full-verified arrays"
echo "run_id=$RUN_ID shards=$SHARD_COUNT array=$ARRAY_SPEC"

for CONDITION in no_context baseline_context; do
  INFER_JOB_ID="$(sbatch --parsable \
    --array "$ARRAY_SPEC" \
    --partition "$INFER_PARTITION" \
    --time "$INFER_TIME" \
    --export=ALL,MODEL_NAME="$MODEL_NAME",RUN_ID="$RUN_ID",CONDITION="$CONDITION",SHARD_DIR="$SHARD_DIR",DATASET_NAME="$DATASET_NAME",SPLIT="$SPLIT",RUNNER="$RUNNER",TIMEOUT_S="$TIMEOUT_S",STEP_LIMIT="$STEP_LIMIT" \
    slurm/run_full_condition_array.sh)"

  EVAL_JOB_ID="$(sbatch --parsable \
    --dependency "afterok:${INFER_JOB_ID}" \
    --array "$ARRAY_SPEC" \
    --partition "$EVAL_PARTITION" \
    --time "$EVAL_TIME" \
    --export=ALL,RUN_ID="$RUN_ID",CONDITION="$CONDITION",SHARD_DIR="$SHARD_DIR",DATASET_NAME="$DATASET_NAME",MAX_WORKERS_EVAL="$MAX_WORKERS_EVAL" \
    slurm/run_full_eval_array.sh)"

  echo "$CONDITION infer=$INFER_JOB_ID eval=$EVAL_JOB_ID"
  echo -e "${CONDITION}\t${INFER_JOB_ID}\t${EVAL_JOB_ID}\t${ARRAY_SPEC}\t${SHARD_DIR}" >> "$JOB_MANIFEST"
done

echo "Wrote job manifest: $JOB_MANIFEST"
echo "Track jobs with: squeue -u $USER"
