#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# SWE-bench Verified evaluation — run selected conditions after tuning.
#
# Submit:
#   MODEL_NAME=openai/gpt-4o EXP_ID=exp001 \
#     sbatch slurm/eval_verified_gpmoo_a_cpu.sh
#
# Required: MODEL_NAME, EXP_ID
# Optional: REPO_CONFIG, CONDITIONS, DATASET_NAME, IDS_FILE,
#           MAX_WORKERS_EVAL, TIMEOUT_S, STEP_LIMIT, ORACLE_ITERATIONS
# ──────────────────────────────────────────────────────────────────────
#SBATCH -J ct_eval
#SBATCH -p gpmoo-a
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gres=gpu:0
#SBATCH -t 08:00:00
#SBATCH -o slurm-eval-%j.out
#SBATCH -e slurm-eval-%j.err
set -euo pipefail

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

log() {
    echo "[$(timestamp)] [eval_cpu] $*"
}

section() {
    echo
    echo "[$(timestamp)] [eval_cpu] ============================================================"
    echo "[$(timestamp)] [eval_cpu] $*"
    echo "[$(timestamp)] [eval_cpu] ============================================================"
}

run_cmd() {
    log "RUN: $*"
    "$@"
    log "DONE: $*"
}

START_TS="$(date +%s)"
log "Job bootstrap started"
log "SLURM_JOB_ID=${SLURM_JOB_ID:-<none>}"
log "SLURM_JOB_NAME=${SLURM_JOB_NAME:-<none>}"
log "SLURM_NODELIST=${SLURM_NODELIST:-<none>}"

REPO_ROOT="${REPO_ROOT:-$PWD}"
ENV_NAME="${ENV_NAME:-contexttune-py311}"
MODEL_NAME="${MODEL_NAME:?set MODEL_NAME}"
EXP_ID="${EXP_ID:?set EXP_ID}"
REPO_CONFIG="${REPO_CONFIG:-artifacts/configs/repos_12.json}"
CONDITIONS="${CONDITIONS:-no_context static_kb oracle_tuned}"
ORACLE_ITERATIONS="${ORACLE_ITERATIONS:-5}"

DATASET_NAME="${DATASET_NAME:-princeton-nlp/SWE-bench_Verified}"
IDS_FILE="${IDS_FILE:-}"
MAX_WORKERS_EVAL="${MAX_WORKERS_EVAL:-4}"
TIMEOUT_S="${TIMEOUT_S:-600}"
STEP_LIMIT="${STEP_LIMIT:-30}"

section "Step 1/6: Environment bootstrap"
log "Changing directory to REPO_ROOT=$REPO_ROOT"
cd "$REPO_ROOT"
log "Current directory: $(pwd)"
log "Initializing conda from /shared/bin/anaconda3/etc/profile.d/conda.sh"
source /shared/bin/anaconda3/etc/profile.d/conda.sh
log "Activating conda env: $ENV_NAME"
conda activate "$ENV_NAME"
log "Python executable: $(command -v python)"
log "Python version: $(python --version 2>&1)"

section "Step 2/6: Run configuration"
log "Experiment: $EXP_ID"
log "Model: $MODEL_NAME"
log "Repo config: $REPO_CONFIG"
log "Conditions: $CONDITIONS"
log "Dataset: $DATASET_NAME"
log "Instance IDs file: ${IDS_FILE:-<not set>}"
log "Max workers eval: $MAX_WORKERS_EVAL"
log "Timeout (s): $TIMEOUT_S"
log "Step limit: $STEP_LIMIT"
log "Oracle iterations: $ORACLE_ITERATIONS"

section "Step 3/6: Input validation"
if [[ ! -f "$REPO_CONFIG" ]]; then
    log "ERROR: repo config not found: $REPO_CONFIG"
    exit 1
fi
log "Repo config found: $REPO_CONFIG"

if [[ -n "$IDS_FILE" ]]; then
    if [[ ! -f "$IDS_FILE" ]]; then
        log "ERROR: IDS_FILE provided but not found: $IDS_FILE"
        exit 1
    fi
    log "Instance IDs file found: $IDS_FILE"
else
    log "No IDS_FILE provided; full split will be used"
fi

section "Step 4/6: Build dynamic flags"
IDS_FLAG=""
if [[ -n "$IDS_FILE" ]]; then
    IDS_FLAG="--eval-instance-ids $IDS_FILE"
    log "Using IDS_FLAG: $IDS_FLAG"
else
    log "IDS_FLAG is empty"
fi

if [[ -n "$IDS_FILE" ]]; then
    section "Step 4.5/6: Repo restriction preview"
    log "Computing repos represented by IDS_FILE (pre-run confirmation)"
    run_cmd python -u - "$DATASET_NAME" "$IDS_FILE" "$REPO_CONFIG" <<'PY'
import json
import sys
from pathlib import Path
from context_policy.datasets.swebench import load_instances, read_instance_ids

dataset_name = sys.argv[1]
split = "test"
ids_file = sys.argv[2]
repo_config = sys.argv[3]

ids = read_instance_ids(ids_file)
instances = load_instances(dataset_name=dataset_name, split=split, instance_ids=ids)
repos_from_ids = sorted({inst["repo"] for inst in instances})

repo_config_rows = json.loads(Path(repo_config).read_text(encoding="utf-8"))
repos_in_config = [row["repo"] for row in repo_config_rows]
effective_repos = [repo for repo in repos_in_config if repo in set(repos_from_ids)]
skipped_count = len(repos_in_config) - len(effective_repos)

print(f"[eval_cpu] Restriction preview: {len(ids)} eval IDs map to {len(repos_from_ids)} repo(s)")
print(f"[eval_cpu] Repos from IDS_FILE: {', '.join(repos_from_ids)}")
print(
    f"[eval_cpu] Effective tuning repos (in REPO_CONFIG): "
    f"{len(effective_repos)}/{len(repos_in_config)} (skipped {skipped_count})"
)
print(f"[eval_cpu] Effective tuning repo list: {', '.join(effective_repos)}")
print(f"[eval_cpu] Repo config source remains: {repo_config}")
PY
fi

if [[ -n "$IDS_FILE" ]]; then
    section "Step 4.6/6: Auto-build + verify SWE-bench Docker images"
    log "Building Docker images for IDS_FILE before inference/eval"
    run_cmd python scripts/build_docker_images.py \
        --instance_ids_file "$IDS_FILE" \
        --dataset_name "$DATASET_NAME" \
        --split test \
        --max_workers "$MAX_WORKERS_EVAL"

    log "Verifying required images exist for every instance in IDS_FILE"
    run_cmd python -u - "$DATASET_NAME" "$IDS_FILE" <<'PY'
import sys
from context_policy.datasets.swebench import load_instances, read_instance_ids
from context_policy.runner.mini_swe_agent_swebench import (
    _docker_image_exists,
    _get_instance_docker_image,
)

dataset_name = sys.argv[1]
ids_file = sys.argv[2]
ids = read_instance_ids(ids_file)
instances = load_instances(dataset_name=dataset_name, split="test", instance_ids=ids)

missing: list[tuple[str, str]] = []
for inst in instances:
    image = _get_instance_docker_image(inst)
    if not _docker_image_exists(image):
        missing.append((inst["instance_id"], image))

if missing:
    print("[eval_cpu] ERROR: Missing required SWE-bench Docker images after build:")
    for iid, image in missing:
        print(f"  - {iid} -> {image}")
    raise SystemExit(1)

print(f"[eval_cpu] Verified Docker images for all {len(instances)} instance(s)")
PY
fi

section "Step 5/6: Execute experiment pipeline"
log "Launching scripts/run_experiment.py"
export PYTHONUNBUFFERED=1
log "PYTHONUNBUFFERED=$PYTHONUNBUFFERED (ensures live orchestrator logs)"
run_cmd python -u scripts/run_experiment.py \
    --model "$MODEL_NAME" \
    --repo-config "$REPO_CONFIG" \
    --experiment-id "$EXP_ID" \
    --conditions $CONDITIONS \
    --oracle-iterations "$ORACLE_ITERATIONS" \
    --eval-dataset "$DATASET_NAME" \
    --max-workers-eval "$MAX_WORKERS_EVAL" \
    --timeout-s "$TIMEOUT_S" \
    --step-limit "$STEP_LIMIT" \
    $IDS_FLAG

section "Step 6/6: Completion"
END_TS="$(date +%s)"
ELAPSED="$((END_TS - START_TS))"
log "Evaluation complete. Check results/${EXP_ID}/experiment_summary.json"
log "Total elapsed seconds: $ELAPSED"
