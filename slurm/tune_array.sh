#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# Per-repo guidance tuning — Slurm array job.
# Each array task runs the oracle loop for one repository.
#
# Submit:
#   MODEL_NAME=openai/gpt-4o EXP_ID=exp001 \
#     sbatch --array=0-11 slurm/tune_array.sh
#
# The array index maps to repo via REPO_LIST.
# Required: MODEL_NAME, EXP_ID
# Optional: ITERATIONS, TIMEOUT_S
# ──────────────────────────────────────────────────────────────────────
#SBATCH -J ct_tune
#SBATCH -p gpmoo-b
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 2-00:00
#SBATCH -o slurm-tune-%A_%a.out
#SBATCH -e slurm-tune-%A_%a.err
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
ENV_NAME="${ENV_NAME:-contexttune-py311}"
MODEL_NAME="${MODEL_NAME:?set MODEL_NAME}"
EXP_ID="${EXP_ID:?set EXP_ID}"

ITERATIONS="${ITERATIONS:-5}"
TIMEOUT_S="${TIMEOUT_S:-120}"

cd "$REPO_ROOT"
source ~/.bashrc || true
conda activate "$ENV_NAME" 2>/dev/null || true

# Repo list — index matches $SLURM_ARRAY_TASK_ID
REPO_LIST=(
    "django/django"
    "astropy/astropy"
    "sympy/sympy"
    "scikit-learn/scikit-learn"
    "matplotlib/matplotlib"
    "pallets/flask"
    "sphinx-doc/sphinx"
    "pylint-dev/pylint"
    "pytest-dev/pytest"
    "psf/requests"
    "pydata/xarray"
    "mwaskom/seaborn"
)

IDX="${SLURM_ARRAY_TASK_ID:-0}"
REPO="${REPO_LIST[$IDX]}"
REPO_SLUG="${REPO//\//__}"
OUTPUT_DIR="results/${EXP_ID}/guidance/${REPO_SLUG}"

# Read commit from commit map if available, else default to HEAD
COMMIT_MAP="artifacts/configs/commit_map_head.txt"
COMMIT="HEAD"
if [[ -f "$COMMIT_MAP" ]]; then
    COMMIT=$(grep "^${REPO} " "$COMMIT_MAP" | awk '{print $2}' || echo "HEAD")
fi

echo "============================================================"
echo "Tuning repo $IDX: $REPO"
echo "Model: $MODEL_NAME"
echo "Oracle iterations: $ITERATIONS"
echo "============================================================"

python scripts/tune_single_repo.py \
    --repo "$REPO" \
    --commit "$COMMIT" \
    --model "$MODEL_NAME" \
    --output-dir "$OUTPUT_DIR" \
    --iterations "$ITERATIONS" \
    --timeout-s "$TIMEOUT_S"

echo "Done tuning $REPO"
