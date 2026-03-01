#!/usr/bin/env bash
#SBATCH -J ct_smoke
#SBATCH -p gpmoo-b
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -t 0-02:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
# ──────────────────────────────────────────────────────────────────────
# Smoke test: run the full experiment pipeline on 1 repo with minimal
# budget to verify everything works end-to-end.
#
# Submit:
#   MODEL_NAME=openai/gpt-4o sbatch slurm/smoke_experiment.sh
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
ENV_NAME="${ENV_NAME:-contexttune-py311}"
MODEL_NAME="${MODEL_NAME:?set MODEL_NAME}"
DRY_RUN="${DRY_RUN:-}"

cd "$REPO_ROOT"
source ~/.bashrc || true
conda activate "$ENV_NAME" 2>/dev/null || true

# Create a minimal repo config with 1 repo
SMOKE_DIR="artifacts/smoke_$$"
mkdir -p "$SMOKE_DIR"

cat > "$SMOKE_DIR/repos.json" << 'EOF'
[
    {"repo": "psf/requests", "commit": "HEAD"}
]
EOF

DRY_FLAG=""
if [[ -n "$DRY_RUN" ]]; then
    DRY_FLAG="--dry-run"
fi

python scripts/run_experiment.py \
    --model "$MODEL_NAME" \
    --repo-config "$SMOKE_DIR/repos.json" \
    --conditions no_context static_kb oracle_tuned \
    --oracle-iterations 1 \
    --timeout-s 300 \
    --step-limit 15 \
    $DRY_FLAG

echo "Smoke test complete."
