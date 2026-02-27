#!/usr/bin/env bash
#SBATCH -J ct_preflight
#SBATCH -p gpmoo-b
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 0-00:20
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
ENV_NAME="${ENV_NAME:-contexttune-py311}"

cd "$REPO_ROOT"
source ~/.bashrc || true
conda activate "$ENV_NAME"

python --version
python -c "import datasets, requests, swebench; print('python deps: ok')"
python -c "from context_policy.utils.paths import PROJECT_ROOT; print('project root:', PROJECT_ROOT)"

if command -v docker >/dev/null 2>&1; then
  docker info >/dev/null && echo "docker: ok" || echo "docker: unavailable in this job"
else
  echo "docker: command not found"
fi

python -c "import os; print('OPENAI_BASE_URL set:', bool(os.getenv('OPENAI_BASE_URL'))); print('OPENAI_API_KEY set:', bool(os.getenv('OPENAI_API_KEY')))"

echo "preflight complete"
