#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/ec2_bootstrap.sh <REPO_URL> [BRANCH] [TARGET_DIR]
#
# Example:
#   ./scripts/ec2_bootstrap.sh https://github.com/you/contexttune.git main contexttune

REPO_URL="${1:-}"
BRANCH="${2:-main}"
TARGET_DIR="${3:-contexttune}"

if [[ -z "${REPO_URL}" ]]; then
  echo "Usage: $0 <REPO_URL> [BRANCH] [TARGET_DIR]"
  exit 2
fi

echo "==> Updating apt + installing base packages..."
sudo apt-get update -y
sudo apt-get install -y \
  git jq curl ca-certificates \
  python3 python3-venv python3-pip \
  build-essential \
  docker.io

echo "==> Enabling Docker..."
sudo systemctl enable --now docker

echo "==> Adding current user to docker group..."
sudo usermod -aG docker "$USER"

echo "==> Checking Python version (need >= 3.11)..."
python3 - <<'PY'
import sys
major, minor = sys.version_info[:2]
print("Python:", sys.version)
if (major, minor) < (3, 11):
    raise SystemExit("ERROR: Python >= 3.11 required.")
print("OK: Python version requirement satisfied.")
PY

echo "==> Cloning or updating repo..."
if [[ -d "${TARGET_DIR}/.git" ]]; then
  cd "${TARGET_DIR}"
  git fetch --all --prune
  git checkout "${BRANCH}"
  git pull --ff-only origin "${BRANCH}"
else
  git clone --branch "${BRANCH}" "${REPO_URL}" "${TARGET_DIR}"
  cd "${TARGET_DIR}"
fi

echo "==> Creating venv..."
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Upgrading pip..."
python -m pip install -U pip wheel setuptools

echo "==> Installing project + deps..."
# Prefer requirements.txt if present; otherwise install editable + core deps
if [[ -f "requirements.txt" ]]; then
  pip install -r requirements.txt
fi

# Install your package (editable is convenient for dev)
pip install -e .

# Ensure common deps used in your pipeline exist (safe if already installed)
pip install -U datasets requests swebench

echo "==> Validating mini-swe-agent installation..."
python -c "from minisweagent.agents.default import DefaultAgent; print('  mini-swe-agent OK')"

echo "==> Validating Docker access..."
if docker info >/dev/null 2>&1; then
  echo "  Docker OK"
else
  echo "  WARNING: docker info failed. Log out and back in for group membership, then re-run."
fi

echo "==> DONE."
echo ""
echo "Next steps:"
echo "1) Log out and back in (or reconnect SSH) so docker group takes effect."
echo "2) Activate venv:  source ${TARGET_DIR}/.venv/bin/activate"
echo "3) Export model env vars:"
echo "   export OPENAI_BASE_URL=https://api.openai.com/v1"
echo "   export OPENAI_API_KEY=YOUR_KEY"
echo "4) Run the smoke test:"
echo "   python scripts/smoke_mini_agent_swebench.py --model openai/<model> --instance_id django__django-10097 --timeout_s 600 --with_context"
echo "5) Run a 3-instance test:"
echo "   python scripts/build_context.py --instance_ids_file scripts/smoke_3_ids.txt --mode baseline"
echo "   python scripts/run_inference.py --model openai/<model> --runner mini_swe_agent_swebench --ablation baseline --instance_ids_file scripts/smoke_3_ids.txt --contexts_root artifacts/contexts --timeout_s 600"
