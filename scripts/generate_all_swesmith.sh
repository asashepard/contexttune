#!/usr/bin/env bash
# Generate SWE-smith tasks for all 12 experiment repos.
#
# Usage:
#   bash scripts/generate_all_swesmith.sh <commit-map-file> [--fallback]
#
# The commit map file has lines of the form:
#   repo/slug  <commit-sha>
#
# Example:
#   django/django  abc123def456
#   astropy/astropy 789012345678
#
# If --fallback is passed, uses git-history stubs instead of swesmith.
set -euo pipefail

COMMIT_MAP="${1:?Usage: $0 <commit-map-file> [--fallback]}"
FALLBACK_FLAG=""
if [[ "${2:-}" == "--fallback" ]]; then
    FALLBACK_FLAG="--fallback"
fi

OUTPUT_ROOT="artifacts/tasks"

while IFS=$' \t' read -r REPO COMMIT; do
    # Skip blank lines and comments
    [[ -z "$REPO" || "$REPO" == \#* ]] && continue
    REPO_DIR="${REPO//\//__}"
    echo "=== $REPO @ $COMMIT ==="
    python scripts/generate_swesmith_tasks.py \
        --repo "$REPO" \
        --commit "$COMMIT" \
        --n-train 200 \
        --n-holdout 50 \
        --output-dir "$OUTPUT_ROOT/$REPO_DIR" \
        $FALLBACK_FLAG
    echo ""
done < "$COMMIT_MAP"

echo "Done. Tasks are in $OUTPUT_ROOT/"
