#!/usr/bin/env bash
# Sanity check: run SWE-bench harness on a single instance with empty patch.
# Usage: sanity_one_instance.sh INSTANCE_ID [DATASET_NAME]

set -euo pipefail

INSTANCE_ID="${1:?Error: INSTANCE_ID required (e.g., django__django-16379)}"
DATASET_NAME="${2:-princeton-nlp/SWE-bench_Verified}"

# Resolve script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Generate run_id: sanity_<INSTANCE_ID>_<YYYYmmdd_HHMMSS>_<4hex>
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RANDOM_SUFFIX="$(head -c 2 /dev/urandom | xxd -p)"
RUN_ID="sanity_${INSTANCE_ID}_${TIMESTAMP}_${RANDOM_SUFFIX}"

echo "============================================"
echo "SWE-bench Sanity Check"
echo "============================================"
echo "Instance ID:  $INSTANCE_ID"
echo "Dataset:      $DATASET_NAME"
echo "Run ID:       $RUN_ID"
echo "============================================"

# Create dummy predictions
PREDS_DIR="$PROJECT_ROOT/artifacts/preds/$RUN_ID"
PREDS_PATH="$PREDS_DIR/preds.jsonl"

echo ""
echo "Step 1: Creating dummy predictions..."
python "$SCRIPT_DIR/make_dummy_preds.py" \
    --instance_id "$INSTANCE_ID" \
    --model_name "dummy" \
    --out "$PREDS_PATH" \
    --run_id "$RUN_ID"

echo ""
echo "Step 2: Running SWE-bench evaluation harness..."
"$SCRIPT_DIR/run_swebench_eval.sh" \
    "$DATASET_NAME" \
    "$PREDS_PATH" \
    "$RUN_ID" \
    1

echo ""
echo "============================================"
echo "Sanity check complete!"
echo "============================================"
echo "Predictions: $PREDS_PATH"
echo "Results:     $PROJECT_ROOT/results/$RUN_ID/"
echo "============================================"
