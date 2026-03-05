#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_FILE="data/baseline_model.json"
S3_DEST="s3://vectorworks-analytics-datalake/tool-rec-data/sagemaker/baseline_model.tar.gz"

if [ ! -f "${MODEL_FILE}" ]; then
    echo "ERROR: ${MODEL_FILE} not found. Run train_baseline.py first."
    exit 1
fi

echo "=== Packaging baseline model ==="
tar -czf /tmp/baseline_model.tar.gz -C data baseline_model.json
ls -lh /tmp/baseline_model.tar.gz

echo "=== Uploading to ${S3_DEST} ==="
aws s3 cp /tmp/baseline_model.tar.gz "${S3_DEST}"

echo "=== Done ==="
