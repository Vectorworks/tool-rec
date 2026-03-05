#!/bin/bash
set -euo pipefail

# Package model artifacts into model.tar.gz and upload to S3.
#
# Usage:
#   cd /home/ec2-user/proj/BIM-Command-Recommendation
#   bash sagemaker/package_model.sh

S3_DEST="s3://vectorworks-analytics-datalake/tool-rec-data/sagemaker/model.tar.gz"
STAGING_DIR=$(mktemp -d)
trap "rm -rf $STAGING_DIR" EXIT

echo "==> Staging model artifacts..."

cp tmp_test/checkpoint-4000/pytorch_model.bin "$STAGING_DIR/"
cp data/command_vocab.json "$STAGING_DIR/"
cp data/command_information_augmentations.csv "$STAGING_DIR/"
cp data/inference_metadata.json "$STAGING_DIR/"

mkdir -p "$STAGING_DIR/processed_nvt"
cp data/processed_nvt/part_0.parquet "$STAGING_DIR/processed_nvt/"
cp data/processed_nvt/schema.pbtxt "$STAGING_DIR/processed_nvt/"

echo "==> Creating model.tar.gz..."
tar -czf model.tar.gz -C "$STAGING_DIR" .

echo "==> model.tar.gz contents:"
tar -tzf model.tar.gz

echo "==> model.tar.gz size:"
du -sh model.tar.gz

echo "==> Uploading to $S3_DEST..."
aws s3 cp model.tar.gz "$S3_DEST"

echo "==> Cleaning up local model.tar.gz..."
rm -f model.tar.gz

echo "==> Done! Model artifact uploaded to $S3_DEST"
