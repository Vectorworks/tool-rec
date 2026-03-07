#!/bin/bash
set -euo pipefail

# Package FAISS retrieval artifacts into model.tar.gz and upload to S3.
#
# Usage:
#   cd /home/ec2-user/proj/BIM-Command-Recommendation
#   bash sagemaker_retrieval/package_model.sh

S3_DEST="s3://vectorworks-analytics-datalake/tool-rec-data/sagemaker/retrieval_model.tar.gz"
STAGING_DIR=$(mktemp -d)
trap "rm -rf $STAGING_DIR" EXIT

echo "==> Staging retrieval artifacts..."

cp data/embeddings/command_embeddings.npy "$STAGING_DIR/"
cp data/embeddings/metadata.json "$STAGING_DIR/"
cp data/embeddings/command.index "$STAGING_DIR/"

echo "==> Creating model.tar.gz..."
tar -czf model.tar.gz -C "$STAGING_DIR" .

echo "==> model.tar.gz contents:"
tar -tzf model.tar.gz

echo "==> model.tar.gz size:"
du -sh model.tar.gz

echo "==> Uploading to $S3_DEST..."
aws s3 cp model.tar.gz "$S3_DEST"

rm -f model.tar.gz

echo "==> Done! Retrieval artifacts uploaded to $S3_DEST"
