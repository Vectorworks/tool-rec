#!/bin/bash
set -euo pipefail

ACCOUNT_ID="167665646931"
REGION="us-east-1"
REPO_NAME="bim-command-rec-baseline"
IMAGE_TAG="latest"
FULL_NAME="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:${IMAGE_TAG}"

echo "=== Logging into ECR ==="
aws ecr get-login-password --region "${REGION}" | \
    docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

echo "=== Creating ECR repo (if needed) ==="
aws ecr create-repository --repository-name "${REPO_NAME}" --region "${REGION}" 2>/dev/null || true

echo "=== Building Docker image ==="
docker build -t "${REPO_NAME}" -f sagemaker_baseline/Dockerfile sagemaker_baseline/

echo "=== Tagging and pushing ==="
docker tag "${REPO_NAME}:latest" "${FULL_NAME}"
docker push "${FULL_NAME}"

echo "=== Done: ${FULL_NAME} ==="
