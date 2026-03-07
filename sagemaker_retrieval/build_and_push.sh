#!/bin/bash
set -euo pipefail

# Build Docker image and push to ECR.
#
# Usage:
#   cd /home/ec2-user/proj/BIM-Command-Recommendation
#   bash sagemaker_retrieval/build_and_push.sh

ACCOUNT_ID="167665646931"
REGION="us-east-1"
REPO_NAME="bim-command-retrieval"
IMAGE_TAG="latest"

FULL_NAME="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:${IMAGE_TAG}"

echo "==> Logging into ECR..."
aws ecr get-login-password --region "${REGION}" | \
    docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

echo "==> Creating ECR repository (if not exists)..."
aws ecr describe-repositories --repository-names "${REPO_NAME}" --region "${REGION}" 2>/dev/null || \
    aws ecr create-repository --repository-name "${REPO_NAME}" --region "${REGION}"

echo "==> Building Docker image..."
docker build -t "${REPO_NAME}" -f sagemaker_retrieval/Dockerfile .

echo "==> Tagging image as ${FULL_NAME}..."
docker tag "${REPO_NAME}:${IMAGE_TAG}" "${FULL_NAME}"

echo "==> Pushing to ECR..."
docker push "${FULL_NAME}"

echo "==> Done! Image pushed to ${FULL_NAME}"
