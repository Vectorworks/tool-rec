"""
deploy.py

Create a SageMaker real-time endpoint for FAISS retrieval.
CPU-only, no GPU needed.

Usage:
    python sagemaker_retrieval/deploy.py
"""

import time
import boto3

ACCOUNT_ID = "167665646931"
REGION = "us-east-1"
REPO_NAME = "bim-command-retrieval"
IMAGE_TAG = "latest"
IMAGE_URI = f"{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/{REPO_NAME}:{IMAGE_TAG}"

MODEL_DATA = "s3://vectorworks-analytics-datalake/tool-rec-data/sagemaker/retrieval_model.tar.gz"
ROLE = "arn:aws:iam::167665646931:role/tools-rec-iam-role"

ENDPOINT_NAME = "bim-command-retrieval"
MODEL_NAME = "bim-command-retrieval"
INSTANCE_TYPE = "ml.c5.large"  # CPU-only, cheapest option sufficient for 1348 vectors
INSTANCE_COUNT = 1


def main():
    sm = boto3.client("sagemaker", region_name=REGION)

    print(f"Image:    {IMAGE_URI}")
    print(f"Model:    {MODEL_DATA}")
    print(f"Instance: {INSTANCE_TYPE}")
    print(f"Endpoint: {ENDPOINT_NAME}")

    # Clean up existing resources if they exist
    for cleanup in [
        lambda: sm.delete_endpoint(EndpointName=ENDPOINT_NAME),
        lambda: sm.get_waiter("endpoint_deleted").wait(EndpointName=ENDPOINT_NAME),
        lambda: sm.delete_endpoint_config(EndpointConfigName=ENDPOINT_NAME),
        lambda: sm.delete_model(ModelName=MODEL_NAME),
    ]:
        try:
            cleanup()
        except sm.exceptions.ClientError:
            pass

    print("Creating model...")
    sm.create_model(
        ModelName=MODEL_NAME,
        PrimaryContainer={
            "Image": IMAGE_URI,
            "ModelDataUrl": MODEL_DATA,
        },
        ExecutionRoleArn=ROLE,
    )

    print("Creating endpoint config...")
    sm.create_endpoint_config(
        EndpointConfigName=ENDPOINT_NAME,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": MODEL_NAME,
                "InstanceType": INSTANCE_TYPE,
                "InitialInstanceCount": INSTANCE_COUNT,
                "ContainerStartupHealthCheckTimeoutInSeconds": 300,
            }
        ],
    )

    print("Creating endpoint...")
    sm.create_endpoint(
        EndpointName=ENDPOINT_NAME,
        EndpointConfigName=ENDPOINT_NAME,
    )

    print("Waiting for endpoint to be InService...")
    while True:
        resp = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
        status = resp["EndpointStatus"]
        print(f"  Status: {status}")
        if status == "InService":
            break
        if status == "Failed":
            print(f"  Failure reason: {resp.get('FailureReason', 'unknown')}")
            return
        time.sleep(30)

    print(f"Endpoint '{ENDPOINT_NAME}' is InService!")


if __name__ == "__main__":
    main()
