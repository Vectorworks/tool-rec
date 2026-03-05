"""
test_endpoint.py

Test the deployed SageMaker endpoint.

Usage:
    python sagemaker/test_endpoint.py
"""

import json
import boto3

ENDPOINT_NAME = "bim-command-rec"
REGION = "us-east-1"


def test_endpoint():
    runtime = boto3.client("sagemaker-runtime", region_name=REGION)

    payload = {
        "commands": [
            "Wall tool",
            "Rectangle tool",
            "Extrude",
            "Move",
            "Rotate",
        ],
        "top_k": 5,
    }

    print(f"==> Invoking endpoint '{ENDPOINT_NAME}'")
    print(f"    Input: {json.dumps(payload, indent=2)}")

    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps(payload),
    )

    result = json.loads(response["Body"].read().decode("utf-8"))
    print(f"\n==> Response:")
    print(json.dumps(result, indent=2))

    # Validate response structure
    assert "predictions" in result, "Missing 'predictions' in response"
    assert "input_length" in result, "Missing 'input_length' in response"
    assert len(result["predictions"]) == payload["top_k"], (
        f"Expected {payload['top_k']} predictions, got {len(result['predictions'])}"
    )

    print(f"\n==> Test passed! Got {len(result['predictions'])} predictions.")
    return result


if __name__ == "__main__":
    test_endpoint()
