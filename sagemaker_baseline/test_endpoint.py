"""Test the baseline SageMaker endpoint."""

import json
import boto3

ENDPOINT_NAME = "bim-command-rec-baseline"
MIXTRAL_ENDPOINT = "bim-command-rec"

payload = {
    "commands": [
        "Tool: Wall",
        "Tool: Rectangle",
        "Menu: Extrude",
        "Menu: Move",
        "Menu: Rotate",
    ],
    "top_k": 5,
}


def invoke(endpoint_name, payload):
    client = boto3.client("sagemaker-runtime", region_name="us-east-1")
    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload),
    )
    return json.loads(response["Body"].read().decode())


def main():
    print(f"=== Baseline ({ENDPOINT_NAME}) ===")
    try:
        result = invoke(ENDPOINT_NAME, payload)
        assert "predictions" in result, "Missing 'predictions' key"
        assert "input_length" in result, "Missing 'input_length' key"
        assert len(result["predictions"]) == payload["top_k"], (
            f"Expected {payload['top_k']} predictions, got {len(result['predictions'])}"
        )
        print(f"Input length: {result['input_length']}")
        for p in result["predictions"]:
            print(f"  #{p['rank']}: {p['command']} ({p['score']:.4f})")
    except Exception as e:
        print(f"Error: {e}")

    print(f"\n=== Mixtral ({MIXTRAL_ENDPOINT}) ===")
    try:
        result = invoke(MIXTRAL_ENDPOINT, payload)
        print(f"Input length: {result['input_length']}")
        for p in result["predictions"]:
            print(f"  #{p['rank']}: {p['command']} ({p['score']:.4f})")
    except Exception as e:
        print(f"Error (may not be deployed): {e}")


if __name__ == "__main__":
    main()
