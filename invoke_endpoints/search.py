import boto3, json
sm = boto3.client("sagemaker-runtime", region_name="us-east-1")
resp = sm.invoke_endpoint(
    EndpointName="bim-command-retrieval",
    ContentType="application/json",
    Body=json.dumps({"action": "search", "query": "design a roof", "top_k": 10})
)
print(json.loads(resp["Body"].read()))