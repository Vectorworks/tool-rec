"""
Test the bim-command-retrieval SageMaker endpoint.

Usage:
    python invoke_endpoints/test_retrieval.py
    python invoke_endpoints/test_retrieval.py --action search --query "draw a wall"
    python invoke_endpoints/test_retrieval.py --action recommend --session "Tool: SelectionTool" "Menu: Undo" "Tool: WallTool"
    python invoke_endpoints/test_retrieval.py --action search --query "lighting" --top-k 5
"""

import argparse
import json
import boto3

ENDPOINT_NAME = "bim-command-retrieval"
REGION = "us-east-1"


def invoke(action: str, query: str | None = None, session: list[str] | None = None, top_k: int = 10):
    sm = boto3.client("sagemaker-runtime", region_name=REGION)
    payload = {"action": action, "top_k": top_k}
    if query:
        payload["query"] = query
    if session:
        payload["session"] = session

    resp = sm.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps(payload),
    )
    return json.loads(resp["Body"].read())


def print_results(result: dict):
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"Action: {result['action']}")
    print(f"Latency: {result['elapsed_ms']} ms")
    print(f"Results ({len(result['results'])}):")
    for i, r in enumerate(result["results"], 1):
        print(f"  {i:2d}. {r['name']:<45s} (id={r['command_id']}, score={r['score']:.4f})")


def main():
    parser = argparse.ArgumentParser(description="Test bim-command-retrieval endpoint")
    parser.add_argument("--action", choices=["search", "recommend"], default="search")
    parser.add_argument("--query", type=str, default=None, help="Text query (for search)")
    parser.add_argument("--session", nargs="+", default=None, help="Command names (for recommend)")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    if args.action == "search" and args.query is None:
        args.query = "design a roof"
        print(f"(using default query: \"{args.query}\")\n")

    if args.action == "recommend" and args.session is None:
        args.session = ["Tool: Select Similar", "Tool: Wall", "Menu: FitWallsToRoof"]
        print(f"(using default session: {args.session})\n")

    result = invoke(args.action, query=args.query, session=args.session, top_k=args.top_k)
    print_results(result)


if __name__ == "__main__":
    main()
