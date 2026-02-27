"""
Replacement for openai_sideinformation.py.
Reads unique commands from S3, calls Claude claude-haiku-4-5 directly (no RAG),
and writes summary/classification/target back to S3 as a CSV.

Usage:
    python generate_augmentation.py
"""

import os
import re
import csv
import json
import time
import boto3
import anthropic
from io import StringIO
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_S3_URI  = "s3://vectorworks-analytics-datalake/athena_query_results/Unsaved/2026/02/19/5a1a7a71-bd2f-40e5-ac4b-7f156ebe0a3a.csv"
OUTPUT_S3_URI = "s3://vectorworks-analytics-datalake/tool-rec-data/command_information_augmentations.csv"
CHECKPOINT_S3_URI = "s3://vectorworks-analytics-datalake/tool-rec-data/augmentation_checkpoint.json"

MODEL = "claude-haiku-4-5-20251001"
MIN_COUNT = 10   # skip commands seen fewer than this many times
# ─────────────────────────────────────────────────────────────────────────────


def parse_s3_uri(uri):
    uri = uri.replace("s3://", "")
    bucket, key = uri.split("/", 1)
    return bucket, key


def s3_read_text(uri):
    bucket, key = parse_s3_uri(uri)
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read().decode("utf-8")


def s3_write_text(uri, text):
    bucket, key = parse_s3_uri(uri)
    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8"))


def strip_ids(msg):
    """Remove trailing localization ID patterns like (-5) (0) or (-217)."""
    msg = re.sub(r'\s*-\s*(\(-?\d+\)\s*)+$', '', msg)
    msg = re.sub(r'\s*(\(-?\d+\)\s*)+$', '', msg)
    return msg.strip().rstrip(' -').strip()


def load_commands(input_uri):
    """Load and deduplicate commands from the Athena CSV."""
    raw = s3_read_text(input_uri)
    reader = csv.DictReader(StringIO(raw))

    commands = {}  # message_clean -> total count
    for row in reader:
        cat = row["cat"]
        msg = strip_ids(row["message_clean"])
        cnt = int(row["cnt"])
        if cat not in ("Tool", "Menu"):
            continue
        if not msg:
            continue
        commands[msg] = commands.get(msg, 0) + cnt

    # filter low-frequency and sort by count descending
    commands = {k: v for k, v in commands.items() if v >= MIN_COUNT}
    return sorted(commands.items(), key=lambda x: -x[1])


def load_checkpoint(uri):
    """Load existing results from checkpoint, return as dict keyed by command."""
    try:
        raw = s3_read_text(uri)
        return json.loads(raw)
    except Exception:
        return {}


def save_checkpoint(uri, checkpoint):
    s3_write_text(uri, json.dumps(checkpoint, ensure_ascii=False))


def augment_command(client, command, existing_classifications, existing_targets):
    """Call Claude to generate summary, classification, and target for one command."""

    prompt = f"""You are an expert in the BIM authoring tool Vectorworks.
For the given Vectorworks command name, provide:
1. A summary: 2-3 sentences describing what the command does at a high level.
2. A classification: a single word describing the primary action type (e.g. Create, Update, Delete, Move, Copy, Export, View, Select, Group, Align, Annotate, other).
   Prefer reusing an existing class if appropriate: {list(set(existing_classifications))[:20]}
3. A target: a single word for the primary object or element type affected (e.g. Object, Layer, Viewport, Text, Group, Symbol, Dimension, Wall, Door, Window, Surface, Line, Polygon, other).
   Prefer reusing an existing target if appropriate: {list(set(existing_targets))[:20]}

Command: {command}

Respond in exactly this JSON format with no other text:
{{"summary": "...", "classification": "...", "target": "..."}}"""

    for attempt in range(3):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            text = response.content[0].text.strip()
            # extract JSON even if surrounded by markdown code fences
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return data.get("summary", ""), data.get("classification", "other"), data.get("target", "other")
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                print(f"  Failed for '{command}': {e}")
    return "", "other", "other"


def write_output(uri, results):
    """Write final CSV to S3."""
    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=["message_content", "summary", "classification", "target"])
    writer.writeheader()
    writer.writerows(results)
    s3_write_text(uri, buf.getvalue())


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)

    print("Loading commands from S3...")
    commands = load_commands(INPUT_S3_URI)
    print(f"Found {len(commands)} unique commands to augment")

    print("Loading checkpoint...")
    checkpoint = load_checkpoint(CHECKPOINT_S3_URI)
    print(f"  {len(checkpoint)} already done")

    existing_classifications = [v["classification"] for v in checkpoint.values() if v.get("classification")]
    existing_targets = [v["target"] for v in checkpoint.values() if v.get("target")]

    todo = [(cmd, cnt) for cmd, cnt in commands if cmd not in checkpoint]
    print(f"  {len(todo)} remaining\n")

    for i, (command, cnt) in enumerate(tqdm(todo, desc="Augmenting")):
        summary, classification, target = augment_command(
            client, command, existing_classifications, existing_targets
        )
        checkpoint[command] = {
            "summary": summary,
            "classification": classification,
            "target": target,
            "cnt": cnt,
        }
        existing_classifications.append(classification)
        existing_targets.append(target)

        # save checkpoint every 50 commands
        if (i + 1) % 50 == 0:
            save_checkpoint(CHECKPOINT_S3_URI, checkpoint)
            tqdm.write(f"  Checkpoint saved ({len(checkpoint)} done)")

    # final checkpoint save
    save_checkpoint(CHECKPOINT_S3_URI, checkpoint)

    # write final CSV
    results = [
        {"message_content": cmd, "summary": v["summary"], "classification": v["classification"], "target": v["target"]}
        for cmd, v in sorted(checkpoint.items(), key=lambda x: -x[1]["cnt"])
    ]
    write_output(OUTPUT_S3_URI, results)
    print(f"\nDone! Output written to {OUTPUT_S3_URI}")


if __name__ == "__main__":
    main()
