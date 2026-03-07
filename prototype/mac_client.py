"""
mac_client.py

Lightweight client for macOS that watches the Vectorworks log file,
extracts command events, and sends them to a SageMaker endpoint.

Requirements (pip install on Mac):
    pip install boto3 watchdog

Usage:
    python mac_client.py --log-file "/path/to/VW User Log.txt"
    python mac_client.py --log-file "/path/to/VW User Log.txt" --endpoint bim-command-rec --region us-east-1

Example log file locations:
    macOS: ~/Library/Application Support/Vectorworks/2026/VW User Log.txt
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import boto3
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler


# ── Log parsing ───────────────────────────────────────────────────────────────

# Categories to keep (UNDO dropped — system bookkeeping, not user undos)
VALID_CATS = {"Tool", "Menu"}

# Prefixes that indicate noise events (dropped entirely)
DROP_PREFIXES = [
    "DestroyEvent: ", "Begin Internal Event: ", "Event:",
    "Event name changed from ", "Beta Undo Alert", "Undo Problem:",
    "Abort Event: ", "Menu: Undo", "Menu: Redo",
]

# Trailing suffixes that indicate noise (zoom, pan, view changes, etc.)
DROP_SUFFIXES = [
    "(242)", "(-242)", "(218)", "(199)", "(236)", "(-241)", "(-240)", "(307)",
    "(201)", "(-303)", "(305)", "(306)", "(193)", "(8)", "(205)", "(203)",
]

# Regex to strip trailing tool IDs like (-217) or (3)
TRAILING_ID_RE = re.compile(r"\s*-?\s*(\(-?\d+\)\s*)+$")
# Regex to strip single quotes around tool names
STRIP_QUOTES_RE = re.compile(r"'([^']+)'")


def parse_log_lines(text: str) -> list[dict]:
    """Parse VW log file (JSON lines) into a list of events."""
    events = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            events.append(obj)
        except json.JSONDecodeError:
            continue
    return events


def clean_message(cat: str, msg: str) -> str | None:
    """Clean a raw log message into vocab format. Returns None if it should be dropped."""
    # Drop noise prefixes
    for prefix in DROP_PREFIXES:
        if msg.startswith(prefix):
            return None

    # Drop noise suffixes
    for suffix in DROP_SUFFIXES:
        if msg.endswith(suffix):
            return None

    # Strip trailing IDs: "Tool: 'Round Rect' (-217)" -> "Tool: 'Round Rect'"
    msg = TRAILING_ID_RE.sub("", msg)

    # Strip quotes: "'Round Rect'" -> "Round Rect"
    msg = STRIP_QUOTES_RE.sub(r"\1", msg)

    msg = msg.strip().rstrip(" -").strip()
    if not msg:
        return None

    # Ensure "Cat: Name" format matching vocab
    if msg.startswith(f"{cat}:"):
        return msg
    return f"{cat}: {msg}"


def filter_events(events: list[dict]) -> list[str]:
    """Extract command names from raw log events."""
    commands = []
    for ev in events:
        cat = ev.get("cat", "")
        msg = ev.get("message", ev.get("msg", ""))
        if cat not in VALID_CATS:
            continue
        cmd = clean_message(cat, msg)
        if cmd:
            commands.append(cmd)
    return commands


# ── SageMaker communication ───────────────────────────────────────────────────

_sm_client = None


def get_sm_client(region: str):
    global _sm_client
    if _sm_client is None:
        _sm_client = boto3.client("sagemaker-runtime", region_name=region)
    return _sm_client


def get_predictions(endpoint_name: str, region: str, commands: list[str], top_k: int = 5) -> dict:
    """Send commands to SageMaker endpoint and return predictions."""
    try:
        client = get_sm_client(region)
        resp = client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=json.dumps({"commands": commands, "top_k": top_k}),
        )
        return json.loads(resp["Body"].read().decode("utf-8"))
    except Exception as e:
        print(f"  Error invoking SageMaker endpoint: {e}")
        return None


def display_status(commands: list[str], result: dict | None = None):
    """In-place terminal display of recent commands and predictions."""
    try:
        cols = os.get_terminal_size().columns
    except OSError:
        cols = 80

    lines = []
    lines.append(f"{'─' * cols}")
    recent = commands[-5:]
    lines.append(f"  Recent commands ({len(commands)} total):")
    for cmd in recent:
        lines.append(f"    > {cmd}")

    if result and result.get("predictions"):
        lines.append("")
        lines.append("  Predicted next commands:")
        for pred in result["predictions"]:
            bar = "█" * int(pred["score"] * 40)
            lines.append(f"    {pred['rank']}. {pred['command']:<40s} {pred['score']:.1%} {bar}")

    lines.append(f"{'─' * cols}")

    # Restore saved cursor position, clear everything below, then redraw
    sys.stdout.write("\033[u\033[J")
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()


# ── File watcher ──────────────────────────────────────────────────────────────

class LogFileHandler(FileSystemEventHandler):
    def __init__(self, log_path: str, filtered_path: str, endpoint_name: str, region: str, top_k: int):
        super().__init__()
        self.log_path = log_path
        self.filtered_path = filtered_path
        self.endpoint_name = endpoint_name
        self.region = region
        self.top_k = top_k
        self.file_pos = 0
        self.last_commands = []

        # Seed position to end of current file so we only process new lines
        try:
            self.file_pos = Path(self.log_path).stat().st_size
        except OSError:
            pass

    def on_modified(self, event):
        if Path(event.src_path).name != Path(self.log_path).name:
            return

        # Read only new bytes since last check (binary mode for reliable seek)
        try:
            size = Path(self.log_path).stat().st_size
            if size < self.file_pos:
                self.file_pos = 0  # file was truncated, reset
            if size == self.file_pos:
                return
            with open(self.log_path, "rb") as f:
                f.seek(self.file_pos)
                new_bytes = f.read()
                self.file_pos = f.tell()
            new_text = new_bytes.decode("utf-8", errors="ignore")
        except Exception:
            return

        if not new_text.strip():
            return

        # Filter new lines and append valid ones to the filtered file
        new_valid = []
        for line in new_text.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            cat = obj.get("cat", "")
            msg = obj.get("message", obj.get("msg", ""))
            if cat not in VALID_CATS:
                continue
            cmd = clean_message(cat, msg)
            if cmd:
                filtered_obj = {"cat": cat, "message": msg, "command": cmd}
                new_valid.append(json.dumps(filtered_obj))

        if new_valid:
            with open(self.filtered_path, "a", encoding="utf-8") as f:
                for line in new_valid:
                    f.write(line + "\n")

        # Build commands from the full filtered file
        try:
            filtered_text = Path(self.filtered_path).read_text(encoding="utf-8", errors="ignore")
        except FileNotFoundError:
            return

        events = parse_log_lines(filtered_text)
        commands = filter_events(events)

        if commands == self.last_commands:
            return
        self.last_commands = commands[:]

        if not commands:
            return

        # Always show recent commands; get predictions if enough history
        result = None
        if len(commands) >= 3:
            result = get_predictions(self.endpoint_name, self.region, commands, self.top_k)
        display_status(commands, result)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BIM Command Recommendation Client")
    parser.add_argument(
        "--log-file", required=True,
        help="Path to Vectorworks log file",
    )
    parser.add_argument(
        "--endpoint", default="bim-command-rec",
        help="SageMaker endpoint name (default: bim-command-rec)",
    )
    parser.add_argument(
        "--region", default="us-east-1",
        help="AWS region (default: us-east-1)",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of predictions to show (default: 5)",
    )
    parser.add_argument(
        "--filtered-file",
        help="Path to write filtered events (default: <log-dir>/filtered_commands.jsonl)",
    )
    parser.add_argument(
        "--poll-interval", type=float, default=1.0,
        help="File poll interval in seconds (default: 1.0)",
    )
    args = parser.parse_args()

    log_path = str(Path(args.log_file).resolve())
    if not Path(log_path).exists():
        print(f"Error: log file not found: {log_path}")
        sys.exit(1)

    # Default filtered file next to the log
    if args.filtered_file:
        filtered_path = str(Path(args.filtered_file).resolve())
    else:
        filtered_path = str(Path(log_path).parent / "filtered_commands.jsonl")

    # Clear filtered file on startup for a fresh session
    Path(filtered_path).write_text("", encoding="utf-8")

    # Test SageMaker endpoint
    print(f"Testing SageMaker endpoint: {args.endpoint} ({args.region})")
    try:
        sm = boto3.client("sagemaker", region_name=args.region)
        status = sm.describe_endpoint(EndpointName=args.endpoint)["EndpointStatus"]
        print(f"  Endpoint status: {status}")
    except Exception as e:
        print(f"  Warning: could not check endpoint ({e}). Will retry on each prediction.")

    print(f"Watching:  {log_path}")
    print(f"Filtered:  {filtered_path}")
    print(f"Endpoint:  {args.endpoint}")
    print(f"Region:    {args.region}")
    print(f"Top-k:     {args.top_k}")
    print(f"Press Ctrl+C to stop.\n")

    # Save cursor position — display_status will restore to here each update
    sys.stdout.write("\033[s")
    sys.stdout.flush()

    handler = LogFileHandler(log_path, filtered_path, args.endpoint, args.region, args.top_k)
    observer = PollingObserver(timeout=args.poll_interval)
    observer.schedule(handler, str(Path(log_path).parent), recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
