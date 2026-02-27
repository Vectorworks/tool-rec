"""
mac_client.py

Lightweight client for macOS that watches the Vectorworks log file,
extracts command events, and sends them to the EC2 inference server.

Requirements (pip install on Mac):
    pip install requests watchdog

Usage:
    python mac_client.py --server http://<ec2-ip>:8000 --log-file "/path/to/VW User Log.txt"

Example log file locations:
    macOS: ~/Library/Application Support/Vectorworks/2024/VW User Log.txt
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler


# ── Log parsing ───────────────────────────────────────────────────────────────

# Categories to keep
VALID_CATS = {"Tool", "Menu", "UNDO"}

# Commands to ignore (navigation, viewport, etc.)
IGNORE_PATTERNS = [
    r"\(242\)", r"\(199\)", r"\(236\)",  # zoom/pan/select IDs
    "Zoom on objects", "Pan", "Selection",
]

# Prefixes to strip
STRIP_PREFIXES = [
    "DestroyEvent: ", "Begin Internal Event: ", "Event: ",
    "End Event: ", "Begin Event: ",
]


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


def filter_events(events: list[dict]) -> list[str]:
    """Extract command names from raw log events."""
    commands = []

    for ev in events:
        cat = ev.get("cat", "")
        msg = ev.get("message", ev.get("msg", ""))

        if cat not in VALID_CATS:
            continue
        if cat == "UNDO":
            # Remove the last command (undo)
            if commands:
                commands.pop()
            continue

        # Skip ignored patterns
        skip = False
        for pat in IGNORE_PATTERNS:
            if re.search(pat, msg):
                skip = True
                break
        if skip:
            continue

        # Strip prefixes
        for prefix in STRIP_PREFIXES:
            if msg.startswith(prefix):
                msg = msg[len(prefix):]
                break

        # Reconstruct as "Tool: X" or "Menu: X" format (matching vocab)
        if cat in ("Tool", "Menu"):
            cmd = f"{cat}: {msg}" if not msg.startswith(f"{cat}:") else msg
        else:
            cmd = msg

        if cmd.strip():
            commands.append(cmd.strip())

    return commands


# ── Server communication ──────────────────────────────────────────────────────

def get_predictions(server_url: str, commands: list[str], top_k: int = 5) -> dict:
    """Send commands to inference server and return predictions."""
    try:
        resp = requests.post(
            f"{server_url}/predict",
            json={"commands": commands, "top_k": top_k},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"  Error contacting server: {e}")
        return None


def display_predictions(result: dict, commands: list[str]):
    """Pretty-print predictions to terminal."""
    if not result or not result.get("predictions"):
        return

    # Show recent commands
    recent = commands[-5:]
    print(f"\n{'─' * 60}")
    print(f"  Recent commands ({result['input_length']} total):")
    for cmd in recent:
        print(f"    > {cmd}")

    print(f"\n  Predicted next commands:")
    for pred in result["predictions"]:
        bar = "█" * int(pred["score"] * 40)
        print(f"    {pred['rank']}. {pred['command']:<40s} {pred['score']:.1%} {bar}")
    print(f"{'─' * 60}")


# ── File watcher ──────────────────────────────────────────────────────────────

class LogFileHandler(FileSystemEventHandler):
    def __init__(self, log_path: str, server_url: str, top_k: int):
        super().__init__()
        self.log_path = log_path
        self.server_url = server_url
        self.top_k = top_k
        self.last_size = 0
        self.last_commands = []

    def on_modified(self, event):
        if event.src_path != self.log_path:
            return

        try:
            text = Path(self.log_path).read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"  Error reading log: {e}")
            return

        # Parse and filter
        events = parse_log_lines(text)
        commands = filter_events(events)

        # Only query if commands changed
        if commands == self.last_commands:
            return
        self.last_commands = commands[:]

        if len(commands) < 3:
            return

        # Get predictions
        result = get_predictions(self.server_url, commands, self.top_k)
        display_predictions(result, commands)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BIM Command Recommendation Client")
    parser.add_argument(
        "--server", required=True,
        help="Inference server URL (e.g. http://ec2-ip:8000)",
    )
    parser.add_argument(
        "--log-file", required=True,
        help="Path to Vectorworks log file",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of predictions to show (default: 5)",
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

    # Test server connection
    print(f"Testing server connection: {args.server}")
    try:
        resp = requests.get(f"{args.server}/health", timeout=5)
        info = resp.json()
        print(f"  Server OK: {info}")
    except Exception as e:
        print(f"  Warning: server not reachable ({e}). Will retry on each prediction.")

    print(f"Watching: {log_path}")
    print(f"Server:   {args.server}")
    print(f"Top-k:    {args.top_k}")
    print(f"Press Ctrl+C to stop.\n")

    handler = LogFileHandler(log_path, args.server, args.top_k)
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
