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
import os
import re
import sys
import time
from pathlib import Path

import requests
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


DISPLAY_LINES = 0  # track how many lines the display occupies


def clear_display():
    """Move cursor up and clear previous display."""
    global DISPLAY_LINES
    if DISPLAY_LINES > 0:
        sys.stdout.write(f"\033[{DISPLAY_LINES}A\033[J")
        sys.stdout.flush()


def display_status(commands: list[str], result: dict | None = None):
    """In-place terminal display of recent commands and predictions."""
    global DISPLAY_LINES
    clear_display()

    lines = []
    try:
        cols = os.get_terminal_size().columns
    except OSError:
        cols = 80

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

    output = "\n".join(lines) + "\n"
    sys.stdout.write(output)
    sys.stdout.flush()
    DISPLAY_LINES = len(lines)


# ── File watcher ──────────────────────────────────────────────────────────────

class LogFileHandler(FileSystemEventHandler):
    def __init__(self, log_path: str, filtered_path: str, server_url: str, top_k: int):
        super().__init__()
        self.log_path = log_path
        self.filtered_path = filtered_path
        self.server_url = server_url
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
            result = get_predictions(self.server_url, commands, self.top_k)
        display_status(commands, result)


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

    # Test server connection
    print(f"Testing server connection: {args.server}")
    try:
        resp = requests.get(f"{args.server}/health", timeout=5)
        info = resp.json()
        print(f"  Server OK: {info}")
    except Exception as e:
        print(f"  Warning: server not reachable ({e}). Will retry on each prediction.")

    print(f"Watching:  {log_path}")
    print(f"Filtered:  {filtered_path}")
    print(f"Server:    {args.server}")
    print(f"Top-k:     {args.top_k}")
    print(f"Press Ctrl+C to stop.\n")

    handler = LogFileHandler(log_path, filtered_path, args.server, args.top_k)
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
