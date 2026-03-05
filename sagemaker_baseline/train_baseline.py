"""Build bigram transition matrix from training data."""

import json
import pandas as pd
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = PROJECT_ROOT / "data" / "preproc_sessions" / "train_new" / "part.0.parquet"
VOCAB_PATH = PROJECT_ROOT / "data" / "command_vocab.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "baseline_model.json"


def main():
    # Load vocab
    with open(VOCAB_PATH) as f:
        vocab = json.load(f)
    reverse_vocab = {v: k for k, v in vocab.items()}
    print(f"Loaded vocab: {len(vocab)} commands")

    # Load training sessions
    df = pd.read_parquet(TRAIN_PATH, columns=["item_id-list"])
    print(f"Loaded {len(df)} sessions")

    # Count bigram transitions and global command frequency
    transitions = defaultdict(lambda: defaultdict(int))
    global_counts = defaultdict(int)
    total_pairs = 0

    for item_ids in df["item_id-list"]:
        for i, cmd_id in enumerate(item_ids):
            global_counts[int(cmd_id)] += 1
            if i > 0:
                prev = int(item_ids[i - 1])
                curr = int(cmd_id)
                transitions[prev][curr] += 1
                total_pairs += 1

    print(f"Total bigram pairs: {total_pairs}")
    print(f"Unique prev commands: {len(transitions)}")

    # Normalize to probabilities
    transition_probs = {}
    for prev_id, next_counts in transitions.items():
        total = sum(next_counts.values())
        transition_probs[str(prev_id)] = {
            str(nid): round(count / total, 6)
            for nid, count in sorted(next_counts.items(), key=lambda x: -x[1])
        }

    # Global top commands (fallback)
    total_cmds = sum(global_counts.values())
    global_top = [
        {"id": cmd_id, "prob": round(count / total_cmds, 6)}
        for cmd_id, count in sorted(global_counts.items(), key=lambda x: -x[1])
    ]

    # Save model
    model = {
        "transitions": transition_probs,
        "global_top": global_top[:100],  # keep top 100 for fallback
        "vocab": vocab,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(model, f)
    print(f"Saved baseline model to {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1024:.1f} KB)")

    # Spot check
    print("\n--- Spot Check ---")
    for cmd_name in ["Tool: Wall", "Tool: Selection", "Menu: Undo"]:
        cmd_id = vocab.get(cmd_name)
        if cmd_id is None:
            print(f"  {cmd_name}: not in vocab")
            continue
        probs = transition_probs.get(str(cmd_id), {})
        top5 = list(probs.items())[:5]
        print(f"  After '{cmd_name}' (id={cmd_id}):")
        for nid, prob in top5:
            name = reverse_vocab.get(int(nid), f"UNK({nid})")
            print(f"    {name}: {prob:.4f}")


if __name__ == "__main__":
    main()
