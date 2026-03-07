"""Generate sentence-transformer embeddings for all BIM commands."""

import json
import csv
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
VOCAB_PATH = DATA_DIR / "command_vocab.json"
AUGMENTATIONS_PATH = DATA_DIR / "command_information_augmentations.csv"
OUTPUT_DIR = DATA_DIR / "embeddings"


def load_augmentations() -> dict[str, dict]:
    """Load command descriptions from augmentations CSV. Returns {command_name: {summary, classification, target}}."""
    augmentations = {}
    with open(AUGMENTATIONS_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["message_content"].strip()
            augmentations[name] = {
                "summary": row.get("summary", "").strip(),
                "classification": row.get("classification", "").strip(),
                "target": row.get("target", "").strip(),
            }
    return augmentations


def generate_embeddings():
    """Generate L2-normalized embeddings for all commands in the vocab."""
    with open(VOCAB_PATH) as f:
        vocab = json.load(f)  # {command_name: command_id}

    augmentations = load_augmentations()
    num_commands = max(vocab.values())  # 1348

    # Row 0 = padding token (zero vector), rows 1..N = commands by ID
    texts = [""] * (num_commands + 1)
    id_to_name = {0: "<pad>"}

    for name, cmd_id in vocab.items():
        id_to_name[cmd_id] = name
        aug = augmentations.get(name)
        if aug and aug["summary"]:
            texts[cmd_id] = f"{name}: {aug['summary']}"
        else:
            texts[cmd_id] = name

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # Embed all texts (skip index 0 padding)
    print(f"Encoding {num_commands} commands...")
    embeddings = model.encode(texts[1:], show_progress_bar=True, normalize_embeddings=True)

    # Prepend zero vector for padding token
    pad_vector = np.zeros((1, EMBEDDING_DIM), dtype=np.float32)
    all_embeddings = np.vstack([pad_vector, embeddings]).astype(np.float32)
    print(f"Embeddings shape: {all_embeddings.shape}")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "command_embeddings.npy", all_embeddings)

    metadata = {
        "model": MODEL_NAME,
        "embedding_dim": EMBEDDING_DIM,
        "num_commands": num_commands,
        "id_to_name": {str(k): v for k, v in id_to_name.items()},
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved embeddings to {OUTPUT_DIR / 'command_embeddings.npy'}")
    print(f"Saved metadata to {OUTPUT_DIR / 'metadata.json'}")
    return all_embeddings, metadata


if __name__ == "__main__":
    generate_embeddings()
