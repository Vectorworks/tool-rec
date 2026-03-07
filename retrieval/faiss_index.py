"""FAISS index for command embedding retrieval."""

import json
import numpy as np
import faiss
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "embeddings"
EMBEDDINGS_PATH = DATA_DIR / "command_embeddings.npy"
METADATA_PATH = DATA_DIR / "metadata.json"
INDEX_PATH = DATA_DIR / "command.index"


class CommandIndex:
    def __init__(self, embeddings: np.ndarray, metadata: dict, index: faiss.Index):
        self.embeddings = embeddings
        self.metadata = metadata
        self.index = index
        self.id_to_name = metadata["id_to_name"]
        self.name_to_id = {v: int(k) for k, v in self.id_to_name.items()}
        self.num_commands = metadata["num_commands"]

    @classmethod
    def build(cls, embeddings: np.ndarray | None = None, metadata: dict | None = None) -> "CommandIndex":
        """Build a new FAISS index from embeddings."""
        if embeddings is None:
            embeddings = np.load(EMBEDDINGS_PATH)
        if metadata is None:
            with open(METADATA_PATH) as f:
                metadata = json.load(f)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        # Add all vectors including padding at index 0
        index.add(embeddings)

        return cls(embeddings, metadata, index)

    def save(self, path: Path | None = None):
        path = path or INDEX_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        print(f"Saved FAISS index to {path} ({self.index.ntotal} vectors)")

    @classmethod
    def load(cls, index_path: Path | None = None) -> "CommandIndex":
        """Load a pre-built FAISS index."""
        index_path = index_path or INDEX_PATH
        embeddings = np.load(EMBEDDINGS_PATH)
        with open(METADATA_PATH) as f:
            metadata = json.load(f)
        index = faiss.read_index(str(index_path))
        return cls(embeddings, metadata, index)

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> list[dict]:
        """Search for nearest commands given a query vector (1, dim)."""
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        query_vector = query_vector.astype(np.float32)

        scores, indices = self.index.search(query_vector, top_k + 1)  # +1 to skip padding if it appears

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx <= 0:  # skip padding
                continue
            name = self.id_to_name.get(str(idx), f"unknown_{idx}")
            results.append({"command_id": int(idx), "name": name, "score": float(score)})
            if len(results) >= top_k:
                break
        return results

    def search_by_ids(self, command_ids: list[int], top_k: int = 10, exclude_input: bool = True) -> list[dict]:
        """Search using the average embedding of given command IDs."""
        valid_ids = [cid for cid in command_ids if 0 < cid <= self.num_commands]
        if not valid_ids:
            return []

        avg_embedding = self.embeddings[valid_ids].mean(axis=0, keepdims=True)
        # Re-normalize
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm

        exclude_set = set(valid_ids) if exclude_input else set()
        # Fetch extra results to account for excluded IDs
        raw_results = self.search(avg_embedding, top_k=top_k + len(exclude_set))
        return [r for r in raw_results if r["command_id"] not in exclude_set][:top_k]

    def search_by_names(self, command_names: list[str], top_k: int = 10) -> list[dict]:
        """Search using command names (resolved to IDs)."""
        command_ids = [self.name_to_id[name] for name in command_names if name in self.name_to_id]
        return self.search_by_ids(command_ids, top_k=top_k)

    def get_embedding(self, command_id: int) -> np.ndarray | None:
        if 0 <= command_id <= self.num_commands:
            return self.embeddings[command_id]
        return None
