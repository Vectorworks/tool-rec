"""
sagemaker_serve.py

FastAPI server for FAISS retrieval SageMaker endpoint.
Endpoints: GET /ping, POST /invocations
Port: 8080 (SageMaker requirement).
Model artifacts loaded from /opt/ml/model/.
"""

import time
import json
import numpy as np
import faiss
from fastapi import FastAPI, Response
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import uvicorn


MODEL_DIR = "/opt/ml/model"

# ── Request/Response models ──────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=100)


class RecommendRequest(BaseModel):
    session: list[str]
    top_k: int = Field(default=10, ge=1, le=100)


class InvocationRequest(BaseModel):
    action: str  # "search" or "recommend"
    query: str | None = None
    session: list[str] | None = None
    top_k: int = Field(default=10, ge=1, le=100)


# ── Load artifacts ───────────────────────────────────────────────────────────

print("Loading embeddings...")
embeddings = np.load(f"{MODEL_DIR}/command_embeddings.npy")

with open(f"{MODEL_DIR}/metadata.json") as f:
    metadata = json.load(f)

id_to_name = metadata["id_to_name"]
name_to_id = {v: int(k) for k, v in id_to_name.items()}
num_commands = metadata["num_commands"]

print("Loading FAISS index...")
index = faiss.read_index(f"{MODEL_DIR}/command.index")
print(f"Index loaded: {index.ntotal} vectors")

print(f"Loading encoder: {metadata['model']}...")
encoder = SentenceTransformer(metadata["model"])
print("Ready.")


# ── Search logic ─────────────────────────────────────────────────────────────

def search_by_vector(query_vector: np.ndarray, top_k: int = 10) -> list[dict]:
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)
    query_vector = query_vector.astype(np.float32)
    scores, indices = index.search(query_vector, top_k + 1)
    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx <= 0:
            continue
        name = id_to_name.get(str(idx), f"unknown_{idx}")
        results.append({"command_id": int(idx), "name": name, "score": float(score)})
        if len(results) >= top_k:
            break
    return results


def search_text(query: str, top_k: int = 10) -> list[dict]:
    query_embedding = encoder.encode([query], normalize_embeddings=True)
    return search_by_vector(query_embedding[0], top_k=top_k)


def recommend_from_session(session: list[str], top_k: int = 10) -> list[dict]:
    command_ids = [name_to_id[name] for name in session if name in name_to_id]
    if not command_ids:
        return []
    avg = embeddings[command_ids].mean(axis=0, keepdims=True)
    norm = np.linalg.norm(avg)
    if norm > 0:
        avg = avg / norm
    exclude = set(command_ids)
    raw = search_by_vector(avg, top_k=top_k + len(exclude))
    return [r for r in raw if r["command_id"] not in exclude][:top_k]


# ── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(title="BIM Command Retrieval")


@app.get("/ping")
def ping():
    return Response(content="", status_code=200)


@app.post("/invocations")
def invocations(req: InvocationRequest):
    t0 = time.perf_counter()

    if req.action == "search":
        if not req.query:
            return {"error": "query is required for search action", "results": []}
        results = search_text(req.query, top_k=req.top_k)
    elif req.action == "recommend":
        if not req.session:
            return {"error": "session is required for recommend action", "results": []}
        results = recommend_from_session(req.session, top_k=req.top_k)
    else:
        return {"error": f"unknown action: {req.action}", "results": []}

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return {"action": req.action, "results": results, "elapsed_ms": round(elapsed_ms, 2)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
