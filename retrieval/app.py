"""FastAPI service for FAISS-based command retrieval."""

import time
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from retrieval.faiss_index import CommandIndex

app = FastAPI(title="BIM Command Retrieval", version="1.0.0")

# Global state loaded at startup
command_index: CommandIndex | None = None
encoder: SentenceTransformer | None = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=100)


class RecommendRequest(BaseModel):
    session: list[str]
    top_k: int = Field(default=10, ge=1, le=100)


@app.on_event("startup")
def startup():
    global command_index, encoder
    print("Loading FAISS index...")
    command_index = CommandIndex.load()
    print(f"Loaded index with {command_index.index.ntotal} vectors")
    print("Loading sentence-transformer model...")
    encoder = SentenceTransformer(command_index.metadata["model"])
    print("Ready.")


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "index_size": command_index.index.ntotal if command_index else 0,
        "model": command_index.metadata["model"] if command_index else None,
        "embedding_dim": command_index.metadata["embedding_dim"] if command_index else None,
        "num_commands": command_index.num_commands if command_index else 0,
    }


@app.post("/search")
def search(req: SearchRequest):
    t0 = time.perf_counter()
    query_embedding = encoder.encode([req.query], normalize_embeddings=True)
    results = command_index.search(query_embedding[0], top_k=req.top_k)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return {"query": req.query, "results": results, "elapsed_ms": round(elapsed_ms, 2)}


@app.post("/recommend")
def recommend(req: RecommendRequest):
    t0 = time.perf_counter()
    results = command_index.search_by_names(req.session, top_k=req.top_k)
    if not results:
        raise HTTPException(status_code=404, detail="No matching commands found for the given session")
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return {"session": req.session, "results": results, "elapsed_ms": round(elapsed_ms, 2)}


@app.get("/embedding/{command_name:path}")
def get_embedding(command_name: str):
    cmd_id = command_index.name_to_id.get(command_name)
    if cmd_id is None:
        raise HTTPException(status_code=404, detail=f"Command not found: {command_name}")
    embedding = command_index.get_embedding(cmd_id)
    return {
        "command_name": command_name,
        "command_id": cmd_id,
        "embedding": embedding.tolist(),
    }
