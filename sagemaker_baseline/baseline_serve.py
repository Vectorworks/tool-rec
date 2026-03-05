"""FastAPI server for bigram baseline model on SageMaker."""

import json
import os
from typing import List, Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Globals loaded at startup
MODEL = None
VOCAB = None
REVERSE_VOCAB = None


class PredictRequest(BaseModel):
    commands: List[str]
    top_k: Optional[int] = 10


class Prediction(BaseModel):
    rank: int
    command: str
    score: float


class PredictResponse(BaseModel):
    predictions: List[Prediction]
    input_length: int


def load_model():
    global MODEL, VOCAB, REVERSE_VOCAB
    model_dir = os.environ.get("MODEL_DIR", "/opt/ml/model")
    model_path = os.path.join(model_dir, "baseline_model.json")
    print(f"Loading baseline model from {model_path}")
    with open(model_path) as f:
        MODEL = json.load(f)
    VOCAB = MODEL["vocab"]
    REVERSE_VOCAB = {v: k for k, v in VOCAB.items()}
    print(f"Loaded: {len(VOCAB)} vocab, {len(MODEL['transitions'])} transition entries")


@app.on_event("startup")
def startup():
    load_model()


@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.post("/invocations", response_model=PredictResponse)
def invocations(request: PredictRequest):
    top_k = request.top_k or 10
    commands = request.commands

    if not commands:
        # No input — return global top
        predictions = _global_top(top_k)
        return PredictResponse(predictions=predictions, input_length=0)

    last_command = commands[-1]
    cmd_id = VOCAB.get(last_command)

    if cmd_id is not None:
        probs = MODEL["transitions"].get(str(cmd_id), {})
    else:
        probs = {}

    if probs:
        # Sort by probability (already sorted in training, but be safe)
        sorted_probs = sorted(probs.items(), key=lambda x: -x[1])[:top_k]
        predictions = [
            Prediction(
                rank=i + 1,
                command=REVERSE_VOCAB.get(int(nid), f"UNK({nid})"),
                score=round(prob, 6),
            )
            for i, (nid, prob) in enumerate(sorted_probs)
        ]
    else:
        # Unknown command or no transitions — fallback
        predictions = _global_top(top_k)

    return PredictResponse(predictions=predictions, input_length=len(commands))


def _global_top(top_k: int) -> list:
    return [
        Prediction(
            rank=i + 1,
            command=REVERSE_VOCAB.get(entry["id"], f"UNK({entry['id']})"),
            score=round(entry["prob"], 6),
        )
        for i, entry in enumerate(MODEL["global_top"][:top_k])
    ]


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
