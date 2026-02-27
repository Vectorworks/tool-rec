"""
inference_server.py

FastAPI server for BIM command recommendation inference.
Runs on EC2 with GPU, serves predictions to remote clients.

Usage:
    CUDA_HOME=/usr/local/cuda conda run -n t4rec_23.06_new_transformers \
        python prototype/inference_server.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import torch
from typing import List, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "tmp_test/checkpoint-4000/pytorch_model.bin"
VOCAB_PATH = "data/command_vocab.json"
AUGMENTATIONS_PATH = "data/command_information_augmentations.csv"
METADATA_PATH = "data/inference_metadata.json"
SCHEMA_DATASET_PATH = "data/processed_nvt/part_0.parquet"
MAX_SEQ_LEN = 110
TOP_K = 10
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# ──────────────────────────────────────────────────────────────────────────────


# ── Request/Response models ───────────────────────────────────────────────────
class PredictRequest(BaseModel):
    commands: List[str]
    top_k: Optional[int] = TOP_K


class Prediction(BaseModel):
    rank: int
    command: str
    score: float


class PredictResponse(BaseModel):
    predictions: List[Prediction]
    input_length: int
# ──────────────────────────────────────────────────────────────────────────────


def load_model():
    """Build model architecture and load trained weights."""
    from numba import config as numba_config
    numba_config.CUDA_LOW_OCCUPANCY_WARNINGS = 0
    import numba.cuda.cudadrv.driver as _numba_driver

    _orig = _numba_driver._ActiveContext.__enter__
    def _patched(self):
        try:
            return _orig(self)
        except _numba_driver.CudaAPIError as e:
            if e.code == 201:
                self._is_top = False
                self.context_handle = None
                self.devnum = None
                return self
            raise
    _numba_driver._ActiveContext.__enter__ = _patched

    from merlin.io import Dataset
    sys.path.insert(0, 'transformers4rec')
    from transformers4rec import torch as tr
    from transformers4rec.torch.masking import CausalLanguageModeling
    from model.patches import _pad_across_processes, _compute_masked_targets_mask_last_item

    CausalLanguageModeling._compute_masked_targets = _compute_masked_targets_mask_last_item

    # Build model architecture (same as training)
    train_ds = Dataset(SCHEMA_DATASET_PATH, engine="parquet", cpu=True)
    schema = train_ds.schema
    schema = schema.select_by_name([
        'item_id-list', 'classification-list', 'target-list',
        'merge_count_norm-list', 'timestamp_interval_norm_global-list',
    ])

    item_cardinality = schema["item_id-list"].int_domain.max + 1
    max_sequence_length, d_model = MAX_SEQ_LEN, 1024

    input_module = tr.TabularSequenceFeatures.from_schema(
        schema,
        max_sequence_length=max_sequence_length,
        continuous_projection=1024,
        embedding_dim_default=1024,
        embedding_dims={"item_id-list": 1024},
        d_output=d_model,
        masking="clm",
        multi_task_labels=True,
        custom_aggregation=True,
        self_attention_agg=True,
        att_pooling=True,
    )

    from transformers4rec.torch.ranking_metric import NDCGAt, RecallAt, MeanReciprocalRankAt
    prediction_task = tr.NextItemPredictionTask(
        weight_tying=False,
        sampled_softmax=False,
        metrics=[
            NDCGAt(top_ks=[3, 5, 10], labels_onehot=True),
            RecallAt(top_ks=[3, 5, 10], labels_onehot=True),
            MeanReciprocalRankAt(top_ks=[3, 5, 10], labels_onehot=True),
        ],
    )

    transformer_config = tr.MixtralConfig.build(
        hidden_size=d_model,
        n_head=16,
        n_layer=2,
        num_experts_per_tok=2,
        num_local_experts=8,
        total_seq_length=max_sequence_length,
        intermediate_size=3584,
    )

    model = transformer_config.to_torch_model(input_module, prediction_task)

    # Load trained weights
    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.top_k = TOP_K
    model.eval()
    model.to(DEVICE)

    print(f"Model loaded on {DEVICE} ({sum(p.numel() for p in model.parameters()):,} params)")
    return model


def load_lookups():
    """Load vocab, augmentations, and encoding metadata."""
    import pandas as pd

    with open(VOCAB_PATH) as f:
        vocab = json.load(f)
    vocab = {str(k): int(v) for k, v in vocab.items()}
    reverse_vocab = {v: k for k, v in vocab.items()}

    aug = pd.read_csv(AUGMENTATIONS_PATH)
    aug_lookup = {}
    for _, row in aug.iterrows():
        cmd = str(row['message_content'])
        aug_lookup[cmd] = {
            'classification': str(row.get('classification', 'other')),
            'target': str(row.get('target', 'other')),
        }

    with open(METADATA_PATH) as f:
        meta = json.load(f)

    return vocab, reverse_vocab, aug_lookup, meta


# ── Global state ──────────────────────────────────────────────────────────────
print("Loading model...")
_model = load_model()
print("Loading lookups...")
_vocab, _reverse_vocab, _aug_lookup, _meta = load_lookups()
print(f"Ready! vocab={len(_vocab)} commands")


# ── Inference logic ───────────────────────────────────────────────────────────
def encode_session(commands: List[str], top_k: int = TOP_K) -> PredictResponse:
    """Encode a command sequence and run model inference."""
    cls_enc = _meta['classification_enc']
    tgt_enc = _meta['target_enc']
    ti_mean = _meta['ti_mean']
    ti_std = _meta['ti_std']

    # Truncate to max sequence length
    commands = commands[-MAX_SEQ_LEN:]
    seq_len = len(commands)

    item_ids = []
    classifications = []
    targets = []
    merge_counts_norm = []
    ti_norms = []

    for cmd in commands:
        # Item ID
        item_id = _vocab.get(cmd, 0)
        item_ids.append(item_id)

        # Classification and target from augmentation
        aug = _aug_lookup.get(cmd, {'classification': 'other', 'target': 'other'})
        classifications.append(cls_enc.get(aug['classification'], cls_enc.get('other', 20)))
        targets.append(tgt_enc.get(aug['target'], tgt_enc.get('other', 18)))

        # Continuous features (merge_count is always 1, ti approximated as 0)
        merge_counts_norm.append(0.0)
        ti_norms.append((0.0 - ti_mean) / ti_std)

    # Pad to MAX_SEQ_LEN
    def pad_int(lst):
        return lst + [0] * (MAX_SEQ_LEN - len(lst))

    def pad_float(lst):
        return lst + [0.0] * (MAX_SEQ_LEN - len(lst))

    # Build input tensors [1, MAX_SEQ_LEN]
    input_dict = {
        'item_id-list': torch.tensor([pad_int(item_ids)], dtype=torch.long),
        'classification-list': torch.tensor([pad_int(classifications)], dtype=torch.long),
        'target-list': torch.tensor([pad_int(targets)], dtype=torch.long),
        'merge_count_norm-list': torch.tensor([pad_float(merge_counts_norm)], dtype=torch.float32),
        'timestamp_interval_norm_global-list': torch.tensor([pad_float(ti_norms)], dtype=torch.float32),
    }

    # Move to device
    input_dict = {k: v.to(DEVICE) for k, v in input_dict.items()}

    # Run inference
    with torch.no_grad():
        output = _model(input_dict)

    # output is (scores, item_ids) tuple when top_k is set
    if isinstance(output, tuple):
        scores, pred_ids = output
    else:
        scores, pred_ids = torch.topk(output, k=top_k, dim=-1)

    scores = scores[0].cpu().numpy()
    pred_ids = pred_ids[0].cpu().numpy()

    # Apply softmax to scores for interpretability
    scores_softmax = np.exp(scores) / np.exp(scores).sum()

    predictions = []
    for i, (pid, score) in enumerate(zip(pred_ids, scores_softmax)):
        cmd_name = _reverse_vocab.get(int(pid), f"unknown_{pid}")
        predictions.append(Prediction(rank=i + 1, command=cmd_name, score=float(score)))

    return PredictResponse(predictions=predictions[:top_k], input_length=seq_len)


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="BIM Command Recommender")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "vocab_size": len(_vocab)}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.commands:
        return PredictResponse(predictions=[], input_length=0)
    top_k = min(req.top_k or TOP_K, TOP_K)
    return encode_session(req.commands, top_k=top_k)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
