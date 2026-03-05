# BIM Command Recommendation — Codebase Overview

## What This System Does

A deep learning system that predicts the next BIM command a Vectorworks user will invoke, based on their recent sequence of tool/menu interactions. Built on NVIDIA's Transformers4Rec framework with a customized Mixtral architecture.

**Input:** Sequence of recent Tool/Menu commands (e.g., `Tool: Line → Menu: Save → Tool: Move by Points → ...`)
**Output:** Top-5 predicted next commands with confidence scores

---

## Data Pipeline

### Raw Data
- **Source:** `s3://vectorworks-analytics-datalake/athena_query_results/...` (81.1 GB CSV)
- **Contents:** Vectorworks application telemetry logs — every tool activation, menu click, undo, etc.
- **Columns:** `session_id`, `ts` (timestamp), `message` (raw command string), `cat` (category: Tool/Menu/UNDO)

### Stage 1: Filtering (`data_processing/stream_stage1_filter.py`)

Parallel S3 ingestion with 32 workers, each reading a byte-range slice of the raw CSV.

**What gets kept:**
- Only `Tool` and `Menu` category rows (`UNDO` dropped entirely)
- Only ASCII-encodable commands (English allowlist)
- Only commands present in the augmentations file (1,803 known commands)

**What gets removed:**
- Navigation noise: zoom, pan, view rotation, scroll commands
- Undo/redo events
- Localization suffixes: `"Tool: Round Rect (-217)"` → `"Tool: Round Rect"`

**Output:** `data/stage1_filtered/` — 4,736,293 rows across 2,575 parquet files (48 MB)

### Stage 2: ETL & Preprocessing (`model/prepare_data.py`)

Orchestrates the full pipeline from filtered data to train-ready parquet:

1. **Load** all stage1 parquets
2. **Compute `timestamp_interval`** — seconds between consecutive actions in each session
3. **Split long sessions** — random split points between 10–100 items; segments <10 items are merged into the previous chunk
4. **Merge augmentations** — joins `classification` and `target` from `command_information_augmentations.csv`
5. **Build vocabulary** — 1,348 unique commands mapped to 1-indexed integers → `data/command_vocab.json`
6. **CPU preprocessing** (`model/preprocess.py:preprocessing_cpu()`):
   - **Categorify:** String columns → integer IDs (item_id, classification, target, cat)
   - **Normalize:** Z-score normalization for `merge_count` and `timestamp_interval`
   - **Groupby:** Aggregate per session into ragged lists (each row = one session's full sequence)
   - **ListSlice:** Truncate to last 200 items per session
   - **Filter:** Remove sessions with fewer than 5 interactions
7. **Generate Merlin schema** (`schema.pbtxt`) — describes column types, domains, and tags for the model loader
8. **Train/val split** (85/15) — balanced so every command appears in both splits

**Output:**
- `data/processed_nvt/part_0.parquet` + `schema.pbtxt`
- `data/preproc_sessions/train_new/` (73,578 sessions)
- `data/preproc_sessions/val_new/` (14,022 sessions)

### Command Augmentations (`data_processing/command_augmentation_and_workflow_generation/generate_augmentation.py`)

Uses Claude (claude-haiku-4-5-20251001) to generate metadata for each command:
- **Summary:** 2-3 sentence description of what the command does
- **Classification:** Action type — one of 20 categories (Create, View, Update, Convert, Export, Render, Group, Import, Annotate, Select, Align, Move, Copy, Connect, Arrange, Delete, Analyze, Combine, Paste, other)
- **Target:** Primary object type — one of 18 categories (Object, Viewport, Palette, Layer, Surface, Text, Symbol, Annotation, Group, Line, Shape, Polygon, Wall, Dimension, Window, Door, Curve, other)

Stored at `data/command_information_augmentations.csv` (1,803 rows).

---

## Model Architecture

### Features Used by the Model

| Feature | Type | Description |
|---|---|---|
| `item_id-list` | Categorical | The command itself (1,348 unique, 1-indexed) |
| `classification-list` | Categorical | Action type from augmentations (20 categories) |
| `target-list` | Categorical | Object type from augmentations (18 categories) |
| `merge_count_norm-list` | Continuous | Normalized merge count (currently all 1s) |
| `timestamp_interval_norm_global-list` | Continuous | Normalized time between actions |

### Supported Architectures (`model/train_eval_full_models.py`)

| Model | Hidden Size | Heads | Layers | Notes |
|---|---|---|---|---|
| **Mixtral** (default) | 1024 | 16 | 2 | 8 experts, 2 active per token |
| LLaMA | 2048 | 32 | 2 | Causal LM |
| LLaMA LoRA | 4096 | 32 | 32 | Pretrained LLaMA2-7B + LoRA adapters |
| BERT | 1024 | 16 | 2 | Masked LM |
| BERT Base | 768 | 12 | 12 | Pretrained BERT-base + fine-tune |
| BERT Large | 1024 | 16 | 24 | Pretrained BERT-large + fine-tune |
| T5 | 1024 | 8 | 2 | Encoder-decoder MLM |

### Input Module

`TabularSequenceFeatures.from_schema()` combines:
- Categorical embeddings (default 1024-dim, item_id explicitly 1024-dim)
- Continuous projection (1024-dim)
- Self-attention aggregation across feature types
- Causal language modeling masking (CLM for autoregressive models, MLM for BERT/T5)
- Multi-task labels for auxiliary classification/target prediction

### Prediction Task

`NextItemPredictionTask` with:
- Full softmax over vocabulary (no sampled softmax)
- Metrics: NDCG@{3,5,10}, Recall@{3,5,10}, MRR@{3,5,10}
- Optional pretrained OpenAI text embeddings (3072-dim) for item_id

### Training Config

- Optimizer: AdamW, lr=3e-5
- FP16 mixed precision
- 10 epochs, early stopping (patience=10)
- Eval every 500 steps, save best model by eval loss
- Logging: TensorBoard + MLflow

### Pretrained Embeddings (Optional)

`model/pretrained_text_embedding.py` generates OpenAI `text-embedding-3-large` vectors (3072-dim) for each command's description. Saved as `data/pre-trained-item-id-new-data_0122.npy`. If the file exists at training time, it's loaded via `EmbeddingOperator`; otherwise the model learns embeddings from scratch.

---

## Differences from Original Deployment Method

### What Changed: GPU Preprocessing → CPU Preprocessing

The original pipeline used **NVTabular** (GPU-accelerated) for preprocessing (`model/preprocess.py:preprocessing()`). This relied on cuDF and Dask-cuDF for GPU dataframe operations.

**Problem:** The EC2 instance has CUDA driver 13.0, which is too new for the cuDF version in the `t4rec_23.06` conda environment. Any cuDF operation triggers a segfault in `numba.cuda.as_cuda_array()` → `safe_cuda_api_call()`.

**Fix:** `preprocessing_cpu()` in `model/preprocess.py` — a pure pandas reimplementation that produces identical output:
- Categorify: Manual dictionary encoding instead of NVTabular `Categorify` op
- Normalize: `(x - mean) / std` instead of NVTabular `Normalize` op
- Groupby: `pandas.groupby().agg(list)` instead of NVTabular `Groupby` op
- Filter/ListSlice: Native pandas instead of NVTabular ops
- Schema generation: Direct Merlin `ColumnSchema` API instead of NVTabular workflow inference

### What Changed: No Triton Serving (Yet)

The original deployment used **NVIDIA Triton Inference Server** via Docker:

```
docker-compose.dev.deploy.yml
  → tritonserver:23.08-py3
  → Mounts ens_models_mixtral/ as model repository
  → Exposes ports 8000 (HTTP), 8001 (gRPC), 8002 (metrics)
```

The Triton ensemble pipeline was:
1. **TransformWorkflow** (NVTabular) — preprocess raw input using saved NVTabular workflow
2. **EmbeddingOperator** — look up pretrained embeddings
3. **PredictPyTorch** — run the model (loaded via cloudpickle, not TorchScript — because `torch.jit.trace` doesn't work with the custom architecture)

**Current state:** Triton deployment is **not active**. The export script (`deployment/ensemble_peft_llm.py`) depends on `nvtabular.Workflow.load()` to load a saved NVTabular workflow, but our CPU preprocessing doesn't produce one. To re-enable Triton, we would need to either:
1. Save a compatible NVTabular workflow artifact from the CPU pipeline, or
2. Write a custom Triton Python backend that replaces the NVTabular preprocessing step

The trained model checkpoint itself (`tmp_test/checkpoint-*/pytorch_model.bin`) is ready; it's the serving wrapper that needs adaptation.

### What Changed: wandb → MLflow

- Original: Weights & Biases (`wandb`) for experiment tracking
- Current: MLflow (server on localhost:5000) + TensorBoard
- `wandb` import is commented out (broken by protobuf version conflict)

### What Stayed the Same

- Model architecture (Mixtral with custom Transformers4Rec modifications)
- Training logic, loss functions, metrics
- Data schema and feature engineering
- Custom monkey-patches in `model/patches.py` (pad_across_processes, CLM masking)
- Session splitting logic in `model/utils.py`

---

## Key Files

```
model/
  prepare_data.py          # Full ETL pipeline (stage1 → vocab → preprocess → split)
  preprocess.py            # preprocessing_cpu() and legacy preprocessing() (GPU)
  train_eval_full_models.py  # Training script (Mixtral default, MLflow enabled)
  train_eval_baseline_models.py  # Ablation baselines (no feature fusion)
  pretrained_text_embedding.py   # Generate OpenAI embeddings for commands
  utils.py                 # Session splitting, save callbacks, param counting
  patches.py               # Monkey-patches for T4Rec bugs (padding, CLM masking)

data_processing/
  stream_stage1_filter.py  # Parallel S3 reader + command filtering (32 workers)
  command_augmentation_and_workflow_generation/
    generate_augmentation.py  # Claude-based command metadata generation
    openai_sideinformation.py # Legacy OpenAI RAG approach (deprecated)
  actual_modeling_flow_tracking_and_log_filtering/
    prefiltering_groupby.py   # Legacy undo/redo tracking (deprecated)

deployment/
  ensemble_peft_llm.py     # Export model → Triton ensemble (LoRA or standard)
  deploy_template/model.py # Triton Python backend (cloudpickle-based)
  triton_inference.py       # Client-side Triton HTTP inference
  local_inference.py        # Local checkpoint evaluation

transformers4rec/           # Customized NVIDIA Merlin fork
  config/transformer.py     # Model config builders (BERT, T5, LLaMA, Mixtral)
  torch/features/tabular.py # Input module (embedding + continuous projection)
  torch/model/prediction_task.py  # NextItemPrediction, multi-task, focal loss
  torch/masking.py          # CLM/MLM masking with auxiliary targets

docker-compose.dev.deploy.yml  # Triton server config (not currently used)
```

---

## Running the Pipeline

```bash
# 1. Filter raw S3 data (one-time, already done)
conda run -n t4rec_23.06_new_transformers python data_processing/stream_stage1_filter.py

# 2. ETL: vocab, preprocess, train/val split (one-time, already done)
conda run -n t4rec_23.06_new_transformers python model/prepare_data.py

# 3. Train
CUDA_HOME=/usr/local/cuda conda run -n t4rec_23.06_new_transformers python model/train_eval_full_models.py

# 4. MLflow UI (already running on port 5000)
conda run -n t4rec_23.06_new_transformers mlflow server \
  --backend-store-uri file:///home/ec2-user/proj/BIM-Command-Recommendation/mlruns \
  --host 0.0.0.0 --port 5000

# 5. TensorBoard
conda run -n t4rec_23.06_new_transformers tensorboard --logdir=./tmp_test --host 0.0.0.0 --port 6006
```
