# Technical Overview: BIM Command Recommendation System

> **Paper**: [Predictive Modeling: BIM Command Recommendation Based on Large-scale Usage Logs](https://arxiv.org/abs/2504.05319)
> **Authors**: Changyu Du, Zihan Deng, Stavros Nousias, André Borrmann
> **Dataset**: 32+ billion rows of real-world Vectorworks BIM log data
> **Best Result**: Recall@10 ≈ 84%

---

## Table of Contents

1. [Problem Formulation](#1-problem-formulation)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Data Processing Pipeline](#3-data-processing-pipeline)
4. [Feature Engineering and Preprocessing](#4-feature-engineering-and-preprocessing)
5. [Input Module and Feature Fusion](#5-input-module-and-feature-fusion)
6. [Transformer Backbones (with Math)](#6-transformer-backbones-with-math)
7. [Loss Functions](#7-loss-functions)
8. [Multi-Task Learning](#8-multi-task-learning)
9. [Parameter-Efficient Fine-Tuning (QLoRA)](#9-parameter-efficient-fine-tuning-qlora)
10. [Masking Strategies](#10-masking-strategies)
11. [Training Configuration](#11-training-configuration)
12. [Evaluation Metrics](#12-evaluation-metrics)
13. [Deployment](#13-deployment)

---

## 1. Problem Formulation

The task is **next-command recommendation** in a BIM authoring tool (Vectorworks). Given a user's sequence of past commands within a session, the model predicts the most likely next command(s).

Formally: given a session history $S = (c_1, c_2, \ldots, c_{T-1})$ with associated side features (timestamps, command categories, LLM-generated metadata), predict $c_T$ — the next command.

This is framed as a **sequential recommendation problem** solved with transformer-based sequence models trained under causal (CLM) or masked (MLM) language modeling objectives.

---

## 2. System Architecture Overview

```
Raw BIM Logs (32B rows)
        │
        ▼
┌─────────────────────────────┐
│   Data Processing Pipeline  │   (data_processing/)
│  - Undo/Redo reconstruction │
│  - Multi-language alignment │
│  - Redundancy removal       │
│  - BPE workflow generation  │
│  - LLM side-info via RAG    │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│   Preprocessing (NVTabular) │   (model/preprocess.py)
│  - Categorify (integer IDs) │
│  - Normalize (continuous)   │
│  - Groupby session          │
│  - Train/val split          │
└────────────┬────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────┐
│              Full Model Architecture               │
│                                                    │
│  ┌─────────────────────────────────────────────┐  │
│  │          Input Module (TabularSequence)     │  │
│  │  item_id embedding  ─────────┐              │  │
│  │  classification emb ─────────┤  Self-Attn   │  │
│  │  target embedding  ──────────┤  Fusion   ──►│  │
│  │  continuous projection ──────┤  + Attn   │  │  │
│  │  pretrained text emb ────────┘  Pooling  │  │  │
│  └──────────────────────────────────────────┘  │  │
│                       │                         │  │
│                       ▼                         │  │
│  ┌──────────────────────────────────────────┐  │  │
│  │        Transformer Backbone              │  │  │
│  │   (LLaMA / Mixtral / BERT / T5)         │  │  │
│  └──────────────────────────────────────────┘  │  │
│                       │                         │  │
│                       ▼                         │  │
│  ┌──────────────────────────────────────────┐  │  │
│  │    Prediction Head (NextItemPrediction)  │  │  │
│  │  Main:  Linear → Softmax → item_id       │  │  │
│  │  Aux 1: MLP → classification             │  │  │
│  │  Aux 2: MLP → target                     │  │  │
│  └──────────────────────────────────────────┘  │  │
└────────────────────────────────────────────────────┘
```

---

## 3. Data Processing Pipeline

The raw logs go through five sequential stages implemented in `data_processing/`.

### Stage 1: Actual Modeling Flow Tracking

Raw logs contain noise: zoom operations, viewport changes, UI events, and undo/redo pairs that distort the true modeling sequence.

**Undo/Redo reconstruction logic:**
- When `"Undo Event: X"` is encountered, find the most recent `"End Event: X"` and mark both for removal.
- When `"Redo Event: X"` is encountered, un-mark the corresponding undo removal (restoring the original event) and mark the redo event itself for removal.

This reconstructs the *actual* sequence of modeling intent, not the raw interaction log.

**Category filter**: Only `UNDO`, `Menu`, and `Tool` categories are retained.

**Regex-based noise removal**: Commands matching patterns like zoom (event code 242), view changes, and internal system events are stripped.

### Stage 2: Multi-Language Alignment

Vectorworks is used globally; logs arrive in many languages. Commands with identical semantics but different language strings are unified:

1. Translate non-English strings to English via Google Translate API.
2. Embed all command strings using Voyage AI embeddings.
3. Cluster with scikit-learn.
4. Produce `command_dictionary.csv` mapping all variants to a canonical form.

### Stage 3: Redundant Command Identification

Some high-level commands always co-occur with low-level ones (e.g., a menu command always triggers a tool command). These redundant high-level commands inflate the sequence and are removed using **association rule mining**:

$$\text{support}(A \to B) = \frac{|\{s : A \in s \wedge B \in s\}|}{|S|}$$

$$\text{confidence}(A \to B) = \frac{|\{s : A \in s \wedge B \in s\}|}{|\{s : A \in s\}|}$$

$$\text{lift}(A \to B) = \frac{\text{confidence}(A \to B)}{\text{support}(B)}$$

Command pairs with high confidence and lift are validated manually in Vectorworks and marked for removal.

### Stage 4: Log Filtering

Apply the command dictionary mapping (Stage 2) and remove redundant commands (Stage 3) to produce the **Aligned Logs**.

### Stage 5: Command Augmentation via RAG

Each unique command is enriched with three LLM-generated metadata fields using a **Retrieval-Augmented Generation (RAG)** pipeline (`data_processing/command_augmentation_and_workflow_generation/openai_sideinformation.py`):

1. BIM documentation (Markdown format) is chunked and embedded with OpenAI `text-embedding-3-large` (768-dim projection).
2. Stored in a **Chroma** vector database.
3. For each command:
   - Retrieve top-3 most relevant documentation chunks (cosine similarity).
   - Prompt **GPT-4o-mini** (temperature=0.2) with the retrieved context.
   - Extract: `summary` (2–3 sentence description), `classification` (1-word: Create/Update/Delete/etc.), `target` (1-word: Object/Group/Layer/etc.).

Additionally, **Byte Pair Encoding (BPE)** is applied to discover frequent command sub-sequences and merge them into *workflow tokens*, expanding the vocabulary with composite commands.

---

## 4. Feature Engineering and Preprocessing

Implemented with **NVTabular** (GPU-accelerated ETL) in `model/preprocess.py`.

### 4.1 Column Preparation

| Raw Column | Renamed To | Type |
|---|---|---|
| `session_anonymized` | `session_id` | string |
| `message_content` | `item_id` | string → int |
| `ts` | `timestamp` | unix float |

### 4.2 Timestamp Interval Feature

For each session, compute the inter-event time delta:

$$\Delta t_i = t_i - t_{i-1}, \quad \Delta t_1 = 0$$

This is stored as `timestamp_interval`.

### 4.3 Categorical Encoding (Categorify)

All categorical columns are integer-encoded with contiguous IDs:

| Feature | Encoding |
|---|---|
| `item_id` | Unique integer per command |
| `classification` | Integer per class label |
| `target` | Integer per target label |
| `cat` | Integer per category |

### 4.4 Continuous Feature Normalization

Z-score normalization (global statistics across all sessions):

$$x_{\text{norm}} = \frac{x - \mu}{\sigma}$$

Applied to:
- `timestamp_interval` → `timestamp_interval_norm_global`
- `merge_count` → `merge_count_norm`

### 4.5 Session Groupby

All per-event features are aggregated into lists per `session_id`, sorted by `timestamp`:

```
session_id  →  item_id-list, classification-list, target-list,
               timestamp_interval-list, timestamp_interval_norm_global-list,
               merge_count-list, merge_count_norm-list, cat-list
```

### 4.6 Session Augmentation (Splitting)

Long sessions (potentially hundreds of commands) are split into shorter sub-sessions to generate more training samples and reduce padding waste. Each session is split into chunks of random length:

$$L_{\text{chunk}} \sim \mathcal{U}(\text{min\_items}=10,\ \text{max\_items}=100)$$

If the leftover tail is shorter than `min_items`, it is merged into the previous chunk. This is performed before NVTabular processing.

### 4.7 Sequence Truncation and Filtering

- **Truncation**: Only the last 200 items of each session are kept (`ListSlice(-200)`).
- **Filtering**: Sessions with fewer than 5 interactions are discarded.

### 4.8 Train/Validation Split (Balanced)

To ensure every command appears in both splits:

1. For each unique `item_id`, randomly select 2 sessions that contain it — assign one to train, one to val (guarantees coverage).
2. All remaining unassigned sessions are split 85% / 15% randomly.

This prevents cold-start artifacts where a command would only appear in one split.

### 4.9 Pretrained Text Embeddings

`model/pretrained_text_embedding.py` produces a lookup table of embeddings for use during training/inference:

1. For each unique command, call OpenAI `text-embedding-3-large` on its RAG-generated description → 3072-dim vector.
2. Truncate to first 1024 dimensions (optional).
3. Apply L2 normalization:

$$\hat{e} = \frac{e}{\|e\|_2}$$

4. Pad rows 0–2 with zero vectors (reserved padding/masking IDs).
5. Save as `.npy` array of shape `(item_cardinality, embedding_dim)`.

At training time, the `EmbeddingOperator` looks up each command's pretrained embedding by its integer ID and injects it into the batch.

---

## 5. Input Module and Feature Fusion

`transformers4rec/torch/features/sequence.py` — `TabularSequenceFeatures`

The input module combines five heterogeneous feature streams into a single per-timestep representation of dimension $d_{\text{model}}$.

### 5.1 Individual Feature Streams

| Stream | Raw Dim | Encoder |
|---|---|---|
| `item_id-list` | trainable embedding | `nn.Embedding(vocab, 1024)` |
| `classification-list` | trainable embedding | `nn.Embedding(vocab, 1024)` |
| `target-list` | trainable embedding | `nn.Embedding(vocab, 1024)` |
| Continuous (`timestamp_interval_norm`, `merge_count_norm`) | scalar | MLP projection → 1024 |
| `pretrained_item_id_embeddings` | 3072 (or 1024) | Linear → 1024 |

### 5.2 Per-Feature Linear Encoders

Each stream $f_k$ is projected to a common dimension $D = 1024$ via:

$$\hat{f}_k = \text{LayerNorm}\!\left(W_k f_k + b_k\right), \quad W_k \in \mathbb{R}^{D \times d_k}$$

### 5.3 Self-Attention Feature Fusion

At each timestep $t$, the $N$ encoded feature vectors are stacked into a feature set $F_t \in \mathbb{R}^{N \times D}$. Multi-head self-attention fuses them:

$$\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

where $Q = K = V = F_t$ (self-attention), with 4 attention heads over $D=1024$ dimensions.

$$d_k = D / h = 1024 / 4 = 256$$

The result is $\tilde{F}_t \in \mathbb{R}^{N \times D}$ — a set of context-enriched feature vectors.

### 5.4 Attention Pooling

A learned query vector $q \in \mathbb{R}^D$ pools the $N$ fused features into a single timestep embedding:

$$\alpha_i = \frac{\exp\!\left(\tilde{F}_{t,i} \cdot q / \sqrt{D}\right)}{\sum_j \exp\!\left(\tilde{F}_{t,j} \cdot q / \sqrt{D}\right)}$$

$$\mathbf{h}_t = \sum_{i=1}^{N} \alpha_i \tilde{F}_{t,i} \in \mathbb{R}^D$$

### 5.5 Projection to Model Dimension

The pooled representation is projected to the backbone's hidden size $d_{\text{model}}$:

$$\mathbf{x}_t = W_{\text{proj}} \mathbf{h}_t + b_{\text{proj}}, \quad W_{\text{proj}} \in \mathbb{R}^{d_{\text{model}} \times D}$$

The sequence $\mathbf{X} = (\mathbf{x}_1, \ldots, \mathbf{x}_T) \in \mathbb{R}^{T \times d_{\text{model}}}$ is fed into the transformer backbone.

---

## 6. Transformer Backbones (with Math)

All models are built on Hugging Face implementations, adapted via the custom `transformers4rec/` fork.

### 6.1 Common Transformer Components

#### Scaled Dot-Product Attention

For queries $Q \in \mathbb{R}^{T \times d_k}$, keys $K \in \mathbb{R}^{T \times d_k}$, values $V \in \mathbb{R}^{T \times d_v}$:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

#### Multi-Head Attention

$$\text{MHA}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

$$\text{head}_i = \text{Attention}(XW_i^Q,\ XW_i^K,\ XW_i^V)$$

where $W_i^Q, W_i^K, W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $d_k = d_{\text{model}} / h$, $W^O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$.

#### Feed-Forward Network (FFN)

$$\text{FFN}(x) = \text{act}(xW_1 + b_1)W_2 + b_2$$

where $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$, $W_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$.

#### Layer Normalization

$$\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$

#### Residual Connection

$$x \leftarrow x + \text{Sublayer}(\text{LayerNorm}(x)) \quad \text{(pre-norm)}$$

---

### 6.2 LLaMA

**Config** (custom, from-scratch training):

| Hyperparameter | Value |
|---|---|
| `d_model` | 2048 |
| Attention heads $h$ | 32 |
| $d_k = d_{\text{model}} / h$ | 64 |
| Layers | 2 |
| Max sequence length | 110 |
| Masking | CLM |

**Architecture differences from standard Transformer:**

1. **RMSNorm** (instead of LayerNorm): More efficient, no re-centering.

$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma, \quad \text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2}$$

2. **SwiGLU activation** in FFN:

$$\text{SwiGLU}(x, W, V, W_2) = (xW \odot \text{Swish}(xV)) W_2$$

$$\text{Swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

where $\odot$ is element-wise multiplication. The gate controls information flow through the FFN.

3. **Grouped Query Attention (GQA)**: Fewer key/value heads than query heads, reducing KV-cache memory during inference.

4. **Rotary Position Embeddings (RoPE)**: Encodes relative position by rotating query and key vectors in 2D subspaces:

$$\text{RoPE}(\mathbf{q}, m) = \mathbf{q} \cdot e^{im\theta}$$

In practice, pairs of dimensions $(q_{2i}, q_{2i+1})$ are rotated by angle $m \cdot \theta_i$ where $\theta_i = 10000^{-2i/d}$:

$$\begin{pmatrix} q_{2i}' \\ q_{2i+1}' \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}$$

This means the dot product $\langle \text{RoPE}(q, m), \text{RoPE}(k, n) \rangle$ depends only on the relative position $m - n$.

**Causal mask**: The attention mask enforces that position $t$ can only attend to positions $\leq t$:

$$M_{ij} = \begin{cases} 0 & \text{if } i \geq j \\ -\infty & \text{if } i < j \end{cases}$$

---

### 6.3 LLaMA + QLoRA (llama_lora)

Uses a full LLaMA2-7B backbone (pretrained) with QLoRA fine-tuning.

**Config**:

| Hyperparameter | Value |
|---|---|
| `d_model` | 4096 |
| Attention heads $h$ | 32 |
| Layers | 32 |
| Max sequence length | 110 |
| Quantization | 4-bit NF4 (bitsandbytes) |
| Masking | CLM |

**LoRA math** (see Section 9 for full details).

---

### 6.4 Mixtral (Mixture of Experts)

**Config**:

| Hyperparameter | Value |
|---|---|
| `d_model` (hidden size) | 1024 |
| Attention heads $h$ | 16 |
| $d_k$ | 64 |
| Layers | 2 |
| Local experts | 8 |
| Active experts per token | 2 |
| FFN intermediate size | 3584 |
| Max sequence length | 110 |
| Masking | CLM |

**Sparse Mixture of Experts (MoE) FFN:**

Instead of a single FFN, Mixtral has $E = 8$ expert FFNs and a learned router. For each token $x_t$:

**Router (gating network):**

$$g(x_t) = \text{Softmax}(W_g x_t) \in \mathbb{R}^E$$

where $W_g \in \mathbb{R}^{E \times d_{\text{model}}}$.

**Top-k selection** ($k = 2$):

$$\mathcal{T}(x_t) = \text{TopK}(g(x_t), k=2)$$

$$\hat{g}_i(x_t) = \begin{cases} g_i(x_t) / \sum_{j \in \mathcal{T}} g_j(x_t) & \text{if } i \in \mathcal{T}(x_t) \\ 0 & \text{otherwise} \end{cases}$$

**Sparse MoE output:**

$$\text{MoE}(x_t) = \sum_{i \in \mathcal{T}(x_t)} \hat{g}_i(x_t) \cdot \text{FFN}_i(x_t)$$

Only 2 of the 8 expert FFNs are evaluated per token, keeping compute cost equivalent to a ~2/8 fraction of a dense model while having 8× the parameter capacity.

**Load balancing auxiliary loss:**

To prevent all tokens from routing to the same few experts, a load-balancing loss $\mathcal{L}_{\text{aux}}$ is added during training:

$$\mathcal{L}_{\text{aux}} = \alpha \cdot E \cdot \sum_{i=1}^{E} f_i \cdot P_i$$

$$f_i = \frac{1}{T} \sum_{t=1}^{T} \mathbb{1}[i \in \mathcal{T}(x_t)], \quad P_i = \frac{1}{T} \sum_{t=1}^{T} g_i(x_t)$$

where $f_i$ is the fraction of tokens dispatched to expert $i$ and $P_i$ is the average router probability for expert $i$.

---

### 6.5 BERT

**Config** (custom, from-scratch or pretrained BERT-large):

| Variant | `d_model` | Heads | Layers | `d_ff` |
|---|---|---|---|---|
| BERT (custom) | 1024 | 16 | 2 | 4096 |
| BERT-base | 768 | 12 | 12 | 3072 |
| BERT-large | 1024 | 16 | 24 | 4096 |

All BERT variants use **bidirectional attention** (no causal mask) and are trained under **MLM** (see Section 10.2).

**BERT Attention** is standard scaled dot-product with no position mask, so each position can attend to all others. Positional information is injected via learned absolute position embeddings:

$$X_{\text{in}} = \text{TokenEmbedding}(c_t) + \text{PosEmbedding}(t) + \text{TypeEmbedding}$$

---

### 6.6 T5

**Config**:

| Hyperparameter | Value |
|---|---|
| `d_model` | 1024 |
| Attention heads $h$ | 8 |
| Layers | 2 |
| Max sequence length | 110 |
| Masking | MLM |
| Architecture | Encoder-Decoder |

T5 uses **relative position biases** (rather than absolute position embeddings) and a simplified architecture without bias terms in attention projections. The encoder processes the masked sequence; the decoder attends to encoder outputs via cross-attention to reconstruct masked tokens.

---

## 7. Loss Functions

### 7.1 Focal Loss (Primary)

Standard cross-entropy weighs all predictions equally, which is problematic for the highly imbalanced command distribution (some commands are far more common). **Focal Loss** down-weights easy examples:

$$\mathcal{L}_{\text{focal}}(p_t) = -(1 - p_t)^\gamma \log(p_t)$$

where:
- $p_t$ = model's predicted probability for the true class
- $\gamma = 2$ (focusing parameter) — higher $\gamma$ places more emphasis on hard, misclassified examples
- $(1 - p_t)^\gamma$ is the **modulating factor** — near zero for easy examples ($p_t \approx 1$), near one for hard examples ($p_t \approx 0$)

For a batch, the full computation is:

$$\mathcal{L}_{\text{focal}} = \frac{1}{N} \sum_{n=1}^{N} (1 - p_{t,n})^\gamma \cdot \text{CE}(logits_n, y_n)$$

This is applied to the main item-prediction head and both auxiliary heads.

### 7.2 Load-Balancing Loss (Mixtral only)

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{focal}} + \alpha_{\text{aux}} \cdot \mathcal{L}_{\text{aux}}$$

where $\mathcal{L}_{\text{aux}}$ encourages uniform expert utilization (see Section 6.4).

### 7.3 Multi-Task Loss

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{focal}}^{\text{item}} + \lambda_1 \mathcal{L}_{\text{focal}}^{\text{class}} + \lambda_2 \mathcal{L}_{\text{focal}}^{\text{target}}$$

where:
- $\mathcal{L}^{\text{item}}$: next command prediction loss
- $\mathcal{L}^{\text{class}}$: auxiliary classification loss (command category: Create/Update/etc.)
- $\mathcal{L}^{\text{target}}$: auxiliary target loss (object type: Object/Group/Layer/etc.)

---

## 8. Multi-Task Learning

The prediction head (`transformers4rec/torch/model/prediction_task.py`) branches into three tasks:

```
Transformer output h_t ∈ ℝ^{d_model}
         │
         ├──► W_item ∈ ℝ^{vocab_size × d_model}  ──► item_id logits (main task)
         │
         ├──► MLP_branch1 ∈ ℝ^{177}  ──► classification logits (177 classes)
         │
         └──► MLP_branch2 ∈ ℝ^{366}  ──► target logits (366 classes)
```

Each auxiliary head is a single linear layer (no bias, no activation — softmax is built into the loss):

$$\hat{y}_{\text{class}} = W_1 h_T, \quad W_1 \in \mathbb{R}^{177 \times d_{\text{model}}}$$

$$\hat{y}_{\text{target}} = W_2 h_T, \quad W_2 \in \mathbb{R}^{366 \times d_{\text{model}}}$$

Both auxiliary tasks also use **Focal Loss** with $\gamma = 2$.

The auxiliary tasks serve as regularizers: learning to predict the *type* and *target* of the next command forces the model to learn generalizable representations of BIM workflows beyond memorizing specific command sequences.

---

## 9. Parameter-Efficient Fine-Tuning (QLoRA)

For the `llama_lora` configuration (LLaMA2-7B, 4096-dim, 32 layers), full fine-tuning is infeasible. **QLoRA** is used instead.

### 9.1 4-bit Quantization (NF4)

The base model weights are quantized to **NF4** (Normal Float 4-bit) format using `bitsandbytes`. NF4 is an information-theoretically optimal quantization for normally distributed weights:

$$W_{\text{quantized}} = \text{NF4}(W_{\text{original}})$$

The quantized weights are kept frozen; only the LoRA adapters (in `bfloat16`) are trained.

### 9.2 LoRA (Low-Rank Adaptation)

For each weight matrix $W_0 \in \mathbb{R}^{d \times k}$ in the attention layers (Q, K, V, O projections), a low-rank decomposition is added:

$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} BA$$

where:
- $B \in \mathbb{R}^{d \times r}$, initialized to zero
- $A \in \mathbb{R}^{r \times k}$, initialized with $\mathcal{N}(0, \sigma^2)$
- $r \ll \min(d, k)$ is the rank (e.g., $r = 8$ or $r = 16$)
- $\alpha$ is a scaling hyperparameter (often $\alpha = r$ in practice)
- $\frac{\alpha}{r}$ scales the contribution of the adapter to avoid needing to tune the learning rate as $r$ changes

**Forward pass:**

$$y = W_0 x + \frac{\alpha}{r} B A x$$

Since $W_0$ is frozen, only $A$ and $B$ receive gradients. Parameter savings:

$$\text{LoRA params} = r(d + k) \ll dk = \text{full params}$$

For a typical LLaMA2-7B attention layer with $d = k = 4096$ and $r = 16$:
$$\text{full} = 4096 \times 4096 = 16.8\text{M params}$$
$$\text{LoRA} = 16 \times (4096 + 4096) = 131\text{K params} \quad (0.78\%)$$

### 9.3 Merging for Inference

After training, adapter weights are merged into the base model for zero-overhead inference:

$$W_{\text{merged}} = W_0 + \frac{\alpha}{r} BA$$

---

## 10. Masking Strategies

### 10.1 Causal Language Modeling (CLM)

Used by: LLaMA, Mixtral (autoregressive models)

The model is trained to predict the **last item** in each sequence. The masking logic in `model/patches.py` (`_compute_masked_targets_mask_last_item`):

1. Identify the last valid (non-padding) position $T^*$ in each sequence.
2. Set the label at $T^*$ as the target; zero out all other labels.
3. Zero out the embedding at position $T^*$ before passing to the transformer (simulating that the model hasn't seen $c_{T^*}$ yet).

$$\text{label}_{i,t} = \begin{cases} c_{T^*_i} & t = T^*_i \\ 0 & \text{otherwise} \end{cases}$$

During inference (not training/testing), padded positions are replaced with a learned `masked_item_embedding`.

The causal self-attention mask ensures position $t$ can only see positions $\leq t$, making this a standard next-token prediction setup restricted to the last position at training time.

### 10.2 Masked Language Modeling (MLM)

Used by: BERT, T5 (bidirectional models)

Random positions in each sequence are masked (replaced with a special `[MASK]` token) and the model must reconstruct them from context. Unlike CLM, the model can attend bidirectionally, giving it access to both past and future context for each masked position:

$$\mathcal{L}_{\text{MLM}} = -\sum_{t \in \mathcal{M}} \log P(c_t \mid c_{\backslash \mathcal{M}})$$

where $\mathcal{M}$ is the set of randomly masked positions.

---

## 11. Training Configuration

```
Optimizer:     AdamW (via HuggingFace Trainer)
Scheduler:     Linear warmup + decay (reset per experiment)
Epochs:        10
Learning rate: 3e-5 (full models) / 1e-4 (baselines)
Batch size:    128
Precision:     fp16 (mixed precision training, fp32 eval)
Early stop:    patience = 10 epochs (metric: eval loss)
Checkpoint:    every 20,000 steps (keep best only)
```

**Experiment tracking**: Weights & Biases (`WANDB_PROJECT="predictive_modeling_large"`)

**GPU**: NVIDIA Quadro RTX 8000 (48 GB VRAM)

**Distributed training**: Single GPU (DeepSpeed multi-GPU planned as future work)

---

## 12. Evaluation Metrics

All metrics are computed over the top-$k$ recommendations on the **last item** of each validation sequence.

### NDCG@k (Normalized Discounted Cumulative Gain)

Measures ranking quality with position discounting:

$$\text{DCG@}k = \sum_{i=1}^{k} \frac{2^{\text{rel}_i} - 1}{\log_2(i+1)}$$

$$\text{NDCG@}k = \frac{\text{DCG@}k}{\text{IDCG@}k}$$

where $\text{rel}_i \in \{0, 1\}$ (binary relevance) and IDCG is the ideal (perfect) ranking. A hit at rank 1 scores $1/\log_2(2) = 1.0$; a hit at rank 5 scores $1/\log_2(6) \approx 0.39$.

### Recall@k

$$\text{Recall@}k = \frac{|\text{relevant items in top-}k|}{|\text{all relevant items}|}$$

For next-item prediction with one ground truth: $\text{Recall@}k \in \{0, 1\}$ per query, averaged over all queries.

### MRR@k (Mean Reciprocal Rank)

$$\text{MRR@}k = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \frac{1}{\text{rank}_q}$$

where $\text{rank}_q$ is the rank of the correct item in the top-$k$ list (0 if not in top-$k$).

All metrics are computed at $k \in \{3, 5, 10\}$.

---

## 13. Deployment

The trained model is exported to **NVIDIA Triton Inference Server** as a multi-stage ensemble:

```
Input (raw sequence features)
        │
        ▼
[TransformWorkflowTriton]   ← NVTabular preprocessing (GPU)
        │
        ▼
[EmbeddingOperator]         ← Pretrained text embedding lookup
        │
        ▼
[PredictPyTorchTriton]      ← Serialized PyTorch model (cloudpickle)
        │
        ▼
Top-5 command recommendations
```

For QLoRA models, LoRA adapters are merged into the base model weights before export:

```python
tr_model = peft_model.merge_and_unload()  # W_merged = W_0 + (α/r)BA
```

The ensemble is served via a Python backend on Triton, with the conda environment packed via `conda-pack`.

A web application (`prototype/`) polls live Vectorworks logs and calls the Triton endpoint to display real-time next-command suggestions to users.

---

## Key File Reference

| File | Purpose |
|---|---|
| `model/preprocess.py` | NVTabular ETL pipeline, session split |
| `model/pretrained_text_embedding.py` | OpenAI embedding generation |
| `model/train_eval_full_models.py` | Full model training (with feature fusion, multi-task, focal loss) |
| `model/train_eval_baseline_models.py` | Ablation baseline training |
| `model/patches.py` | CLM masking bug fix, distributed padding |
| `model/utils.py` | Session augmentation, PEFT callback |
| `transformers4rec/torch/features/sequence.py` | Feature fusion module (self-attn + attn-pooling) |
| `transformers4rec/torch/model/prediction_task.py` | Focal loss, multi-task heads, MoE aux loss |
| `transformers4rec/config/transformer.py` | LLaMA, Mixtral, BERT, T5 config + QLoRA setup |
| `transformers4rec/torch/masking.py` | CLM/MLM masking logic |
| `data_processing/` | Raw log filtering, alignment, augmentation, RAG |
| `peft/` | Custom LoRA/QLoRA implementation |
| `deployment/` | Triton ensemble export, local inference |
