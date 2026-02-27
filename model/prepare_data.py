"""
prepare_data.py

Bridges the stage1_filtered parquet files to the input format expected by
preprocess.py's save_random_split_balance_distribution().

Steps:
  1. Read all data/stage1_filtered/*.parquet
  2. Compute timestamp_interval per session
  3. Split long sessions (min=10, max=100 items)
  4. Join classification + target from augmentations CSV (S3)
  5. Build command vocabulary and map message_content -> integer item_id
  6. Write data/preprocess_input.parquet
  7. Run CPU-based preprocessing (pandas replacement for NVTabular)
  8. Run save_random_split_balance_distribution (train/val split)

Usage:
    conda run -n t4rec_23.06_new_transformers python model/prepare_data.py
"""

import os
import sys
import json
import shutil
import glob
import numpy as np
import boto3
import pandas as pd
import dask.dataframe as dd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.preprocess import save_random_split_balance_distribution
from model.utils import split_sessions_with_correlated_ids

# ── Config ─────────────────────────────────────────────────────────────────
STAGE1_DIR           = "data/stage1_filtered"
AUGMENTATIONS_S3_URI = "s3://vectorworks-analytics-datalake/tool-rec-data/command_information_augmentations.csv"
AUGMENTATIONS_LOCAL  = "data/command_information_augmentations.csv"
VOCAB_PATH           = "data/command_vocab.json"
MERGED_PATH          = "data/preprocess_input.parquet"
NVT_DATA_PATH        = "data/processed_nvt"
NVT_WORKFLOW_PATH    = "data/workflow_etl"
SPLITS_PATH          = "data/preproc_sessions"
# ───────────────────────────────────────────────────────────────────────────


def load_stage1(directory):
    files = sorted(glob.glob(os.path.join(directory, "*.parquet")))
    print(f"  Reading {len(files)} parquet files...")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"  {len(df):,} rows, {df['session_id'].nunique():,} sessions, "
          f"{df['message_content'].nunique()} unique commands")
    return df


def compute_timestamp_interval(df):
    df = df.sort_values(['session_id', 'ts'])
    df['timestamp_interval'] = (
        df.groupby('session_id')['ts'].diff().fillna(0)
    )
    return df


def download_augmentations(s3_uri, local_path):
    if os.path.exists(local_path):
        print(f"  Using cached {local_path}")
        return pd.read_csv(local_path)
    print(f"  Downloading from S3...")
    bucket = s3_uri.replace("s3://", "").split("/")[0]
    key    = "/".join(s3_uri.replace("s3://", "").split("/")[1:])
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(obj['Body'])
    df.to_csv(local_path, index=False)
    return df


def build_vocab(commands):
    """Assign a stable integer ID (1-indexed) to each command string."""
    vocab = {cmd: i + 1 for i, cmd in enumerate(sorted(commands))}
    return vocab


SESSIONS_SPLIT_PATH = "data/sessions_split.parquet"
SESSIONS_MAX_LENGTH  = 200
MINIMUM_SESSION_LENGTH = 5


def preprocessing_cpu(inter_data_path, data_path, workflow_path):
    """
    CPU-only replacement for NVTabular preprocessing().

    Produces identical column names and Merlin schema.pbtxt so that
    train_eval_full_models.py can read the output without changes.
    """
    from merlin.schema import Schema, ColumnSchema, Tags
    from merlin.schema.io.tensorflow_metadata import TensorflowMetadata
    import merlin.dtypes as md

    os.makedirs(data_path, exist_ok=True)
    os.makedirs(workflow_path, exist_ok=True)

    print("  Reading input parquet...")
    df = pd.read_parquet(inter_data_path)

    # ── 1. Categorify: encode string columns to 1-indexed integers ──────────
    cat_sizes = {}
    for col in ['classification', 'target', 'cat']:
        if col in df.columns:
            uniq = sorted(df[col].dropna().unique())
            enc  = {v: i + 1 for i, v in enumerate(uniq)}
            df[col] = df[col].map(enc).fillna(0).astype(np.int64)
            cat_sizes[col] = len(uniq) + 1   # cardinality (including 0=null)
            print(f"    {col}: {len(uniq)} categories → 1..{len(uniq)}")

    item_id_max = int(df['item_id'].max())
    print(f"    item_id: max={item_id_max}")

    # ── 2. Normalize: timestamp_interval and merge_count ───────────────────
    print("  Normalising continuous features...")
    ti_mean = float(df['timestamp_interval'].mean())
    ti_std  = max(float(df['timestamp_interval'].std()), 1e-8)
    mc_mean = float(df['merge_count'].astype(float).mean())
    mc_std  = max(float(df['merge_count'].astype(float).std()), 1e-8)

    df['timestamp_interval_norm_global'] = (
        (df['timestamp_interval'] - ti_mean) / ti_std
    ).astype(np.float64)
    df['merge_count']      = df['merge_count'].astype(np.float64)
    df['merge_count_norm'] = (
        (df['merge_count'] - mc_mean) / mc_std
    ).astype(np.float64)

    # ── 3. Groupby: sort by timestamp, aggregate into lists ─────────────────
    print("  Grouping by session_id into sequence lists...")
    df = df.sort_values(['session_id', 'timestamp'])

    seq_cols = [
        'item_id', 'timestamp',
        'timestamp_interval', 'timestamp_interval_norm_global',
        'classification', 'target',
        'merge_count', 'merge_count_norm', 'cat',
    ]

    def tail_list(s):
        return s.tolist()[-SESSIONS_MAX_LENGTH:]

    grouped = df.groupby('session_id', sort=False).agg(
        {col: tail_list for col in seq_cols}
    ).reset_index()

    grouped.rename(columns={c: f"{c}-list" for c in seq_cols}, inplace=True)
    grouped['item_id-count'] = grouped['item_id-list'].apply(len)

    # ── 4. Filter: remove sessions shorter than MINIMUM_SESSION_LENGTH ──────
    before = len(grouped)
    grouped = grouped[grouped['item_id-count'] > MINIMUM_SESSION_LENGTH].reset_index(drop=True)
    print(f"  Filtered {before - len(grouped)} short sessions → {len(grouped)} kept")

    # ── 5. Convert list columns to typed numpy arrays ───────────────────────
    int_list_cols   = ['item_id-list', 'classification-list', 'target-list', 'cat-list']
    float_list_cols = [
        'timestamp_interval-list', 'timestamp_interval_norm_global-list',
        'merge_count-list', 'merge_count_norm-list',
    ]
    for col in int_list_cols:
        grouped[col] = grouped[col].apply(lambda x: np.array(x, dtype=np.int64))
    for col in float_list_cols:
        grouped[col] = grouped[col].apply(lambda x: np.array(x, dtype=np.float64))

    # ── 6. Write parquet ─────────────────────────────────────────────────────
    out_cols = [
        'session_id', 'item_id-count',
        'item_id-list', 'timestamp-list',
        'timestamp_interval-list', 'timestamp_interval_norm_global-list',
        'classification-list', 'target-list',
        'merge_count-list', 'merge_count_norm-list',
        'cat-list',
    ]
    out_path = os.path.join(data_path, "part_0.parquet")
    grouped[out_cols].to_parquet(out_path, index=False, engine='pyarrow')
    print(f"  Saved {len(grouped):,} sessions → {out_path}")

    # ── 7. Generate Merlin schema.pbtxt ──────────────────────────────────────
    def _cat_col(name, max_val, extra_tags=None):
        tags = [Tags.LIST, Tags.CATEGORICAL] + (extra_tags or [])
        return (
            ColumnSchema(name, tags=tags, dtype=md.int64, is_list=True, is_ragged=True)
            .with_properties({'domain': {'min': 1, 'max': max_val}})
        )

    def _cont_col(name):
        return ColumnSchema(name, tags=[Tags.LIST, Tags.CONTINUOUS],
                            dtype=md.float64, is_list=True, is_ragged=True)

    schema = Schema([
        ColumnSchema('session_id', tags=[Tags.CATEGORICAL], dtype=md.string),
        ColumnSchema('item_id-count', dtype=md.int32),
        _cat_col('item_id-list', item_id_max, extra_tags=[Tags.ITEM_ID, Tags.ITEM, Tags.ID]),
        _cont_col('timestamp_interval-list'),
        _cont_col('timestamp_interval_norm_global-list'),
        _cont_col('merge_count-list'),
        _cont_col('merge_count_norm-list'),
        _cat_col('cat-list',            cat_sizes.get('cat', 3) - 1),
        _cat_col('classification-list', cat_sizes.get('classification', 2) - 1),
        _cat_col('target-list',         cat_sizes.get('target', 2) - 1),
    ])

    proto_text = TensorflowMetadata.from_merlin_schema(schema).to_proto_text()
    for dest in [data_path, workflow_path]:
        schema_path = os.path.join(dest, "schema.pbtxt")
        with open(schema_path, 'w') as f:
            f.write(proto_text)
        print(f"  Schema written → {schema_path}")


def main():
    os.makedirs("data", exist_ok=True)

    if os.path.exists(SESSIONS_SPLIT_PATH):
        # Fast path: skip the expensive split step
        print(f"Loading existing {SESSIONS_SPLIT_PATH} (skipping re-split)...")
        df = pd.read_parquet(SESSIONS_SPLIT_PATH)
        print(f"  {len(df):,} rows, {df['session_id'].nunique():,} sessions")
    else:
        # 1. Load all stage1 parquets
        print("Loading stage1 filtered data...")
        df = load_stage1(STAGE1_DIR)

        # 2. Compute timestamp_interval
        print("Computing timestamp intervals...")
        df = compute_timestamp_interval(df)

        # Rename to match preprocess.py conventions
        df.rename(columns={
            'session_id': 'session_id',   # already correct
            'ts':         'timestamp',
            'message_content': 'item_id',  # will be string for now
        }, inplace=True)

        # 3. Split long sessions
        print("Splitting long sessions (min=10, max=100)...")
        df = split_sessions_with_correlated_ids(
            df, min_items=10, max_items=100,
            output_file=SESSIONS_SPLIT_PATH,
        )
        print(f"  After split: {len(df):,} rows, {df['session_id'].nunique():,} sessions")

    # 4. Join augmentations (classification, target)
    print("Joining augmentations (classification, target)...")
    aug = download_augmentations(AUGMENTATIONS_S3_URI, AUGMENTATIONS_LOCAL)
    aug = aug[['message_content', 'classification', 'target']].rename(
        columns={'message_content': 'item_id'}
    )
    df = df.merge(aug, on='item_id', how='left')
    null_class = df['classification'].isna().sum()
    if null_class > 0:
        print(f"  Warning: {null_class:,} rows with no classification (filling 'other')")
        df['classification'].fillna('other', inplace=True)
        df['target'].fillna('other', inplace=True)
    print(f"  Classifications: {df['classification'].nunique()}, "
          f"Targets: {df['target'].nunique()}")

    # 5. Build vocabulary and map item_id -> integer
    print("Building command vocabulary...")
    unique_commands = sorted(df['item_id'].unique())
    vocab = build_vocab(unique_commands)
    with open(VOCAB_PATH, 'w') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"  {len(vocab)} commands -> {VOCAB_PATH}")

    df['item_id'] = df['item_id'].map(vocab)
    print(f"  item_id range: {df['item_id'].min()} - {df['item_id'].max()}")

    # 6. Write merged parquet
    cols = ['session_id', 'item_id', 'timestamp', 'merge_count', 'cat',
            'timestamp_interval', 'classification', 'target']
    df[cols].to_parquet(MERGED_PATH, index=False)
    print(f"  Saved -> {MERGED_PATH}")

    # 7. CPU preprocessing (pandas replacement for NVTabular)
    print("\nRunning CPU preprocessing...")
    preprocessing_cpu(
        inter_data_path=MERGED_PATH,
        data_path=NVT_DATA_PATH,
        workflow_path=NVT_WORKFLOW_PATH,
    )

    # 8. Train/val split
    print("\nCreating train/val splits...")
    df_nvt = dd.read_parquet(f"{NVT_DATA_PATH}/*.parquet")
    save_random_split_balance_distribution(
        df=df_nvt,
        output_dir=SPLITS_PATH,
        train_size=0.85,
        val_size=0.15,
        overwrite=True,
    )

    print("\nAll done. Ready to train.")
    print(f"  Train: {SPLITS_PATH}/train_new/")
    print(f"  Val:   {SPLITS_PATH}/val_new/")
    print(f"  Schema parquet: {NVT_DATA_PATH}/")


if __name__ == "__main__":
    main()
