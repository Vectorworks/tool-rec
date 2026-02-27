"""
stream_stage1_filter.py

32 workers each make an independent S3 byte-range GET request and process
their slice of the raw log CSV in parallel, then write filtered parquet files
to data/stage1_filtered/.

What it does:
  - Keeps only Tool and Menu category rows (UNDO dropped entirely)
  - Applies prefix/suffix noise removal (zoom, pan, view changes, etc.)
  - Strips trailing localization IDs: "Tool: Round Rect (-217)" -> "Tool: Round Rect"
  - Filters to the English allowlist from command_information_augmentations.csv
  - Adds merge_count=1 (BPE workflow merging is skipped)
  - Converts timestamps to unix int

Output parquet columns:
  session_id, ts, message_content, cat, merge_count

Usage:
    conda run -n t4rec_23.06_new_transformers python data_processing/stream_stage1_filter.py
"""

import os
import re
import time
import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
from io import TextIOWrapper
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── Config ─────────────────────────────────────────────────────────────────
RAW_LOG_S3_URI       = "s3://vectorworks-analytics-datalake/athena_query_results/Unsaved/2026/02/19/3611baa5-ada5-45f0-a087-752c78002850.csv"
AUGMENTATIONS_S3_URI = "s3://vectorworks-analytics-datalake/tool-rec-data/command_information_augmentations.csv"
OUTPUT_DIR           = "data/stage1_filtered"
N_WORKERS            = 32
WORKER_CHUNK_ROWS    = 200_000   # pandas chunksize within each worker
MAX_LINE_BYTES       = 8_192     # generous max length of one CSV row
# ───────────────────────────────────────────────────────────────────────────

CSV_COLUMNS = [
    'sn_anonymized', 'session_anonymized', 'mac_id_anonymized',
    'ts', 'log_lvl', 'vw_ver', 'platform', 'os_ver', 'type', 'cat', 'message'
]

PREFIXES_TO_REMOVE = [
    "DestroyEvent: ", "Begin Internal Event: ", "Event:",
    "Event name changed from ", "Beta Undo Alert", "Undo Problem:",
    "Abort Event: ", "Menu: Undo", "Menu: Redo",
]

SUFFIXES_TO_REMOVE = [
    "(242)", "(-242)", "(218)", "(199)", "(236)", "(-241)", "(-240)", "(307)",
    "End Event: Select Similar: Undo not pos (166)",
    "End Event: Fit To Objects (166)",
    "(201)", "(-303)", "(305)", "(306)", "(193)", "(8)", "(205)", "(203)",
]

PREFIX_RE       = re.compile(r'^(?:{})'.format('|'.join(re.escape(p) for p in PREFIXES_TO_REMOVE)))
SUFFIX_RE       = re.compile(r'(?:{})$'.format('|'.join(re.escape(s) for s in SUFFIXES_TO_REMOVE)))
MAX_RE          = re.compile(r'\(MAX-\d+\)')
TRAILING_ID_RE1 = re.compile(r'\s*-\s*(\(-?\d+\)\s*)+$')
TRAILING_ID_RE2 = re.compile(r'\s*(\(-?\d+\)\s*)+$')

OUTPUT_SCHEMA = pa.schema([
    pa.field("session_id",      pa.string()),
    pa.field("ts",              pa.int64()),
    pa.field("message_content", pa.string()),
    pa.field("cat",             pa.string()),
    pa.field("merge_count",     pa.int32()),
])


def parse_s3_uri(uri):
    uri = uri.replace("s3://", "")
    bucket, key = uri.split("/", 1)
    return bucket, key


def strip_ids(msg):
    msg = TRAILING_ID_RE1.sub('', msg)
    msg = TRAILING_ID_RE2.sub('', msg)
    return msg.strip().rstrip(' -').strip()


def to_unix(time_str):
    try:
        if isinstance(time_str, pd.Timestamp):
            return int(time_str.timestamp())
        dt = datetime.strptime(str(time_str), "%Y-%m-%d %H:%M:%S.%f")
        return int(dt.timestamp())
    except Exception:
        return 0


def load_allowlist(uri):
    bucket, key = parse_s3_uri(uri)
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(obj['Body'])
    english = set()
    for cmd in df['message_content'].dropna():
        try:
            cmd.encode('ascii')
            english.add(cmd)
        except UnicodeEncodeError:
            pass
    return english


def find_split_points(bucket, key, file_size, n):
    """
    Find byte positions of complete line starts nearest to equal splits.
    Makes n-1 tiny S3 range requests (8 KB each) to locate newlines.
    """
    s3 = boto3.client('s3')
    points = [0]
    for i in range(1, n):
        approx = i * file_size // n
        win_s = max(0, approx - MAX_LINE_BYTES)
        win_e = min(file_size - 1, approx + MAX_LINE_BYTES)
        resp = s3.get_object(Bucket=bucket, Key=key, Range=f"bytes={win_s}-{win_e}")
        window = resp['Body'].read()
        offset = approx - win_s
        nl = window.find(b'\n', offset)
        points.append(win_s + nl + 1 if nl != -1 else approx)
    points.append(file_size)
    return [(points[i], points[i + 1] - 1) for i in range(n)]


def process_chunk(df, allowlist):
    df = df[df['cat'].isin(['Tool', 'Menu'])].copy()
    if df.empty:
        return None
    df = df[~df['message'].str.match(PREFIX_RE, na=False)]
    if df.empty:
        return None
    df = df[~df['message'].str.contains(SUFFIX_RE, regex=True, na=False)]
    if df.empty:
        return None
    df['message'] = df['message'].str.replace(MAX_RE, '', regex=True)
    df['message_content'] = df['message'].apply(strip_ids)
    df = df[df['message_content'].isin(allowlist)]
    if df.empty:
        return None
    df['ts'] = df['ts'].apply(to_unix)
    df['merge_count'] = 1
    df = df[['session_anonymized', 'ts', 'message_content', 'cat', 'merge_count']].copy()
    df.rename(columns={'session_anonymized': 'session_id'}, inplace=True)
    return df


def worker(args):
    """
    Each worker:
      1. Opens an S3 byte-range GET for its slice of the file
      2. Reads in WORKER_CHUNK_ROWS-sized pandas chunks
      3. Filters and writes parquet files to OUTPUT_DIR
    """
    worker_id, byte_start, byte_end, allowlist, output_dir = args

    bucket, key = parse_s3_uri(RAW_LOG_S3_URI)
    s3 = boto3.client('s3')

    resp = s3.get_object(Bucket=bucket, Key=key, Range=f"bytes={byte_start}-{byte_end}")
    stream = TextIOWrapper(resp['Body'], encoding='utf-8', errors='replace')

    # Worker 0 has the CSV header row; all others need column names supplied.
    # dtype=str prevents pandas from inferring hex session IDs as int64.
    if worker_id == 0:
        reader = pd.read_csv(stream, chunksize=WORKER_CHUNK_ROWS,
                             low_memory=False, dtype=str)
    else:
        reader = pd.read_csv(
            stream, names=CSV_COLUMNS, header=None,
            chunksize=WORKER_CHUNK_ROWS, low_memory=False, dtype=str
        )

    part_idx  = 0
    total_out = 0

    for chunk in reader:
        result = process_chunk(chunk, allowlist)
        if result is not None and len(result) > 0:
            out_path = os.path.join(output_dir, f"part_{worker_id:04d}_{part_idx:04d}.parquet")
            table = pa.Table.from_pandas(result, schema=OUTPUT_SCHEMA, preserve_index=False)
            pq.write_table(table, out_path, compression='snappy')
            total_out += len(result)
            part_idx += 1

    return worker_id, total_out


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading English allowlist from S3...")
    allowlist = load_allowlist(AUGMENTATIONS_S3_URI)
    print(f"  {len(allowlist)} allowed commands")

    bucket, key = parse_s3_uri(RAW_LOG_S3_URI)
    s3 = boto3.client('s3')

    print("Getting file size...")
    head = s3.head_object(Bucket=bucket, Key=key)
    file_size = head['ContentLength']
    print(f"  {file_size / 1e9:.1f} GB")

    print(f"Finding {N_WORKERS} split points...")
    splits = find_split_points(bucket, key, file_size, N_WORKERS)
    sizes_gb = [(e - s) / 1e9 for s, e in splits]
    print(f"  Range sizes: min={min(sizes_gb):.2f} GB, max={max(sizes_gb):.2f} GB")

    args = [
        (i, start, end, frozenset(allowlist), OUTPUT_DIR)
        for i, (start, end) in enumerate(splits)
    ]

    print(f"\nLaunching {N_WORKERS} parallel workers...")
    t0 = time.time()

    total_out = 0
    done      = 0

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(worker, a): a[0] for a in args}
        for future in as_completed(futures):
            wid, count = future.result()
            total_out += count
            done += 1
            elapsed = time.time() - t0
            print(f"  Worker {wid:2d} done | {count:>8,} rows kept | "
                  f"{done:2d}/{N_WORKERS} complete | {elapsed:.0f}s elapsed")

    elapsed = time.time() - t0
    print(f"\nFinished in {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"Total rows written: {total_out:,}")
    print(f"Output: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
