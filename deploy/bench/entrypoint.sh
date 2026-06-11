#!/bin/bash
set -e

DATASET="${DATASET:-sift-128-euclidean}"
BENCH_GROUPS="${BENCH_GROUPS:-test}"
K="${K:-10}"
# Auto-detect GPU mode: remote-index-builder only appears in Docker DNS when
# started via --profile gpu. DNS entries are registered at network setup time
# (before containers run), so this check is reliable by the time entrypoint
# executes (OpenSearch healthy check alone takes 30+ seconds).
if getent hosts remote-index-builder > /dev/null 2>&1; then
    echo "remote-index-builder detected — waiting for it to be ready..."
    until python3 -c 'import socket; socket.create_connection(("remote-index-builder", 1025), 2).close()' 2>/dev/null; do
        sleep 5
    done
    echo "remote-index-builder is ready."
    export REMOTE_INDEX_BUILD=true
else
    echo "remote-index-builder not available — using CPU build mode."
    export REMOTE_INDEX_BUILD=false
fi

# Step 1: Download dataset (skipped automatically if already present)
python -m cuvs_bench.get_dataset \
    --dataset "$DATASET" \
    --dataset-path /data/datasets

# Step 2: Run benchmark (build + search + writes result JSON files)
python -u run.py

# Step 3: Export JSON → CSV (required by cuvs_bench.plot)
python -m cuvs_bench.run --data-export \
    --dataset "$DATASET" \
    --dataset-path /data/datasets \
    --algorithms opensearch_faiss_hnsw \
    --groups "$BENCH_GROUPS" \
    --count "$K" \
    --batch-size 10000 \
    --search-mode latency

# Step 4: Plot — PNGs written to /data/datasets (mounted from host $DATASET_PATH)
python -m cuvs_bench.plot \
    --dataset "$DATASET" \
    --dataset-path /data/datasets \
    --algorithms opensearch_faiss_hnsw \
    --groups "$BENCH_GROUPS" \
    --count "$K" \
    --output-filepath /data/datasets
