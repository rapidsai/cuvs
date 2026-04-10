# OpenSearch GPU Remote Index Build Benchmark

Docker Compose benchmark of [OpenSearch's GPU-accelerated remote index build](https://docs.opensearch.org/latest/vector-search/remote-index-build/) feature using [cuvs-bench](https://github.com/jrbourbeau/cuvs/tree/main/python/cuvs_bench). Spins up all required services, builds a kNN index on a GPU, and runs a cuvs-bench search benchmark sweep.

## How it works

OpenSearch's kNN plugin can offload Faiss HNSW index construction to a dedicated GPU service. Rather than building the index in-process on the OpenSearch node, the workflow is:

```
OpenSearch flushes a segment
  → uploads raw vectors + doc-IDs to S3 (MinIO in this demo)
  → POSTs /_build to the remote-index-builder service
      → service downloads vectors from MinIO
      → builds Faiss HNSW index on GPU
      → uploads finished index back to MinIO
  → OpenSearch downloads the GPU-built index and merges it into the shard
```

## Services

| Service | Image | Purpose |
|---|---|---|
| `minio` | `minio/minio` | S3-compatible object store — staging area for vectors and built indexes |
| `minio-init` | `minio/mc` | One-shot: creates the `opensearch-vectors` bucket |
| `opensearch` | custom build of `opensearchproject/opensearch` | OpenSearch node with kNN plugin and `repository-s3` plugin |
| `remote-index-builder` | `opensearchproject/remote-vector-index-builder:api-latest` | FastAPI service that builds Faiss indexes on the GPU |
| `bench` | custom Python | Registers repo + cluster settings, runs cuvs-bench build/search benchmark |

## Requirements

- **NVIDIA GPU** with CUDA support
- **NVIDIA Container Toolkit** — [installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- **Docker Compose v2**
- **ANN benchmark dataset** in binary format (`.fbin`) — see [Dataset format](#dataset-format)

## Usage

Set the host kernel parameter required by OpenSearch (once per reboot):

```bash
sudo sysctl -w vm.max_map_count=262144
```

Set required environment variables:

```bash
export DATASET_PATH=/path/to/ann-benchmark-datasets   # directory containing dataset files
```

Optionally configure the benchmark:

```bash
export DATASET=sift-128-euclidean   # default
export BENCH_GROUPS=test            # test | base | large (default: test)
export K=10                         # number of neighbors (default: 10)
```

Download the dataset (one-time setup, skipped automatically if already present):

```bash
docker compose build bench
docker compose run --rm --no-deps bench python -m cuvs_bench.get_dataset \
    --dataset ${DATASET:-sift-128-euclidean} \
    --dataset-path /data/datasets
```

Start all services:

```bash
docker compose up --build
```

The `bench` container logs its progress through index build, GPU verification, and search. When complete you'll see a results table:

```
════════════════════════════════════════════════════════════
  Benchmark complete!
════════════════════════════════════════════════════════════

  OpenSearch : http://opensearch:9200
  MinIO console : http://localhost:9001  (minioadmin / minioadmin)
```

To tear everything down:

```bash
docker compose down -v
```

## What the bench script does

1. Registers MinIO as an OpenSearch S3 snapshot repository
2. Applies cluster settings to enable remote index build and point OpenSearch at the builder service
3. Runs `cuvs-bench` build phase (handled entirely by the OpenSearch backend):
   - Creates the kNN index and bulk-ingests dataset vectors
   - Force-merges the index to trigger the GPU build
   - Polls MinIO every 5 s for `.faiss` files confirming GPU build completion
   - Records total build time (ingestion + GPU build) in the result
4. Runs `cuvs-bench` search phase and prints a recall/QPS/latency table

## Dataset format

cuvs-bench reads binary vector files with a simple header:

```
[4 bytes: n_rows as uint32]
[4 bytes: n_cols as uint32]
[n_rows × n_cols × itemsize bytes: vector data]
```

Supported extensions: `.fbin` (float32), `.f16bin` (float16), `.u8bin` (uint8), `.i8bin` (int8).

`DATASET_PATH` should be a directory where each dataset lives in its own subdirectory named after the dataset, e.g.:

```
$DATASET_PATH/
  sift-128-euclidean/
    base.fbin
    query.fbin
    groundtruth.neighbors.ibin
```

## Key configuration

**Cluster settings** (applied by `bench/run.py`):

```json
{
  "persistent": {
    "knn.remote_index_build.enabled": true,
    "knn.remote_index_build.repository": "vector-repo",
    "knn.remote_index_build.service.endpoint": "http://remote-index-builder:1025"
  }
}
```

**Parameter groups** (`BENCH_GROUPS`):

| Group | Build params | Search params | Use case |
|---|---|---|---|
| `test` | 1 combo (m=16, ef_construction=100) | ef_search: 50, 100 | Quick smoke test |
| `base` | 6 combos (m × ef_construction sweep) | ef_search: 50–512 | Standard benchmark |
| `large` | 2 combos (larger m, ef_construction) | ef_search: 100–1024 | High-recall benchmark |

## GPU build verification

The cuvs-bench OpenSearch backend polls MinIO every 5 seconds for `.faiss` files under `s3://opensearch-vectors/knn-indexes/`. The remote-index-builder is the only component that writes `.faiss` files back to the bucket, so their presence is definitive proof the GPU build completed.

The build raises a `TimeoutError` (causing the `bench` container to exit with code 1) if the expected number of `.faiss` files does not appear within 600 seconds.

## Running without a GPU

This is intentionally not supported. The `bench` container will exit 1 if `.faiss` files do not appear in MinIO within 600 s. If you want to experiment without hardware, remove the `deploy.resources.reservations` block from `remote-index-builder` in `docker-compose.yml` and be aware the benchmark will fail at the verification step.

## Running tests

The cuvs-bench OpenSearch backend has three tiers of tests. All run inside the `bench` container so no local Python environment is needed.

**Build the bench image first** (or after any code changes):

```bash
docker compose build --no-cache bench
```

### Unit tests (no server required)

```bash
docker compose run --rm --no-deps bench \
    pytest /opt/cuvs/python/cuvs_bench/cuvs_bench/tests/test_opensearch.py -v
```

### Integration tests (live OpenSearch node)

Requires a running OpenSearch node. Start it with:

```bash
docker compose up -d --wait opensearch
```

Then run:

```bash
docker compose run --rm --no-deps \
    -e OPENSEARCH_URL=http://opensearch:9200 \
    bench \
    pytest /opt/cuvs/python/cuvs_bench/cuvs_bench/tests/test_opensearch.py -v -m integration
```

### Remote index build integration tests (full GPU stack)

Requires the full stack (OpenSearch, MinIO, and the remote index builder). Start all services with:

```bash
docker compose up -d --wait minio minio-init opensearch remote-index-builder
```

Then run:

```bash
docker compose run --rm --no-deps \
    -e OPENSEARCH_URL=http://opensearch:9200 \
    -e BUILDER_URL=http://remote-index-builder:1025 \
    -e S3_ENDPOINT=http://minio:9000 \
    -e S3_BUCKET=opensearch-vectors \
    -e S3_ACCESS_KEY=minioadmin \
    -e S3_SECRET_KEY=minioadmin \
    bench \
    pytest /opt/cuvs/python/cuvs_bench/cuvs_bench/tests/test_opensearch.py -v -m integration
```

This runs all integration tests including `TestOpenSearchRemoteIndexBuildIntegration`, which verifies the full GPU build flow end-to-end.

## Ports

| Port | Service |
|---|---|
| `9200` | OpenSearch REST API |
| `9000` | MinIO S3 API |
| `9001` | MinIO web console |
| `1025` | Remote index builder API |
