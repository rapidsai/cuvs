# Usage

This page shows how to run cuVS Bench after installation, including Python and Docker workflows. For dataset formats, built-in datasets, custom dataset descriptors, and ground-truth generation, see [Benchmark Datasets](datasets.md).

## End-to-end: smaller-scale benchmarks (&lt;1M to 10M)

This example downloads a built-in dataset and runs benchmarks on a 10M-vector subset of Yandex Deep-1B.

```bash
# (1) Prepare a built-in dataset.
python -m cuvs_bench.get_dataset --dataset deep-image-96-angular --normalize
```

```python
# (2) Build and search index.
from cuvs_bench.orchestrator import BenchmarkOrchestrator

orchestrator = BenchmarkOrchestrator(backend_type="cpp_gbench")
results = orchestrator.run_benchmark(
    dataset="deep-image-96-inner",
    algorithms="cuvs_cagra",
    count=10,
    batch_size=10,
    build=True,
    search=True,
)
```

```bash
# (3) Export data.
python -m cuvs_bench.run --data-export --dataset deep-image-96-inner

# (4) Plot results.
python -m cuvs_bench.plot --dataset deep-image-96-inner
```

For available built-in datasets and ground-truth details, see [Benchmark Datasets](datasets.md#built-in-datasets).

## End-to-end: large-scale benchmarks (>10M vectors)

`cuvs_bench.get_dataset` does not download billion-scale datasets. Prepare the dataset first using the guidance in [Benchmark Datasets](datasets.md#dataset-sources); after that, the run commands are the same. Datasets at this scale are best suited to large-memory GPUs such as A100 or H100.

```python
# Build and search index.
from cuvs_bench.orchestrator import BenchmarkOrchestrator

orchestrator = BenchmarkOrchestrator(backend_type="cpp_gbench")
results = orchestrator.run_benchmark(
    dataset="deep-1B",
    algorithms="cuvs_cagra",
    count=10,
    batch_size=10,
    build=True,
    search=True,
)
```

```bash
# Export data.
python -m cuvs_bench.run --data-export --dataset deep-1B

# Plot results.
python -m cuvs_bench.plot --dataset deep-1B
```

## Running with Docker containers

Docker supports end-to-end runs and manual execution inside the container. See [Installation](install.md#docker) for available images and tag guidance.

### End-to-end run on GPU

Without a custom entrypoint, the container runs the full smaller-scale workflow above.

For GPU systems, set `DATA_FOLDER` to a dedicated local folder. Datasets are stored in `$DATA_FOLDER/datasets` and results in `$DATA_FOLDER/result`.

```bash
export DATA_FOLDER=path/to/store/datasets/and/results
docker run --gpus all --rm -it -u $(id -u)                      \
    -v $DATA_FOLDER:/data/benchmarks                            \
    rapidsai/cuvs-bench:26.06a-cuda12-py3.13              \
    "--dataset deep-image-96-angular"                           \
    "--normalize"                                               \
    "--algorithms cuvs_cagra,cuvs_ivf_pq --batch-size 10 -k 10" \
    ""
```

Usage of the above command is as follows:

| Argument | Description |
| --- | --- |
| `rapidsai/cuvs-bench:26.06a-cuda12-py3.13` | Image to use. See [Installation](install.md#docker) for links to lists of available tags. |
| `"--dataset deep-image-96-angular"` | Dataset name |
| `"--normalize"` | Whether to normalize the dataset |
| `"--algorithms cuvs_cagra,hnswlib --batch-size 10 -k 10"` | Arguments passed to the `run` script, such as the algorithms to benchmark, the batch size, and `k` |
| `""` | Additional (optional) arguments that will be passed to the `plot` script. |

***Note about user and file permissions:*** The flag `-u $(id -u)` allows the user inside the container to match the `uid` of the user outside the container, allowing the container to read and write to the mounted volume indicated by the `$DATA_FOLDER` variable.

### End-to-end run on CPU

Use the same argument pattern with the CPU-only container on systems without a GPU.

***Note:*** Use the `cuvs-bench-cpu` image and omit `--gpus all`:

```bash
export DATA_FOLDER=path/to/store/datasets/and/results
docker run  --rm -it -u $(id -u)                  \
    -v $DATA_FOLDER:/data/benchmarks              \
    rapidsai/cuvs-bench-cpu:26.06a-py3.13     \
     "--dataset deep-image-96-angular"            \
     "--normalize"                                \
     "--algorithms hnswlib --batch-size 10 -k 10" \
     ""
```

### Manually run the scripts inside the container

All `cuvs-bench` images include the Conda packages, so you can open a shell and run commands directly:

```bash
export DATA_FOLDER=path/to/store/datasets/and/results
docker run --gpus all --rm -it -u $(id -u)          \
    --entrypoint /bin/bash                          \
    --workdir /data/benchmarks                      \
    -v $DATA_FOLDER:/data/benchmarks                \
    rapidsai/cuvs-bench:26.06a-cuda12-py3.13
```

This opens a container shell with the `cuvs-bench` Python package ready to use:

```bash
(base) root@00b068fbb862:/data/benchmarks# python -m cuvs_bench.get_dataset --dataset deep-image-96-angular --normalize
```

Containers can also run in detached mode.

## Evaluating results

Build benchmarks report these measurements:

| Name | Description |
| --- | --- |
| Benchmark | A name that uniquely identifies the benchmark instance |
| Time | Wall-time spent training the index |
| CPU | CPU time spent training the index |
| Iterations | Number of iterations (this is usually 1) |
| GPU | GPU time spent building |
| index_size | Number of vectors used to train index |

Search benchmarks report these measurements. The most important fields are `Latency`, `items_per_second`, and `end_to_end`.

| Name | Description |
| --- | --- |
| Benchmark | A name that uniquely identifies the benchmark instance |
| Time | The wall-clock time of a single iteration (batch) divided by the number of threads. |
| CPU | The average CPU time (user + sys time). This does not include idle time, which can also happen while waiting for GPU sync. |
| Iterations | Total number of batches. This is going to be `total_queries` / `n_queries`. |
| GPU | GPU latency of a single batch (seconds). In throughput mode this is averaged over multiple threads. |
| Latency | Latency of a single batch (seconds), calculated from wall-clock time. In throughput mode this is averaged over multiple threads. |
| Recall | Proportion of correct neighbors to ground truth neighbors. Note this column is only present if groundtruth file is specified in dataset configuration. |
| items_per_second | Total throughput, a.k.a Queries per second (QPS). This is approximately `total_queries` / `end_to_end`. |
| k | Number of neighbors being queried in each iteration |
| end_to_end | Total time taken to run all batches for all iterations |
| n_queries | Total number of query vectors in each batch |
| total_queries | Total number of vector queries across all iterations ( = `iterations` * `n_queries`) |

Notes:

- `Time` and `end_to_end` are measured slightly differently, so `end_to_end = Time * Iterations` is only approximate.
- Output tables may also include hyper-parameters for each benchmarked configuration.
- Recall can fluctuate when fewer neighbors are processed than are available in the benchmark, because processed query count depends on iteration count.
