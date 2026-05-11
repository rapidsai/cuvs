# Benchmark Datasets

cuVS Bench datasets define the vectors to index, the queries to run, and the exact ground truth used to measure recall. Use this page to prepare built-in datasets, convert external datasets into the cuVS Bench binary format, and describe custom datasets in YAML.

## Dataset files

cuVS Bench datasets usually contain four binary files:

| File | Purpose |
| --- | --- |
| `base.fbin` | Database vectors used to build the index |
| `query.fbin` | Query vectors used during search |
| `groundtruth.neighbors.ibin` | Exact nearest-neighbor ids |
| `groundtruth.distances.fbin` | Exact nearest-neighbor distances |

The vector files are used for build and search. The ground-truth files are tied to a distance metric and are used to evaluate recall.

## Binary format

Dataset suffixes describe the stored type:

| Suffix | Type |
| --- | --- |
| `.fbin` | `float32` |
| `.f16bin` | `float16` |
| `.ibin` | `int32` |
| `.u8bin` | `uint8` |
| `.i8bin` | `int8` |

All binary files are little-endian. The first 8 bytes store `num_vectors` and `num_dimensions` as `uint32_t` values. The remaining bytes store `num_vectors * num_dimensions` values in row-major order.

Some implementations can use `float16` vectors for better performance. Convert `float32` files with:

```bash
python/cuvs_bench/cuvs_bench/get_dataset/fbin_to_f16bin.py
```

## Built-in datasets

The `cuvs_bench.get_dataset` helper can download and prepare several smaller benchmark datasets. Datasets are stored under `RAPIDS_DATASET_ROOT_DIR` when set, otherwise under a local `datasets` subdirectory.

```bash
python -m cuvs_bench.get_dataset --dataset deep-image-96-angular --normalize
```

Common built-in datasets include:

| Dataset name | Train rows | Columns | Test rows | Distance |
| --- | --- | --- | --- | --- |
| `deep-image-96-angular` | 10M | 96 | 10K | Angular |
| `fashion-mnist-784-euclidean` | 60K | 784 | 10K | Euclidean |
| `glove-50-angular` | 1.1M | 50 | 10K | Angular |
| `glove-100-angular` | 1.1M | 100 | 10K | Angular |
| `mnist-784-euclidean` | 60K | 784 | 10K | Euclidean |
| `nytimes-256-angular` | 290K | 256 | 10K | Angular |
| `sift-128-euclidean` | 1M | 128 | 10K | Euclidean |

These datasets include ground-truth test sets with 100 neighbors, so `k` must be less than or equal to 100.

## Dataset sources

Million-scale datasets are available from [ann-benchmarks](https://github.com/erikbern/ann-benchmarks#data-sets). These datasets are distributed as HDF5 files. Convert them to cuVS Bench binary files with:

```bash
python/cuvs_bench/cuvs_bench/get_dataset/hdf5_to_fbin.py [-n] <input>.hdf5
```

Use `-n` to normalize base and query vectors. This is useful for angular datasets because normalized cosine search can be measured as inner-product search.

Billion-scale datasets are available from [big-ann-benchmarks](http://big-ann-benchmarks.com). Their ground-truth files contain both neighbors and distances, so split them before running benchmarks:

```bash
python -m cuvs_bench.split_groundtruth --groundtruth deep_new_groundtruth.public.10K.bin
```

This produces `groundtruth.neighbors.ibin` and `groundtruth.distances.fbin`.

The `wiki-all` dataset contains 88M 768-dimensional vectors for realistic RAG/LLM-scale benchmarking, plus 1M and 10M subsets. See the [Wiki-all Dataset Guide](wiki_all_dataset.md) to download it.

## Generate ground truth

If a dataset does not include ground truth, generate it with `cuvs_bench.generate_groundtruth`:

```bash
# With an existing query file
python -m cuvs_bench.generate_groundtruth --dataset /dataset/base.fbin --output=groundtruth_dir --queries=/dataset/query.public.10K.fbin

# With randomly generated queries
python -m cuvs_bench.generate_groundtruth --dataset /dataset/base.fbin --output=groundtruth_dir --queries=random --n_queries=10000

# With random queries selected from a subset of the dataset
python -m cuvs_bench.generate_groundtruth --dataset /dataset/base.fbin --nrows=2000000 --output=groundtruth_dir --queries=random-choice --n_queries=10000
```

For billion-scale sources that provide ground truth for only the first 10M or 100M base vectors, use `subset_size` in the dataset configuration so the benchmark uses the matching prefix of the base file.

## Dataset configurations

Each benchmark dataset needs a descriptor with file names and basic dataset properties. Descriptors for several common datasets are already available in [datasets.yaml](https://github.com/rapidsai/cuvs/blob/branch-25.04/python/cuvs_bench/cuvs_bench/config/datasets/datasets.yaml).

The default `${CUVS_HOME}/python/cuvs_bench/cuvs_bench/config/datasets/datasets.yaml` includes entries such as:

```yaml
- name: sift-128-euclidean
  base_file: sift-128-euclidean/base.fbin
  query_file: sift-128-euclidean/query.fbin
  groundtruth_neighbors_file: sift-128-euclidean/groundtruth.neighbors.ibin
  dims: 128
  distance: euclidean
```

For a new dataset, create a descriptor such as `mydataset.yaml`:

```yaml
- name: mydata-1M
  base_file: mydata-1M/base.100M.u8bin
  subset_size: 1000000
  dims: 128
  query_file: mydata-10M/queries.u8bin
  groundtruth_neighbors_file: mydata-1M/groundtruth.neighbors.ibin
  distance: euclidean
```

Choose any `name` and pass it as `--dataset`. File paths are relative to `--dataset-path`. The optional `subset_size` uses the first `subset_size` vectors, which lets you benchmark multiple subsets without duplicating vectors. Ground truth must be generated separately for each subset.

To run the benchmark on the newly defined `mydata-1M` dataset, use:

```bash
python -m cuvs_bench.run --dataset mydata-1M --dataset-path=/path/to/data/folder --dataset-configuration=mydataset.yaml --algorithms=cuvs_cagra
```
