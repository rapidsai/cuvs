# cuVS Bench Datasets

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

## Dataset sources

Million-scale datasets are available from [ann-benchmarks](https://github.com/erikbern/ann-benchmarks#data-sets). These datasets are distributed as HDF5 files. Convert them to cuVS Bench binary files with:

```bash
python/cuvs_bench/cuvs_bench/get_dataset/hdf5_to_fbin.py [-n] <input>.hdf5
```

Use `-n` to normalize base and query vectors. This is useful for angular datasets because normalized cosine search can be measured as inner-product search.

Billion-scale datasets are available from [big-ann-benchmarks](http://big-ann-benchmarks.com). Their ground-truth files contain both neighbors and distances, so split them before running benchmarks:

```bash
python -m cuvs_bench.split_groundtruth deep_new_groundtruth.public.10K.bin groundtruth
```

This produces `groundtruth.neighbors.ibin` and `groundtruth.distances.fbin`.

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
