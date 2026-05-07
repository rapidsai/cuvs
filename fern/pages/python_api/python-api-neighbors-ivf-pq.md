---
slug: api-reference/python-api-neighbors-ivf-pq
---

# IVF PQ

_Python module: `cuvs.neighbors.ivf_pq`_

## Index

```python
cdef class Index
```

IvfPq index object. This object stores the trained IvfPq index state
which can be used to perform nearest neighbors searches.

**Members**

| Name | Kind | Source |
| --- | --- | --- |
| `trained` | property | `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:256` |
| `n_lists` | property | `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:263` |
| `dim` | property | `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:270` |
| `pq_dim` | property | `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:277` |
| `pq_len` | property | `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:284` |
| `pq_bits` | property | `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:292` |
| `centers` | property | `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:305` |
| `centers_padded` | property | `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:319` |
| `pq_centers` | property | `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:335` |
| `centers_rot` | property | `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:349` |
| `rotation_matrix` | property | `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:364` |
| `list_sizes` | property | `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:379` |
| `lists` | method | `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:392` |
| `list_data` | method | `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:411` |
| `list_indices` | method | `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:451` |

### trained

```python
def trained(self)
```

_Source: `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:256`_

### n_lists

```python
def n_lists(self)
```

The number of inverted lists (clusters)

_Source: `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:263`_

### dim

```python
def dim(self)
```

dimensionality of the cluster centers

_Source: `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:270`_

### pq_dim

```python
def pq_dim(self)
```

The dimensionality of an encoded vector after compression by PQ

_Source: `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:277`_

### pq_len

```python
def pq_len(self)
```

The dimensionality of a subspace, i.e. the number of vector
components mapped to a subspace

_Source: `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:284`_

### pq_bits

```python
def pq_bits(self)
```

The bit length of an encoded vector element after
compression by PQ.

_Source: `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:292`_

### centers

```python
def centers(self)
```

Get the cluster centers corresponding to the lists in the
original space

_Source: `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:305`_

### centers_padded

```python
def centers_padded(self)
```

Get the padded cluster centers [n_lists, dim_ext]
where dim_ext = round_up(dim + 1, 8).
This returns contiguous data suitable for build_precomputed.

_Source: `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:319`_

### pq_centers

```python
def pq_centers(self)
```

Get the PQ cluster centers

_Source: `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:335`_

### centers_rot

```python
def centers_rot(self)
```

Get the rotated cluster centers [n_lists, rot_dim]
where rot_dim = pq_len * pq_dim

_Source: `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:349`_

### rotation_matrix

```python
def rotation_matrix(self)
```

Get the rotation matrix [rot_dim, dim]
Transform matrix (original space -&gt; rotated padded space)

_Source: `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:364`_

### list_sizes

```python
def list_sizes(self)
```

Get the sizes of each list

_Source: `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:379`_

### lists

```python
def lists(self, resources=None)
```

Iterates through the pq-encoded list data

This function returns an iterator over each list,
with each value being the pq-encoded data for the
entire list

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `resources` | `cuvs.common.Resources, optional` |  |

_Source: `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:392`_

### list_data

```python
def list_data(self, label, n_rows=0, offset=0, out_codes=None, resources=None)
```

Gets unpacked list data for a single list (cluster)

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `label`, `int` |  | The cluster to get data for |
| `n_rows`, `int` |  | The number of rows to return for the cluster (0 is all rows) |
| `offset`, `int` |  | The row to start getting data at out_codes, CAI Optional buffer to hold memory. Will be created if None |
| `resources` | `cuvs.common.Resources, optional` |  |

_Source: `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:411`_

### list_indices

```python
def list_indices(self, label, n_rows=0)
```

Gets indices for a single cluster (list)

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `label`, `int` |  | The cluster to get data for n_rows, int, optional Number of rows in the list |

_Source: `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:451`_

_Source: `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:239`_

## IndexParams

```python
cdef class IndexParams
```

Parameters to build index for IvfPq nearest neighbor search

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `n_lists` | `int, default = 1024` | The number of clusters used in the coarse quantizer. |
| `metric` | `str, default="sqeuclidean"` | String denoting the metric type. Valid values for metric: ["sqeuclidean", "inner_product", "euclidean", "cosine"], where:<br /><br />- sqeuclidean is the euclidean distance without the square root operation, i.e.: distance(a,b) = \\sum_i (a_i - b_i)^2,<br />- euclidean is the euclidean distance<br />- inner product distance is defined as distance(a, b) = \\sum_i a_i * b_i.<br />- cosine distance is defined as distance(a, b) = 1 - \\sum_i a_i * b_i / ( \|\|a\|\|_2 * \|\|b\|\|_2). |
| `kmeans_n_iters` | `int, default = 20` | The number of iterations searching for kmeans centers during index building. |
| `kmeans_trainset_fraction` | `int, default = 0.5` | If kmeans_trainset_fraction is less than 1, then the dataset is subsampled, and only n_samples * kmeans_trainset_fraction rows are used for training. |
| `pq_bits` | `int, default = 8` | The bit length of the vector element after quantization. |
| `pq_dim` | `int, default = 0` | The dimensionality of a the vector after product quantization. When zero, an optimal value is selected using a heuristic. Note pq_dim * pq_bits must be a multiple of 8. Hint: a smaller 'pq_dim' results in a smaller index size and better search performance, but lower recall. If 'pq_bits' is 8, 'pq_dim' can be set to any number, but multiple of 8 are desirable for good performance. If 'pq_bits' is not 8, 'pq_dim' should be a multiple of 8. For good performance, it is desirable that 'pq_dim' is a multiple of 32. Ideally, 'pq_dim' should be also a divisor of the dataset dim. |
| `codebook_kind` | `string, default = "subspace"` | Valid values ["subspace", "cluster"] |
| `force_random_rotation` | `bool, default = False` | Apply a random rotation matrix on the input data and queries even if `dim % pq_dim == 0`. Note: if `dim` is not multiple of `pq_dim`, a random rotation is always applied to the input data and queries to transform the working space from `dim` to `rot_dim`, which may be slightly larger than the original space and and is a multiple of `pq_dim` (`rot_dim % pq_dim == 0`). However, this transform is not necessary when `dim` is multiple of `pq_dim` (`dim == rot_dim`, hence no need in adding "extra" data columns / features). By default, if `dim == rot_dim`, the rotation transform is initialized with the identity matrix. When `force_random_rotation == True`, a random orthogonal transform matrix is generated regardless of the values of `dim` and `pq_dim`. |
| `add_data_on_build` | `bool, default = True` | After training the coarse and fine quantizers, we will populate the index with the dataset if add_data_on_build == True, otherwise the index is left empty, and the extend method can be used to add new vectors to the index. |
| `conservative_memory_allocation` | `bool, default = True` | By default, the algorithm allocates more space than necessary for individual clusters (`list_data`). This allows to amortize the cost of memory allocation and reduce the number of data copies during repeated calls to `extend` (extending the database). To disable this behavior and use as little GPU memory for the database as possible, set this flat to `True`. |
| `max_train_points_per_pq_code` | `int, default = 256` | The max number of data points to use per PQ code during PQ codebook training. Using more data points per PQ code may increase the quality of PQ codebook but may also increase the build time. The parameter is applied to both PQ codebook generation methods, i.e., PER_SUBSPACE and PER_CLUSTER. In both cases, we will use pq_book_size * max_train_points_per_pq_code training points to train each codebook. |
| `codes_layout` | `string, default = "interleaved"` | Memory layout of the IVF-PQ list data. Valid values ["flat", "interleaved"]<br /><br />- flat: Codes are stored contiguously, one vector's codes after another.<br />- interleaved: Codes are interleaved for optimized search performance. This is the default and recommended for search workloads. |

**Constructor**

```python
def __init__(self, *, n_lists=1024, metric="sqeuclidean", metric_arg=2.0, kmeans_n_iters=20, kmeans_trainset_fraction=0.5, pq_bits=8, pq_dim=0, codebook_kind="subspace", force_random_rotation=False, add_data_on_build=True, conservative_memory_allocation=False, max_train_points_per_pq_code=256, codes_layout="interleaved")
```

**Members**

| Name | Kind | Source |
| --- | --- | --- |
| `get_handle` | method | `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:174` |
| `metric` | property | `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:178` |
| `metric_arg` | property | `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:182` |
| `add_data_on_build` | property | `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:186` |
| `n_lists` | property | `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:190` |
| `kmeans_n_iters` | property | `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:194` |
| `kmeans_trainset_fraction` | property | `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:198` |
| `pq_bits` | property | `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:202` |
| `pq_dim` | property | `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:206` |
| `codebook_kind` | property | `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:210` |
| `force_random_rotation` | property | `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:214` |
| `add_data_on_build` | property | `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:218` |
| `conservative_memory_allocation` | property | `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:222` |
| `max_train_points_per_pq_code` | property | `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:226` |
| `codes_layout` | property | `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:230` |
| `get_handle` | method | `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:236` |

### get_handle

```python
def get_handle(self)
```

_Source: `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:174`_

### metric

```python
def metric(self)
```

_Source: `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:178`_

### metric_arg

```python
def metric_arg(self)
```

_Source: `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:182`_

### add_data_on_build

```python
def add_data_on_build(self)
```

_Source: `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:186`_

### n_lists

```python
def n_lists(self)
```

_Source: `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:190`_

### kmeans_n_iters

```python
def kmeans_n_iters(self)
```

_Source: `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:194`_

### kmeans_trainset_fraction

```python
def kmeans_trainset_fraction(self)
```

_Source: `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:198`_

### pq_bits

```python
def pq_bits(self)
```

_Source: `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:202`_

### pq_dim

```python
def pq_dim(self)
```

_Source: `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:206`_

### codebook_kind

```python
def codebook_kind(self)
```

_Source: `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:210`_

### force_random_rotation

```python
def force_random_rotation(self)
```

_Source: `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:214`_

### add_data_on_build

```python
def add_data_on_build(self)
```

_Source: `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:218`_

### conservative_memory_allocation

```python
def conservative_memory_allocation(self)
```

_Source: `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:222`_

### max_train_points_per_pq_code

```python
def max_train_points_per_pq_code(self)
```

_Source: `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:226`_

### codes_layout

```python
def codes_layout(self)
```

_Source: `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:230`_

### get_handle

```python
def get_handle(self)
```

_Source: `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:236`_

_Source: `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:40`_

## SearchParams

```python
cdef class SearchParams
```

Supplemental parameters to search IVF-Pq index

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `n_probes` | `int` | The number of clusters to search. |
| `lut_dtype` | `default = np.float32` | Data type of look up table to be created dynamically at search time. The use of low-precision types reduces the amount of shared memory required at search time, so fast shared memory kernels can be used even for datasets with large dimansionality. Note that the recall is slightly degraded when low-precision type is selected. Possible values [np.float32, np.float16, np.uint8] |
| `internal_distance_dtype` | `default = np.float32` | Storage data type for distance/similarity computation. Possible values [np.float32, np.float16] |
| `coarse_search_dtype` | `default = np.float32` | [Experimental] The data type to use as the GEMM element type when searching the clusters to probe. Possible values: [np.float32, np.float16, np.int8].<br />- Legacy default: np.float32<br />- Recommended for performance: np.float16 (half)<br />- Experimental/low-precision: np.int8 |
| `max_internal_batch_size` | `default = 4096` | Set the internal batch size to improve GPU utilization at the cost of larger memory footprint. |

**Constructor**

```python
def __init__(self, *, n_probes=20, lut_dtype=np.float32, internal_distance_dtype=np.float32, coarse_search_dtype=np.float32, max_internal_batch_size=4096)
```

**Members**

| Name | Kind | Source |
| --- | --- | --- |
| `get_handle` | method | `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:716` |
| `n_probes` | property | `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:720` |
| `lut_dtype` | property | `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:724` |
| `internal_distance_dtype` | property | `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:728` |
| `coarse_search_dtype` | property | `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:732` |
| `max_internal_batch_size` | property | `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:736` |
| `get_handle` | method | `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:739` |

### get_handle

```python
def get_handle(self)
```

_Source: `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:716`_

### n_probes

```python
def n_probes(self)
```

_Source: `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:720`_

### lut_dtype

```python
def lut_dtype(self)
```

_Source: `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:724`_

### internal_distance_dtype

```python
def internal_distance_dtype(self)
```

_Source: `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:728`_

### coarse_search_dtype

```python
def coarse_search_dtype(self)
```

_Source: `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:732`_

### max_internal_batch_size

```python
def max_internal_batch_size(self)
```

_Source: `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:736`_

### get_handle

```python
def get_handle(self)
```

_Source: `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:739`_

_Source: `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:667`_

## build

`@auto_sync_resources`

```python
def build(IndexParams index_params, dataset, resources=None)
```

Build the IvfPq index from the dataset for efficient search.

The input dataset array can be either CUDA array interface compliant matrix
or an array interface compliant matrix in host memory.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `index_params` | `cuvs.neighbors.ivf_pq.IndexParams` | Parameters on how to build the index |
| `dataset` | `Array interface compliant matrix shape (n_samples, dim)` | Supported dtype [float32, float16, int8, uint8] |
| `resources` | `cuvs.common.Resources, optional` |  |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| `index` | `cuvs.neighbors.ivf_pq.Index` |  |

**Examples**

```python
>>> import cupy as cp
>>> from cuvs.neighbors import ivf_pq
>>> n_samples = 50000
>>> n_features = 50
>>> n_queries = 1000
>>> k = 10
>>> dataset = cp.random.random_sample((n_samples, n_features),
...                                   dtype=cp.float32)
>>> build_params = ivf_pq.IndexParams(metric="sqeuclidean")
>>> index = ivf_pq.build(build_params, dataset)
>>> distances, neighbors = ivf_pq.search(ivf_pq.SearchParams(),
...                                        index, dataset,
...                                        k)
>>> distances = cp.asarray(distances)
>>> neighbors = cp.asarray(neighbors)
```

_Source: `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:477`_

## build_precomputed

`@auto_sync_resources`

```python
def build_precomputed(IndexParams index_params, uint32_t dim, pq_centers, centers, centers_rot, rotation_matrix, resources=None)
```

Build a view-type IVF-PQ index from precomputed centroids and codebook.

This function creates a non-owning index that stores a reference to the provided device data.
All parameters must be provided with correct extents. The caller is responsible for ensuring
the lifetime of the input data exceeds the lifetime of the returned index.

The index_params must be consistent with the provided matrices. Specifically:
- index_params.codebook_kind determines the expected shape of pq_centers
- index_params.metric will be stored in the index
- index_params.conservative_memory_allocation will be stored in the index

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `index_params` | `cuvs.neighbors.ivf_pq.IndexParams` | Parameters that must be consistent with the provided matrices |
| `dim` | `int` | Dimensionality of the input data |
| `pq_centers` | `CUDA array interface compliant tensor` | PQ codebook on device memory with required shape:<br />- codebook_kind "subspace": [pq_dim, pq_len, pq_book_size]<br />- codebook_kind "cluster":  [n_lists, pq_len, pq_book_size] Supported dtype: float32 |
| `centers` | `CUDA array interface compliant matrix` | Cluster centers in the original space [n_lists, dim_ext] where dim_ext = round_up(dim + 1, 8). Supported dtype: float32 |
| `centers_rot` | `CUDA array interface compliant matrix` | Rotated cluster centers [n_lists, rot_dim] where rot_dim = pq_len * pq_dim. Supported dtype: float32 |
| `rotation_matrix` | `CUDA array interface compliant matrix` | Transform matrix (original space -&gt; rotated padded space) [rot_dim, dim]. Supported dtype: float32 |
| `resources` | `cuvs.common.Resources, optional` |  |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| `index` | `cuvs.neighbors.ivf_pq.Index` |  |

**Examples**

```python
>>> import cupy as cp
>>> from cuvs.neighbors import ivf_pq
>>> n_lists = 100
>>> dim = 128
>>> pq_dim = 16
>>> pq_bits = 8
>>> pq_len = (dim + pq_dim - 1) // pq_dim  # ceil division
>>> pq_book_size = 1 << pq_bits
>>> rot_dim = pq_len * pq_dim
>>> dim_ext = ((dim + 1 + 7) // 8) * 8  # round_up(dim + 1, 8)
>>>
>>> # Prepare precomputed matrices (example with random data)
>>> pq_centers = cp.random.random((pq_dim, pq_len, pq_book_size),
...                               dtype=cp.float32)
>>> centers = cp.random.random((n_lists, dim_ext), dtype=cp.float32)
>>> centers_rot = cp.random.random((n_lists, rot_dim), dtype=cp.float32)
>>> rotation_matrix = cp.random.random((rot_dim, dim), dtype=cp.float32)
>>>
>>> # Build index from precomputed data
>>> build_params = ivf_pq.IndexParams(n_lists=n_lists, pq_dim=pq_dim,
...                                    pq_bits=pq_bits,
...                                    codebook_kind="subspace")
>>> index = ivf_pq.build_precomputed(build_params, dim, pq_centers,
...                                   centers, centers_rot, rotation_matrix)
```

_Source: `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:543`_

## extend

`@auto_sync_resources`

```python
def extend(Index index, new_vectors, new_indices, resources=None)
```

Extend an existing index with new vectors.

The input array can be either CUDA array interface compliant matrix or
array interface compliant matrix in host memory.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `index` | `ivf_pq.Index` | Trained ivf_pq object. |
| `new_vectors` | `array interface compliant matrix shape (n_samples, dim)` | Supported dtype [float, int8, uint8] |
| `new_indices` | `array interface compliant vector shape (n_samples)` | Supported dtype [int64] |
| `resources` | `cuvs.common.Resources, optional` |  |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| `index` | `cuvs.neighbors.ivf_pq.Index` |  |

**Examples**

```python
>>> import cupy as cp
>>> from cuvs.neighbors import ivf_pq
>>> n_samples = 50000
>>> n_features = 50
>>> n_queries = 1000
>>> dataset = cp.random.random_sample((n_samples, n_features),
...                                   dtype=cp.float32)
>>> index = ivf_pq.build(ivf_pq.IndexParams(), dataset)
>>> n_rows = 100
>>> more_data = cp.random.random_sample((n_rows, n_features),
...                                     dtype=cp.float32)
>>> indices = n_samples + cp.arange(n_rows, dtype=cp.int64)
>>> index = ivf_pq.extend(index, more_data, indices)
>>> # Search using the built index
>>> queries = cp.random.random_sample((n_queries, n_features),
...                                   dtype=cp.float32)
>>> distances, neighbors = ivf_pq.search(ivf_pq.SearchParams(),
...                                      index, queries,
...                                      k=10)
```

_Source: `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:913`_

## load

`@auto_sync_resources`

```python
def load(filename, resources=None)
```

Loads index from file.

Saving / loading the index is experimental. The serialization format is
subject to change, therefore loading an index saved with a previous
version of cuvs is not guaranteed to work.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `filename` | `string` | Name of the file. |
| `resources` | `cuvs.common.Resources, optional` |  |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| `index` | `Index` |  |

_Source: `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:880`_

## save

`@auto_sync_resources`

```python
def save(filename, Index index, bool include_dataset=True, resources=None)
```

Saves the index to a file.

Saving / loading the index is experimental. The serialization format is
subject to change.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `filename` | `string` | Name of the file. |
| `index` | `Index` | Trained IVF-PQ index. |
| `resources` | `cuvs.common.Resources, optional` |  |

**Examples**

```python
>>> import cupy as cp
>>> from cuvs.neighbors import ivf_pq
>>> n_samples = 50000
>>> n_features = 50
>>> dataset = cp.random.random_sample((n_samples, n_features),
...                                   dtype=cp.float32)
>>> # Build index
>>> index = ivf_pq.build(ivf_pq.IndexParams(), dataset)
>>> # Serialize and deserialize the ivf_pq index built
>>> ivf_pq.save("my_index.bin", index)
>>> index_loaded = ivf_pq.load("my_index.bin")
```

_Source: `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:843`_

## search

`@auto_sync_resources`
`@auto_convert_output`

```python
def search(SearchParams search_params, Index index, queries, k, neighbors=None, distances=None, resources=None)
```

Find the k nearest neighbors for each query.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `search_params` | `cuvs.neighbors.ivf_pq.SearchParams` | Parameters on how to search the index |
| `index` | `cuvs.neighbors.ivf_pq.Index` | Trained IvfPq index. |
| `queries` | `CUDA array interface compliant matrix shape (n_samples, dim)` | Supported dtype [float, int8, uint8] |
| `k` | `int` | The number of neighbors. |
| `neighbors` | `Optional CUDA array interface compliant matrix shape` | (n_queries, k), dtype int64_t. If supplied, neighbor indices will be written here in-place. (default None) |
| `distances` | `Optional CUDA array interface compliant matrix shape` | (n_queries, k) If supplied, the distances to the neighbors will be written here in-place. (default None) |
| `resources` | `cuvs.common.Resources, optional` |  |

**Examples**

```python
>>> import cupy as cp
>>> from cuvs.neighbors import ivf_pq
>>> n_samples = 50000
>>> n_features = 50
>>> n_queries = 1000
>>> dataset = cp.random.random_sample((n_samples, n_features),
...                                   dtype=cp.float32)
>>> # Build the index
>>> index = ivf_pq.build(ivf_pq.IndexParams(), dataset)
>>>
>>> # Search using the built index
>>> queries = cp.random.random_sample((n_queries, n_features),
...                                   dtype=cp.float32)
>>> k = 10
>>> search_params = ivf_pq.SearchParams(n_probes=20)
>>>
>>> distances, neighbors = ivf_pq.search(search_params, index, queries,
...                                     k)
```

_Source: `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:745`_

## transform

`@auto_sync_resources`

```python
def transform(Index index, input_dataset, output_labels=None, output_dataset=None, resources=None)
```

Transform a dataset by applying pq-encoding to the vectors.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `index` | `ivf_pq.Index` | Trained ivf_pq object. |
| `input_dataset` | `array interface compliant matrix shape (n_samples, dim)` | Supported dtype [float] |
| `new_indices` | `Optional array interface compliant vector shape (n_samples)` | Supported dtype [uint32] |
| `output_dataset` | `Optional array interface compliant matrix shape (n_samples, pq_dim)` | Supported dtype [uint8] |
| `resources` | `cuvs.common.Resources, optional` |  |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| `output_labels`, `output_dataset` |  | The cluster that each point in the dataset belongs to, and the transformed dataset |

_Source: `python/cuvs/cuvs/neighbors/ivf_pq/ivf_pq.pyx:987`_
