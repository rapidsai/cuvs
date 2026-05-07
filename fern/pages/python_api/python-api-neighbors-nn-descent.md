---
slug: api-reference/python-api-neighbors-nn-descent
---

# NN Descent

_Python module: `cuvs.neighbors.nn_descent`_

## Index

```python
cdef class Index
```

NN-Descent index object. This object stores the trained NN-Descent index,
which can be used to get the NN-Descent graph and distances after
building

**Members**

| Name | Kind | Source |
| --- | --- | --- |
| `trained` | property | `python/cuvs/cuvs/neighbors/nn_descent/nn_descent.pyx:165` |
| `graph` | property | `python/cuvs/cuvs/neighbors/nn_descent/nn_descent.pyx:169` |
| `distances` | property | `python/cuvs/cuvs/neighbors/nn_descent/nn_descent.pyx:173` |

### trained

```python
def trained(self)
```

_Source: `python/cuvs/cuvs/neighbors/nn_descent/nn_descent.pyx:165`_

### graph

```python
def graph(self)
```

_Source: `python/cuvs/cuvs/neighbors/nn_descent/nn_descent.pyx:169`_

### distances

```python
def distances(self)
```

_Source: `python/cuvs/cuvs/neighbors/nn_descent/nn_descent.pyx:173`_

_Source: `python/cuvs/cuvs/neighbors/nn_descent/nn_descent.pyx:143`_

## IndexParams

```python
cdef class IndexParams
```

Parameters to build NN-Descent Index

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `metric` | `str, default = "sqeuclidean"` | String denoting the metric type. Supported metrics are `l2`, `euclidean`, `sqeuclidean`, `inner_product`, `cosine`, and `bitwise_hamming` (`bitwise_hamming` is for int8 and uint8 data types only) |
| `graph_degree` | `int` | For an input dataset of dimensions (N, D), determines the final dimensions of the all-neighbors knn graph which turns out to be of dimensions (N, graph_degree) |
| `intermediate_graph_degree` | `int` | Internally, nn-descent builds an all-neighbors knn graph of dimensions (N, intermediate_graph_degree) before selecting the final `graph_degree` neighbors. It's recommended that `intermediate_graph_degree` &gt;= 1.5 * graph_degree |
| `max_iterations` | `int` | The number of iterations that nn-descent will refine the graph for. More iterations produce a better quality graph at cost of performance |
| `termination_threshold` | `float` | The delta at which nn-descent will terminate its iterations |
| `return_distances` | `bool` | Whether to return distances array |
| `dist_comp_dtype` | `str, default = "auto"` | Dtype to use for distance computation. Supported dtypes are `auto`, `fp32`, and `fp16` `auto` automatically determines the best dtype for distance computation based on the dataset dimensions. `fp32` uses fp32 distance computation for better precision at the cost of performance and memory usage. This option is only valid when data type is fp32. `fp16` uses fp16 distance computation for better performance and memory usage at the cost of precision. |

**Constructor**

```python
def __init__(self, *, metric=None, metric_arg=None, graph_degree=None, intermediate_graph_degree=None, max_iterations=None, termination_threshold=None, return_distances=None, dist_comp_dtype="auto" )
```

**Members**

| Name | Kind | Source |
| --- | --- | --- |
| `metric` | property | `python/cuvs/cuvs/neighbors/nn_descent/nn_descent.pyx:116` |
| `metric_arg` | property | `python/cuvs/cuvs/neighbors/nn_descent/nn_descent.pyx:120` |
| `graph_degree` | property | `python/cuvs/cuvs/neighbors/nn_descent/nn_descent.pyx:124` |
| `intermediate_graph_degree` | property | `python/cuvs/cuvs/neighbors/nn_descent/nn_descent.pyx:128` |
| `max_iterations` | property | `python/cuvs/cuvs/neighbors/nn_descent/nn_descent.pyx:132` |
| `termination_threshold` | property | `python/cuvs/cuvs/neighbors/nn_descent/nn_descent.pyx:136` |
| `get_handle` | method | `python/cuvs/cuvs/neighbors/nn_descent/nn_descent.pyx:139` |

### metric

```python
def metric(self)
```

_Source: `python/cuvs/cuvs/neighbors/nn_descent/nn_descent.pyx:116`_

### metric_arg

```python
def metric_arg(self)
```

_Source: `python/cuvs/cuvs/neighbors/nn_descent/nn_descent.pyx:120`_

### graph_degree

```python
def graph_degree(self)
```

_Source: `python/cuvs/cuvs/neighbors/nn_descent/nn_descent.pyx:124`_

### intermediate_graph_degree

```python
def intermediate_graph_degree(self)
```

_Source: `python/cuvs/cuvs/neighbors/nn_descent/nn_descent.pyx:128`_

### max_iterations

```python
def max_iterations(self)
```

_Source: `python/cuvs/cuvs/neighbors/nn_descent/nn_descent.pyx:132`_

### termination_threshold

```python
def termination_threshold(self)
```

_Source: `python/cuvs/cuvs/neighbors/nn_descent/nn_descent.pyx:136`_

### get_handle

```python
def get_handle(self)
```

Get a pointer to the underlying C object.

_Source: `python/cuvs/cuvs/neighbors/nn_descent/nn_descent.pyx:139`_

_Source: `python/cuvs/cuvs/neighbors/nn_descent/nn_descent.pyx:39`_

## build

`@auto_sync_resources`

```python
def build(IndexParams index_params, dataset, graph=None, resources=None)
```

Build KNN graph from the dataset

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `index_params` | `cuvs.neighbors.nn_descent.IndexParams` |  |
| `dataset` | `Array interface compliant matrix, on either host or device memory` | Supported dtype [float, int8, uint8] |
| `graph` | `Optional host matrix for storing output graph` |  |
| `resources` | `cuvs.common.Resources, optional` |  |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| `index` | `cuvs.neighbors.nn_descent.Index` |  |

**Examples**

```python
>>> import cupy as cp
>>> from cuvs.neighbors import nn_descent
>>> n_samples = 50000
>>> n_features = 50
>>> n_queries = 1000
>>> k = 10
>>> dataset = cp.random.random_sample((n_samples, n_features),
...                                   dtype=cp.float32)
>>> build_params = nn_descent.IndexParams(metric="sqeuclidean")
>>> index = nn_descent.build(build_params, dataset)
>>> graph = index.graph
```

_Source: `python/cuvs/cuvs/neighbors/nn_descent/nn_descent.pyx:210`_
