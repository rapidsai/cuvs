---
slug: api-reference/cpp-api-cluster-spectral
---

# Spectral

_Source header: `cpp/include/cuvs/cluster/spectral.hpp`_

## Spectral Clustering Parameters

_Doxygen group: `spectral_params`_

<a id="cuvs-cluster-spectral-params"></a>
### cuvs::cluster::spectral::params

Parameters for spectral clustering

```cpp
struct params { ... } ;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `n_clusters` | `int` | Number of clusters to find |
| `n_components` | `int` | Number of eigenvectors to use for the spectral embedding (typically equal to n_clusters) |
| `n_init` | `int` | Number of k-means runs with different centroid seeds |
| `n_neighbors` | `int` | Number of nearest neighbors for constructing the connectivity graph |
| `tolerance` | `float` | Tolerance for the eigenvalue solver |
| `rng_state` | `raft::random::RngState` | Random number generator state for reproducibility |

_Source: `cpp/include/cuvs/cluster/spectral.hpp:22`_

## Spectral Clustering

_Doxygen group: `spectral`_

<a id="cuvs-cluster-spectral-fit-predict"></a>
### cuvs::cluster::spectral::fit_predict

Perform spectral clustering on a connectivity graph

```cpp
void fit_predict(raft::resources const& handle,
params config,
raft::device_coo_matrix_view<float, int, int, int> connectivity_graph,
raft::device_vector_view<int, int> labels);
```

n_clusters-1)

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | RAFT resource handle |
| `config` | in | [`params`](/api-reference/cpp-api-cluster-spectral#cuvs-cluster-spectral-params) | Spectral clustering parameters |
| `connectivity_graph` | in | `raft::device_coo_matrix_view<float, int, int, int>` | Sparse COO matrix representing connectivity between data points |
| `labels` | out | `raft::device_vector_view<int, int>` | Device vector of size n_samples to store cluster assignments (0 to |

**Returns**

`void`

_Source: `cpp/include/cuvs/cluster/spectral.hpp:84`_

**Additional overload:** `cuvs::cluster::spectral::fit_predict`

Perform spectral clustering on a connectivity graph

```cpp
void fit_predict(raft::resources const& handle,
params config,
raft::device_coo_matrix_view<double, int, int, int> connectivity_graph,
raft::device_vector_view<int, int> labels);
```

n_clusters-1)

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | RAFT resource handle |
| `config` | in | [`params`](/api-reference/cpp-api-cluster-spectral#cuvs-cluster-spectral-params) | Spectral clustering parameters |
| `connectivity_graph` | in | `raft::device_coo_matrix_view<double, int, int, int>` | Sparse COO matrix representing connectivity between data points |
| `labels` | out | `raft::device_vector_view<int, int>` | Device vector of size n_samples to store cluster assignments (0 to |

**Returns**

`void`

_Source: `cpp/include/cuvs/cluster/spectral.hpp:122`_

**Additional overload:** `cuvs::cluster::spectral::fit_predict`

Perform spectral clustering on a dense dataset

```cpp
void fit_predict(raft::resources const& handle,
params config,
raft::device_matrix_view<float, int, raft::row_major> dataset,
raft::device_vector_view<int, int> labels);
```

This overload automatically constructs the connectivity graph from the input dataset using k-nearest neighbors.

n_clusters-1)

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | RAFT resource handle |
| `config` | in | [`params`](/api-reference/cpp-api-cluster-spectral#cuvs-cluster-spectral-params) | Spectral clustering parameters |
| `dataset` | in | `raft::device_matrix_view<float, int, raft::row_major>` | Dense row-major matrix of shape (n_samples, n_features) |
| `labels` | out | `raft::device_vector_view<int, int>` | Device vector of size n_samples to store cluster assignments (0 to |

**Returns**

`void`

_Source: `cpp/include/cuvs/cluster/spectral.hpp:155`_
