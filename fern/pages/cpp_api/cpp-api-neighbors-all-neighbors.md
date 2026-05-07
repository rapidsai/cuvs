---
slug: api-reference/cpp-api-neighbors-all-neighbors
---

# All Neighbors

_Source header: `cpp/include/cuvs/neighbors/all_neighbors.hpp`_

## The all-neighbors algorithm parameters.

_Doxygen group: `all_neighbors_cpp_params`_

### GraphBuildParams

The all-neighbors algorithm parameters.

```cpp
using GraphBuildParams = std::variant<graph_build_params::ivf_pq_params,
graph_build_params::nn_descent_params,
graph_build_params::brute_force_params>;
```

_Source: `cpp/include/cuvs/neighbors/all_neighbors.hpp:22`_

### cuvs::neighbors::all_neighbors::all_neighbors_params

Parameters used to build an all-neighbors graph (find nearest neighbors for all the

training vectors). For scalability, the all-neighbors graph construction algorithm partitions a set of training vectors into overlapping clusters, computes a local knn graph on each cluster, and merges the local graphs into a single global graph. Device memory usage and accuracy can be configured by changing the `overlap_factor` and `n_clusters`. The algorithm used to build each local graph is also configurable.

```cpp
struct all_neighbors_params { ... } ;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `graph_build_params` | `GraphBuildParams` | Parameters for knn graph building algorithm |
| `overlap_factor` | `size_t` | Number of nearest clusters each data point will be assigned to in the batching algorithm. |
| `n_clusters` | `size_t` | Number of total clusters (aka batches) to split the data into. If set to 1, algorithm creates |
| `metric` | `cuvs::distance::DistanceType` | Metric used. |

_Source: `cpp/include/cuvs/neighbors/all_neighbors.hpp:37`_

## The all-neighbors knn graph build

_Doxygen group: `all_neighbors_cpp_build`_

### cuvs::neighbors::all_neighbors::build

Builds an approximate all-neighbors knn graph  (find nearest neighbors for all the

```cpp
void build(
const raft::resources& handle,
const all_neighbors_params& params,
raft::host_matrix_view<const float, int64_t, row_major> dataset,
raft::device_matrix_view<int64_t, int64_t, row_major> indices,
std::optional<raft::device_matrix_view<float, int64_t, row_major>> distances      = std::nullopt,
std::optional<raft::device_vector_view<float, int64_t, row_major>> core_distances = std::nullopt,
float alpha                                                                       = 1.0);
```

training vectors) Usage example: compute core_distances. If core_distances is given, the resulting indices and distances will be mutual reachability space.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `const raft::resources&` | raft::resources is an object managing resources |
| `params` | in | `const all_neighbors_params&` | an instance of all_neighbors::all_neighbors_params that are parameters to build all-neighbors knn graph |
| `dataset` | in | `raft::host_matrix_view<const float, int64_t, row_major>` | raft::host_matrix_view input dataset expected to be located in host memory |
| `indices` | out | `raft::device_matrix_view<int64_t, int64_t, row_major>` | nearest neighbor indices of shape [n_row x k] |
| `distances` | out | `std::optional<raft::device_matrix_view<float, int64_t, row_major>>` | nearest neighbor distances [n_row x k] Default: `std::nullopt`. |
| `core_distances` | out | `std::optional<raft::device_vector_view<float, int64_t, row_major>>` | array for core distances of size [n_row]. Requires distances matrix to Default: `std::nullopt`. |
| `alpha` | in | `float` | distance scaling parameter as used in robust single linkage. Default: `1.0`. |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/all_neighbors.hpp:126`_

**Additional overload:** `cuvs::neighbors::all_neighbors::build`

Builds an approximate all-neighbors knn graph (find nearest neighbors for all the training

```cpp
void build(
const raft::resources& handle,
const all_neighbors_params& params,
raft::device_matrix_view<const float, int64_t, row_major> dataset,
raft::device_matrix_view<int64_t, int64_t, row_major> indices,
std::optional<raft::device_matrix_view<float, int64_t, row_major>> distances      = std::nullopt,
std::optional<raft::device_vector_view<float, int64_t, row_major>> core_distances = std::nullopt,
float alpha                                                                       = 1.0);
```

vectors) params.n_clusters should be 1 for data on device. To use a larger params.n_clusters for efficient device memory usage, put data on host RAM. Usage example: compute core_distances. If core_distances is given, the resulting indices and distances will be mutual reachability space.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `const raft::resources&` | raft::resources is an object managing resources |
| `params` | in | `const all_neighbors_params&` | an instance of all_neighbors::all_neighbors_params that are parameters to build all-neighbors knn graph |
| `dataset` | in | `raft::device_matrix_view<const float, int64_t, row_major>` | raft::device_matrix_view input dataset expected to be located in device memory |
| `indices` | out | `raft::device_matrix_view<int64_t, int64_t, row_major>` | nearest neighbor indices of shape [n_row x k] |
| `distances` | out | `std::optional<raft::device_matrix_view<float, int64_t, row_major>>` | nearest neighbor distances [n_row x k] Default: `std::nullopt`. |
| `core_distances` | out | `std::optional<raft::device_vector_view<float, int64_t, row_major>>` | array for core distances of size [n_row]. Requires distances matrix to Default: `std::nullopt`. |
| `alpha` | in | `float` | distance scaling parameter as used in robust single linkage. Default: `1.0`. |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/all_neighbors.hpp:162`_
