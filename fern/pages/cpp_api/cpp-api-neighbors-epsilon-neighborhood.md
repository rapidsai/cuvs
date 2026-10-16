---
slug: api-reference/cpp-api-neighbors-epsilon-neighborhood
---

# Epsilon Neighborhood

_Source header: `cuvs/neighbors/epsilon_neighborhood.hpp`_

## Epsilon Neighborhood L2 Operations

<a id="neighbors-epsilon-neighborhood-compute"></a>
### neighbors::epsilon_neighborhood::compute

Computes epsilon neighborhood for the given distance metric and ball size. The epsilon neighbors is represented by a dense boolean adjacency matrix of size m * n and an array of degrees for each vertex, which can be used as a compressed sparse row (CSR) indptr array.

```cpp
template <typename value_t, typename idx_t, typename matrix_idx_t>
void compute(raft::resources const& handle,
raft::device_matrix_view<const value_t, matrix_idx_t, raft::row_major> x,
raft::device_matrix_view<const value_t, matrix_idx_t, raft::row_major> y,
raft::device_matrix_view<bool, matrix_idx_t, raft::row_major> adj,
raft::device_vector_view<idx_t, matrix_idx_t> vd,
value_t eps,
cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded);
```

Currently, only L2Unexpanded (L2-squared) distance metric is supported. Other metrics will throw an exception.

```cpp
 #include <cuvs/neighbors/epsilon_neighborhood.hpp>
 #include <raft/core/resources.hpp>
 #include <raft/core/device_mdarray.hpp>
 using namespace cuvs::neighbors;
 raft::resources handle;
 ...
 auto x = raft::make_device_matrix<float, int64_t>(handle, m, k);
 auto y = raft::make_device_matrix<float, int64_t>(handle, n, k);
 auto adj = raft::make_device_matrix<bool, int64_t>(handle, m, n);
 auto vd = raft::make_device_vector<int, int64_t>(handle, m + 1);
 epsilon_neighborhood::compute(handle, x.view(), y.view(), adj.view(), vd.view(),
                               eps, cuvs::distance::DistanceType::L2Unexpanded);
```

**Template Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `value_t` | `` | IO and math type |
| `idx_t` | `` | Index type |
| `matrix_idx_t` | `` | matrix indexing type |

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | raft handle to manage library resources |
| `x` | in | `raft::device_matrix_view<const value_t, matrix_idx_t, raft::row_major>` | first matrix [row-major] [on device] [dim = m x k] |
| `y` | in | `raft::device_matrix_view<const value_t, matrix_idx_t, raft::row_major>` | second matrix [row-major] [on device] [dim = n x k] |
| `adj` | out | `raft::device_matrix_view<bool, matrix_idx_t, raft::row_major>` | adjacency matrix [row-major] [on device] [dim = m x n] |
| `vd` | out | `raft::device_vector_view<idx_t, matrix_idx_t>` | vertex degree array [on device] [len = m + 1] `vd + m` stores the total number of edges in the adjacency matrix. Pass a nullptr if you don't need this info. |
| `eps` | in | `value_t` | defines epsilon neighborhood radius (should be passed as squared when using L2Unexpanded metric) |
| `metric` | in | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) | distance metric to use. Currently only L2Unexpanded is supported.<br />Default: `cuvs::distance::DistanceType::L2Unexpanded`. |

**Returns**

`void`
