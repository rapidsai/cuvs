---
slug: api-reference/cpp-api-distance-distance
---

# Distance

_Source header: `cpp/include/cuvs/distance/distance.hpp`_

## Pairwise Distances API

_Doxygen group: `pairwise_distance`_

### kernels::pairwise_distance

Compute pairwise distances for two matrices

```cpp
void pairwise_distance(
raft::resources const& handle,
raft::device_matrix_view<const float, std::int64_t, raft::layout_c_contiguous> const x,
raft::device_matrix_view<const float, std::int64_t, raft::layout_c_contiguous> const y,
raft::device_matrix_view<float, std::int64_t, raft::layout_c_contiguous> dist,
cuvs::distance::DistanceType metric,
float metric_arg = 2.0f);
```

Note: Only contiguous row- or column-major layouts supported currently. Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | raft handle for managing expensive resources |
| `x` | in | `raft::device_matrix_view<const float, std::int64_t, raft::layout_c_contiguous> const` | first set of points (size n*k) |
| `y` | in | `raft::device_matrix_view<const float, std::int64_t, raft::layout_c_contiguous> const` | second set of points (size m*k) |
| `dist` | out | `raft::device_matrix_view<float, std::int64_t, raft::layout_c_contiguous>` | output distance matrix (size n*m) |
| `metric` | in | `cuvs::distance::DistanceType` | distance to evaluate |
| `metric_arg` | in | `float` | metric argument (used for Minkowski distance) Default: `2.0f`. |

**Returns**

`void`

_Source: `cpp/include/cuvs/distance/distance.hpp:161`_

### kernels::pairwise_distance

Compute pairwise distances for two matrices

```cpp
void pairwise_distance(
raft::resources const& handle,
raft::device_matrix_view<const double, std::int64_t, raft::layout_c_contiguous> const x,
raft::device_matrix_view<const double, std::int64_t, raft::layout_c_contiguous> const y,
raft::device_matrix_view<double, std::int64_t, raft::layout_c_contiguous> dist,
cuvs::distance::DistanceType metric,
double metric_arg = 2.0f);
```

Note: Only contiguous row- or column-major layouts supported currently. Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | raft handle for managing expensive resources |
| `x` | in | `raft::device_matrix_view<const double, std::int64_t, raft::layout_c_contiguous> const` | first set of points (size n*k) |
| `y` | in | `raft::device_matrix_view<const double, std::int64_t, raft::layout_c_contiguous> const` | second set of points (size m*k) |
| `dist` | out | `raft::device_matrix_view<double, std::int64_t, raft::layout_c_contiguous>` | output distance matrix (size n*m) |
| `metric` | in | `cuvs::distance::DistanceType` | distance to evaluate |
| `metric_arg` | in | `double` | metric argument (used for Minkowski distance) Default: `2.0f`. |

**Returns**

`void`

_Source: `cpp/include/cuvs/distance/distance.hpp:205`_

### kernels::pairwise_distance

Compute pairwise distances for two matrices

```cpp
void pairwise_distance(
raft::resources const& handle,
raft::device_matrix_view<const half, std::int64_t, raft::layout_c_contiguous> const x,
raft::device_matrix_view<const half, std::int64_t, raft::layout_c_contiguous> const y,
raft::device_matrix_view<float, std::int64_t, raft::layout_c_contiguous> dist,
cuvs::distance::DistanceType metric,
float metric_arg = 2.0f);
```

Note: Only contiguous row- or column-major layouts supported currently. Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | raft handle for managing expensive resources |
| `x` | in | `raft::device_matrix_view<const half, std::int64_t, raft::layout_c_contiguous> const` | first set of points (size n*k) |
| `y` | in | `raft::device_matrix_view<const half, std::int64_t, raft::layout_c_contiguous> const` | second set of points (size m*k) |
| `dist` | out | `raft::device_matrix_view<float, std::int64_t, raft::layout_c_contiguous>` | output distance matrix (size n*m) |
| `metric` | in | `cuvs::distance::DistanceType` | distance to evaluate |
| `metric_arg` | in | `float` | metric argument (used for Minkowski distance) Default: `2.0f`. |

**Returns**

`void`

_Source: `cpp/include/cuvs/distance/distance.hpp:248`_

### kernels::pairwise_distance

Compute pairwise distances for two matrices

```cpp
void pairwise_distance(
raft::resources const& handle,
raft::device_matrix_view<const float, std::int64_t, raft::layout_f_contiguous> const x,
raft::device_matrix_view<const float, std::int64_t, raft::layout_f_contiguous> const y,
raft::device_matrix_view<float, std::int64_t, raft::layout_f_contiguous> dist,
cuvs::distance::DistanceType metric,
float metric_arg = 2.0f);
```

Note: Only contiguous row- or column-major layouts supported currently. Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | raft handle for managing expensive resources |
| `x` | in | `raft::device_matrix_view<const float, std::int64_t, raft::layout_f_contiguous> const` | first set of points (size n*k) |
| `y` | in | `raft::device_matrix_view<const float, std::int64_t, raft::layout_f_contiguous> const` | second set of points (size m*k) |
| `dist` | out | `raft::device_matrix_view<float, std::int64_t, raft::layout_f_contiguous>` | output distance matrix (size n*m) |
| `metric` | in | `cuvs::distance::DistanceType` | distance to evaluate |
| `metric_arg` | in | `float` | metric argument (used for Minkowski distance) Default: `2.0f`. |

**Returns**

`void`

_Source: `cpp/include/cuvs/distance/distance.hpp:292`_

### kernels::pairwise_distance

Compute pairwise distances for two matrices

```cpp
void pairwise_distance(
raft::resources const& handle,
raft::device_matrix_view<const double, std::int64_t, raft::layout_f_contiguous> const x,
raft::device_matrix_view<const double, std::int64_t, raft::layout_f_contiguous> const y,
raft::device_matrix_view<double, std::int64_t, raft::layout_f_contiguous> dist,
cuvs::distance::DistanceType metric,
double metric_arg = 2.0f);
```

Note: Only contiguous row- or column-major layouts supported currently. Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | raft handle for managing expensive resources |
| `x` | in | `raft::device_matrix_view<const double, std::int64_t, raft::layout_f_contiguous> const` | first set of points (size n*k) |
| `y` | in | `raft::device_matrix_view<const double, std::int64_t, raft::layout_f_contiguous> const` | second set of points (size m*k) |
| `dist` | out | `raft::device_matrix_view<double, std::int64_t, raft::layout_f_contiguous>` | output distance matrix (size n*m) |
| `metric` | in | `cuvs::distance::DistanceType` | distance to evaluate |
| `metric_arg` | in | `double` | metric argument (used for Minkowski distance) Default: `2.0f`. |

**Returns**

`void`

_Source: `cpp/include/cuvs/distance/distance.hpp:335`_

### kernels::pairwise_distance

Compute pairwise distances for two matrices

```cpp
void pairwise_distance(
raft::resources const& handle,
raft::device_matrix_view<const half, std::int64_t, raft::layout_f_contiguous> const x,
raft::device_matrix_view<const half, std::int64_t, raft::layout_f_contiguous> const y,
raft::device_matrix_view<float, std::int64_t, raft::layout_f_contiguous> dist,
cuvs::distance::DistanceType metric,
float metric_arg = 2.0f);
```

Note: Only contiguous row- or column-major layouts supported currently. Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | raft handle for managing expensive resources |
| `x` | in | `raft::device_matrix_view<const half, std::int64_t, raft::layout_f_contiguous> const` | first set of points (size n*k) |
| `y` | in | `raft::device_matrix_view<const half, std::int64_t, raft::layout_f_contiguous> const` | second set of points (size m*k) |
| `dist` | out | `raft::device_matrix_view<float, std::int64_t, raft::layout_f_contiguous>` | output distance matrix (size n*m) |
| `metric` | in | `cuvs::distance::DistanceType` | distance to evaluate |
| `metric_arg` | in | `float` | metric argument (used for Minkowski distance) Default: `2.0f`. |

**Returns**

`void`

_Source: `cpp/include/cuvs/distance/distance.hpp:378`_

### kernels::pairwise_distance

Compute sparse pairwise distances between x and y, using the provided

```cpp
void pairwise_distance(raft::resources const& handle,
raft::device_csr_matrix_view<const float, int, int, int> x,
raft::device_csr_matrix_view<const float, int, int, int> y,
raft::device_matrix_view<float, int, raft::row_major> dist,
cuvs::distance::DistanceType metric,
float metric_arg = 2.0f);
```

input configuration and distance function.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | raft::resources |
| `x` | in | `raft::device_csr_matrix_view<const float, int, int, int>` | raft::device_csr_matrix_view |
| `y` | in | `raft::device_csr_matrix_view<const float, int, int, int>` | raft::device_csr_matrix_view |
| `dist` | out | `raft::device_matrix_view<float, int, raft::row_major>` | raft::device_matrix_view dense matrix |
| `metric` | in | `cuvs::distance::DistanceType` | distance metric to use |
| `metric_arg` | in | `float` | metric argument (used for Minkowski distance) Default: `2.0f`. |

**Returns**

`void`

_Source: `cpp/include/cuvs/distance/distance.hpp:419`_

### kernels::pairwise_distance

Compute sparse pairwise distances between x and y, using the provided

```cpp
void pairwise_distance(raft::resources const& handle,
raft::device_csr_matrix_view<const double, int, int, int> x,
raft::device_csr_matrix_view<const double, int, int, int> y,
raft::device_matrix_view<double, int, raft::row_major> dist,
cuvs::distance::DistanceType metric,
float metric_arg = 2.0f);
```

input configuration and distance function.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | raft::resources |
| `x` | in | `raft::device_csr_matrix_view<const double, int, int, int>` | raft::device_csr_matrix_view |
| `y` | in | `raft::device_csr_matrix_view<const double, int, int, int>` | raft::device_csr_matrix_view |
| `dist` | out | `raft::device_matrix_view<double, int, raft::row_major>` | raft::device_matrix_view dense matrix |
| `metric` | in | `cuvs::distance::DistanceType` | distance metric to use |
| `metric_arg` | in | `float` | metric argument (used for Minkowski distance) Default: `2.0f`. |

**Returns**

`void`

_Source: `cpp/include/cuvs/distance/distance.hpp:459`_
