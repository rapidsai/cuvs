---
slug: api-reference/cpp-api-neighbors-ivf-flat
---

# IVF Flat

_Source header: `cpp/include/cuvs/neighbors/ivf_flat.hpp`_

## IVF-Flat index search parameters

_Doxygen group: `ivf_flat_cpp_search_params`_

### cuvs::neighbors::ivf_flat::search_params

IVF-Flat index search parameters

```cpp
struct search_params : cuvs::neighbors::search_params { ... } ;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `n_probes` | `uint32_t` | The number of clusters to search. |
| `metric_udf` | `std::optional<std::string>` | Custom metric UDF code. |

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:73`_

### cuvs::neighbors::ivf_flat::make_list_extents

Determine the extents of an array enough to hold a given amount of data.

```cpp
constexpr auto make_list_extents(SizeT n_rows) const -> list_extents;
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `n_rows` |  | `SizeT` |  |

**Returns**

`list_extents`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:108`_

## IVF-Flat index

_Doxygen group: `ivf_flat_cpp_index`_

### cuvs::neighbors::ivf_flat::index

Construct an empty index.

```cpp
index(raft::resources const& res);
```

Constructs an empty index. This index will either need to be trained with `build` or loaded from a saved copy with `deserialize`

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:154`_

**Additional overload:** `cuvs::neighbors::ivf_flat::index`

Construct an empty index. It needs to be trained and then populated.

```cpp
index(raft::resources const& res, const index_params& params, uint32_t dim);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `params` |  | `const index_params&` |  |
| `dim` |  | `uint32_t` |  |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:157`_

**Additional overload:** `cuvs::neighbors::ivf_flat::index`

Construct an empty index. It needs to be trained and then populated.

```cpp
index(raft::resources const& res,
cuvs::distance::DistanceType metric,
uint32_t n_lists,
bool adaptive_centers,
bool conservative_memory_allocation,
uint32_t dim);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `metric` |  | `cuvs::distance::DistanceType` |  |
| `n_lists` |  | `uint32_t` |  |
| `adaptive_centers` |  | `bool` |  |
| `conservative_memory_allocation` |  | `bool` |  |
| `dim` |  | `uint32_t` |  |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:159`_

### cuvs::neighbors::ivf_flat::veclen

Vectorized load/store size in elements, determines the size of interleaved data chunks.

```cpp
uint32_t veclen() const noexcept;
```

**Returns**

`uint32_t`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:169`_

### cuvs::neighbors::ivf_flat::metric

Distance metric used for clustering.

```cpp
cuvs::distance::DistanceType metric() const noexcept;
```

**Returns**

`cuvs::distance::DistanceType`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:172`_

### cuvs::neighbors::ivf_flat::adaptive_centers

Whether `centers()` change upon extending the index (ivf_flat::extend).

```cpp
bool adaptive_centers() const noexcept;
```

**Returns**

`bool`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:175`_

### cuvs::neighbors::ivf_flat::list_sizes

Sizes of the lists (clusters) [n_lists]

```cpp
raft::device_vector_view<uint32_t, uint32_t> list_sizes() noexcept;
```

NB: This may differ from the actual list size if the shared lists have been extended by another index

**Returns**

`raft::device_vector_view<uint32_t, uint32_t>`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:204`_

### cuvs::neighbors::ivf_flat::centers

k-means cluster centers corresponding to the lists [n_lists, dim]

```cpp
raft::device_matrix_view<float, uint32_t, raft::row_major> centers() noexcept;
```

**Returns**

`raft::device_matrix_view<float, uint32_t, raft::row_major>`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:208`_

### cuvs::neighbors::ivf_flat::center_norms

(Optional) Precomputed norms of the `centers` w.r.t. the chosen distance metric [n_lists].

```cpp
std::optional<raft::device_vector_view<float, uint32_t>> center_norms() noexcept;
```

NB: this may be empty if the index is empty or if the metric does not require the center norms calculation.

**Returns**

`std::optional<raft::device_vector_view<float, uint32_t>>`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:217`_

### cuvs::neighbors::ivf_flat::accum_sorted_sizes

Accumulated list sizes, sorted in descending order [n_lists + 1].

```cpp
auto accum_sorted_sizes() noexcept -> raft::host_vector_view<IdxT, uint32_t>;
```

The last value contains the total length of the index. The value at index zero is always zero. That is, the content of this span is as if the `list_sizes` was sorted and then accumulated. This span is used during search to estimate the maximum size of the workspace.

**Returns**

`raft::host_vector_view<IdxT, uint32_t>`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:229`_

### cuvs::neighbors::ivf_flat::size

Total length of the index.

```cpp
IdxT size() const noexcept;
```

**Returns**

`IdxT`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:234`_

### cuvs::neighbors::ivf_flat::dim

Dimensionality of the data.

```cpp
uint32_t dim() const noexcept;
```

**Returns**

`uint32_t`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:237`_

### cuvs::neighbors::ivf_flat::n_lists

Number of clusters/inverted lists.

```cpp
uint32_t n_lists() const noexcept;
```

**Returns**

`uint32_t`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:240`_

### cuvs::neighbors::ivf_flat::inds_ptrs

Pointers to the inverted lists (clusters) indices  [n_lists].

```cpp
raft::device_vector_view<IdxT*, uint32_t> inds_ptrs() noexcept;
```

**Returns**

`raft::device_vector_view<IdxT*, uint32_t>`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:245`_

### cuvs::neighbors::ivf_flat::conservative_memory_allocation

Whether to use conservative memory allocation when extending the list (cluster) data

```cpp
bool conservative_memory_allocation() const noexcept;
```

(see index_params.conservative_memory_allocation).

**Returns**

`bool`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:252`_

### cuvs::neighbors::ivf_flat::lists

Lists' data and indices.

```cpp
std::vector<std::shared_ptr<list_data<T, IdxT>>>& lists() noexcept;
```

**Returns**

`std::vector<std::shared_ptr<list_data<T, IdxT>>>&`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:257`_

## IVF-Flat index build

_Doxygen group: `ivf_flat_cpp_index_build`_

### cuvs::neighbors::ivf_flat::build

Build the index from the dataset for efficient search.

```cpp
auto build(raft::resources const& handle,
const cuvs::neighbors::ivf_flat::index_params& index_params,
raft::device_matrix_view<const float, int64_t, raft::row_major> dataset)
-> cuvs::neighbors::ivf_flat::index<float, int64_t>;
```

NB: Currently, the following distance metrics are supported: - L2Expanded - L2Unexpanded - InnerProduct - CosineExpanded Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::ivf_flat::index_params&` | configure the index building |
| `dataset` | in | `raft::device_matrix_view<const float, int64_t, raft::row_major>` | a device pointer to a row-major matrix [n_rows, dim] |

**Returns**

`cuvs::neighbors::ivf_flat::index<float, int64_t>`

the constructed ivf-flat index

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:325`_

**Additional overload:** `cuvs::neighbors::ivf_flat::build`

Build the index from the dataset for efficient search.

```cpp
void build(raft::resources const& handle,
const cuvs::neighbors::ivf_flat::index_params& index_params,
raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
cuvs::neighbors::ivf_flat::index<float, int64_t>& idx);
```

NB: Currently, the following distance metrics are supported: - L2Expanded - L2Unexpanded - InnerProduct - CosineExpanded Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::ivf_flat::index_params&` | configure the index building |
| `dataset` | in | `raft::device_matrix_view<const float, int64_t, raft::row_major>` | raft::device_matrix_view to a row-major matrix [n_rows, dim] |
| `idx` | out | `cuvs::neighbors::ivf_flat::index<float, int64_t>&` | reference to ivf_flat::index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:355`_

**Additional overload:** `cuvs::neighbors::ivf_flat::build`

Build the index from the dataset for efficient search.

```cpp
auto build(raft::resources const& handle,
const cuvs::neighbors::ivf_flat::index_params& index_params,
raft::device_matrix_view<const half, int64_t, raft::row_major> dataset)
-> cuvs::neighbors::ivf_flat::index<half, int64_t>;
```

NB: Currently, the following distance metrics are supported: - L2Expanded - L2Unexpanded - InnerProduct - CosineExpanded Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::ivf_flat::index_params&` | configure the index building |
| `dataset` | in | `raft::device_matrix_view<const half, int64_t, raft::row_major>` | a device pointer to a row-major matrix [n_rows, dim] |

**Returns**

`cuvs::neighbors::ivf_flat::index<half, int64_t>`

the constructed ivf-flat index

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:384`_

**Additional overload:** `cuvs::neighbors::ivf_flat::build`

Build the index from the dataset for efficient search.

```cpp
void build(raft::resources const& handle,
const cuvs::neighbors::ivf_flat::index_params& index_params,
raft::device_matrix_view<const half, int64_t, raft::row_major> dataset,
cuvs::neighbors::ivf_flat::index<half, int64_t>& idx);
```

NB: Currently, the following distance metrics are supported: - L2Expanded - L2Unexpanded - InnerProduct - CosineExpanded Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::ivf_flat::index_params&` | configure the index building |
| `dataset` | in | `raft::device_matrix_view<const half, int64_t, raft::row_major>` | raft::device_matrix_view to a row-major matrix [n_rows, dim] |
| `idx` | out | `cuvs::neighbors::ivf_flat::index<half, int64_t>&` | reference to ivf_flat::index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:414`_

**Additional overload:** `cuvs::neighbors::ivf_flat::build`

Build the index from the dataset for efficient search.

```cpp
auto build(raft::resources const& handle,
const cuvs::neighbors::ivf_flat::index_params& index_params,
raft::device_matrix_view<const int8_t, int64_t, raft::row_major> dataset)
-> cuvs::neighbors::ivf_flat::index<int8_t, int64_t>;
```

NB: Currently, the following distance metrics are supported: - L2Expanded - L2Unexpanded - InnerProduct - CosineExpanded Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::ivf_flat::index_params&` | configure the index building |
| `dataset` | in | `raft::device_matrix_view<const int8_t, int64_t, raft::row_major>` | a device pointer to a row-major matrix [n_rows, dim] |

**Returns**

`cuvs::neighbors::ivf_flat::index<int8_t, int64_t>`

the constructed ivf-flat index

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:443`_

**Additional overload:** `cuvs::neighbors::ivf_flat::build`

Build the index from the dataset for efficient search.

```cpp
void build(raft::resources const& handle,
const cuvs::neighbors::ivf_flat::index_params& index_params,
raft::device_matrix_view<const int8_t, int64_t, raft::row_major> dataset,
cuvs::neighbors::ivf_flat::index<int8_t, int64_t>& idx);
```

NB: Currently, the following distance metrics are supported: - L2Expanded - L2Unexpanded - InnerProduct - CosineExpanded Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::ivf_flat::index_params&` | configure the index building |
| `dataset` | in | `raft::device_matrix_view<const int8_t, int64_t, raft::row_major>` | raft::device_matrix_view to a row-major matrix [n_rows, dim] |
| `idx` | out | `cuvs::neighbors::ivf_flat::index<int8_t, int64_t>&` | reference to ivf_flat::index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:473`_

**Additional overload:** `cuvs::neighbors::ivf_flat::build`

Build the index from the dataset for efficient search.

```cpp
auto build(raft::resources const& handle,
const cuvs::neighbors::ivf_flat::index_params& index_params,
raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> dataset)
-> cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>;
```

NB: Currently, the following distance metrics are supported: - L2Expanded - L2Unexpanded - InnerProduct - CosineExpanded Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::ivf_flat::index_params&` | configure the index building |
| `dataset` | in | `raft::device_matrix_view<const uint8_t, int64_t, raft::row_major>` | a device pointer to a row-major matrix [n_rows, dim] |

**Returns**

`cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>`

the constructed ivf-flat index

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:502`_

**Additional overload:** `cuvs::neighbors::ivf_flat::build`

Build the index from the dataset for efficient search.

```cpp
void build(raft::resources const& handle,
const cuvs::neighbors::ivf_flat::index_params& index_params,
raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> dataset,
cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>& idx);
```

NB: Currently, the following distance metrics are supported: - L2Expanded - L2Unexpanded - InnerProduct - CosineExpanded Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::ivf_flat::index_params&` | configure the index building |
| `dataset` | in | `raft::device_matrix_view<const uint8_t, int64_t, raft::row_major>` | raft::device_matrix_view to a row-major matrix [n_rows, dim] |
| `idx` | out | `cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>&` | reference to ivf_flat::index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:532`_

**Additional overload:** `cuvs::neighbors::ivf_flat::build`

Build the index from the dataset for efficient search.

```cpp
auto build(raft::resources const& handle,
const cuvs::neighbors::ivf_flat::index_params& index_params,
raft::host_matrix_view<const float, int64_t, raft::row_major> dataset)
-> cuvs::neighbors::ivf_flat::index<float, int64_t>;
```

NB: Currently, the following distance metrics are supported: - L2Expanded - L2Unexpanded - InnerProduct - CosineExpanded Note, if index_params.add_data_on_build is set to true, the user can set a stream pool in the input raft::resource with at least one stream to enable kernel and copy overlapping. Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::ivf_flat::index_params&` | configure the index building |
| `dataset` | in | `raft::host_matrix_view<const float, int64_t, raft::row_major>` | raft::host_matrix_view to a row-major matrix [n_rows, dim] |

**Returns**

`cuvs::neighbors::ivf_flat::index<float, int64_t>`

the constructed ivf-flat index

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:568`_

**Additional overload:** `cuvs::neighbors::ivf_flat::build`

Build the index from the dataset for efficient search.

```cpp
void build(raft::resources const& handle,
const cuvs::neighbors::ivf_flat::index_params& index_params,
raft::host_matrix_view<const float, int64_t, raft::row_major> dataset,
cuvs::neighbors::ivf_flat::index<float, int64_t>& idx);
```

NB: Currently, the following distance metrics are supported: - L2Expanded - L2Unexpanded - InnerProduct - CosineExpanded Note, if index_params.add_data_on_build is set to true, the user can set a stream pool in the input raft::resource with at least one stream to enable kernel and copy overlapping. Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::ivf_flat::index_params&` | configure the index building |
| `dataset` | in | `raft::host_matrix_view<const float, int64_t, raft::row_major>` | raft::host_matrix_view to a row-major matrix [n_rows, dim] |
| `idx` | out | `cuvs::neighbors::ivf_flat::index<float, int64_t>&` | reference to ivf_flat::index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:605`_

**Additional overload:** `cuvs::neighbors::ivf_flat::build`

Build the index from the dataset for efficient search.

```cpp
auto build(raft::resources const& handle,
const cuvs::neighbors::ivf_flat::index_params& index_params,
raft::host_matrix_view<const half, int64_t, raft::row_major> dataset)
-> cuvs::neighbors::ivf_flat::index<half, int64_t>;
```

NB: Currently, the following distance metrics are supported: - L2Expanded - L2Unexpanded - InnerProduct - CosineExpanded Note, if index_params.add_data_on_build is set to true, the user can set a stream pool in the input raft::resource with at least one stream to enable kernel and copy overlapping. Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::ivf_flat::index_params&` | configure the index building |
| `dataset` | in | `raft::host_matrix_view<const half, int64_t, raft::row_major>` | raft::host_matrix_view to a row-major matrix [n_rows, dim] |

**Returns**

`cuvs::neighbors::ivf_flat::index<half, int64_t>`

the constructed ivf-flat index

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:641`_

**Additional overload:** `cuvs::neighbors::ivf_flat::build`

Build the index from the dataset for efficient search.

```cpp
void build(raft::resources const& handle,
const cuvs::neighbors::ivf_flat::index_params& index_params,
raft::host_matrix_view<const half, int64_t, raft::row_major> dataset,
cuvs::neighbors::ivf_flat::index<half, int64_t>& idx);
```

NB: Currently, the following distance metrics are supported: - L2Expanded - L2Unexpanded - InnerProduct - CosineExpanded Note, if index_params.add_data_on_build is set to true, the user can set a stream pool in the input raft::resource with at least one stream to enable kernel and copy overlapping. Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::ivf_flat::index_params&` | configure the index building |
| `dataset` | in | `raft::host_matrix_view<const half, int64_t, raft::row_major>` | raft::host_matrix_view to a row-major matrix [n_rows, dim] |
| `idx` | out | `cuvs::neighbors::ivf_flat::index<half, int64_t>&` | reference to ivf_flat::index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:678`_

**Additional overload:** `cuvs::neighbors::ivf_flat::build`

Build the index from the dataset for efficient search.

```cpp
auto build(raft::resources const& handle,
const cuvs::neighbors::ivf_flat::index_params& index_params,
raft::host_matrix_view<const int8_t, int64_t, raft::row_major> dataset)
-> cuvs::neighbors::ivf_flat::index<int8_t, int64_t>;
```

NB: Currently, the following distance metrics are supported: - L2Expanded - L2Unexpanded - InnerProduct - CosineExpanded Note, if index_params.add_data_on_build is set to true, the user can set a stream pool in the input raft::resource with at least one stream to enable kernel and copy overlapping. Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::ivf_flat::index_params&` | configure the index building |
| `dataset` | in | `raft::host_matrix_view<const int8_t, int64_t, raft::row_major>` | a host pointer to a row-major matrix [n_rows, dim] |

**Returns**

`cuvs::neighbors::ivf_flat::index<int8_t, int64_t>`

the constructed ivf-flat index

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:714`_

**Additional overload:** `cuvs::neighbors::ivf_flat::build`

Build the index from the dataset for efficient search.

```cpp
void build(raft::resources const& handle,
const cuvs::neighbors::ivf_flat::index_params& index_params,
raft::host_matrix_view<const int8_t, int64_t, raft::row_major> dataset,
cuvs::neighbors::ivf_flat::index<int8_t, int64_t>& idx);
```

NB: Currently, the following distance metrics are supported: - L2Expanded - L2Unexpanded - InnerProduct - CosineExpanded Note, if index_params.add_data_on_build is set to true, the user can set a stream pool in the input raft::resource with at least one stream to enable kernel and copy overlapping. Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::ivf_flat::index_params&` | configure the index building |
| `dataset` | in | `raft::host_matrix_view<const int8_t, int64_t, raft::row_major>` | raft::host_matrix_view to a row-major matrix [n_rows, dim] |
| `idx` | out | `cuvs::neighbors::ivf_flat::index<int8_t, int64_t>&` | reference to ivf_flat::index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:751`_

**Additional overload:** `cuvs::neighbors::ivf_flat::build`

Build the index from the dataset for efficient search.

```cpp
auto build(raft::resources const& handle,
const cuvs::neighbors::ivf_flat::index_params& index_params,
raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> dataset)
-> cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>;
```

NB: Currently, the following distance metrics are supported: - L2Expanded - L2Unexpanded - InnerProduct - CosineExpanded Note, if index_params.add_data_on_build is set to true, the user can set a stream pool in the input raft::resource with at least one stream to enable kernel and copy overlapping. Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::ivf_flat::index_params&` | configure the index building |
| `dataset` | in | `raft::host_matrix_view<const uint8_t, int64_t, raft::row_major>` | a host pointer to a row-major matrix [n_rows, dim] |

**Returns**

`cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>`

the constructed ivf-flat index

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:787`_

**Additional overload:** `cuvs::neighbors::ivf_flat::build`

Build the index from the dataset for efficient search.

```cpp
void build(raft::resources const& handle,
const cuvs::neighbors::ivf_flat::index_params& index_params,
raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> dataset,
cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>& idx);
```

NB: Currently, the following distance metrics are supported: - L2Expanded - L2Unexpanded - InnerProduct - CosineExpanded Note, if index_params.add_data_on_build is set to true, the user can set a stream pool in the input raft::resource with at least one stream to enable kernel and copy overlapping. Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::ivf_flat::index_params&` | configure the index building |
| `dataset` | in | `raft::host_matrix_view<const uint8_t, int64_t, raft::row_major>` | raft::host_matrix_view to a row-major matrix [n_rows, dim] |
| `idx` | out | `cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>&` | reference to ivf_flat::index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:824`_

## IVF-Flat index extend

_Doxygen group: `ivf_flat_cpp_index_extend`_

### cuvs::neighbors::ivf_flat::extend

Build a new index containing the data of the original plus new extra vectors.

```cpp
auto extend(raft::resources const& handle,
raft::device_matrix_view<const float, int64_t, raft::row_major> new_vectors,
std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
const cuvs::neighbors::ivf_flat::index<float, int64_t>& idx)
-> cuvs::neighbors::ivf_flat::index<float, int64_t>;
```

Implementation note: The new data is clustered according to existing kmeans clusters, then the cluster centers are adjusted to match the newly labeled data. Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `new_vectors` | in | `raft::device_matrix_view<const float, int64_t, raft::row_major>` | raft::device_matrix_view to a row-major matrix [n_rows, index.dim()] |
| `new_indices` | in | `std::optional<raft::device_vector_view<const int64_t, int64_t>>` | optional raft::device_vector_view to a vector of indices [n_rows]. If the original index is empty (`orig_index.size() == 0`), you can pass `std::nullopt` here to imply a continuous range `[0...n_rows)`. |
| `idx` | in | `const cuvs::neighbors::ivf_flat::index<float, int64_t>&` | original index |

**Returns**

`cuvs::neighbors::ivf_flat::index<float, int64_t>`

the constructed extended ivf-flat index

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:866`_

**Additional overload:** `cuvs::neighbors::ivf_flat::extend`

Extend the index in-place with the new data.

```cpp
void extend(raft::resources const& handle,
raft::device_matrix_view<const float, int64_t, raft::row_major> new_vectors,
std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
cuvs::neighbors::ivf_flat::index<float, int64_t>* idx);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `new_vectors` | in | `raft::device_matrix_view<const float, int64_t, raft::row_major>` | raft::device_matrix_view to a row-major matrix [n_rows, index.dim()] |
| `new_indices` | in | `std::optional<raft::device_vector_view<const int64_t, int64_t>>` | optional raft::device_vector_view to a vector of indices [n_rows]. If the original index is empty (`orig_index.size() == 0`), you can pass `std::nullopt` here to imply a continuous range `[0...n_rows)`. |
| `idx` | inout | `cuvs::neighbors::ivf_flat::index<float, int64_t>*` | pointer to index, to be overwritten in-place |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:896`_

**Additional overload:** `cuvs::neighbors::ivf_flat::extend`

Build a new index containing the data of the original plus new extra vectors.

```cpp
auto extend(raft::resources const& handle,
raft::device_matrix_view<const half, int64_t, raft::row_major> new_vectors,
std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
const cuvs::neighbors::ivf_flat::index<half, int64_t>& idx)
-> cuvs::neighbors::ivf_flat::index<half, int64_t>;
```

Implementation note: The new data is clustered according to existing kmeans clusters, then the cluster centers are adjusted to match the newly labeled data. Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `new_vectors` | in | `raft::device_matrix_view<const half, int64_t, raft::row_major>` | raft::device_matrix_view to a row-major matrix [n_rows, index.dim()] |
| `new_indices` | in | `std::optional<raft::device_vector_view<const int64_t, int64_t>>` | optional raft::device_vector_view to a vector of indices [n_rows]. If the original index is empty (`orig_index.size() == 0`), you can pass `std::nullopt` here to imply a continuous range `[0...n_rows)`. |
| `idx` | in | `const cuvs::neighbors::ivf_flat::index<half, int64_t>&` | original index |

**Returns**

`cuvs::neighbors::ivf_flat::index<half, int64_t>`

the constructed extended ivf-flat index

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:930`_

**Additional overload:** `cuvs::neighbors::ivf_flat::extend`

Extend the index in-place with the new data.

```cpp
void extend(raft::resources const& handle,
raft::device_matrix_view<const half, int64_t, raft::row_major> new_vectors,
std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
cuvs::neighbors::ivf_flat::index<half, int64_t>* idx);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `new_vectors` | in | `raft::device_matrix_view<const half, int64_t, raft::row_major>` | raft::device_matrix_view to a row-major matrix [n_rows, index.dim()] |
| `new_indices` | in | `std::optional<raft::device_vector_view<const int64_t, int64_t>>` | optional raft::device_vector_view to a vector of indices [n_rows]. If the original index is empty (`orig_index.size() == 0`), you can pass `std::nullopt` here to imply a continuous range `[0...n_rows)`. |
| `idx` | inout | `cuvs::neighbors::ivf_flat::index<half, int64_t>*` | pointer to index, to be overwritten in-place |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:960`_

**Additional overload:** `cuvs::neighbors::ivf_flat::extend`

Build a new index containing the data of the original plus new extra vectors.

```cpp
auto extend(raft::resources const& handle,
raft::device_matrix_view<const int8_t, int64_t, raft::row_major> new_vectors,
std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
const cuvs::neighbors::ivf_flat::index<int8_t, int64_t>& idx)
-> cuvs::neighbors::ivf_flat::index<int8_t, int64_t>;
```

Implementation note: The new data is clustered according to existing kmeans clusters, then the cluster centers are adjusted to match the newly labeled data. Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `new_vectors` | in | `raft::device_matrix_view<const int8_t, int64_t, raft::row_major>` | raft::device_matrix_view to a row-major matrix [n_rows, index.dim()] |
| `new_indices` | in | `std::optional<raft::device_vector_view<const int64_t, int64_t>>` | optional raft::device_vector_view to a vector of indices [n_rows]. If the original index is empty (`orig_index.size() == 0`), you can pass `std::nullopt` here to imply a continuous range `[0...n_rows)`. |
| `idx` | in | `const cuvs::neighbors::ivf_flat::index<int8_t, int64_t>&` | original index |

**Returns**

`cuvs::neighbors::ivf_flat::index<int8_t, int64_t>`

the constructed extended ivf-flat index

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:994`_

**Additional overload:** `cuvs::neighbors::ivf_flat::extend`

Extend the index in-place with the new data.

```cpp
void extend(raft::resources const& handle,
raft::device_matrix_view<const int8_t, int64_t, raft::row_major> new_vectors,
std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
cuvs::neighbors::ivf_flat::index<int8_t, int64_t>* idx);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `new_vectors` | in | `raft::device_matrix_view<const int8_t, int64_t, raft::row_major>` | raft::device_matrix_view to a row-major matrix [n_rows, index.dim()] |
| `new_indices` | in | `std::optional<raft::device_vector_view<const int64_t, int64_t>>` | optional raft::device_vector_view to a vector of indices [n_rows]. If the original index is empty (`orig_index.size() == 0`), you can pass `std::nullopt` here to imply a continuous range `[0...n_rows)`. |
| `idx` | inout | `cuvs::neighbors::ivf_flat::index<int8_t, int64_t>*` | pointer to index, to be overwritten in-place |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:1025`_

**Additional overload:** `cuvs::neighbors::ivf_flat::extend`

Build a new index containing the data of the original plus new extra vectors.

```cpp
auto extend(raft::resources const& handle,
raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> new_vectors,
std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
const cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>& idx)
-> cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>;
```

Implementation note: The new data is clustered according to existing kmeans clusters, then the cluster centers are adjusted to match the newly labeled data. Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `new_vectors` | in | `raft::device_matrix_view<const uint8_t, int64_t, raft::row_major>` | raft::device_matrix_view to a row-major matrix [n_rows, index.dim()] |
| `new_indices` | in | `std::optional<raft::device_vector_view<const int64_t, int64_t>>` | optional raft::device_vector_view to a vector of indices [n_rows]. If the original index is empty (`orig_index.size() == 0`), you can pass `std::nullopt` here to imply a continuous range `[0...n_rows)`. |
| `idx` | in | `const cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>&` | original index |

**Returns**

`cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>`

the constructed extended ivf-flat index

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:1059`_

**Additional overload:** `cuvs::neighbors::ivf_flat::extend`

Extend the index in-place with the new data.

```cpp
void extend(raft::resources const& handle,
raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> new_vectors,
std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>* idx);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `new_vectors` | in | `raft::device_matrix_view<const uint8_t, int64_t, raft::row_major>` | raft::device_matrix_view to a row-major matrix [n_rows, index.dim()] |
| `new_indices` | in | `std::optional<raft::device_vector_view<const int64_t, int64_t>>` | optional raft::device_vector_view to a vector of indices [n_rows]. If the original index is empty (`orig_index.size() == 0`), you can pass `std::nullopt` here to imply a continuous range `[0...n_rows)`. |
| `idx` | inout | `cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>*` | pointer to index, to be overwritten in-place |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:1089`_

**Additional overload:** `cuvs::neighbors::ivf_flat::extend`

Build a new index containing the data of the original plus new extra vectors.

```cpp
auto extend(raft::resources const& handle,
raft::host_matrix_view<const float, int64_t, raft::row_major> new_vectors,
std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
const cuvs::neighbors::ivf_flat::index<float, int64_t>& idx)
-> cuvs::neighbors::ivf_flat::index<float, int64_t>;
```

Note, the user can set a stream pool in the input raft::resource with at least one stream to enable kernel and copy overlapping. Implementation note: The new data is clustered according to existing kmeans clusters, then the cluster centers are adjusted to match the newly labeled data. Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `new_vectors` | in | `raft::host_matrix_view<const float, int64_t, raft::row_major>` | raft::host_matrix_view to a row-major matrix [n_rows, index.dim()] |
| `new_indices` | in | `std::optional<raft::host_vector_view<const int64_t, int64_t>>` | optional raft::host_vector_view to a vector of indices [n_rows]. If the original index is empty (`orig_index.size() == 0`), you can pass `std::nullopt` here to imply a continuous range `[0...n_rows)`. |
| `idx` | in | `const cuvs::neighbors::ivf_flat::index<float, int64_t>&` | original index |

**Returns**

`cuvs::neighbors::ivf_flat::index<float, int64_t>`

the constructed extended ivf-flat index

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:1129`_

**Additional overload:** `cuvs::neighbors::ivf_flat::extend`

Extend the index in-place with the new data.

```cpp
void extend(raft::resources const& handle,
raft::host_matrix_view<const float, int64_t, raft::row_major> new_vectors,
std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
cuvs::neighbors::ivf_flat::index<float, int64_t>* idx);
```

Note, the user can set a stream pool in the input raft::resource with at least one stream to enable kernel and copy overlapping. Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `new_vectors` | in | `raft::host_matrix_view<const float, int64_t, raft::row_major>` | raft::host_matrix_view to a row-major matrix [n_rows, index.dim()] |
| `new_indices` | in | `std::optional<raft::host_vector_view<const int64_t, int64_t>>` | optional raft::host_vector_view to a vector of indices [n_rows]. If the original index is empty (`orig_index.size() == 0`), you can pass `std::nullopt` here to imply a continuous range `[0...n_rows)`. |
| `idx` | inout | `cuvs::neighbors::ivf_flat::index<float, int64_t>*` | pointer to index, to be overwritten in-place |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:1165`_

**Additional overload:** `cuvs::neighbors::ivf_flat::extend`

Build a new index containing the data of the original plus new extra vectors.

```cpp
auto extend(raft::resources const& handle,
raft::host_matrix_view<const half, int64_t, raft::row_major> new_vectors,
std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
const cuvs::neighbors::ivf_flat::index<half, int64_t>& idx)
-> cuvs::neighbors::ivf_flat::index<half, int64_t>;
```

Note, the user can set a stream pool in the input raft::resource with at least one stream to enable kernel and copy overlapping. Implementation note: The new data is clustered according to existing kmeans clusters, then the cluster centers are adjusted to match the newly labeled data. Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `new_vectors` | in | `raft::host_matrix_view<const half, int64_t, raft::row_major>` | raft::host_matrix_view to a row-major matrix [n_rows, index.dim()] |
| `new_indices` | in | `std::optional<raft::host_vector_view<const int64_t, int64_t>>` | optional raft::host_vector_view to a vector of indices [n_rows]. If the original index is empty (`orig_index.size() == 0`), you can pass `std::nullopt` here to imply a continuous range `[0...n_rows)`. |
| `idx` | in | `const cuvs::neighbors::ivf_flat::index<half, int64_t>&` | original index |

**Returns**

`cuvs::neighbors::ivf_flat::index<half, int64_t>`

the constructed extended ivf-flat index

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:1205`_

**Additional overload:** `cuvs::neighbors::ivf_flat::extend`

Extend the index in-place with the new data.

```cpp
void extend(raft::resources const& handle,
raft::host_matrix_view<const half, int64_t, raft::row_major> new_vectors,
std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
cuvs::neighbors::ivf_flat::index<half, int64_t>* idx);
```

Note, the user can set a stream pool in the input raft::resource with at least one stream to enable kernel and copy overlapping. Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `new_vectors` | in | `raft::host_matrix_view<const half, int64_t, raft::row_major>` | raft::host_matrix_view to a row-major matrix [n_rows, index.dim()] |
| `new_indices` | in | `std::optional<raft::host_vector_view<const int64_t, int64_t>>` | optional raft::host_vector_view to a vector of indices [n_rows]. If the original index is empty (`orig_index.size() == 0`), you can pass `std::nullopt` here to imply a continuous range `[0...n_rows)`. |
| `idx` | inout | `cuvs::neighbors::ivf_flat::index<half, int64_t>*` | pointer to index, to be overwritten in-place |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:1241`_

**Additional overload:** `cuvs::neighbors::ivf_flat::extend`

Build a new index containing the data of the original plus new extra vectors.

```cpp
auto extend(raft::resources const& handle,
raft::host_matrix_view<const int8_t, int64_t, raft::row_major> new_vectors,
std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
const cuvs::neighbors::ivf_flat::index<int8_t, int64_t>& idx)
-> cuvs::neighbors::ivf_flat::index<int8_t, int64_t>;
```

Note, the user can set a stream pool in the input raft::resource with at least one stream to enable kernel and copy overlapping. Implementation note: The new data is clustered according to existing kmeans clusters, then the cluster centers are adjusted to match the newly labeled data. Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `new_vectors` | in | `raft::host_matrix_view<const int8_t, int64_t, raft::row_major>` | raft::host_matrix_view to a row-major matrix [n_rows, index.dim()] |
| `new_indices` | in | `std::optional<raft::host_vector_view<const int64_t, int64_t>>` | optional raft::host_vector_view to a vector of indices [n_rows]. If the original index is empty (`orig_index.size() == 0`), you can pass `std::nullopt` here to imply a continuous range `[0...n_rows)`. |
| `idx` | in | `const cuvs::neighbors::ivf_flat::index<int8_t, int64_t>&` | original index |

**Returns**

`cuvs::neighbors::ivf_flat::index<int8_t, int64_t>`

the constructed extended ivf-flat index

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:1281`_

**Additional overload:** `cuvs::neighbors::ivf_flat::extend`

Extend the index in-place with the new data.

```cpp
void extend(raft::resources const& handle,
raft::host_matrix_view<const int8_t, int64_t, raft::row_major> new_vectors,
std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
cuvs::neighbors::ivf_flat::index<int8_t, int64_t>* idx);
```

Note, the user can set a stream pool in the input raft::resource with at least one stream to enable kernel and copy overlapping. Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `new_vectors` | in | `raft::host_matrix_view<const int8_t, int64_t, raft::row_major>` | raft::host_matrix_view to a row-major matrix [n_rows, index.dim()] |
| `new_indices` | in | `std::optional<raft::host_vector_view<const int64_t, int64_t>>` | optional raft::host_vector_view to a vector of indices [n_rows]. If the original index is empty (`orig_index.size() == 0`), you can pass `std::nullopt` here to imply a continuous range `[0...n_rows)`. |
| `idx` | inout | `cuvs::neighbors::ivf_flat::index<int8_t, int64_t>*` | pointer to index, to be overwritten in-place |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:1317`_

**Additional overload:** `cuvs::neighbors::ivf_flat::extend`

Build a new index containing the data of the original plus new extra vectors.

```cpp
auto extend(raft::resources const& handle,
raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> new_vectors,
std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
const cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>& idx)
-> cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>;
```

Note, the user can set a stream pool in the input raft::resource with at least one stream to enable kernel and copy overlapping. Implementation note: The new data is clustered according to existing kmeans clusters, then the cluster centers are adjusted to match the newly labeled data. Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `new_vectors` | in | `raft::host_matrix_view<const uint8_t, int64_t, raft::row_major>` | raft::host_matrix_view to a row-major matrix [n_rows, index.dim()] |
| `new_indices` | in | `std::optional<raft::host_vector_view<const int64_t, int64_t>>` | optional raft::host_vector_view to a vector of indices [n_rows]. If the original index is empty (`orig_index.size() == 0`), you can pass `std::nullopt` here to imply a continuous range `[0...n_rows)`. |
| `idx` | in | `const cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>&` | original index |

**Returns**

`cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>`

the constructed extended ivf-flat index

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:1357`_

**Additional overload:** `cuvs::neighbors::ivf_flat::extend`

Extend the index in-place with the new data.

```cpp
void extend(raft::resources const& handle,
raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> new_vectors,
std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>* idx);
```

Note, the user can set a stream pool in the input raft::resource with at least one stream to enable kernel and copy overlapping. Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `new_vectors` | in | `raft::host_matrix_view<const uint8_t, int64_t, raft::row_major>` | raft::host_matrix_view to a row-major matrix [n_rows, index.dim()] |
| `new_indices` | in | `std::optional<raft::host_vector_view<const int64_t, int64_t>>` | optional raft::host_vector_view to a vector of indices [n_rows]. If the original index is empty (`orig_index.size() == 0`), you can pass `std::nullopt` here to imply a continuous range `[0...n_rows)`. |
| `idx` | inout | `cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>*` | pointer to index, to be overwritten in-place |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:1393`_

## IVF-Flat index serialize

_Doxygen group: `ivf_flat_cpp_serialize`_

### cuvs::neighbors::ivf_flat::serialize

Save the index to file.

```cpp
void serialize(raft::resources const& handle,
const std::string& filename,
const cuvs::neighbors::ivf_flat::index<float, int64_t>& index);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `filename` | in | `const std::string&` | the file name for saving the index |
| `index` | in | `const cuvs::neighbors::ivf_flat::index<float, int64_t>&` | IVF-Flat index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:1612`_

### cuvs::neighbors::ivf_flat::deserialize

Load index from file.

```cpp
void deserialize(raft::resources const& handle,
const std::string& filename,
cuvs::neighbors::ivf_flat::index<float, int64_t>* index);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `filename` | in | `const std::string&` | the name of the file that stores the index |
| `index` | in | `cuvs::neighbors::ivf_flat::index<float, int64_t>*` | IVF-Flat index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:1640`_

**Additional overload:** `cuvs::neighbors::ivf_flat::serialize`

Write the index to an output stream

```cpp
void serialize(raft::resources const& handle,
std::ostream& os,
const cuvs::neighbors::ivf_flat::index<float, int64_t>& index);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `os` | in | `std::ostream&` | output stream |
| `index` | in | `const cuvs::neighbors::ivf_flat::index<float, int64_t>&` | IVF-Flat index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:1666`_

**Additional overload:** `cuvs::neighbors::ivf_flat::deserialize`

Load index from input stream

```cpp
void deserialize(raft::resources const& handle,
std::istream& is,
cuvs::neighbors::ivf_flat::index<float, int64_t>* index);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `is` | in | `std::istream&` | input stream |
| `index` | in | `cuvs::neighbors::ivf_flat::index<float, int64_t>*` | IVF-Flat index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:1694`_

**Additional overload:** `cuvs::neighbors::ivf_flat::serialize`

Save the index to file.

```cpp
void serialize(raft::resources const& handle,
const std::string& filename,
const cuvs::neighbors::ivf_flat::index<half, int64_t>& index);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `filename` | in | `const std::string&` | the file name for saving the index |
| `index` | in | `const cuvs::neighbors::ivf_flat::index<half, int64_t>&` | IVF-Flat index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:1720`_

**Additional overload:** `cuvs::neighbors::ivf_flat::deserialize`

Load index from file.

```cpp
void deserialize(raft::resources const& handle,
const std::string& filename,
cuvs::neighbors::ivf_flat::index<half, int64_t>* index);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `filename` | in | `const std::string&` | the name of the file that stores the index |
| `index` | in | `cuvs::neighbors::ivf_flat::index<half, int64_t>*` | IVF-Flat index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:1748`_

**Additional overload:** `cuvs::neighbors::ivf_flat::serialize`

Write the index to an output stream

```cpp
void serialize(raft::resources const& handle,
std::ostream& os,
const cuvs::neighbors::ivf_flat::index<half, int64_t>& index);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `os` | in | `std::ostream&` | output stream |
| `index` | in | `const cuvs::neighbors::ivf_flat::index<half, int64_t>&` | IVF-Flat index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:1774`_

**Additional overload:** `cuvs::neighbors::ivf_flat::deserialize`

Load index from input stream

```cpp
void deserialize(raft::resources const& handle,
std::istream& is,
cuvs::neighbors::ivf_flat::index<half, int64_t>* index);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `is` | in | `std::istream&` | input stream |
| `index` | in | `cuvs::neighbors::ivf_flat::index<half, int64_t>*` | IVF-Flat index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:1802`_

**Additional overload:** `cuvs::neighbors::ivf_flat::serialize`

Save the index to file.

```cpp
void serialize(raft::resources const& handle,
const std::string& filename,
const cuvs::neighbors::ivf_flat::index<int8_t, int64_t>& index);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `filename` | in | `const std::string&` | the file name for saving the index |
| `index` | in | `const cuvs::neighbors::ivf_flat::index<int8_t, int64_t>&` | IVF-Flat index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:1828`_

**Additional overload:** `cuvs::neighbors::ivf_flat::deserialize`

Load index from file.

```cpp
void deserialize(raft::resources const& handle,
const std::string& filename,
cuvs::neighbors::ivf_flat::index<int8_t, int64_t>* index);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `filename` | in | `const std::string&` | the name of the file that stores the index |
| `index` | in | `cuvs::neighbors::ivf_flat::index<int8_t, int64_t>*` | IVF-Flat index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:1856`_

**Additional overload:** `cuvs::neighbors::ivf_flat::serialize`

Write the index to an output stream

```cpp
void serialize(raft::resources const& handle,
std::ostream& os,
const cuvs::neighbors::ivf_flat::index<int8_t, int64_t>& index);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `os` | in | `std::ostream&` | output stream |
| `index` | in | `const cuvs::neighbors::ivf_flat::index<int8_t, int64_t>&` | IVF-Flat index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:1882`_

**Additional overload:** `cuvs::neighbors::ivf_flat::deserialize`

Load index from input stream

```cpp
void deserialize(raft::resources const& handle,
std::istream& is,
cuvs::neighbors::ivf_flat::index<int8_t, int64_t>* index);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `is` | in | `std::istream&` | input stream |
| `index` | in | `cuvs::neighbors::ivf_flat::index<int8_t, int64_t>*` | IVF-Flat index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:1910`_

**Additional overload:** `cuvs::neighbors::ivf_flat::serialize`

Save the index to file.

```cpp
void serialize(raft::resources const& handle,
const std::string& filename,
const cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>& index);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `filename` | in | `const std::string&` | the file name for saving the index |
| `index` | in | `const cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>&` | IVF-Flat index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:1936`_

**Additional overload:** `cuvs::neighbors::ivf_flat::deserialize`

Load index from file.

```cpp
void deserialize(raft::resources const& handle,
const std::string& filename,
cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>* index);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `filename` | in | `const std::string&` | the name of the file that stores the index |
| `index` | in | `cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>*` | IVF-Flat index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:1964`_

**Additional overload:** `cuvs::neighbors::ivf_flat::serialize`

Write the index to an output stream

```cpp
void serialize(raft::resources const& handle,
std::ostream& os,
const cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>& index);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `os` | in | `std::ostream&` | output stream |
| `index` | in | `const cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>&` | IVF-Flat index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:1990`_

**Additional overload:** `cuvs::neighbors::ivf_flat::deserialize`

Load index from input stream

```cpp
void deserialize(raft::resources const& handle,
std::istream& is,
cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>* index);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `is` | in | `std::istream&` | input stream |
| `index` | in | `cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>*` | IVF-Flat index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:2018`_

## Helper functions for IVF Flat

_Doxygen group: `ivf_flat_helpers`_

### namespace codepacker \{

Helper functions for IVF Flat

```cpp
namespace codepacker {
```

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:2484`_

### codepacker::pack

Write flat codes into an existing list by the given offset.

```cpp
void pack(raft::resources const& res,
raft::device_matrix_view<const float, uint32_t, raft::row_major> codes,
uint32_t veclen,
uint32_t offset,
raft::device_mdspan<float,
typename list_spec<uint32_t, float, int64_t>::list_extents,
raft::row_major> list_data);
```

NB: no memory allocation happens here; the list must fit the data (offset + n_vec). Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` |  |
| `codes` | in | `raft::device_matrix_view<const float, uint32_t, raft::row_major>` | flat codes [n_vec, dim] |
| `veclen` | in | `uint32_t` | size of interleaved data chunks |
| `offset` | in | `uint32_t` | how many records to skip before writing the data into the list |
| `list_data` | inout | `raft::device_mdspan<float, typename list_spec<uint32_t, float, int64_t>::list_extents, raft::row_major>` | block to write into |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:2508`_

**Additional overload:** `codepacker::pack`

Write flat codes into an existing list by the given offset.

```cpp
void pack(raft::resources const& res,
raft::device_matrix_view<const half, uint32_t, raft::row_major> codes,
uint32_t veclen,
uint32_t offset,
raft::device_mdspan<half,
typename list_spec<uint32_t, half, int64_t>::list_extents,
raft::row_major> list_data);
```

NB: no memory allocation happens here; the list must fit the data (offset + n_vec). Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` |  |
| `codes` | in | `raft::device_matrix_view<const half, uint32_t, raft::row_major>` | flat codes [n_vec, dim] |
| `veclen` | in | `uint32_t` | size of interleaved data chunks |
| `offset` | in | `uint32_t` | how many records to skip before writing the data into the list |
| `list_data` | inout | `raft::device_mdspan<half, typename list_spec<uint32_t, half, int64_t>::list_extents, raft::row_major>` | block to write into |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:2538`_

**Additional overload:** `codepacker::pack`

Write flat codes into an existing list by the given offset.

```cpp
void pack(raft::resources const& res,
raft::device_matrix_view<const int8_t, uint32_t, raft::row_major> codes,
uint32_t veclen,
uint32_t offset,
raft::device_mdspan<int8_t,
typename list_spec<uint32_t, int8_t, int64_t>::list_extents,
raft::row_major> list_data);
```

NB: no memory allocation happens here; the list must fit the data (offset + n_vec). Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` |  |
| `codes` | in | `raft::device_matrix_view<const int8_t, uint32_t, raft::row_major>` | flat codes [n_vec, dim] |
| `veclen` | in | `uint32_t` | size of interleaved data chunks |
| `offset` | in | `uint32_t` | how many records to skip before writing the data into the list |
| `list_data` | inout | `raft::device_mdspan<int8_t, typename list_spec<uint32_t, int8_t, int64_t>::list_extents, raft::row_major>` | block to write into |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:2568`_

**Additional overload:** `codepacker::pack`

Write flat codes into an existing list by the given offset.

```cpp
void pack(raft::resources const& res,
raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major> codes,
uint32_t veclen,
uint32_t offset,
raft::device_mdspan<uint8_t,
typename list_spec<uint32_t, uint8_t, int64_t>::list_extents,
raft::row_major> list_data);
```

NB: no memory allocation happens here; the list must fit the data (offset + n_vec). Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` |  |
| `codes` | in | `raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major>` | flat codes [n_vec, dim] |
| `veclen` | in | `uint32_t` | size of interleaved data chunks |
| `offset` | in | `uint32_t` | how many records to skip before writing the data into the list |
| `list_data` | inout | `raft::device_mdspan<uint8_t, typename list_spec<uint32_t, uint8_t, int64_t>::list_extents, raft::row_major>` | block to write into |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:2598`_

### codepacker::unpack

Unpack `n_take` consecutive records of a single list (cluster) in the compressed index

```cpp
void unpack(raft::resources const& res,
raft::device_mdspan<const float,
typename list_spec<uint32_t, float, int64_t>::list_extents,
raft::row_major> list_data,
uint32_t veclen,
uint32_t offset,
raft::device_matrix_view<float, uint32_t, raft::row_major> codes);
```

starting at given `offset`. Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `list_data` | in | `raft::device_mdspan<const float, typename list_spec<uint32_t, float, int64_t>::list_extents, raft::row_major>` | block to read from |
| `veclen` | in | `uint32_t` | size of interleaved data chunks |
| `offset` | in | `uint32_t` | How many records in the list to skip. |
| `codes` | inout | `raft::device_matrix_view<float, uint32_t, raft::row_major>` | the destination buffer [n_take, index.dim()]. The length `n_take` defines how many records to unpack, it must be &lt;= the list size. |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:2631`_

**Additional overload:** `codepacker::unpack`

Unpack `n_take` consecutive records of a single list (cluster) in the compressed index

```cpp
void unpack(raft::resources const& res,
raft::device_mdspan<const half,
typename list_spec<uint32_t, half, int64_t>::list_extents,
raft::row_major> list_data,
uint32_t veclen,
uint32_t offset,
raft::device_matrix_view<half, uint32_t, raft::row_major> codes);
```

starting at given `offset`. Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `list_data` | in | `raft::device_mdspan<const half, typename list_spec<uint32_t, half, int64_t>::list_extents, raft::row_major>` | block to read from |
| `veclen` | in | `uint32_t` | size of interleaved data chunks |
| `offset` | in | `uint32_t` | How many records in the list to skip. |
| `codes` | inout | `raft::device_matrix_view<half, uint32_t, raft::row_major>` | the destination buffer [n_take, index.dim()]. The length `n_take` defines how many records to unpack, it must be &lt;= the list size. |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:2664`_

**Additional overload:** `codepacker::unpack`

Unpack `n_take` consecutive records of a single list (cluster) in the compressed index

```cpp
void unpack(raft::resources const& res,
raft::device_mdspan<const int8_t,
typename list_spec<uint32_t, int8_t, int64_t>::list_extents,
raft::row_major> list_data,
uint32_t veclen,
uint32_t offset,
raft::device_matrix_view<int8_t, uint32_t, raft::row_major> codes);
```

starting at given `offset`. Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `list_data` | in | `raft::device_mdspan<const int8_t, typename list_spec<uint32_t, int8_t, int64_t>::list_extents, raft::row_major>` | block to read from |
| `veclen` | in | `uint32_t` | size of interleaved data chunks |
| `offset` | in | `uint32_t` | How many records in the list to skip. |
| `codes` | inout | `raft::device_matrix_view<int8_t, uint32_t, raft::row_major>` | the destination buffer [n_take, index.dim()]. The length `n_take` defines how many records to unpack, it must be &lt;= the list size. |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:2696`_

**Additional overload:** `codepacker::unpack`

Unpack `n_take` consecutive records of a single list (cluster) in the compressed index

```cpp
void unpack(raft::resources const& res,
raft::device_mdspan<const uint8_t,
typename list_spec<uint32_t, uint8_t, int64_t>::list_extents,
raft::row_major> list_data,
uint32_t veclen,
uint32_t offset,
raft::device_matrix_view<uint8_t, uint32_t, raft::row_major> codes);
```

starting at given `offset`. Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `list_data` | in | `raft::device_mdspan<const uint8_t, typename list_spec<uint32_t, uint8_t, int64_t>::list_extents, raft::row_major>` | block to read from |
| `veclen` | in | `uint32_t` | size of interleaved data chunks |
| `offset` | in | `uint32_t` | How many records in the list to skip. |
| `codes` | inout | `raft::device_matrix_view<uint8_t, uint32_t, raft::row_major>` | the destination buffer [n_take, index.dim()]. The length `n_take` defines how many records to unpack, it must be &lt;= the list size. |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:2729`_

### codepacker::pack_1

Write one flat code into a block by the given offset. The offset indicates the id of the record

```cpp
void pack_1(const float* flat_code, float* block, uint32_t dim, uint32_t veclen, uint32_t offset);
```

in the list. This function interleaves the code and is intended to later copy the interleaved codes over to the IVF list on device. NB: no memory allocation happens here; the block must fit the record (offset + 1).

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `flat_code` | in | `const float*` | input flat code |
| `block` | out | `float*` | block of memory to write interleaved codes to |
| `dim` | in | `uint32_t` | dimension of the flat code |
| `veclen` | in | `uint32_t` | size of interleaved data chunks |
| `offset` | in | `uint32_t` | how many records to skip before writing the data into the list |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:2749`_

**Additional overload:** `codepacker::pack_1`

Write one flat code into a block by the given offset. The offset indicates the id of the record

```cpp
void pack_1(const half* flat_code, half* block, uint32_t dim, uint32_t veclen, uint32_t offset);
```

in the list. This function interleaves the code and is intended to later copy the interleaved codes over to the IVF list on device. NB: no memory allocation happens here; the block must fit the record (offset + 1).

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `flat_code` | in | `const half*` | input flat code |
| `block` | out | `half*` | block of memory to write interleaved codes to |
| `dim` | in | `uint32_t` | dimension of the flat code |
| `veclen` | in | `uint32_t` | size of interleaved data chunks |
| `offset` | in | `uint32_t` | how many records to skip before writing the data into the list |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:2763`_

**Additional overload:** `codepacker::pack_1`

Write one flat code into a block by the given offset. The offset indicates the id of the record

```cpp
void pack_1(const int8_t* flat_code, int8_t* block, uint32_t dim, uint32_t veclen, uint32_t offset);
```

in the list. This function interleaves the code and is intended to later copy the interleaved codes over to the IVF list on device. NB: no memory allocation happens here; the block must fit the record (offset + 1).

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `flat_code` | in | `const int8_t*` | input flat code |
| `block` | out | `int8_t*` | block of memory to write interleaved codes to |
| `dim` | in | `uint32_t` | dimension of the flat code |
| `veclen` | in | `uint32_t` | size of interleaved data chunks |
| `offset` | in | `uint32_t` | how many records to skip before writing the data into the list |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:2777`_

**Additional overload:** `codepacker::pack_1`

Write one flat code into a block by the given offset. The offset indicates the id of the record

```cpp
void pack_1(
const uint8_t* flat_code, uint8_t* block, uint32_t dim, uint32_t veclen, uint32_t offset);
```

in the list. This function interleaves the code and is intended to later copy the interleaved codes over to the IVF list on device. NB: no memory allocation happens here; the block must fit the record (offset + 1).

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `flat_code` | in | `const uint8_t*` | input flat code |
| `block` | out | `uint8_t*` | block of memory to write interleaved codes to |
| `dim` | in | `uint32_t` | dimension of the flat code |
| `veclen` | in | `uint32_t` | size of interleaved data chunks |
| `offset` | in | `uint32_t` | how many records to skip before writing the data into the list |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:2791`_

### codepacker::unpack_1

Unpack 1 record of a single list (cluster) in the index to fetch the flat code. The offset

```cpp
void unpack_1(const float* block, float* flat_code, uint32_t dim, uint32_t veclen, uint32_t offset);
```

indicates the id of the record. This function fetches one flat code from an interleaved code. interleaved format.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `block` | in | `const float*` | interleaved block. The block can be thought of as the whole inverted list in |
| `flat_code` | out | `float*` | output flat code |
| `dim` | in | `uint32_t` | dimension of the flat code |
| `veclen` | in | `uint32_t` | size of interleaved data chunks |
| `offset` | in | `uint32_t` | fetch the flat code by the given offset |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:2805`_

**Additional overload:** `codepacker::unpack_1`

Unpack 1 record of a single list (cluster) in the index to fetch the flat code. The offset

```cpp
void unpack_1(const half* block, half* flat_code, uint32_t dim, uint32_t veclen, uint32_t offset);
```

indicates the id of the record. This function fetches one flat code from an interleaved code. interleaved format.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `block` | in | `const half*` | interleaved block. The block can be thought of as the whole inverted list in |
| `flat_code` | out | `half*` | output flat code |
| `dim` | in | `uint32_t` | dimension of the flat code |
| `veclen` | in | `uint32_t` | size of interleaved data chunks |
| `offset` | in | `uint32_t` | fetch the flat code by the given offset |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:2818`_

**Additional overload:** `codepacker::unpack_1`

Unpack 1 record of a single list (cluster) in the index to fetch the flat code. The offset

```cpp
void unpack_1(
const int8_t* block, int8_t* flat_code, uint32_t dim, uint32_t veclen, uint32_t offset);
```

indicates the id of the record. This function fetches one flat code from an interleaved code. interleaved format.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `block` | in | `const int8_t*` | interleaved block. The block can be thought of as the whole inverted list in |
| `flat_code` | out | `int8_t*` | output flat code |
| `dim` | in | `uint32_t` | dimension of the flat code |
| `veclen` | in | `uint32_t` | size of interleaved data chunks |
| `offset` | in | `uint32_t` | fetch the flat code by the given offset |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:2831`_

**Additional overload:** `codepacker::unpack_1`

Unpack 1 record of a single list (cluster) in the index to fetch the flat code. The offset

```cpp
void unpack_1(
const uint8_t* block, uint8_t* flat_code, uint32_t dim, uint32_t veclen, uint32_t offset);
```

indicates the id of the record. This function fetches one flat code from an interleaved code. interleaved format.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `block` | in | `const uint8_t*` | interleaved block. The block can be thought of as the whole inverted list in |
| `flat_code` | out | `uint8_t*` | output flat code |
| `dim` | in | `uint32_t` | dimension of the flat code |
| `veclen` | in | `uint32_t` | size of interleaved data chunks |
| `offset` | in | `uint32_t` | fetch the flat code by the given offset |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:2845`_

### codepacker::reset_index

Public helper API to reset the data and indices ptrs, and the list sizes. Useful for

```cpp
void reset_index(const raft::resources& res, index<float, int64_t>* index);
```

externally modifying the index without going through the build stage. The data and indices of the IVF lists will be lost. Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `const raft::resources&` | raft resource |
| `index` | inout | `index<float, int64_t>*` | pointer to IVF-Flat index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:2870`_

**Additional overload:** `codepacker::reset_index`

Public helper API to reset the data and indices ptrs, and the list sizes. Useful for

```cpp
void reset_index(const raft::resources& res, index<half, int64_t>* index);
```

externally modifying the index without going through the build stage. The data and indices of the IVF lists will be lost. Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `const raft::resources&` | raft resource |
| `index` | inout | `index<half, int64_t>*` | pointer to IVF-Flat index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:2892`_

**Additional overload:** `codepacker::reset_index`

Public helper API to reset the data and indices ptrs, and the list sizes. Useful for

```cpp
void reset_index(const raft::resources& res, index<int8_t, int64_t>* index);
```

externally modifying the index without going through the build stage. The data and indices of the IVF lists will be lost. Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `const raft::resources&` | raft resource |
| `index` | inout | `index<int8_t, int64_t>*` | pointer to IVF-Flat index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:2914`_

**Additional overload:** `codepacker::reset_index`

Public helper API to reset the data and indices ptrs, and the list sizes. Useful for

```cpp
void reset_index(const raft::resources& res, index<uint8_t, int64_t>* index);
```

externally modifying the index without going through the build stage. The data and indices of the IVF lists will be lost. Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `const raft::resources&` | raft resource |
| `index` | inout | `index<uint8_t, int64_t>*` | pointer to IVF-Flat index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:2936`_

### codepacker::recompute_internal_state

Helper exposing the re-computation of list sizes and related arrays if IVF lists have been

```cpp
void recompute_internal_state(const raft::resources& res, index<float, int64_t>* index);
```

modified externally. Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `const raft::resources&` | raft resource |
| `index` | inout | `index<float, int64_t>*` | pointer to IVF-Flat index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:2964`_

**Additional overload:** `codepacker::recompute_internal_state`

Helper exposing the re-computation of list sizes and related arrays if IVF lists have been

```cpp
void recompute_internal_state(const raft::resources& res, index<half, int64_t>* index);
```

modified externally. Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `const raft::resources&` | raft resource |
| `index` | inout | `index<half, int64_t>*` | pointer to IVF-Flat index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:2992`_

**Additional overload:** `codepacker::recompute_internal_state`

Helper exposing the re-computation of list sizes and related arrays if IVF lists have been

```cpp
void recompute_internal_state(const raft::resources& res, index<int8_t, int64_t>* index);
```

modified externally. Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `const raft::resources&` | raft resource |
| `index` | inout | `index<int8_t, int64_t>*` | pointer to IVF-Flat index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:3020`_

**Additional overload:** `codepacker::recompute_internal_state`

Helper exposing the re-computation of list sizes and related arrays if IVF lists have been

```cpp
void recompute_internal_state(const raft::resources& res, index<uint8_t, int64_t>* index);
```

modified externally. Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `const raft::resources&` | raft resource |
| `index` | inout | `index<uint8_t, int64_t>*` | pointer to IVF-Flat index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_flat.hpp:3048`_
