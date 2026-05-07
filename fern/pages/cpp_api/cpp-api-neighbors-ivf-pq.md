---
slug: api-reference/cpp-api-neighbors-ivf-pq
---

# IVF PQ

_Source header: `cpp/include/cuvs/neighbors/ivf_pq.hpp`_

## IVF-PQ index build parameters

_Doxygen group: `ivf_pq_cpp_index_params`_

### cuvs::neighbors::ivf_pq::codebook_gen

A type for specifying how PQ codebooks are created.

```cpp
enum class codebook_gen { ... } ;
```

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:31`_

### cuvs::neighbors::ivf_pq::list_layout

A type for specifying the memory layout of PQ codes in IVF lists.

```cpp
enum class list_layout { ... } ;
```

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:37`_

### cuvs::neighbors::ivf_pq::from_dataset

Creates index_params based on shape of the input dataset.

```cpp
static index_params from_dataset(
raft::matrix_extent<int64_t> dataset,
cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `dataset` |  | `raft::matrix_extent<int64_t>` |  |
| `metric` |  | `cuvs::distance::DistanceType` | Default: `cuvs::distance::DistanceType::L2Expanded`. |

**Returns**

`static index_params`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:145`_

## IVF-PQ index search parameters

_Doxygen group: `ivf_pq_cpp_search_params`_

### cuvs::neighbors::ivf_pq::search_params

IVF-PQ index search parameters

```cpp
struct search_params : cuvs::neighbors::search_params { ... } ;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `n_probes` | `uint32_t` | The number of clusters to search. |
| `lut_dtype` | `cudaDataType_t` | Data type of look up table to be created dynamically at search time. |
| `internal_distance_dtype` | `cudaDataType_t` | Storage data type for distance/similarity computed at search time. |
| `preferred_shmem_carveout` | `double` | Preferred fraction of SM's unified memory / L1 cache to be used as shared memory. |
| `coarse_search_dtype` | `cudaDataType_t` | [Experimental] The data type to use as the GEMM element type when searching the clusters to |
| `max_internal_batch_size` | `uint32_t` | Set the internal batch size to improve GPU utilization at the cost of larger memory footprint. |

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:157`_

## IVF-PQ index

_Doxygen group: `ivf_pq_cpp_index`_

### cuvs::neighbors::ivf_pq::index

Construct an empty index.

```cpp
index(raft::resources const& handle);
```

Constructs an empty index. This index will either need to be trained with `build` or loaded from a saved copy with `deserialize`

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:493`_

**Additional overload:** `cuvs::neighbors::ivf_pq::index`

Construct an index with specified parameters.

```cpp
index(raft::resources const& handle,
cuvs::distance::DistanceType metric,
codebook_gen codebook_kind,
uint32_t n_lists,
uint32_t dim,
uint32_t pq_bits                    = 8,
uint32_t pq_dim                     = 0,
bool conservative_memory_allocation = false);
```

This constructor creates an owning index with the given parameters.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` | RAFT resources handle |
| `metric` |  | `cuvs::distance::DistanceType` | Distance metric for clustering |
| `codebook_kind` |  | `codebook_gen` | How PQ codebooks are created |
| `n_lists` |  | `uint32_t` | Number of inverted lists (clusters) |
| `dim` |  | `uint32_t` | Dimensionality of the input data |
| `pq_bits` |  | `uint32_t` | Bit length of vector elements after PQ compression Default: `8`. |
| `pq_dim` |  | `uint32_t` | Dimensionality after PQ compression (0 = auto-select) Default: `0`. |
| `conservative_memory_allocation` |  | `bool` | Memory allocation strategy Default: `false`. |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:509`_

**Additional overload:** `cuvs::neighbors::ivf_pq::index`

Construct an index from index parameters.

```cpp
index(raft::resources const& handle, const index_params& params, uint32_t dim);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` | RAFT resources handle |
| `params` |  | `const index_params&` | Index parameters |
| `dim` |  | `uint32_t` | Dimensionality of the input data |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:525`_

### cuvs::neighbors::ivf_pq::size

Total length of the index.

```cpp
IdxT size() const noexcept override;
```

**Returns**

`IdxT`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:528`_

### cuvs::neighbors::ivf_pq::dim

Dimensionality of the input data.

```cpp
uint32_t dim() const noexcept override;
```

**Returns**

`uint32_t`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:531`_

### cuvs::neighbors::ivf_pq::dim_ext

Dimensionality of the cluster centers:

```cpp
uint32_t dim_ext() const noexcept;
```

input data dim extended with vector norms and padded to 8 elems.

**Returns**

`uint32_t`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:537`_

### cuvs::neighbors::ivf_pq::rot_dim

Dimensionality of the data after transforming it for PQ processing

```cpp
uint32_t rot_dim() const noexcept;
```

(rotated and augmented to be muplitple of `pq_dim`).

**Returns**

`uint32_t`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:543`_

### cuvs::neighbors::ivf_pq::pq_bits

The bit length of an encoded vector element after compression by PQ.

```cpp
uint32_t pq_bits() const noexcept override;
```

**Returns**

`uint32_t`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:546`_

### cuvs::neighbors::ivf_pq::pq_dim

The dimensionality of an encoded vector after compression by PQ.

```cpp
uint32_t pq_dim() const noexcept override;
```

**Returns**

`uint32_t`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:549`_

### cuvs::neighbors::ivf_pq::pq_len

Dimensionality of a subspace, i.e. the number of vector components mapped to a subspace

```cpp
uint32_t pq_len() const noexcept;
```

**Returns**

`uint32_t`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:552`_

### cuvs::neighbors::ivf_pq::pq_book_size

The number of vectors in a PQ codebook (`1 &lt;&lt; pq_bits`).

```cpp
uint32_t pq_book_size() const noexcept;
```

**Returns**

`uint32_t`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:555`_

### cuvs::neighbors::ivf_pq::metric

Distance metric used for clustering.

```cpp
cuvs::distance::DistanceType metric() const noexcept override;
```

**Returns**

`cuvs::distance::DistanceType`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:558`_

### cuvs::neighbors::ivf_pq::codebook_kind

How PQ codebooks are created.

```cpp
codebook_gen codebook_kind() const noexcept override;
```

**Returns**

`codebook_gen`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:561`_

### cuvs::neighbors::ivf_pq::codes_layout

Memory layout of PQ codes in IVF lists.

```cpp
list_layout codes_layout() const noexcept override;
```

**Returns**

`list_layout`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:564`_

### cuvs::neighbors::ivf_pq::n_lists

Number of clusters/inverted lists (first level quantization).

```cpp
uint32_t n_lists() const noexcept;
```

**Returns**

`uint32_t`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:567`_

### cuvs::neighbors::ivf_pq::conservative_memory_allocation

Whether to use conservative memory allocation when extending the list (cluster) data

```cpp
bool conservative_memory_allocation() const noexcept override;
```

(see index_params.conservative_memory_allocation).

**Returns**

`bool`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:573`_

### cuvs::neighbors::ivf_pq::pq_centers

PQ cluster centers

```cpp
raft::device_mdspan<const float, pq_centers_extents, raft::row_major> pq_centers()
const noexcept override;
```

- codebook_gen::PER_SUBSPACE: [pq_dim , pq_len, pq_book_size]
- codebook_gen::PER_CLUSTER:  [n_lists, pq_len, pq_book_size]

**Returns**

`raft::device_mdspan<const float, pq_centers_extents, raft::row_major>`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:581`_

### cuvs::neighbors::ivf_pq::lists

Lists' data and indices (polymorphic, works for both FLAT and INTERLEAVED layouts).

```cpp
std::vector<std::shared_ptr<list_data_base<IdxT>>>& lists() noexcept override;
```

**Returns**

`std::vector<std::shared_ptr<list_data_base<IdxT>>>&`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:585`_

### cuvs::neighbors::ivf_pq::data_ptrs

Pointers to the inverted lists (clusters) data  [n_lists].

```cpp
raft::device_vector_view<uint8_t*, uint32_t, raft::row_major> data_ptrs() noexcept override;
```

**Returns**

`raft::device_vector_view<uint8_t*, uint32_t, raft::row_major>`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:589`_

### cuvs::neighbors::ivf_pq::inds_ptrs

Pointers to the inverted lists (clusters) indices  [n_lists].

```cpp
raft::device_vector_view<IdxT*, uint32_t, raft::row_major> inds_ptrs() noexcept override;
```

**Returns**

`raft::device_vector_view<IdxT*, uint32_t, raft::row_major>`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:594`_

### cuvs::neighbors::ivf_pq::rotation_matrix

The transform matrix (original space -&gt; rotated padded space) [rot_dim, dim]

```cpp
raft::device_matrix_view<const float, uint32_t, raft::row_major> rotation_matrix()
const noexcept override;
```

**Returns**

`raft::device_matrix_view<const float, uint32_t, raft::row_major>`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:599`_

### cuvs::neighbors::ivf_pq::accum_sorted_sizes

Accumulated list sizes, sorted in descending order [n_lists + 1].

```cpp
raft::host_vector_view<IdxT, uint32_t, raft::row_major> accum_sorted_sizes() noexcept override;
```

The last value contains the total length of the index. The value at index zero is always zero.

That is, the content of this span is as if the `list_sizes` was sorted and then accumulated.

This span is used during search to estimate the maximum size of the workspace.

**Returns**

`raft::host_vector_view<IdxT, uint32_t, raft::row_major>`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:616`_

### cuvs::neighbors::ivf_pq::list_sizes

Sizes of the lists [n_lists].

```cpp
raft::device_vector_view<uint32_t, uint32_t, raft::row_major> list_sizes() noexcept override;
```

**Returns**

`raft::device_vector_view<uint32_t, uint32_t, raft::row_major>`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:621`_

### cuvs::neighbors::ivf_pq::centers

Cluster centers corresponding to the lists in the original space [n_lists, dim_ext]

```cpp
raft::device_matrix_view<const float, uint32_t, raft::row_major> centers()
const noexcept override;
```

**Returns**

`raft::device_matrix_view<const float, uint32_t, raft::row_major>`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:626`_

### cuvs::neighbors::ivf_pq::centers_rot

Cluster centers corresponding to the lists in the rotated space [n_lists, rot_dim]

```cpp
raft::device_matrix_view<const float, uint32_t, raft::row_major> centers_rot()
const noexcept override;
```

**Returns**

`raft::device_matrix_view<const float, uint32_t, raft::row_major>`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:635`_

### cuvs::neighbors::ivf_pq::get_list_size_in_bytes

fetch size of a particular IVF list in bytes using the list extents.

```cpp
uint32_t get_list_size_in_bytes(uint32_t label) const override;
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `label` | in | `uint32_t` | list ID |

**Returns**

`uint32_t`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:654`_

**Additional overload:** `cuvs::neighbors::ivf_pq::index`

Construct index from implementation pointer.

```cpp
explicit index(std::unique_ptr<index_iface<IdxT>> impl);
```

This constructor is used internally by build/extend/deserialize functions.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `impl` |  | `std::unique_ptr<index_iface<IdxT>>` | Implementation pointer (owning or view) |

**Returns**

`explicit`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:663`_

## IVF-PQ index build

_Doxygen group: `ivf_pq_cpp_index_build`_

### cuvs::neighbors::ivf_pq::build

Build the index from the dataset for efficient search.

```cpp
auto build(raft::resources const& handle,
const cuvs::neighbors::ivf_pq::index_params& index_params,
raft::device_matrix_view<const float, int64_t, raft::row_major> dataset)
-> cuvs::neighbors::ivf_pq::index<int64_t>;
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::ivf_pq::index_params&` | configure the index building |
| `dataset` | in | `raft::device_matrix_view<const float, int64_t, raft::row_major>` | a device matrix view to a row-major matrix [n_rows, dim] |

**Returns**

`cuvs::neighbors::ivf_pq::index<int64_t>`

the constructed ivf-pq index

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:699`_

**Additional overload:** `cuvs::neighbors::ivf_pq::build`

Build the index from the dataset for efficient search.

```cpp
void build(raft::resources const& handle,
const cuvs::neighbors::ivf_pq::index_params& index_params,
raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
cuvs::neighbors::ivf_pq::index<int64_t>* idx);
```

NB: Currently, the following distance metrics are supported:

- L2Expanded
- L2Unexpanded
- InnerProduct
- CosineExpanded

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::ivf_pq::index_params&` | configure the index building |
| `dataset` | in | `raft::device_matrix_view<const float, int64_t, raft::row_major>` | raft::device_matrix_view to a row-major matrix [n_rows, dim] |
| `idx` | out | `cuvs::neighbors::ivf_pq::index<int64_t>*` | reference to ivf_pq::index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:729`_

**Additional overload:** `cuvs::neighbors::ivf_pq::build`

Build the index from the dataset for efficient search.

```cpp
auto build(raft::resources const& handle,
const cuvs::neighbors::ivf_pq::index_params& index_params,
raft::device_matrix_view<const half, int64_t, raft::row_major> dataset)
-> cuvs::neighbors::ivf_pq::index<int64_t>;
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::ivf_pq::index_params&` | configure the index building |
| `dataset` | in | `raft::device_matrix_view<const half, int64_t, raft::row_major>` | a device matrix view to a row-major matrix [n_rows, dim] |

**Returns**

`cuvs::neighbors::ivf_pq::index<int64_t>`

the constructed ivf-pq index

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:752`_

**Additional overload:** `cuvs::neighbors::ivf_pq::build`

Build the index from the dataset for efficient search.

```cpp
void build(raft::resources const& handle,
const cuvs::neighbors::ivf_pq::index_params& index_params,
raft::device_matrix_view<const half, int64_t, raft::row_major> dataset,
cuvs::neighbors::ivf_pq::index<int64_t>* idx);
```

NB: Currently, the following distance metrics are supported:

- L2Expanded
- L2Unexpanded
- InnerProduct
- CosineExpanded

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::ivf_pq::index_params&` | configure the index building |
| `dataset` | in | `raft::device_matrix_view<const half, int64_t, raft::row_major>` | raft::device_matrix_view to a row-major matrix [n_rows, dim] |
| `idx` | out | `cuvs::neighbors::ivf_pq::index<int64_t>*` | reference to ivf_pq::index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:782`_

**Additional overload:** `cuvs::neighbors::ivf_pq::build`

Build the index from the dataset for efficient search.

```cpp
auto build(raft::resources const& handle,
const cuvs::neighbors::ivf_pq::index_params& index_params,
raft::device_matrix_view<const int8_t, int64_t, raft::row_major> dataset)
-> cuvs::neighbors::ivf_pq::index<int64_t>;
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::ivf_pq::index_params&` | configure the index building |
| `dataset` | in | `raft::device_matrix_view<const int8_t, int64_t, raft::row_major>` | a device matrix view to a row-major matrix [n_rows, dim] |

**Returns**

`cuvs::neighbors::ivf_pq::index<int64_t>`

the constructed ivf-pq index

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:804`_

**Additional overload:** `cuvs::neighbors::ivf_pq::build`

Build the index from the dataset for efficient search.

```cpp
void build(raft::resources const& handle,
const cuvs::neighbors::ivf_pq::index_params& index_params,
raft::device_matrix_view<const int8_t, int64_t, raft::row_major> dataset,
cuvs::neighbors::ivf_pq::index<int64_t>* idx);
```

NB: Currently, the following distance metrics are supported:

- L2Expanded
- L2Unexpanded
- InnerProduct
- CosineExpanded

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::ivf_pq::index_params&` | configure the index building |
| `dataset` | in | `raft::device_matrix_view<const int8_t, int64_t, raft::row_major>` | raft::device_matrix_view to a row-major matrix [n_rows, dim] |
| `idx` | out | `cuvs::neighbors::ivf_pq::index<int64_t>*` | reference to ivf_pq::index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:834`_

**Additional overload:** `cuvs::neighbors::ivf_pq::build`

Build the index from the dataset for efficient search.

```cpp
auto build(raft::resources const& handle,
const cuvs::neighbors::ivf_pq::index_params& index_params,
raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> dataset)
-> cuvs::neighbors::ivf_pq::index<int64_t>;
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::ivf_pq::index_params&` | configure the index building |
| `dataset` | in | `raft::device_matrix_view<const uint8_t, int64_t, raft::row_major>` | a device matrix view to a row-major matrix [n_rows, dim] |

**Returns**

`cuvs::neighbors::ivf_pq::index<int64_t>`

the constructed ivf-pq index

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:857`_

**Additional overload:** `cuvs::neighbors::ivf_pq::build`

Build the index from the dataset for efficient search.

```cpp
void build(raft::resources const& handle,
const cuvs::neighbors::ivf_pq::index_params& index_params,
raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> dataset,
cuvs::neighbors::ivf_pq::index<int64_t>* idx);
```

NB: Currently, the following distance metrics are supported:

- L2Expanded
- L2Unexpanded
- InnerProduct
- CosineExpanded

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::ivf_pq::index_params&` | configure the index building |
| `dataset` | in | `raft::device_matrix_view<const uint8_t, int64_t, raft::row_major>` | raft::device_matrix_view to a row-major matrix [n_rows, dim] |
| `idx` | out | `cuvs::neighbors::ivf_pq::index<int64_t>*` | reference to ivf_pq::index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:887`_

**Additional overload:** `cuvs::neighbors::ivf_pq::build`

Build the index from the dataset for efficient search.

```cpp
auto build(raft::resources const& handle,
const cuvs::neighbors::ivf_pq::index_params& index_params,
raft::host_matrix_view<const float, int64_t, raft::row_major> dataset)
-> cuvs::neighbors::ivf_pq::index<int64_t>;
```

Note, if index_params.add_data_on_build is set to true, the user can set a stream pool in the input raft::resource with at least one stream to enable kernel and copy overlapping.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::ivf_pq::index_params&` | configure the index building |
| `dataset` | in | `raft::host_matrix_view<const float, int64_t, raft::row_major>` | a host_matrix_view to a row-major matrix [n_rows, dim] |

**Returns**

`cuvs::neighbors::ivf_pq::index<int64_t>`

the constructed ivf-pq index

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:916`_

**Additional overload:** `cuvs::neighbors::ivf_pq::build`

Build the index from the dataset for efficient search.

```cpp
void build(raft::resources const& handle,
const cuvs::neighbors::ivf_pq::index_params& index_params,
raft::host_matrix_view<const float, int64_t, raft::row_major> dataset,
cuvs::neighbors::ivf_pq::index<int64_t>* idx);
```

NB: Currently, the following distance metrics are supported:

- L2Expanded
- L2Unexpanded
- InnerProduct
- CosineExpanded

Note, if index_params.add_data_on_build is set to true, the user can set a stream pool in the input raft::resource with at least one stream to enable kernel and copy overlapping.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::ivf_pq::index_params&` | configure the index building |
| `dataset` | in | `raft::host_matrix_view<const float, int64_t, raft::row_major>` | raft::host_matrix_view to a row-major matrix [n_rows, dim] |
| `idx` | out | `cuvs::neighbors::ivf_pq::index<int64_t>*` | reference to ivf_pq::index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:953`_

**Additional overload:** `cuvs::neighbors::ivf_pq::build`

Build the index from the dataset for efficient search.

```cpp
auto build(raft::resources const& handle,
const cuvs::neighbors::ivf_pq::index_params& index_params,
raft::host_matrix_view<const half, int64_t, raft::row_major> dataset)
-> cuvs::neighbors::ivf_pq::index<int64_t>;
```

Note, if index_params.add_data_on_build is set to true, the user can set a stream pool in the input raft::resource with at least one stream to enable kernel and copy overlapping.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::ivf_pq::index_params&` | configure the index building |
| `dataset` | in | `raft::host_matrix_view<const half, int64_t, raft::row_major>` | a host_matrix_view to a row-major matrix [n_rows, dim] |

**Returns**

`cuvs::neighbors::ivf_pq::index<int64_t>`

the constructed ivf-pq index

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:983`_

**Additional overload:** `cuvs::neighbors::ivf_pq::build`

Build the index from the dataset for efficient search.

```cpp
void build(raft::resources const& handle,
const cuvs::neighbors::ivf_pq::index_params& index_params,
raft::host_matrix_view<const half, int64_t, raft::row_major> dataset,
cuvs::neighbors::ivf_pq::index<int64_t>* idx);
```

NB: Currently, the following distance metrics are supported:

- L2Expanded
- L2Unexpanded
- InnerProduct
- CosineExpanded

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::ivf_pq::index_params&` | configure the index building |
| `dataset` | in | `raft::host_matrix_view<const half, int64_t, raft::row_major>` | raft::host_matrix_view to a row-major matrix [n_rows, dim] |
| `idx` | out | `cuvs::neighbors::ivf_pq::index<int64_t>*` | reference to ivf_pq::index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:1013`_

**Additional overload:** `cuvs::neighbors::ivf_pq::build`

Build the index from the dataset for efficient search.

```cpp
auto build(raft::resources const& handle,
const cuvs::neighbors::ivf_pq::index_params& index_params,
raft::host_matrix_view<const int8_t, int64_t, raft::row_major> dataset)
-> cuvs::neighbors::ivf_pq::index<int64_t>;
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::ivf_pq::index_params&` | configure the index building |
| `dataset` | in | `raft::host_matrix_view<const int8_t, int64_t, raft::row_major>` | a host_matrix_view to a row-major matrix [n_rows, dim] |

**Returns**

`cuvs::neighbors::ivf_pq::index<int64_t>`

the constructed ivf-pq index

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:1036`_

**Additional overload:** `cuvs::neighbors::ivf_pq::build`

Build the index from the dataset for efficient search.

```cpp
void build(raft::resources const& handle,
const cuvs::neighbors::ivf_pq::index_params& index_params,
raft::host_matrix_view<const int8_t, int64_t, raft::row_major> dataset,
cuvs::neighbors::ivf_pq::index<int64_t>* idx);
```

NB: Currently, the following distance metrics are supported:

- L2Expanded
- L2Unexpanded
- InnerProduct
- CosineExpanded

Note, if index_params.add_data_on_build is set to true, the user can set a stream pool in the input raft::resource with at least one stream to enable kernel and copy overlapping.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::ivf_pq::index_params&` | configure the index building |
| `dataset` | in | `raft::host_matrix_view<const int8_t, int64_t, raft::row_major>` | raft::host_matrix_view to a row-major matrix [n_rows, dim] |
| `idx` | out | `cuvs::neighbors::ivf_pq::index<int64_t>*` | reference to ivf_pq::index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:1073`_

**Additional overload:** `cuvs::neighbors::ivf_pq::build`

Build the index from the dataset for efficient search.

```cpp
auto build(raft::resources const& handle,
const cuvs::neighbors::ivf_pq::index_params& index_params,
raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> dataset)
-> cuvs::neighbors::ivf_pq::index<int64_t>;
```

Note, if index_params.add_data_on_build is set to true, the user can set a stream pool in the input raft::resource with at least one stream to enable kernel and copy overlapping.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::ivf_pq::index_params&` | configure the index building |
| `dataset` | in | `raft::host_matrix_view<const uint8_t, int64_t, raft::row_major>` | a host_matrix_view to a row-major matrix [n_rows, dim] |

**Returns**

`cuvs::neighbors::ivf_pq::index<int64_t>`

the constructed ivf-pq index

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:1103`_

**Additional overload:** `cuvs::neighbors::ivf_pq::build`

Build the index from the dataset for efficient search.

```cpp
void build(raft::resources const& handle,
const cuvs::neighbors::ivf_pq::index_params& index_params,
raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> dataset,
cuvs::neighbors::ivf_pq::index<int64_t>* idx);
```

NB: Currently, the following distance metrics are supported:

- L2Expanded
- L2Unexpanded
- InnerProduct
- CosineExpanded

Note, if index_params.add_data_on_build is set to true, the user can set a stream pool in the input raft::resource with at least one stream to enable kernel and copy overlapping.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::ivf_pq::index_params&` | configure the index building |
| `dataset` | in | `raft::host_matrix_view<const uint8_t, int64_t, raft::row_major>` | raft::host_matrix_view to a row-major matrix [n_rows, dim] |
| `idx` | out | `cuvs::neighbors::ivf_pq::index<int64_t>*` | reference to ivf_pq::index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:1140`_

**Additional overload:** `cuvs::neighbors::ivf_pq::build`

Build a view-type IVF-PQ index from device memory centroids and codebook.

```cpp
auto build(raft::resources const& handle,
const cuvs::neighbors::ivf_pq::index_params& index_params,
const uint32_t dim,
raft::device_mdspan<const float, raft::extent_3d<uint32_t>, raft::row_major> pq_centers,
raft::device_matrix_view<const float, uint32_t, raft::row_major> centers,
raft::device_matrix_view<const float, uint32_t, raft::row_major> centers_rot,
raft::device_matrix_view<const float, uint32_t, raft::row_major> rotation_matrix)
-> cuvs::neighbors::ivf_pq::index<int64_t>;
```

This function creates a non-owning index that stores a reference to the provided device data. All parameters must be provided with correct extents. The caller is responsible for ensuring the lifetime of the input data exceeds the lifetime of the returned index.

The index_params must be consistent with the provided matrices. Specifically:

- index_params.codebook_kind determines the expected shape of pq_centers
- index_params.metric will be stored in the index
- index_params.conservative_memory_allocation will be stored in the index The function will verify consistency between index_params, dim, and the matrix extents.

dim]

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | raft resources handle |
| `index_params` | in | `const cuvs::neighbors::ivf_pq::index_params&` | configure the index (metric, codebook_kind, etc.). Must be consistent with the provided matrices. |
| `dim` | in | `const uint32_t` | dimensionality of the input data |
| `pq_centers` | in | `raft::device_mdspan<const float, raft::extent_3d<uint32_t>, raft::row_major>` | PQ codebook on device memory with required extents:<br />- codebook_gen::PER_SUBSPACE: [pq_dim, pq_len, pq_book_size]<br />- codebook_gen::PER_CLUSTER:  [n_lists, pq_len, pq_book_size] |
| `centers` | in | `raft::device_matrix_view<const float, uint32_t, raft::row_major>` | Cluster centers in the original space [n_lists, dim_ext] where dim_ext = round_up(dim + 1, 8) |
| `centers_rot` | in | `raft::device_matrix_view<const float, uint32_t, raft::row_major>` | Rotated cluster centers [n_lists, rot_dim] where rot_dim = pq_len * pq_dim |
| `rotation_matrix` | in | `raft::device_matrix_view<const float, uint32_t, raft::row_major>` | Transform matrix (original space -&gt; rotated padded space) [rot_dim, |

**Returns**

`cuvs::neighbors::ivf_pq::index<int64_t>`

A view-type ivf_pq index that references the provided data

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:1174`_

**Additional overload:** `cuvs::neighbors::ivf_pq::build`

Build an IVF-PQ index from device memory centroids and codebook.

```cpp
void build(raft::resources const& handle,
const cuvs::neighbors::ivf_pq::index_params& index_params,
const uint32_t dim,
raft::device_mdspan<const float, raft::extent_3d<uint32_t>, raft::row_major> pq_centers,
raft::device_matrix_view<const float, uint32_t, raft::row_major> centers,
raft::device_matrix_view<const float, uint32_t, raft::row_major> centers_rot,
raft::device_matrix_view<const float, uint32_t, raft::row_major> rotation_matrix,
cuvs::neighbors::ivf_pq::index<int64_t>* idx);
```

This function creates a non-owning index that references the provided device data directly. All parameters must be provided with correct extents. The caller is responsible for ensuring the lifetime of the input data exceeds the lifetime of the returned index.

The index_params must be consistent with the provided matrices. Specifically:

- index_params.codebook_kind determines the expected shape of pq_centers
- index_params.metric will be stored in the index
- index_params.conservative_memory_allocation will be stored in the index The function will verify consistency between index_params, dim, and the matrix extents.

dim]

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | raft resources handle |
| `index_params` | in | `const cuvs::neighbors::ivf_pq::index_params&` | configure the index (metric, codebook_kind, etc.). Must be consistent with the provided matrices. |
| `dim` | in | `const uint32_t` | dimensionality of the input data |
| `pq_centers` | in | `raft::device_mdspan<const float, raft::extent_3d<uint32_t>, raft::row_major>` | PQ codebook on device memory with required extents:<br />- codebook_gen::PER_SUBSPACE: [pq_dim, pq_len, pq_book_size]<br />- codebook_gen::PER_CLUSTER:  [n_lists, pq_len, pq_book_size] |
| `centers` | in | `raft::device_matrix_view<const float, uint32_t, raft::row_major>` | Cluster centers in the original space [n_lists, dim_ext] where dim_ext = round_up(dim + 1, 8) |
| `centers_rot` | in | `raft::device_matrix_view<const float, uint32_t, raft::row_major>` | Rotated cluster centers [n_lists, rot_dim] where rot_dim = pq_len * pq_dim |
| `rotation_matrix` | in | `raft::device_matrix_view<const float, uint32_t, raft::row_major>` | Transform matrix (original space -&gt; rotated padded space) [rot_dim, |
| `idx` | out | `cuvs::neighbors::ivf_pq::index<int64_t>*` | pointer to ivf_pq::index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:1211`_

**Additional overload:** `cuvs::neighbors::ivf_pq::build`

Build an IVF-PQ index from host memory centroids and codebook (in-place).

```cpp
auto build(
raft::resources const& handle,
const cuvs::neighbors::ivf_pq::index_params& index_params,
const uint32_t dim,
raft::host_mdspan<const float, raft::extent_3d<uint32_t>, raft::row_major> pq_centers,
raft::host_matrix_view<const float, uint32_t, raft::row_major> centers,
std::optional<raft::host_matrix_view<const float, uint32_t, raft::row_major>> centers_rot,
std::optional<raft::host_matrix_view<const float, uint32_t, raft::row_major>> rotation_matrix)
-> cuvs::neighbors::ivf_pq::index<int64_t>;
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | raft resources handle |
| `index_params` | in | `const cuvs::neighbors::ivf_pq::index_params&` | configure the index building |
| `dim` | in | `const uint32_t` | dimensionality of the input data |
| `pq_centers` | in | `raft::host_mdspan<const float, raft::extent_3d<uint32_t>, raft::row_major>` | PQ codebook |
| `centers` | in | `raft::host_matrix_view<const float, uint32_t, raft::row_major>` | Cluster centers |
| `centers_rot` | in | `std::optional<raft::host_matrix_view<const float, uint32_t, raft::row_major>>` | Optional rotated cluster centers |
| `rotation_matrix` | in | `std::optional<raft::host_matrix_view<const float, uint32_t, raft::row_major>>` | Optional rotation matrix |

**Returns**

`cuvs::neighbors::ivf_pq::index<int64_t>`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:1231`_

**Additional overload:** `cuvs::neighbors::ivf_pq::build`

Build an IVF-PQ index from host memory centroids and codebook (in-place).

```cpp
void build(
raft::resources const& handle,
const cuvs::neighbors::ivf_pq::index_params& index_params,
const uint32_t dim,
raft::host_mdspan<const float, raft::extent_3d<uint32_t>, raft::row_major> pq_centers,
raft::host_matrix_view<const float, uint32_t, raft::row_major> centers,
std::optional<raft::host_matrix_view<const float, uint32_t, raft::row_major>> centers_rot,
std::optional<raft::host_matrix_view<const float, uint32_t, raft::row_major>> rotation_matrix,
cuvs::neighbors::ivf_pq::index<int64_t>* idx);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | raft resources handle |
| `index_params` | in | `const cuvs::neighbors::ivf_pq::index_params&` | configure the index building |
| `dim` | in | `const uint32_t` | dimensionality of the input data |
| `pq_centers` | in | `raft::host_mdspan<const float, raft::extent_3d<uint32_t>, raft::row_major>` | PQ codebook on host memory |
| `centers` | in | `raft::host_matrix_view<const float, uint32_t, raft::row_major>` | Cluster centers on host memory |
| `centers_rot` | in | `std::optional<raft::host_matrix_view<const float, uint32_t, raft::row_major>>` | Optional rotated cluster centers on host |
| `rotation_matrix` | in | `std::optional<raft::host_matrix_view<const float, uint32_t, raft::row_major>>` | Optional rotation matrix on host |
| `idx` | out | `cuvs::neighbors::ivf_pq::index<int64_t>*` | pointer to IVF-PQ index to be built |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:1253`_

## IVF-PQ index extend

_Doxygen group: `ivf_pq_cpp_index_extend`_

### cuvs::neighbors::ivf_pq::extend

Extend the index with the new data.

```cpp
auto extend(raft::resources const& handle,
raft::device_matrix_view<const float, int64_t, raft::row_major> new_vectors,
std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
const cuvs::neighbors::ivf_pq::index<int64_t>& idx)
-> cuvs::neighbors::ivf_pq::index<int64_t>;
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `new_vectors` | in | `raft::device_matrix_view<const float, int64_t, raft::row_major>` | a device matrix view to a row-major matrix [n_rows, idx.dim()] |
| `new_indices` | in | `std::optional<raft::device_vector_view<const int64_t, int64_t>>` | a device vector view to a vector of indices [n_rows]. If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt` here to imply a continuous range `[0...n_rows)`. |
| `idx` | inout | `const cuvs::neighbors::ivf_pq::index<int64_t>&` |  |

**Returns**

`cuvs::neighbors::ivf_pq::index<int64_t>`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:1293`_

**Additional overload:** `cuvs::neighbors::ivf_pq::extend`

Extend the index with the new data.

```cpp
void extend(raft::resources const& handle,
raft::device_matrix_view<const float, int64_t, raft::row_major> new_vectors,
std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
cuvs::neighbors::ivf_pq::index<int64_t>* idx);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `new_vectors` | in | `raft::device_matrix_view<const float, int64_t, raft::row_major>` | a device matrix view to a row-major matrix [n_rows, idx.dim()] |
| `new_indices` | in | `std::optional<raft::device_vector_view<const int64_t, int64_t>>` | a device vector view to a vector of indices [n_rows]. If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt` here to imply a continuous range `[0...n_rows)`. |
| `idx` | inout | `cuvs::neighbors::ivf_pq::index<int64_t>*` |  |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:1322`_

**Additional overload:** `cuvs::neighbors::ivf_pq::extend`

Extend the index with the new data.

```cpp
auto extend(raft::resources const& handle,
raft::device_matrix_view<const half, int64_t, raft::row_major> new_vectors,
std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
const cuvs::neighbors::ivf_pq::index<int64_t>& idx)
-> cuvs::neighbors::ivf_pq::index<int64_t>;
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `new_vectors` | in | `raft::device_matrix_view<const half, int64_t, raft::row_major>` | a device matrix view to a row-major matrix [n_rows, idx.dim()] |
| `new_indices` | in | `std::optional<raft::device_vector_view<const int64_t, int64_t>>` | a device vector view to a vector of indices [n_rows]. If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt` here to imply a continuous range `[0...n_rows)`. |
| `idx` | inout | `const cuvs::neighbors::ivf_pq::index<int64_t>&` |  |

**Returns**

`cuvs::neighbors::ivf_pq::index<int64_t>`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:1350`_

**Additional overload:** `cuvs::neighbors::ivf_pq::extend`

Extend the index with the new data.

```cpp
void extend(raft::resources const& handle,
raft::device_matrix_view<const half, int64_t, raft::row_major> new_vectors,
std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
cuvs::neighbors::ivf_pq::index<int64_t>* idx);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `new_vectors` | in | `raft::device_matrix_view<const half, int64_t, raft::row_major>` | a device matrix view to a row-major matrix [n_rows, idx.dim()] |
| `new_indices` | in | `std::optional<raft::device_vector_view<const int64_t, int64_t>>` | a device vector view to a vector of indices [n_rows]. If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt` here to imply a continuous range `[0...n_rows)`. |
| `idx` | inout | `cuvs::neighbors::ivf_pq::index<int64_t>*` |  |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:1379`_

**Additional overload:** `cuvs::neighbors::ivf_pq::extend`

Extend the index with the new data.

```cpp
auto extend(raft::resources const& handle,
raft::device_matrix_view<const int8_t, int64_t, raft::row_major> new_vectors,
std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
const cuvs::neighbors::ivf_pq::index<int64_t>& idx)
-> cuvs::neighbors::ivf_pq::index<int64_t>;
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `new_vectors` | in | `raft::device_matrix_view<const int8_t, int64_t, raft::row_major>` | a device matrix view to a row-major matrix [n_rows, idx.dim()] |
| `new_indices` | in | `std::optional<raft::device_vector_view<const int64_t, int64_t>>` | a device vector view to a vector of indices [n_rows]. If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt` here to imply a continuous range `[0...n_rows)`. |
| `idx` | inout | `const cuvs::neighbors::ivf_pq::index<int64_t>&` |  |

**Returns**

`cuvs::neighbors::ivf_pq::index<int64_t>`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:1407`_

**Additional overload:** `cuvs::neighbors::ivf_pq::extend`

Extend the index with the new data.

```cpp
void extend(raft::resources const& handle,
raft::device_matrix_view<const int8_t, int64_t, raft::row_major> new_vectors,
std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
cuvs::neighbors::ivf_pq::index<int64_t>* idx);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `new_vectors` | in | `raft::device_matrix_view<const int8_t, int64_t, raft::row_major>` | a device matrix view to a row-major matrix [n_rows, idx.dim()] |
| `new_indices` | in | `std::optional<raft::device_vector_view<const int64_t, int64_t>>` | a device vector view to a vector of indices [n_rows]. If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt` here to imply a continuous range `[0...n_rows)`. |
| `idx` | inout | `cuvs::neighbors::ivf_pq::index<int64_t>*` |  |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:1436`_

**Additional overload:** `cuvs::neighbors::ivf_pq::extend`

Extend the index with the new data.

```cpp
auto extend(raft::resources const& handle,
raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> new_vectors,
std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
const cuvs::neighbors::ivf_pq::index<int64_t>& idx)
-> cuvs::neighbors::ivf_pq::index<int64_t>;
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `new_vectors` | in | `raft::device_matrix_view<const uint8_t, int64_t, raft::row_major>` | a device matrix view to a row-major matrix [n_rows, idx.dim()] |
| `new_indices` | in | `std::optional<raft::device_vector_view<const int64_t, int64_t>>` | a device vector view to a vector of indices [n_rows]. If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt` here to imply a continuous range `[0...n_rows)`. |
| `idx` | inout | `const cuvs::neighbors::ivf_pq::index<int64_t>&` |  |

**Returns**

`cuvs::neighbors::ivf_pq::index<int64_t>`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:1464`_

**Additional overload:** `cuvs::neighbors::ivf_pq::extend`

Extend the index with the new data.

```cpp
void extend(raft::resources const& handle,
raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> new_vectors,
std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
cuvs::neighbors::ivf_pq::index<int64_t>* idx);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `new_vectors` | in | `raft::device_matrix_view<const uint8_t, int64_t, raft::row_major>` | a device matrix view to a row-major matrix [n_rows, idx.dim()] |
| `new_indices` | in | `std::optional<raft::device_vector_view<const int64_t, int64_t>>` | a device vector view to a vector of indices [n_rows]. If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt` here to imply a continuous range `[0...n_rows)`. |
| `idx` | inout | `cuvs::neighbors::ivf_pq::index<int64_t>*` |  |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:1493`_

**Additional overload:** `cuvs::neighbors::ivf_pq::extend`

Extend the index with the new data.

```cpp
auto extend(raft::resources const& handle,
raft::host_matrix_view<const float, int64_t, raft::row_major> new_vectors,
std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
const cuvs::neighbors::ivf_pq::index<int64_t>& idx)
-> cuvs::neighbors::ivf_pq::index<int64_t>;
```

Note, the user can set a stream pool in the input raft::resource with at least one stream to enable kernel and copy overlapping.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `new_vectors` | in | `raft::host_matrix_view<const float, int64_t, raft::row_major>` | a host matrix view to a row-major matrix [n_rows, idx.dim()] |
| `new_indices` | in | `std::optional<raft::host_vector_view<const int64_t, int64_t>>` | a host vector view to a vector of indices [n_rows]. If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt` here to imply a continuous range `[0...n_rows)`. |
| `idx` | inout | `const cuvs::neighbors::ivf_pq::index<int64_t>&` |  |

**Returns**

`cuvs::neighbors::ivf_pq::index<int64_t>`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:1527`_

**Additional overload:** `cuvs::neighbors::ivf_pq::extend`

Extend the index with the new data.

```cpp
void extend(raft::resources const& handle,
raft::host_matrix_view<const float, int64_t, raft::row_major> new_vectors,
std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
cuvs::neighbors::ivf_pq::index<int64_t>* idx);
```

Note, the user can set a stream pool in the input raft::resource with at least one stream to enable kernel and copy overlapping.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `new_vectors` | in | `raft::host_matrix_view<const float, int64_t, raft::row_major>` | a host matrix view to a row-major matrix [n_rows, idx.dim()] |
| `new_indices` | in | `std::optional<raft::host_vector_view<const int64_t, int64_t>>` | a host vector view to a vector of indices [n_rows]. If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt` here to imply a continuous range `[0...n_rows)`. |
| `idx` | inout | `cuvs::neighbors::ivf_pq::index<int64_t>*` |  |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:1562`_

**Additional overload:** `cuvs::neighbors::ivf_pq::extend`

Extend the index with the new data.

```cpp
auto extend(raft::resources const& handle,
raft::host_matrix_view<const half, int64_t, raft::row_major> new_vectors,
std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
const cuvs::neighbors::ivf_pq::index<int64_t>& idx)
-> cuvs::neighbors::ivf_pq::index<int64_t>;
```

Note, the user can set a stream pool in the input raft::resource with at least one stream to enable kernel and copy overlapping.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `new_vectors` | in | `raft::host_matrix_view<const half, int64_t, raft::row_major>` | a host matrix view to a row-major matrix [n_rows, idx.dim()] |
| `new_indices` | in | `std::optional<raft::host_vector_view<const int64_t, int64_t>>` | a host vector view to a vector of indices [n_rows]. If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt` here to imply a continuous range `[0...n_rows)`. |
| `idx` | inout | `const cuvs::neighbors::ivf_pq::index<int64_t>&` |  |

**Returns**

`cuvs::neighbors::ivf_pq::index<int64_t>`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:1596`_

**Additional overload:** `cuvs::neighbors::ivf_pq::extend`

Extend the index with the new data.

```cpp
void extend(raft::resources const& handle,
raft::host_matrix_view<const half, int64_t, raft::row_major> new_vectors,
std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
cuvs::neighbors::ivf_pq::index<int64_t>* idx);
```

Note, the user can set a stream pool in the input raft::resource with at least one stream to enable kernel and copy overlapping.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `new_vectors` | in | `raft::host_matrix_view<const half, int64_t, raft::row_major>` | a host matrix view to a row-major matrix [n_rows, idx.dim()] |
| `new_indices` | in | `std::optional<raft::host_vector_view<const int64_t, int64_t>>` | a host vector view to a vector of indices [n_rows]. If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt` here to imply a continuous range `[0...n_rows)`. |
| `idx` | inout | `cuvs::neighbors::ivf_pq::index<int64_t>*` |  |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:1631`_

**Additional overload:** `cuvs::neighbors::ivf_pq::extend`

Extend the index with the new data.

```cpp
auto extend(raft::resources const& handle,
raft::host_matrix_view<const int8_t, int64_t, raft::row_major> new_vectors,
std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
const cuvs::neighbors::ivf_pq::index<int64_t>& idx)
-> cuvs::neighbors::ivf_pq::index<int64_t>;
```

Note, the user can set a stream pool in the input raft::resource with at least one stream to enable kernel and copy overlapping.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `new_vectors` | in | `raft::host_matrix_view<const int8_t, int64_t, raft::row_major>` | a host matrix view to a row-major matrix [n_rows, idx.dim()] |
| `new_indices` | in | `std::optional<raft::host_vector_view<const int64_t, int64_t>>` | a host vector view to a vector of indices [n_rows]. If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt` here to imply a continuous range `[0...n_rows)`. |
| `idx` | inout | `const cuvs::neighbors::ivf_pq::index<int64_t>&` |  |

**Returns**

`cuvs::neighbors::ivf_pq::index<int64_t>`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:1665`_

**Additional overload:** `cuvs::neighbors::ivf_pq::extend`

Extend the index with the new data.

```cpp
void extend(raft::resources const& handle,
raft::host_matrix_view<const int8_t, int64_t, raft::row_major> new_vectors,
std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
cuvs::neighbors::ivf_pq::index<int64_t>* idx);
```

Note, the user can set a stream pool in the input raft::resource with at least one stream to enable kernel and copy overlapping.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `new_vectors` | in | `raft::host_matrix_view<const int8_t, int64_t, raft::row_major>` | a host matrix view to a row-major matrix [n_rows, idx.dim()] |
| `new_indices` | in | `std::optional<raft::host_vector_view<const int64_t, int64_t>>` | a host vector view to a vector of indices [n_rows]. If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt` here to imply a continuous range `[0...n_rows)`. |
| `idx` | inout | `cuvs::neighbors::ivf_pq::index<int64_t>*` |  |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:1700`_

**Additional overload:** `cuvs::neighbors::ivf_pq::extend`

Extend the index with the new data.

```cpp
auto extend(raft::resources const& handle,
raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> new_vectors,
std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
const cuvs::neighbors::ivf_pq::index<int64_t>& idx)
-> cuvs::neighbors::ivf_pq::index<int64_t>;
```

Note, the user can set a stream pool in the input raft::resource with at least one stream to enable kernel and copy overlapping.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `new_vectors` | in | `raft::host_matrix_view<const uint8_t, int64_t, raft::row_major>` | a host matrix view to a row-major matrix [n_rows, idx.dim()] |
| `new_indices` | in | `std::optional<raft::host_vector_view<const int64_t, int64_t>>` | a host vector view to a vector of indices [n_rows]. If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt` here to imply a continuous range `[0...n_rows)`. |
| `idx` | inout | `const cuvs::neighbors::ivf_pq::index<int64_t>&` |  |

**Returns**

`cuvs::neighbors::ivf_pq::index<int64_t>`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:1734`_

**Additional overload:** `cuvs::neighbors::ivf_pq::extend`

Extend the index with the new data.

```cpp
void extend(raft::resources const& handle,
raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> new_vectors,
std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
cuvs::neighbors::ivf_pq::index<int64_t>* idx);
```

Note, the user can set a stream pool in the input raft::resource with at least one stream to enable kernel and copy overlapping.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `new_vectors` | in | `raft::host_matrix_view<const uint8_t, int64_t, raft::row_major>` | a host matrix view to a row-major matrix [n_rows, idx.dim()] |
| `new_indices` | in | `std::optional<raft::host_vector_view<const int64_t, int64_t>>` | a host vector view to a vector of indices [n_rows]. If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt` here to imply a continuous range `[0...n_rows)`. |
| `idx` | inout | `cuvs::neighbors::ivf_pq::index<int64_t>*` |  |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:1770`_

## IVF-PQ index transform

_Doxygen group: `ivf_pq_cpp_transform`_

### cuvs::neighbors::ivf_pq::transform

Transform a dataset by applying pq-encoding to each vector

```cpp
void transform(raft::resources const& handle,
const cuvs::neighbors::ivf_pq::index<int64_t>& index,
raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
raft::device_vector_view<uint32_t, int64_t> output_labels,
raft::device_matrix_view<uint8_t, int64_t> output_dataset);
```

cluster ids (labels) for each vector in the input dataset index.pq_bits(), 8)]] that will get populated with the pq-encoded dataset

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index` | in | `const cuvs::neighbors::ivf_pq::index<int64_t>&` | ivf-pq constructed index |
| `dataset` | in | `raft::device_matrix_view<const float, int64_t, raft::row_major>` | a device matrix view to a row-major matrix [n_rows, index.dim()] |
| `output_labels` | out | `raft::device_vector_view<uint32_t, int64_t>` | a device vector view [n_rows] that will get populaterd with the |
| `output_dataset` | out | `raft::device_matrix_view<uint8_t, int64_t>` | a device matrix view [n_rows, ceildiv(index.pq_dim() * |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:1981`_

**Additional overload:** `cuvs::neighbors::ivf_pq::transform`

```cpp
void transform(raft::resources const& handle,
const cuvs::neighbors::ivf_pq::index<int64_t>& index,
raft::device_matrix_view<const half, int64_t, raft::row_major> dataset,
raft::device_vector_view<uint32_t, int64_t> output_labels,
raft::device_matrix_view<uint8_t, int64_t> output_dataset);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |
| `index` |  | `const cuvs::neighbors::ivf_pq::index<int64_t>&` |  |
| `dataset` |  | `raft::device_matrix_view<const half, int64_t, raft::row_major>` |  |
| `output_labels` |  | `raft::device_vector_view<uint32_t, int64_t>` |  |
| `output_dataset` |  | `raft::device_matrix_view<uint8_t, int64_t>` |  |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:1987`_

**Additional overload:** `cuvs::neighbors::ivf_pq::transform`

```cpp
void transform(raft::resources const& handle,
const cuvs::neighbors::ivf_pq::index<int64_t>& index,
raft::device_matrix_view<const int8_t, int64_t, raft::row_major> dataset,
raft::device_vector_view<uint32_t, int64_t> output_labels,
raft::device_matrix_view<uint8_t, int64_t> output_dataset);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |
| `index` |  | `const cuvs::neighbors::ivf_pq::index<int64_t>&` |  |
| `dataset` |  | `raft::device_matrix_view<const int8_t, int64_t, raft::row_major>` |  |
| `output_labels` |  | `raft::device_vector_view<uint32_t, int64_t>` |  |
| `output_dataset` |  | `raft::device_matrix_view<uint8_t, int64_t>` |  |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:1993`_

**Additional overload:** `cuvs::neighbors::ivf_pq::transform`

```cpp
void transform(raft::resources const& handle,
const cuvs::neighbors::ivf_pq::index<int64_t>& index,
raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> dataset,
raft::device_vector_view<uint32_t, int64_t> output_labels,
raft::device_matrix_view<uint8_t, int64_t> output_dataset);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` |  | `raft::resources const&` |  |
| `index` |  | `const cuvs::neighbors::ivf_pq::index<int64_t>&` |  |
| `dataset` |  | `raft::device_matrix_view<const uint8_t, int64_t, raft::row_major>` |  |
| `output_labels` |  | `raft::device_vector_view<uint32_t, int64_t>` |  |
| `output_dataset` |  | `raft::device_matrix_view<uint8_t, int64_t>` |  |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:1999`_

## IVF-PQ index serialize

_Doxygen group: `ivf_pq_cpp_serialize`_

### cuvs::neighbors::ivf_pq::serialize

Write the index to an output stream

```cpp
void serialize(raft::resources const& handle,
std::ostream& os,
const cuvs::neighbors::ivf_pq::index<int64_t>& index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `os` | in | `std::ostream&` | output stream |
| `index` | in | `const cuvs::neighbors::ivf_pq::index<int64_t>&` | IVF-PQ index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:2031`_

**Additional overload:** `cuvs::neighbors::ivf_pq::serialize`

Save the index to file.

```cpp
void serialize(raft::resources const& handle,
const std::string& filename,
const cuvs::neighbors::ivf_pq::index<int64_t>& index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `filename` | in | `const std::string&` | the file name for saving the index |
| `index` | in | `const cuvs::neighbors::ivf_pq::index<int64_t>&` | IVF-PQ index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:2054`_

### cuvs::neighbors::ivf_pq::deserialize

Load index from input stream

```cpp
void deserialize(raft::resources const& handle,
std::istream& str,
cuvs::neighbors::ivf_pq::index<int64_t>* index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `str` | in | `std::istream&` | the name of the file that stores the index |
| `index` | out | `cuvs::neighbors::ivf_pq::index<int64_t>*` | IVF-PQ index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:2081`_

**Additional overload:** `cuvs::neighbors::ivf_pq::deserialize`

Load index from file.

```cpp
void deserialize(raft::resources const& handle,
const std::string& filename,
cuvs::neighbors::ivf_pq::index<int64_t>* index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `filename` | in | `const std::string&` | the name of the file that stores the index |
| `index` | out | `cuvs::neighbors::ivf_pq::index<int64_t>*` | IVF-PQ index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:2106`_

## IVF-PQ helper methods

_Doxygen group: `ivf_pq_cpp_helpers`_

### namespace codepacker \{

IVF-PQ helper methods

```cpp
namespace codepacker {
```

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:2561`_

### codepacker::unpack

Unpack `n_take` consecutive records of a single list (cluster) in the compressed index

```cpp
void unpack(raft::resources const& res,
raft::device_mdspan<const uint8_t,
list_spec_interleaved<uint32_t, uint32_t>::list_extents,
raft::row_major> list_data,
uint32_t pq_bits,
uint32_t offset,
raft::device_matrix_view<uint8_t, uint32_t, raft::row_major> codes);
```

starting at given `offset`.

Bit compression is removed, which means output will have pq_dim dimensional vectors (one code per byte, instead of ceildiv(pq_dim * pq_bits, 8) bytes of pq codes).

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `list_data` | in | `raft::device_mdspan<const uint8_t, list_spec_interleaved<uint32_t, uint32_t>::list_extents, raft::row_major>` | block to read from |
| `pq_bits` | in | `uint32_t` | bit length of encoded vector elements |
| `offset` | in | `uint32_t` | How many records in the list to skip. |
| `codes` | out | `raft::device_matrix_view<uint8_t, uint32_t, raft::row_major>` | the destination buffer [n_take, index.pq_dim()]. The length `n_take` defines how many records to unpack, it must be smaller than the list size. |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:2594`_

### codepacker::unpack_contiguous

Unpack `n_rows` consecutive records of a single list (cluster) in the compressed index

```cpp
void unpack_contiguous(raft::resources const& res,
raft::device_mdspan<const uint8_t,
list_spec_interleaved<uint32_t, uint32_t>::list_extents,
raft::row_major> list_data,
uint32_t pq_bits,
uint32_t offset,
uint32_t n_rows,
uint32_t pq_dim,
uint8_t* codes);
```

starting at given `offset`. The output codes of a single vector are contiguous, not expanded to one code per byte, which means the output has ceildiv(pq_dim * pq_bits, 8) bytes per PQ encoded vector.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `list_data` | in | `raft::device_mdspan<const uint8_t, list_spec_interleaved<uint32_t, uint32_t>::list_extents, raft::row_major>` | block to read from |
| `pq_bits` | in | `uint32_t` | bit length of encoded vector elements |
| `offset` | in | `uint32_t` | How many records in the list to skip. |
| `n_rows` | in | `uint32_t` | How many records to unpack |
| `pq_dim` | in | `uint32_t` | The dimensionality of the PQ compressed records |
| `codes` | out | `uint8_t*` | the destination buffer [n_rows, ceildiv(pq_dim * pq_bits, 8)]. The length `n_rows` defines how many records to unpack, it must be smaller than the list size. |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:2634`_

### codepacker::pack

Write flat PQ codes into an existing list by the given offset.

```cpp
void pack(raft::resources const& res,
raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major> codes,
uint32_t pq_bits,
uint32_t offset,
raft::device_mdspan<uint8_t,
list_spec_interleaved<uint32_t, uint32_t>::list_extents,
raft::row_major> list_data);
```

NB: no memory allocation happens here; the list must fit the data (offset + n_vec).

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `codes` | in | `raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major>` | flat PQ codes, one code per byte [n_vec, pq_dim] |
| `pq_bits` | in | `uint32_t` | bit length of encoded vector elements |
| `offset` | in | `uint32_t` | how many records to skip before writing the data into the list |
| `list_data` | in | `raft::device_mdspan<uint8_t, list_spec_interleaved<uint32_t, uint32_t>::list_extents, raft::row_major>` | block to write into |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:2666`_

### codepacker::pack_contiguous

Write flat PQ codes into an existing list by the given offset. The input codes of a single vector

```cpp
void pack_contiguous(raft::resources const& res,
const uint8_t* codes,
uint32_t n_rows,
uint32_t pq_dim,
uint32_t pq_bits,
uint32_t offset,
raft::device_mdspan<uint8_t,
list_spec_interleaved<uint32_t, uint32_t>::list_extents,
raft::row_major> list_data);
```

are contiguous (not expanded to one code per byte).

NB: no memory allocation happens here; the list must fit the data (offset + n_rows records).

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `codes` | in | `const uint8_t*` | flat PQ codes, [n_vec, ceildiv(pq_dim * pq_bits, 8)] |
| `n_rows` | in | `uint32_t` | number of records |
| `pq_dim` | in | `uint32_t` |  |
| `pq_bits` | in | `uint32_t` | bit length of encoded vector elements |
| `offset` | in | `uint32_t` | how many records to skip before writing the data into the list |
| `list_data` | in | `raft::device_mdspan<uint8_t, list_spec_interleaved<uint32_t, uint32_t>::list_extents, raft::row_major>` | block to write into |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:2702`_

### codepacker::pack_list_data

Write flat PQ codes into an existing list by the given offset.

```cpp
void pack_list_data(raft::resources const& res,
index<int64_t>* index,
raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major> codes,
uint32_t label,
uint32_t offset);
```

The list is identified by its label.

NB: no memory allocation happens here; the list must fit the data (offset + n_vec).

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `index` | inout | `index<int64_t>*` | IVF-PQ index. |
| `codes` | in | `raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major>` | flat PQ codes, one code per byte [n_rows, pq_dim] |
| `label` | in | `uint32_t` | The id of the list (cluster) into which we write. |
| `offset` | in | `uint32_t` | how many records to skip before writing the data into the list |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:2736`_

### codepacker::pack_contiguous_list_data

Write flat PQ codes into an existing list by the given offset. Use this when the input

```cpp
void pack_contiguous_list_data(raft::resources const& res,
index<int64_t>* index,
uint8_t* codes,
uint32_t n_rows,
uint32_t label,
uint32_t offset);
```

vectors are PQ encoded and not expanded to one code per byte.

The list is identified by its label.

NB: no memory allocation happens here; the list into which the vectors are packed must fit offset + n_rows rows.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `index` | inout | `index<int64_t>*` | pointer to IVF-PQ index |
| `codes` | in | `uint8_t*` | flat contiguous PQ codes [n_rows, ceildiv(pq_dim * pq_bits, 8)] |
| `n_rows` | in | `uint32_t` | how many records to pack |
| `label` | in | `uint32_t` | The id of the list (cluster) into which we write. |
| `offset` | in | `uint32_t` | how many records to skip before writing the data into the list |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:2779`_

### codepacker::unpack_list_data

Unpack `n_take` consecutive records of a single list (cluster) in the compressed index

```cpp
void unpack_list_data(raft::resources const& res,
const index<int64_t>& index,
raft::device_matrix_view<uint8_t, uint32_t, raft::row_major> out_codes,
uint32_t label,
uint32_t offset);
```

starting at given `offset`, one code per byte (independently of pq_bits).

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` |  |
| `index` | in | `const index<int64_t>&` |  |
| `out_codes` | out | `raft::device_matrix_view<uint8_t, uint32_t, raft::row_major>` | the destination buffer [n_take, index.pq_dim()]. The length `n_take` defines how many records to unpack, it must be smaller than the list size. |
| `label` | in | `uint32_t` | The id of the list (cluster) to decode. |
| `offset` | in | `uint32_t` | How many records in the list to skip. |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:2815`_

**Additional overload:** `codepacker::unpack_list_data`

Unpack a series of records of a single list (cluster) in the compressed index

```cpp
void unpack_list_data(raft::resources const& res,
const index<int64_t>& index,
raft::device_vector_view<const uint32_t> in_cluster_indices,
raft::device_matrix_view<uint8_t, uint32_t, raft::row_major> out_codes,
uint32_t label);
```

by their in-list offsets, one code per byte (independently of pq_bits).

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `index` | in | `const index<int64_t>&` | IVF-PQ index (passed by reference) |
| `in_cluster_indices` | in | `raft::device_vector_view<const uint32_t>` | The offsets of the selected indices within the cluster. |
| `out_codes` | out | `raft::device_matrix_view<uint8_t, uint32_t, raft::row_major>` | the destination buffer [n_take, index.pq_dim()]. The length `n_take` defines how many records to unpack, it must be smaller than the list size. |
| `label` | in | `uint32_t` | The id of the list (cluster) to decode. |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:2851`_

### codepacker::unpack_contiguous_list_data

Unpack `n_rows` consecutive PQ encoded vectors of a single list (cluster) in the

```cpp
void unpack_contiguous_list_data(raft::resources const& res,
const index<int64_t>& index,
uint8_t* out_codes,
uint32_t n_rows,
uint32_t label,
uint32_t offset);
```

compressed index starting at given `offset`, not expanded to one code per byte. Each code in the output buffer occupies ceildiv(index.pq_dim() * index.pq_bits(), 8) bytes.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `index` | in | `const index<int64_t>&` | IVF-PQ index (passed by reference) |
| `out_codes` | out | `uint8_t*` | the destination buffer [n_rows, ceildiv(index.pq_dim() * index.pq_bits(), 8)]. The length `n_rows` defines how many records to unpack, offset + n_rows must be smaller than or equal to the list size. |
| `n_rows` | in | `uint32_t` | how many codes to unpack |
| `label` | in | `uint32_t` | The id of the list (cluster) to decode. |
| `offset` | in | `uint32_t` | How many records in the list to skip. |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:2892`_

### codepacker::reconstruct_list_data

Decode `n_take` consecutive records of a single list (cluster) in the compressed index

```cpp
void reconstruct_list_data(raft::resources const& res,
const index<int64_t>& index,
raft::device_matrix_view<float, uint32_t, raft::row_major> out_vectors,
uint32_t label,
uint32_t offset);
```

starting at given `offset`.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` |  |
| `index` | in | `const index<int64_t>&` |  |
| `out_vectors` | out | `raft::device_matrix_view<float, uint32_t, raft::row_major>` | the destination buffer [n_take, index.dim()]. The length `n_take` defines how many records to reconstruct, it must be smaller than the list size. |
| `label` | in | `uint32_t` | The id of the list (cluster) to decode. |
| `offset` | in | `uint32_t` | How many records in the list to skip. |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:2929`_

**Additional overload:** `codepacker::reconstruct_list_data`

Decode a series of records of a single list (cluster) in the compressed index

```cpp
void reconstruct_list_data(raft::resources const& res,
const index<int64_t>& index,
raft::device_vector_view<const uint32_t> in_cluster_indices,
raft::device_matrix_view<float, uint32_t, raft::row_major> out_vectors,
uint32_t label);
```

by their in-list offsets.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` |  |
| `index` | in | `const index<int64_t>&` |  |
| `in_cluster_indices` | in | `raft::device_vector_view<const uint32_t>` | The offsets of the selected indices within the cluster. |
| `out_vectors` | out | `raft::device_matrix_view<float, uint32_t, raft::row_major>` | the destination buffer [n_take, index.dim()]. The length `n_take` defines how many records to reconstruct, it must be smaller than the list size. |
| `label` | in | `uint32_t` | The id of the list (cluster) to decode. |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:2984`_

### codepacker::extend_list_with_codes

Extend one list of the index in-place, by the list label, skipping the classification and

```cpp
void extend_list_with_codes(
raft::resources const& res,
index<int64_t>* index,
raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major> new_codes,
raft::device_vector_view<const int64_t, uint32_t, raft::row_major> new_indices,
uint32_t label);
```

encoding steps.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` |  |
| `index` | inout | `index<int64_t>*` |  |
| `new_codes` | in | `raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major>` | flat PQ codes, one code per byte [n_rows, index.pq_dim()] |
| `new_indices` | in | `raft::device_vector_view<const int64_t, uint32_t, raft::row_major>` | source indices [n_rows] |
| `label` | in | `uint32_t` | the id of the target list (cluster). |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:3032`_

### codepacker::extend_list_with_contiguous_codes

Extend one list of the index in-place, by the list label, skipping the classification and

```cpp
void extend_list_with_contiguous_codes(
raft::resources const& res,
index<int64_t>* index,
raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major> new_codes,
raft::device_vector_view<const int64_t, uint32_t, raft::row_major> new_indices,
uint32_t label);
```

encoding steps. Uses contiguous/packed codes format.

This is similar to extend_list_with_codes but takes codes in contiguous packed format [n_rows, ceildiv(pq_dim * pq_bits, 8)] instead of unpacked format [n_rows, pq_dim]. This works correctly with any pq_bits value.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` |  |
| `index` | inout | `index<int64_t>*` |  |
| `new_codes` | in | `raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major>` | flat contiguous PQ codes [n_rows, ceildiv(pq_dim * pq_bits, 8)] |
| `new_indices` | in | `raft::device_vector_view<const int64_t, uint32_t, raft::row_major>` | source indices [n_rows] |
| `label` | in | `uint32_t` | the id of the target list (cluster). |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:3071`_

### codepacker::extend_list

Extend one list of the index in-place, by the list label, skipping the classification

```cpp
void extend_list(raft::resources const& res,
index<int64_t>* index,
raft::device_matrix_view<const float, uint32_t, raft::row_major> new_vectors,
raft::device_vector_view<const int64_t, uint32_t, raft::row_major> new_indices,
uint32_t label);
```

step.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` |  |
| `index` | inout | `index<int64_t>*` |  |
| `new_vectors` | in | `raft::device_matrix_view<const float, uint32_t, raft::row_major>` | data to encode [n_rows, index.dim()] |
| `new_indices` | in | `raft::device_vector_view<const int64_t, uint32_t, raft::row_major>` | source indices [n_rows] |
| `label` | in | `uint32_t` | the id of the target list (cluster). |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:3106`_

### codepacker::erase_list

Remove all data from a single list (cluster) in the index.

```cpp
void erase_list(raft::resources const& res, index<int64_t>* index, uint32_t label);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` |  |
| `index` | inout | `index<int64_t>*` |  |
| `label` | in | `uint32_t` | the id of the target list (cluster). |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:3141`_

### codepacker::reset_index

Public helper API to reset the data and indices ptrs, and the list sizes. Useful for

```cpp
void reset_index(const raft::resources& res, index<int64_t>* index);
```

externally modifying the index without going through the build stage. The data and indices of the IVF lists will be lost.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `const raft::resources&` | raft resource |
| `index` | inout | `index<int64_t>*` | pointer to IVF-PQ index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:3163`_

### codepacker::pad_centers_with_norms

Pad cluster centers with their L2 norms for efficient GEMM operations.

```cpp
void pad_centers_with_norms(
raft::resources const& res,
raft::device_matrix_view<const float, uint32_t, raft::row_major> centers,
raft::device_matrix_view<float, uint32_t, raft::row_major> padded_centers);
```

This function takes cluster centers and pads them with their L2 norms to create extended centers suitable for coarse search operations. The output has dimensions [n_centers, dim_ext] where dim_ext = round_up(dim + 1, 8).

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `centers` | in | `raft::device_matrix_view<const float, uint32_t, raft::row_major>` | cluster centers [n_centers, dim] |
| `padded_centers` | out | `raft::device_matrix_view<float, uint32_t, raft::row_major>` | padded centers with norms [n_centers, dim_ext] |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:3176`_

**Additional overload:** `codepacker::pad_centers_with_norms`

Pad cluster centers with their L2 norms for efficient GEMM operations.

```cpp
void pad_centers_with_norms(
raft::resources const& res,
raft::host_matrix_view<const float, uint32_t, raft::row_major> centers,
raft::device_matrix_view<float, uint32_t, raft::row_major> padded_centers);
```

This function takes cluster centers and pads them with their L2 norms to create extended centers suitable for coarse search operations. The output has dimensions [n_centers, dim_ext] where dim_ext = round_up(dim + 1, 8).

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `centers` | in | `raft::host_matrix_view<const float, uint32_t, raft::row_major>` | cluster centers [n_centers, dim] |
| `padded_centers` | out | `raft::device_matrix_view<float, uint32_t, raft::row_major>` | padded centers with norms [n_centers, dim_ext] |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:3192`_

### codepacker::rotate_padded_centers

Rotate padded centers with the rotation matrix.

```cpp
void rotate_padded_centers(
raft::resources const& res,
raft::device_matrix_view<const float, uint32_t, raft::row_major> padded_centers,
raft::device_matrix_view<const float, uint32_t, raft::row_major> rotation_matrix,
raft::device_matrix_view<float, uint32_t, raft::row_major> rotated_centers);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `padded_centers` | in | `raft::device_matrix_view<const float, uint32_t, raft::row_major>` | padded centers [n_centers, dim_ext] |
| `rotation_matrix` | in | `raft::device_matrix_view<const float, uint32_t, raft::row_major>` | rotation matrix [rot_dim, dim] |
| `rotated_centers` | out | `raft::device_matrix_view<float, uint32_t, raft::row_major>` | rotated centers [n_centers, rot_dim] |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:3205`_

### codepacker::extract_centers

Public helper API for fetching a trained index's IVF centroids

```cpp
void extract_centers(raft::resources const& res,
const index<int64_t>& index,
raft::device_matrix_view<float, int64_t, raft::row_major> cluster_centers);
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `index` | in | `const index<int64_t>&` | IVF-PQ index (passed by reference) |
| `cluster_centers` | out | `raft::device_matrix_view<float, int64_t, raft::row_major>` | IVF cluster centers [index.n_lists(), index.dim] |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:3228`_

**Additional overload:** `codepacker::extract_centers`

```cpp
void extract_centers(raft::resources const& res,
const index<int64_t>& index,
raft::host_matrix_view<float, uint32_t, raft::row_major> cluster_centers);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `index` |  | `const index<int64_t>&` |  |
| `cluster_centers` |  | `raft::host_matrix_view<float, uint32_t, raft::row_major>` |  |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:3233`_

### codepacker::recompute_internal_state

Helper exposing the re-computation of list sizes and related arrays if IVF lists have been

```cpp
void recompute_internal_state(const raft::resources& res, index<int64_t>* index);
```

modified externally.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `const raft::resources&` | raft resource |
| `index` | inout | `index<int64_t>*` | pointer to IVF-PQ index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:3262`_

### codepacker::make_rotation_matrix

Generate a rotation matrix into user-provided buffer (standalone version).

```cpp
void make_rotation_matrix(
raft::resources const& res,
raft::device_matrix_view<float, uint32_t, raft::row_major> rotation_matrix,
bool force_random_rotation);
```

This standalone helper generates a rotation matrix without requiring an index object. Users can call this to prepare a rotation matrix before building from precomputed data.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `rotation_matrix` | out | `raft::device_matrix_view<float, uint32_t, raft::row_major>` | Output buffer [rot_dim, dim] for the rotation matrix |
| `force_random_rotation` | in | `bool` | If false and rot_dim == dim, creates identity matrix. If true or rot_dim != dim, creates random orthogonal matrix. |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:3289`_

### codepacker::resize_list

Resize an IVF-PQ list with flat layout.

```cpp
void resize_list(raft::resources const& res,
std::shared_ptr<list_data_base<int64_t, uint32_t>>& orig_list,
const list_spec_flat<uint32_t, int64_t>& spec,
uint32_t new_used_size,
uint32_t old_used_size);
```

This helper resizes an IVF list that uses the flat (non-interleaved) PQ code layout. If the new size exceeds the current capacity, a new list is allocated and existing data is copied. The function handles the type casting internally.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `orig_list` | inout | `std::shared_ptr<list_data_base<int64_t, uint32_t>>&` | the list to resize (may be replaced with a new allocation) |
| `spec` | in | `const list_spec_flat<uint32_t, int64_t>&` | the list specification containing pq_bits, pq_dim, and allocation settings |
| `new_used_size` | in | `uint32_t` | the new size of the list (number of vectors) |
| `old_used_size` | in | `uint32_t` | the current size of the list (data up to this size is preserved) |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:3319`_

**Additional overload:** `codepacker::resize_list`

Resize an IVF-PQ list with interleaved layout.

```cpp
void resize_list(raft::resources const& res,
std::shared_ptr<list_data_base<int64_t, uint32_t>>& orig_list,
const list_spec_interleaved<uint32_t, int64_t>& spec,
uint32_t new_used_size,
uint32_t old_used_size);
```

This helper resizes an IVF list that uses the interleaved PQ code layout (default). If the new size exceeds the current capacity, a new list is allocated and existing data is copied. The function handles the type casting internally.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resource |
| `orig_list` | inout | `std::shared_ptr<list_data_base<int64_t, uint32_t>>&` | the list to resize (may be replaced with a new allocation) |
| `spec` | in | `const list_spec_interleaved<uint32_t, int64_t>&` | the list specification containing pq_bits, pq_dim, and allocation settings |
| `new_used_size` | in | `uint32_t` | the new size of the list (number of vectors) |
| `old_used_size` | in | `uint32_t` | the current size of the list (data up to this size is preserved) |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/ivf_pq.hpp:3350`_
