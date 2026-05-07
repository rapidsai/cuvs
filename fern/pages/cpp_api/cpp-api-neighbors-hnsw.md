---
slug: api-reference/cpp-api-neighbors-hnsw
---

# HNSW

_Source header: `cpp/include/cuvs/neighbors/hnsw.hpp`_

## hnswlib index wrapper params

<a id="cuvs-neighbors-hnsw-hnswhierarchy"></a>
### cuvs::neighbors::hnsw::HnswHierarchy

Hierarchy for HNSW index when converting from CAGRA index

NOTE: When the value is `NONE`, the HNSW index is built as a base-layer-only index.

```cpp
enum class HnswHierarchy { ... };
```

**Values**

| Name | Value |
| --- | --- |
| `NONE` | `` |

_Source: `cpp/include/cuvs/neighbors/hnsw.hpp:40`_

<a id="cuvs-neighbors-hnsw-deprecated"></a>
### cuvs::neighbors::hnsw::[[deprecated

Create a CAGRA index parameters compatible with HNSW index

```cpp
[[deprecated("Use cagra::index_params::from_hnsw_params instead")]]
cuvs::neighbors::cagra::index_params to_cagra_params(
raft::matrix_extent<int64_t> dataset,
int M,
int ef_construction,
cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded);
```

* IMPORTANT NOTE *

The reference HNSW index and the corresponding from-CAGRA generated HNSW index will NOT produce the same recalls and QPS for the same parameter `ef`. The graphs are different internally. For the same `ef`, the from-CAGRA index likely has a slightly higher recall and slightly lower QPS. However, the Recall-QPS curves should be similar (i.e. the points are just shifted along the curve).

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `arg1` |  | `"Use cagra::index_params::from_hnsw_params instead"` |  |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/hnsw.hpp:114`_

## hnswlib index wrapper

<a id="cuvs-neighbors-hnsw-index"></a>
### cuvs::neighbors::hnsw::index

hnswlib index wrapper

```cpp
template <typename T>
struct index : cuvs::neighbors::index { ... };
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `metric_` | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#cuvs-distance-distancetype) |  |
| `hierarchy_` | [`HnswHierarchy`](/api-reference/cpp-api-neighbors-hnsw#cuvs-neighbors-hnsw-hnswhierarchy) |  |

_Source: `cpp/include/cuvs/neighbors/hnsw.hpp:130`_

<a id="cuvs-neighbors-hnsw-index-index"></a>
### cuvs::neighbors::hnsw::index::index

load a base-layer-only hnswlib index originally saved from a built CAGRA index.

```cpp
index(int dim, cuvs::distance::DistanceType metric, HnswHierarchy hierarchy = HnswHierarchy::NONE)
: dim_;
```

This is a virtual class and it cannot be used directly. To create an index, use the factory function `cuvs::neighbors::hnsw::from_cagra` from the header `cuvs/neighbors/hnsw.hpp`

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `dim` | in | `int` | dimensions of the training dataset |
| `metric` | in | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#cuvs-distance-distancetype) | distance metric to search. Supported metrics ("L2Expanded", "InnerProduct") |
| `hierarchy` | in | [`HnswHierarchy`](/api-reference/cpp-api-neighbors-hnsw#cuvs-neighbors-hnsw-hnswhierarchy) | hierarchy used for upper HNSW layers Default: `HnswHierarchy::NONE`. |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/hnsw.hpp:143`_

<a id="cuvs-neighbors-hnsw-index-get-index"></a>
### cuvs::neighbors::hnsw::index::get_index

Get underlying index

```cpp
virtual void const* get_index() const = 0;
```

**Returns**

`virtual void const*`

_Source: `cpp/include/cuvs/neighbors/hnsw.hpp:153`_

<a id="cuvs-neighbors-hnsw-index-set-ef"></a>
### cuvs::neighbors::hnsw::index::set_ef

Set ef for search

```cpp
virtual void set_ef(int ef) const = 0;
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `ef` |  | `int` |  |

**Returns**

`virtual void`

_Source: `cpp/include/cuvs/neighbors/hnsw.hpp:164`_

<a id="cuvs-neighbors-hnsw-index-file-path"></a>
### cuvs::neighbors::hnsw::index::file_path

Get file path for disk-backed index

```cpp
virtual std::string file_path() const;
```

**Returns**

`virtual std::string`

_Source: `cpp/include/cuvs/neighbors/hnsw.hpp:169`_

## HNSW index extend parameters

<a id="cuvs-neighbors-hnsw-extend-params"></a>
### cuvs::neighbors::hnsw::extend_params

HNSW index extend parameters

```cpp
struct extend_params { ... };
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `num_threads` | `int` | Number of host threads to use to add additional vectors to the index. Value of 0 automatically maximizes parallelism. |

_Source: `cpp/include/cuvs/neighbors/hnsw.hpp:186`_

## Build HNSW index on the GPU

<a id="cuvs-neighbors-hnsw-build"></a>
### cuvs::neighbors::hnsw::build

Build an HNSW index on the GPU

```cpp
std::unique_ptr<index<float>> build(
raft::resources const& res,
const index_params& params,
raft::host_matrix_view<const float, int64_t, raft::row_major> dataset);
```

The resulting graph is compatible for HNSW search, but is not an exact equivalent of the graph built by the HNSW.

The HNSW index construction parameters `M` and `ef_construction` are the main parameters to control the graph degree and graph quality.  We have additional options that can be used to fine tune graph building on the GPU (see `cuvs::neighbors::cagra::index_params`). In case the index does not fit the host or GPU memory,  we would use disk as temporary storage. In such cases it is important to set `ace_params.build_dir` to a fast disk with sufficient storage size.

NOTE: This function requires CUDA headers to be available at compile time.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | `const index_params&` | hnsw index parameters including ACE configuration |
| `dataset` | in | `raft::host_matrix_view<const float, int64_t, raft::row_major>` | a host matrix view to a row-major matrix [n_rows, dim] |

**Returns**

[`std::unique_ptr<index<float>>`](/api-reference/cpp-api-neighbors-hnsw#cuvs-neighbors-hnsw-index)

_Source: `cpp/include/cuvs/neighbors/hnsw.hpp:254`_

**Additional overload:** `cuvs::neighbors::hnsw::build`

Build an HNSW index on the GPU

```cpp
std::unique_ptr<index<half>> build(
raft::resources const& res,
const index_params& params,
raft::host_matrix_view<const half, int64_t, raft::row_major> dataset);
```

The resulting graph is compatible for HNSW search, but is not an exact equivalent of the graph built by the HNSW.

The HNSW index construction parameters `M` and `ef_construction` are the main parameters to control the graph degree and graph quality.  We have additional options that can be used to fine tune graph building on the GPU (see `cuvs::neighbors::cagra::index_params`). In case the index does not fit the host or GPU memory,  we would use disk as temporary storage. In such cases it is important to set `ace_params.build_dir` to a fast disk with sufficient storage size.

NOTE: This function requires CUDA headers to be available at compile time.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | `const index_params&` | hnsw index parameters including ACE configuration |
| `dataset` | in | `raft::host_matrix_view<const half, int64_t, raft::row_major>` | a host matrix view to a row-major matrix [n_rows, dim] |

**Returns**

[`std::unique_ptr<index<half>>`](/api-reference/cpp-api-neighbors-hnsw#cuvs-neighbors-hnsw-index)

_Source: `cpp/include/cuvs/neighbors/hnsw.hpp:312`_

**Additional overload:** `cuvs::neighbors::hnsw::build`

Build an HNSW index on the GPU

```cpp
std::unique_ptr<index<uint8_t>> build(
raft::resources const& res,
const index_params& params,
raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> dataset);
```

The resulting graph is compatible for HNSW search, but is not an exact equivalent of the graph built by the HNSW.

The HNSW index construction parameters `M` and `ef_construction` are the main parameters to control the graph degree and graph quality.  We have additional options that can be used to fine tune graph building on the GPU (see `cuvs::neighbors::cagra::index_params`). In case the index does not fit the host or GPU memory,  we would use disk as temporary storage. In such cases it is important to set `ace_params.build_dir` to a fast disk with sufficient storage size.

NOTE: This function requires CUDA headers to be available at compile time.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | `const index_params&` | hnsw index parameters including ACE configuration |
| `dataset` | in | `raft::host_matrix_view<const uint8_t, int64_t, raft::row_major>` | a host matrix view to a row-major matrix [n_rows, dim] |

**Returns**

[`std::unique_ptr<index<uint8_t>>`](/api-reference/cpp-api-neighbors-hnsw#cuvs-neighbors-hnsw-index)

_Source: `cpp/include/cuvs/neighbors/hnsw.hpp:370`_

**Additional overload:** `cuvs::neighbors::hnsw::build`

Build an HNSW index on the GPU

```cpp
std::unique_ptr<index<int8_t>> build(
raft::resources const& res,
const index_params& params,
raft::host_matrix_view<const int8_t, int64_t, raft::row_major> dataset);
```

The resulting graph is compatible for HNSW search, but is not an exact equivalent of the graph built by the HNSW.

The HNSW index construction parameters `M` and `ef_construction` are the main parameters to control the graph degree and graph quality.  We have additional options that can be used to fine tune graph building on the GPU (see `cuvs::neighbors::cagra::index_params`). In case the index does not fit the host or GPU memory,  we would use disk as temporary storage. In such cases it is important to set `ace_params.build_dir` to a fast disk with sufficient storage size.

NOTE: This function requires CUDA headers to be available at compile time.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | `const index_params&` | hnsw index parameters including ACE configuration |
| `dataset` | in | `raft::host_matrix_view<const int8_t, int64_t, raft::row_major>` | a host matrix view to a row-major matrix [n_rows, dim] |

**Returns**

[`std::unique_ptr<index<int8_t>>`](/api-reference/cpp-api-neighbors-hnsw#cuvs-neighbors-hnsw-index)

_Source: `cpp/include/cuvs/neighbors/hnsw.hpp:428`_

## Load CAGRA index as hnswlib index

<a id="cuvs-neighbors-hnsw-from-cagra"></a>
### cuvs::neighbors::hnsw::from_cagra

Construct an hnswlib index from a CAGRA index

```cpp
std::unique_ptr<index<float>> from_cagra(
raft::resources const& res,
const index_params& params,
const cuvs::neighbors::cagra::index<float, uint32_t>& cagra_index,
std::optional<raft::host_matrix_view<const float, int64_t, raft::row_major>> dataset =
std::nullopt);
```

NOTE: When `hnsw::index_params.hierarchy` is:

1. `NONE`: This method uses the filesystem to write the CAGRA index in `/tmp/&lt;random_number&gt;.bin` before reading it as an hnswlib index, then deleting the temporary file. The returned index is immutable and can only be searched by the hnswlib wrapper in cuVS, as the format is not compatible with the original hnswlib.
2. `CPU`: The returned index is mutable and can be extended with additional vectors. The serialized index is also compatible with the original hnswlib library.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | `const index_params&` | hnsw index parameters |
| `cagra_index` | in | [`const cuvs::neighbors::cagra::index<float, uint32_t>&`](/api-reference/cpp-api-neighbors-cagra#cuvs-neighbors-cagra-index) | cagra index |
| `dataset` | in | `std::optional<raft::host_matrix_view<const float, int64_t, raft::row_major>>` | optional dataset to avoid extra memory copy when hierarchy is `CPU` Default: `std::nullopt`. |

**Returns**

[`std::unique_ptr<index<float>>`](/api-reference/cpp-api-neighbors-hnsw#cuvs-neighbors-hnsw-index)

_Source: `cpp/include/cuvs/neighbors/hnsw.hpp:471`_

**Additional overload:** `cuvs::neighbors::hnsw::from_cagra`

Construct an hnswlib index from a CAGRA index

```cpp
std::unique_ptr<index<half>> from_cagra(
raft::resources const& res,
const index_params& params,
const cuvs::neighbors::cagra::index<half, uint32_t>& cagra_index,
std::optional<raft::host_matrix_view<const half, int64_t, raft::row_major>> dataset =
std::nullopt);
```

NOTE: When `hnsw::index_params.hierarchy` is:

1. `NONE`: This method uses the filesystem to write the CAGRA index in `/tmp/&lt;random_number&gt;.bin` before reading it as an hnswlib index, then deleting the temporary file. The returned index is immutable and can only be searched by the hnswlib wrapper in cuVS, as the format is not compatible with the original hnswlib.
2. `CPU`: The returned index is mutable and can be extended with additional vectors. The serialized index is also compatible with the original hnswlib library.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | `const index_params&` | hnsw index parameters |
| `cagra_index` | in | [`const cuvs::neighbors::cagra::index<half, uint32_t>&`](/api-reference/cpp-api-neighbors-cagra#cuvs-neighbors-cagra-index) | cagra index |
| `dataset` | in | `std::optional<raft::host_matrix_view<const half, int64_t, raft::row_major>>` | optional dataset to avoid extra memory copy when hierarchy is `CPU` Default: `std::nullopt`. |

**Returns**

[`std::unique_ptr<index<half>>`](/api-reference/cpp-api-neighbors-hnsw#cuvs-neighbors-hnsw-index)

_Source: `cpp/include/cuvs/neighbors/hnsw.hpp:507`_

**Additional overload:** `cuvs::neighbors::hnsw::from_cagra`

Construct an hnswlib index from a CAGRA index

```cpp
std::unique_ptr<index<uint8_t>> from_cagra(
raft::resources const& res,
const index_params& params,
const cuvs::neighbors::cagra::index<uint8_t, uint32_t>& cagra_index,
std::optional<raft::host_matrix_view<const uint8_t, int64_t, raft::row_major>> dataset =
std::nullopt);
```

NOTE: When `hnsw::index_params.hierarchy` is:

1. `NONE`: This method uses the filesystem to write the CAGRA index in `/tmp/&lt;random_number&gt;.bin` before reading it as an hnswlib index, then deleting the temporary file. The returned index is immutable and can only be searched by the hnswlib wrapper in cuVS, as the format is not compatible with the original hnswlib.
2. `CPU`: The returned index is mutable and can be extended with additional vectors. The serialized index is also compatible with the original hnswlib library.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | `const index_params&` | hnsw index parameters |
| `cagra_index` | in | [`const cuvs::neighbors::cagra::index<uint8_t, uint32_t>&`](/api-reference/cpp-api-neighbors-cagra#cuvs-neighbors-cagra-index) | cagra index |
| `dataset` | in | `std::optional<raft::host_matrix_view<const uint8_t, int64_t, raft::row_major>>` | optional dataset to avoid extra memory copy when hierarchy is `CPU` Default: `std::nullopt`. |

**Returns**

[`std::unique_ptr<index<uint8_t>>`](/api-reference/cpp-api-neighbors-hnsw#cuvs-neighbors-hnsw-index)

_Source: `cpp/include/cuvs/neighbors/hnsw.hpp:543`_

**Additional overload:** `cuvs::neighbors::hnsw::from_cagra`

Construct an hnswlib index from a CAGRA index

```cpp
std::unique_ptr<index<int8_t>> from_cagra(
raft::resources const& res,
const index_params& params,
const cuvs::neighbors::cagra::index<int8_t, uint32_t>& cagra_index,
std::optional<raft::host_matrix_view<const int8_t, int64_t, raft::row_major>> dataset =
std::nullopt);
```

NOTE: When `hnsw::index_params.hierarchy` is:

1. `NONE`: This method uses the filesystem to write the CAGRA index in `/tmp/&lt;random_number&gt;.bin` before reading it as an hnswlib index, then deleting the temporary file. The returned index is immutable and can only be searched by the hnswlib wrapper in cuVS, as the format is not compatible with the original hnswlib.
2. `CPU`: The returned index is mutable and can be extended with additional vectors. The serialized index is also compatible with the original hnswlib library.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | `const index_params&` | hnsw index parameters |
| `cagra_index` | in | [`const cuvs::neighbors::cagra::index<int8_t, uint32_t>&`](/api-reference/cpp-api-neighbors-cagra#cuvs-neighbors-cagra-index) | cagra index |
| `dataset` | in | `std::optional<raft::host_matrix_view<const int8_t, int64_t, raft::row_major>>` | optional dataset to avoid extra memory copy when hierarchy is `CPU` Default: `std::nullopt`. |

**Returns**

[`std::unique_ptr<index<int8_t>>`](/api-reference/cpp-api-neighbors-hnsw#cuvs-neighbors-hnsw-index)

_Source: `cpp/include/cuvs/neighbors/hnsw.hpp:579`_

## Extend HNSW index with additional vectors

<a id="cuvs-neighbors-hnsw-extend"></a>
### cuvs::neighbors::hnsw::extend

Add new vectors to an HNSW index

```cpp
void extend(raft::resources const& res,
const extend_params& params,
raft::host_matrix_view<const float, int64_t, raft::row_major> additional_dataset,
index<float>& idx);
```

NOTE: The HNSW index can only be extended when the `hnsw::index_params.hierarchy` is `CPU` when converting from a CAGRA index.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | [`const extend_params&`](/api-reference/cpp-api-neighbors-hnsw#cuvs-neighbors-hnsw-extend-params) | configure the extend |
| `additional_dataset` | in | `raft::host_matrix_view<const float, int64_t, raft::row_major>` | a host matrix view to a row-major matrix [n_rows, index-&gt;dim()] |
| `idx` | inout | [`index<float>&`](/api-reference/cpp-api-neighbors-hnsw#cuvs-neighbors-hnsw-index) | HNSW index to extend |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/hnsw.hpp:624`_

**Additional overload:** `cuvs::neighbors::hnsw::extend`

Add new vectors to an HNSW index

```cpp
void extend(raft::resources const& res,
const extend_params& params,
raft::host_matrix_view<const half, int64_t, raft::row_major> additional_dataset,
index<half>& idx);
```

NOTE: The HNSW index can only be extended when the `hnsw::index_params.hierarchy` is `CPU` when converting from a CAGRA index.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | [`const extend_params&`](/api-reference/cpp-api-neighbors-hnsw#cuvs-neighbors-hnsw-extend-params) | configure the extend |
| `additional_dataset` | in | `raft::host_matrix_view<const half, int64_t, raft::row_major>` | a host matrix view to a row-major matrix [n_rows, index-&gt;dim()] |
| `idx` | inout | [`index<half>&`](/api-reference/cpp-api-neighbors-hnsw#cuvs-neighbors-hnsw-index) | HNSW index to extend |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/hnsw.hpp:658`_

**Additional overload:** `cuvs::neighbors::hnsw::extend`

Add new vectors to an HNSW index

```cpp
void extend(raft::resources const& res,
const extend_params& params,
raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> additional_dataset,
index<uint8_t>& idx);
```

NOTE: The HNSW index can only be extended when the `hnsw::index_params.hierarchy` is `CPU` when converting from a CAGRA index.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | [`const extend_params&`](/api-reference/cpp-api-neighbors-hnsw#cuvs-neighbors-hnsw-extend-params) | configure the extend |
| `additional_dataset` | in | `raft::host_matrix_view<const uint8_t, int64_t, raft::row_major>` | a host matrix view to a row-major matrix [n_rows, index-&gt;dim()] |
| `idx` | inout | [`index<uint8_t>&`](/api-reference/cpp-api-neighbors-hnsw#cuvs-neighbors-hnsw-index) | HNSW index to extend |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/hnsw.hpp:692`_

**Additional overload:** `cuvs::neighbors::hnsw::extend`

Add new vectors to an HNSW index

```cpp
void extend(raft::resources const& res,
const extend_params& params,
raft::host_matrix_view<const int8_t, int64_t, raft::row_major> additional_dataset,
index<int8_t>& idx);
```

NOTE: The HNSW index can only be extended when the `hnsw::index_params.hierarchy` is `CPU` when converting from a CAGRA index.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | [`const extend_params&`](/api-reference/cpp-api-neighbors-hnsw#cuvs-neighbors-hnsw-extend-params) | configure the extend |
| `additional_dataset` | in | `raft::host_matrix_view<const int8_t, int64_t, raft::row_major>` | a host matrix view to a row-major matrix [n_rows, index-&gt;dim()] |
| `idx` | inout | [`index<int8_t>&`](/api-reference/cpp-api-neighbors-hnsw#cuvs-neighbors-hnsw-index) | HNSW index to extend |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/hnsw.hpp:726`_

## Build CAGRA index and search with hnswlib

<a id="cuvs-neighbors-hnsw-search-params"></a>
### cuvs::neighbors::hnsw::search_params

Build CAGRA index and search with hnswlib

```cpp
struct search_params : cuvs::neighbors::search_params { ... };
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `ef` | `int` |  |
| `num_threads` | `int` |  |

_Source: `cpp/include/cuvs/neighbors/hnsw.hpp:740`_

## Search hnswlib index

<a id="cuvs-neighbors-hnsw-search"></a>
### cuvs::neighbors::hnsw::search

Search HNSW index constructed from a CAGRA index

```cpp
void search(raft::resources const& res,
const search_params& params,
const index<float>& idx,
raft::host_matrix_view<const float, int64_t, raft::row_major> queries,
raft::host_matrix_view<uint64_t, int64_t, raft::row_major> neighbors,
raft::host_matrix_view<float, int64_t, raft::row_major> distances);
```

NOTE: The HNSW index can only be searched by the hnswlib wrapper in cuVS when the hierarchy is `NONE`, as the format is not compatible with the original hnswlib.

[n_queries, k] k]

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | [`const search_params&`](/api-reference/cpp-api-neighbors-hnsw#cuvs-neighbors-hnsw-search-params) | configure the search |
| `idx` | in | [`const index<float>&`](/api-reference/cpp-api-neighbors-hnsw#cuvs-neighbors-hnsw-index) | cagra index |
| `queries` | in | `raft::host_matrix_view<const float, int64_t, raft::row_major>` | a host matrix view to a row-major matrix [n_queries, index-&gt;dim()] |
| `neighbors` | out | `raft::host_matrix_view<uint64_t, int64_t, raft::row_major>` | a host matrix view to the indices of the neighbors in the source dataset |
| `distances` | out | `raft::host_matrix_view<float, int64_t, raft::row_major>` | a host matrix view to the distances to the selected neighbors [n_queries, |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/hnsw.hpp:794`_

**Additional overload:** `cuvs::neighbors::hnsw::search`

Search HNSW index constructed from a CAGRA index

```cpp
void search(raft::resources const& res,
const search_params& params,
const index<half>& idx,
raft::host_matrix_view<const half, int64_t, raft::row_major> queries,
raft::host_matrix_view<uint64_t, int64_t, raft::row_major> neighbors,
raft::host_matrix_view<float, int64_t, raft::row_major> distances);
```

NOTE: The HNSW index can only be searched by the hnswlib wrapper in cuVS when the hierarchy is `NONE`, as the format is not compatible with the original hnswlib.

[n_queries, k] k]

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | [`const search_params&`](/api-reference/cpp-api-neighbors-hnsw#cuvs-neighbors-hnsw-search-params) | configure the search |
| `idx` | in | [`const index<half>&`](/api-reference/cpp-api-neighbors-hnsw#cuvs-neighbors-hnsw-index) | cagra index |
| `queries` | in | `raft::host_matrix_view<const half, int64_t, raft::row_major>` | a host matrix view to a row-major matrix [n_queries, index-&gt;dim()] |
| `neighbors` | out | `raft::host_matrix_view<uint64_t, int64_t, raft::row_major>` | a host matrix view to the indices of the neighbors in the source dataset |
| `distances` | out | `raft::host_matrix_view<float, int64_t, raft::row_major>` | a host matrix view to the distances to the selected neighbors [n_queries, |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/hnsw.hpp:838`_

**Additional overload:** `cuvs::neighbors::hnsw::search`

Search HNSWindex constructed from a CAGRA index

```cpp
void search(raft::resources const& res,
const search_params& params,
const index<uint8_t>& idx,
raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> queries,
raft::host_matrix_view<uint64_t, int64_t, raft::row_major> neighbors,
raft::host_matrix_view<float, int64_t, raft::row_major> distances);
```

NOTE: The HNSW index can only be searched by the hnswlib wrapper in cuVS when the hierarchy is `NONE`, as the format is not compatible with the original hnswlib.

[n_queries, k] k]

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | [`const search_params&`](/api-reference/cpp-api-neighbors-hnsw#cuvs-neighbors-hnsw-search-params) | configure the search |
| `idx` | in | [`const index<uint8_t>&`](/api-reference/cpp-api-neighbors-hnsw#cuvs-neighbors-hnsw-index) | cagra index |
| `queries` | in | `raft::host_matrix_view<const uint8_t, int64_t, raft::row_major>` | a host matrix view to a row-major matrix [n_queries, index-&gt;dim()] |
| `neighbors` | out | `raft::host_matrix_view<uint64_t, int64_t, raft::row_major>` | a host matrix view to the indices of the neighbors in the source dataset |
| `distances` | out | `raft::host_matrix_view<float, int64_t, raft::row_major>` | a host matrix view to the distances to the selected neighbors [n_queries, |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/hnsw.hpp:882`_

**Additional overload:** `cuvs::neighbors::hnsw::search`

Search HNSW index constructed from a CAGRA index

```cpp
void search(raft::resources const& res,
const search_params& params,
const index<int8_t>& idx,
raft::host_matrix_view<const int8_t, int64_t, raft::row_major> queries,
raft::host_matrix_view<uint64_t, int64_t, raft::row_major> neighbors,
raft::host_matrix_view<float, int64_t, raft::row_major> distances);
```

NOTE: The HNSW index can only be searched by the hnswlib wrapper in cuVS when the hierarchy is `NONE`, as the format is not compatible with the original hnswlib.

[n_queries, k] k]

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | [`const search_params&`](/api-reference/cpp-api-neighbors-hnsw#cuvs-neighbors-hnsw-search-params) | configure the search |
| `idx` | in | [`const index<int8_t>&`](/api-reference/cpp-api-neighbors-hnsw#cuvs-neighbors-hnsw-index) | cagra index |
| `queries` | in | `raft::host_matrix_view<const int8_t, int64_t, raft::row_major>` | a host matrix view to a row-major matrix [n_queries, index-&gt;dim()] |
| `neighbors` | out | `raft::host_matrix_view<uint64_t, int64_t, raft::row_major>` | a host matrix view to the indices of the neighbors in the source dataset |
| `distances` | out | `raft::host_matrix_view<float, int64_t, raft::row_major>` | a host matrix view to the distances to the selected neighbors [n_queries, |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/hnsw.hpp:926`_

## Deserialize CAGRA index as hnswlib index

<a id="cuvs-neighbors-hnsw-serialize"></a>
### cuvs::neighbors::hnsw::serialize

Serialize the HNSW index to file

```cpp
void serialize(raft::resources const& res, const std::string& filename, const index<float>& idx);
```

NOTE: When hierarchy is `NONE`, the saved hnswlib index is immutable and can only be read by the hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib. However, when hierarchy is `CPU`, the saved hnswlib index is compatible with the original hnswlib library.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `filename` | in | `const std::string&` | path to the file to save the serialized CAGRA index |
| `idx` | in | [`const index<float>&`](/api-reference/cpp-api-neighbors-hnsw#cuvs-neighbors-hnsw-index) | cagra index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/hnsw.hpp:969`_

**Additional overload:** `cuvs::neighbors::hnsw::serialize`

Serialize the HNSW index to file

```cpp
void serialize(raft::resources const& res, const std::string& filename, const index<half>& idx);
```

NOTE: When hierarchy is `NONE`, the saved hnswlib index is immutable and can only be read by the hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib. However, when hierarchy is `CPU`, the saved hnswlib index is compatible with the original hnswlib library.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `filename` | in | `const std::string&` | path to the file to save the serialized CAGRA index |
| `idx` | in | [`const index<half>&`](/api-reference/cpp-api-neighbors-hnsw#cuvs-neighbors-hnsw-index) | cagra index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/hnsw.hpp:998`_

**Additional overload:** `cuvs::neighbors::hnsw::serialize`

Serialize the HNSW index to file

```cpp
void serialize(raft::resources const& res, const std::string& filename, const index<uint8_t>& idx);
```

NOTE: When hierarchy is `NONE`, the saved hnswlib index is immutable and can only be read by the hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib. However, when hierarchy is `CPU`, the saved hnswlib index is compatible with the original hnswlib library.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `filename` | in | `const std::string&` | path to the file to save the serialized CAGRA index |
| `idx` | in | [`const index<uint8_t>&`](/api-reference/cpp-api-neighbors-hnsw#cuvs-neighbors-hnsw-index) | cagra index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/hnsw.hpp:1027`_

**Additional overload:** `cuvs::neighbors::hnsw::serialize`

Serialize the HNSW index to file

```cpp
void serialize(raft::resources const& res, const std::string& filename, const index<int8_t>& idx);
```

NOTE: When hierarchy is `NONE`, the saved hnswlib index is immutable and can only be read by the hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib. However, when hierarchy is `CPU`, the saved hnswlib index is compatible with the original hnswlib library.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `filename` | in | `const std::string&` | path to the file to save the serialized CAGRA index |
| `idx` | in | [`const index<int8_t>&`](/api-reference/cpp-api-neighbors-hnsw#cuvs-neighbors-hnsw-index) | cagra index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/hnsw.hpp:1056`_

<a id="cuvs-neighbors-hnsw-deserialize"></a>
### cuvs::neighbors::hnsw::deserialize

De-serialize a CAGRA index saved to a file as an hnswlib index

```cpp
void deserialize(raft::resources const& res,
const index_params& params,
const std::string& filename,
int dim,
cuvs::distance::DistanceType metric,
index<float>** index);
```

NOTE: When hierarchy is `NONE`, the saved hnswlib index is immutable and can only be read by the hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib. However, when hierarchy is `CPU`, the saved hnswlib index is compatible with the original hnswlib library.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | `const index_params&` | hnsw index parameters |
| `filename` | in | `const std::string&` | path to the file containing the serialized CAGRA index |
| `dim` | in | `int` | dimensions of the training dataset |
| `metric` | in | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#cuvs-distance-distancetype) | distance metric to search. Supported metrics ("L2Expanded", "InnerProduct") |
| `index` | out | [`index<float>**`](/api-reference/cpp-api-neighbors-hnsw#cuvs-neighbors-hnsw-index) | hnsw index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/hnsw.hpp:1094`_

**Additional overload:** `cuvs::neighbors::hnsw::deserialize`

De-serialize a CAGRA index saved to a file as an hnswlib index

```cpp
void deserialize(raft::resources const& res,
const index_params& params,
const std::string& filename,
int dim,
cuvs::distance::DistanceType metric,
index<half>** index);
```

NOTE: When hierarchy is `NONE`, the saved hnswlib index is immutable and can only be read by the hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib. However, when hierarchy is `CPU`, the saved hnswlib index is compatible with the original hnswlib library.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | `const index_params&` | hnsw index parameters |
| `filename` | in | `const std::string&` | path to the file containing the serialized CAGRA index |
| `dim` | in | `int` | dimensions of the training dataset |
| `metric` | in | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#cuvs-distance-distancetype) | distance metric to search. Supported metrics ("L2Expanded", "InnerProduct") |
| `index` | out | [`index<half>**`](/api-reference/cpp-api-neighbors-hnsw#cuvs-neighbors-hnsw-index) | hnsw index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/hnsw.hpp:1137`_

**Additional overload:** `cuvs::neighbors::hnsw::deserialize`

De-serialize a CAGRA index saved to a file as an hnswlib index

```cpp
void deserialize(raft::resources const& res,
const index_params& params,
const std::string& filename,
int dim,
cuvs::distance::DistanceType metric,
index<uint8_t>** index);
```

NOTE: When hierarchy is `NONE`, the saved hnswlib index is immutable and can only be read by the hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib. However, when hierarchy is `CPU`, the saved hnswlib index is compatible with the original hnswlib library.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | `const index_params&` | hnsw index parameters |
| `filename` | in | `const std::string&` | path to the file containing the serialized CAGRA index |
| `dim` | in | `int` | dimensions of the training dataset |
| `metric` | in | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#cuvs-distance-distancetype) | distance metric to search. Supported metrics ("L2Expanded", "InnerProduct") |
| `index` | out | [`index<uint8_t>**`](/api-reference/cpp-api-neighbors-hnsw#cuvs-neighbors-hnsw-index) | hnsw index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/hnsw.hpp:1180`_

**Additional overload:** `cuvs::neighbors::hnsw::deserialize`

De-serialize a CAGRA index saved to a file as an hnswlib index

```cpp
void deserialize(raft::resources const& res,
const index_params& params,
const std::string& filename,
int dim,
cuvs::distance::DistanceType metric,
index<int8_t>** index);
```

NOTE: When hierarchy is `NONE`, the saved hnswlib index is immutable and can only be read by the hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib. However, when hierarchy is `CPU`, the saved hnswlib index is compatible with the original hnswlib library.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | `const index_params&` | hnsw index parameters |
| `filename` | in | `const std::string&` | path to the file containing the serialized CAGRA index |
| `dim` | in | `int` | dimensions of the training dataset |
| `metric` | in | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#cuvs-distance-distancetype) | distance metric to search. Supported metrics ("L2Expanded", "InnerProduct") |
| `index` | out | [`index<int8_t>**`](/api-reference/cpp-api-neighbors-hnsw#cuvs-neighbors-hnsw-index) | hnsw index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/hnsw.hpp:1223`_
