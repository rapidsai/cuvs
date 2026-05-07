---
slug: api-reference/cpp-api-neighbors-common
---

# Common

_Source header: `cpp/include/cuvs/neighbors/common.hpp`_

## Approximate Nearest Neighbors Types

<a id="cuvs-neighbors-index"></a>
### cuvs::neighbors::index

The base for approximate KNN index structures.

```cpp
struct index { ... };
```

_Source: `cpp/include/cuvs/neighbors/common.hpp:107`_

<a id="cuvs-neighbors-index-params"></a>
### cuvs::neighbors::index_params

The base for KNN index parameters.

```cpp
struct index_params { ... };
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `metric` | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#cuvs-distance-distancetype) | Distance type. |
| `metric_arg` | `float` | The argument used by some distance metrics. |

_Source: `cpp/include/cuvs/neighbors/common.hpp:110`_

<a id="cuvs-neighbors-mergestrategy"></a>
### cuvs::neighbors::MergeStrategy

Strategy for merging indices.

This enum is declared separately to avoid namespace pollution when including common.hpp. It provides a generic merge strategy that can be used across different index types.

```cpp
enum class MergeStrategy { ... };
```

**Values**

| Name | Value |
| --- | --- |
| `MERGE_STRATEGY_PHYSICAL` | `0` |
| `MERGE_STRATEGY_LOGICAL` | `1` |

_Source: `cpp/include/cuvs/neighbors/common.hpp:125`_

## Types

<a id="cuvs-neighbors-dataset"></a>
### cuvs::neighbors::dataset

Two-dimensional dataset; maybe owning, maybe compressed, maybe strided.

```cpp
template <typename IdxT>
struct dataset { ... };
```

_Source: `cpp/include/cuvs/neighbors/common.hpp:135`_

<a id="cuvs-neighbors-vpq-dataset"></a>
### cuvs::neighbors::vpq_dataset

VPQ compressed dataset.

The dataset is compressed using two level quantization

1. Vector Quantization
2. Product Quantization of residuals

```cpp
template <typename MathT, typename IdxT>
struct vpq_dataset : public dataset<IdxT> { ... };
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `vq_code_book` | `raft::device_matrix<math_type, uint32_t, raft::row_major>` | Vector Quantization codebook - "coarse cluster centers". |
| `pq_code_book` | `raft::device_matrix<math_type, uint32_t, raft::row_major>` | Product Quantization codebook - "fine cluster centers". |
| `data` | `raft::device_matrix<uint8_t, index_type, raft::row_major>` | Compressed dataset. |

_Source: `cpp/include/cuvs/neighbors/common.hpp:406`_

<a id="cuvs-neighbors-ivf-list-base"></a>
### cuvs::neighbors::ivf::list_base

Abstract base class for IVF list data.

This allows polymorphic access to list data regardless of the underlying layout.

TODO: Make this struct internal (tracking issue: https://github.com/rapidsai/cuvs/issues/1726)

```cpp
template <typename ValueT, typename IdxT, typename SizeT = uint32_t>
struct list_base { ... };
```

_Source: `cpp/include/cuvs/neighbors/common.hpp:719`_

<a id="cuvs-neighbors-ivf-list"></a>
### cuvs::neighbors::ivf::list

The data for a single IVF list.

```cpp
template <template <typename, typename...> typename SpecT,
typename SizeT,
typename... SpecExtraArgs>
struct list : public list_base<typename SpecT<SizeT, SpecExtraArgs...>::value_type,
typename SpecT<SizeT, SpecExtraArgs...>::index_type,
SizeT> { ... };
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `data` | [`raft::device_mdarray<value_type, list_extents, raft::row_major>`](/api-reference/cpp-api-neighbors-ivf-pq#list-extents) | Possibly encoded data; it's layout is defined by `SpecT`. |
| `indices` | `raft::device_mdarray<index_type, raft::extent_1d<size_type>, raft::row_major>` | Source indices. |
| `size` | `std::atomic<size_type>` | The actual size of the content. |

_Source: `cpp/include/cuvs/neighbors/common.hpp:749`_

## Filtering for ANN Types

<a id="cuvs-neighbors-filtering-filtertype"></a>
### cuvs::neighbors::filtering::FilterType

Filtering for ANN Types

```cpp
enum class FilterType { ... };
```

**Values**

| Name | Value |
| --- | --- |
| `None` | `` |
| `Bitmap` | `` |
| `Bitset` | `` |

_Source: `cpp/include/cuvs/neighbors/common.hpp:496`_

<a id="cuvs-neighbors-filtering-none-sample-filter-operator"></a>
### cuvs::neighbors::filtering::none_sample_filter::operator

```cpp
constexpr __forceinline__ _RAFT_HOST_DEVICE bool operator()(
// query index
const uint32_t query_ix,
// the current inverted list index
const uint32_t cluster_ix,
// the index of the current sample inside the current inverted list
const uint32_t sample_ix) const;
```

**Returns**

`constexpr __forceinline__ _RAFT_HOST_DEVICE bool`

_Source: `cpp/include/cuvs/neighbors/common.hpp:506`_

<a id="cuvs-neighbors-filtering-none-sample-filter-get-filter-type"></a>
### cuvs::neighbors::filtering::none_sample_filter::get_filter_type

```cpp
FilterType get_filter_type() const override;
```

**Returns**

[`FilterType`](/api-reference/cpp-api-neighbors-common#cuvs-neighbors-filtering-filtertype)

_Source: `cpp/include/cuvs/neighbors/common.hpp:520`_

<a id="cuvs-neighbors-filtering-ivf-to-sample-filter"></a>
### cuvs::neighbors::filtering::ivf_to_sample_filter

Filter used to convert the cluster index and sample index

of an IVF search into a sample index. This can be used as an intermediate filter.

```cpp
template <typename index_t, typename filter_t>
struct ivf_to_sample_filter : public base_filter { ... };
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `inds_ptrs_` | `const index_t* const*` |  |
| `next_filter_` | `const filter_t` |  |

_Source: `cpp/include/cuvs/neighbors/common.hpp:531`_

<a id="cuvs-neighbors-filtering-ivf-to-sample-filter-operator"></a>
### cuvs::neighbors::filtering::ivf_to_sample_filter::operator

If the original filter takes three arguments, then don't modify the arguments.

```cpp
inline _RAFT_HOST_DEVICE bool operator()(
// query index
const uint32_t query_ix,
// the current inverted list index
const uint32_t cluster_ix,
// the index of the current sample inside the current inverted list
const uint32_t sample_ix) const;
```

If the original filter takes two arguments, then we are using `inds_ptr_` to obtain the sample index.

**Returns**

`inline _RAFT_HOST_DEVICE bool`

_Source: `cpp/include/cuvs/neighbors/common.hpp:544`_

<a id="cuvs-neighbors-filtering-bitmap-filter"></a>
### cuvs::neighbors::filtering::bitmap_filter

Filter an index with a bitmap

```cpp
template <typename bitmap_t, typename index_t>
struct bitmap_filter : public base_filter { ... };
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `bitmap_view_` | `const view_t` |  |

_Source: `cpp/include/cuvs/neighbors/common.hpp:562`_

<a id="cuvs-neighbors-filtering-bitmap-filter-operator"></a>
### cuvs::neighbors::filtering::bitmap_filter::operator

```cpp
inline _RAFT_HOST_DEVICE bool operator()(
// query index
const uint32_t query_ix,
// the index of the current sample
const uint32_t sample_ix) const;
```

**Returns**

`inline _RAFT_HOST_DEVICE bool`

_Source: `cpp/include/cuvs/neighbors/common.hpp:571`_

<a id="cuvs-neighbors-filtering-bitmap-filter-get-filter-type"></a>
### cuvs::neighbors::filtering::bitmap_filter::get_filter_type

```cpp
FilterType get_filter_type() const override;
```

**Returns**

[`FilterType`](/api-reference/cpp-api-neighbors-common#cuvs-neighbors-filtering-filtertype)

_Source: `cpp/include/cuvs/neighbors/common.hpp:578`_

<a id="cuvs-neighbors-filtering-bitset-filter"></a>
### cuvs::neighbors::filtering::bitset_filter

Filter an index with a bitset

```cpp
template <typename bitset_t, typename index_t>
struct bitset_filter : public base_filter { ... };
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `bitset_view_` | `const view_t` |  |

_Source: `cpp/include/cuvs/neighbors/common.hpp:592`_

<a id="cuvs-neighbors-filtering-bitset-filter-bitset-filter"></a>
### cuvs::neighbors::filtering::bitset_filter::bitset_filter

```cpp
_RAFT_HOST_DEVICE bitset_filter(const view_t bitset_for_filtering);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `bitset_for_filtering` |  | `const view_t` |  |

**Returns**

`_RAFT_HOST_DEVICE`

_Source: `cpp/include/cuvs/neighbors/common.hpp:600`_

<a id="cuvs-neighbors-filtering-bitset-filter-get-filter-type"></a>
### cuvs::neighbors::filtering::bitset_filter::get_filter_type

```cpp
FilterType get_filter_type() const override;
```

**Returns**

[`FilterType`](/api-reference/cpp-api-neighbors-common#cuvs-neighbors-filtering-filtertype)

_Source: `cpp/include/cuvs/neighbors/common.hpp:608`_

## ANN MG index build parameters

<a id="cuvs-neighbors-distribution-mode"></a>
### cuvs::neighbors::distribution_mode

```cpp
enum distribution_mode { ... };
```

_Source: `cpp/include/cuvs/neighbors/common.hpp:904`_

<a id="cuvs-neighbors-mg-index-params"></a>
### cuvs::neighbors::mg_index_params

```cpp
template <typename Upstream>
struct mg_index_params : public Upstream { ... };
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `mode` | [`cuvs::neighbors::distribution_mode`](/api-reference/cpp-api-neighbors-common#cuvs-neighbors-distribution-mode) | Distribution mode |

_Source: `cpp/include/cuvs/neighbors/common.hpp:933`_

## ANN MG search parameters

<a id="cuvs-neighbors-replicated-search-mode"></a>
### cuvs::neighbors::replicated_search_mode

```cpp
enum replicated_search_mode { ... };
```

**Values**

| Name | Value |
| --- | --- |
| `LOAD_BALANCER` | `` |
| `ROUND_ROBIN` | `` |

_Source: `cpp/include/cuvs/neighbors/common.hpp:915`_

<a id="cuvs-neighbors-sharded-merge-mode"></a>
### cuvs::neighbors::sharded_merge_mode

```cpp
enum sharded_merge_mode { ... };
```

**Values**

| Name | Value |
| --- | --- |
| `MERGE_ON_ROOT_RANK` | `` |
| `TREE_MERGE` | `` |

_Source: `cpp/include/cuvs/neighbors/common.hpp:924`_

<a id="cuvs-neighbors-mg-search-params"></a>
### cuvs::neighbors::mg_search_params

```cpp
template <typename Upstream>
struct mg_search_params : public Upstream { ... };
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `search_mode` | [`cuvs::neighbors::replicated_search_mode`](/api-reference/cpp-api-neighbors-common#cuvs-neighbors-replicated-search-mode) | Replicated search mode |
| `merge_mode` | [`cuvs::neighbors::sharded_merge_mode`](/api-reference/cpp-api-neighbors-common#cuvs-neighbors-sharded-merge-mode) | Sharded merge mode |
| `n_rows_per_batch` | `int64_t` | Number of rows per batch |

_Source: `cpp/include/cuvs/neighbors/common.hpp:945`_
