---
slug: api-reference/cpp-api-neighbors-common
---

# Common

_Source header: `cpp/include/cuvs/neighbors/common.hpp`_

## Approximate Nearest Neighbors Types

_Doxygen group: `neighbors_index`_

### cuvs::neighbors::index

The base for approximate KNN index structures.

```cpp
struct index { ... } ;
```

_Source: `cpp/include/cuvs/neighbors/common.hpp:107`_

### cuvs::neighbors::index_params

The base for KNN index parameters.

```cpp
struct index_params { ... } ;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `metric` | `cuvs::distance::DistanceType` | Distance type. |
| `metric_arg` | `float` | The argument used by some distance metrics. |

_Source: `cpp/include/cuvs/neighbors/common.hpp:110`_

### cuvs::neighbors::MergeStrategy

Strategy for merging indices.

This enum is declared separately to avoid namespace pollution when including common.hpp. It provides a generic merge strategy that can be used across different index types.

```cpp
enum class MergeStrategy { ... } ;
```

**Values**

| Name | Value |
| --- | --- |
| `MERGE_STRATEGY_PHYSICAL` | `0` |
| `MERGE_STRATEGY_LOGICAL` | `1` |

_Source: `cpp/include/cuvs/neighbors/common.hpp:125`_

## Filtering for ANN Types

_Doxygen group: `neighbors_filtering`_

### filtering::FilterType

Filtering for ANN Types

```cpp
enum class FilterType { ... } ;
```

**Values**

| Name | Value |
| --- | --- |
| `None` | `` |
| `Bitmap` | `` |
| `Bitset` | `` |

_Source: `cpp/include/cuvs/neighbors/common.hpp:496`_

### filtering::operator

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

### filtering::get_filter_type

```cpp
FilterType get_filter_type() const override;
```

**Returns**

`FilterType`

_Source: `cpp/include/cuvs/neighbors/common.hpp:520`_

**Additional overload:** `filtering::operator`

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

**Additional overload:** `filtering::operator`

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

**Additional overload:** `filtering::get_filter_type`

```cpp
FilterType get_filter_type() const override;
```

**Returns**

`FilterType`

_Source: `cpp/include/cuvs/neighbors/common.hpp:578`_

### filtering::bitset_filter

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

**Additional overload:** `filtering::get_filter_type`

```cpp
FilterType get_filter_type() const override;
```

**Returns**

`FilterType`

_Source: `cpp/include/cuvs/neighbors/common.hpp:608`_

## ANN MG index build parameters

_Doxygen group: `mg_cpp_index_params`_

### ivf::distribution_mode

```cpp
enum distribution_mode { ... } ;
```

_Source: `cpp/include/cuvs/neighbors/common.hpp:904`_

## ANN MG search parameters

_Doxygen group: `mg_cpp_search_params`_

### ivf::replicated_search_mode

```cpp
enum replicated_search_mode { ... } ;
```

**Values**

| Name | Value |
| --- | --- |
| `LOAD_BALANCER` | `` |
| `ROUND_ROBIN` | `` |

_Source: `cpp/include/cuvs/neighbors/common.hpp:915`_

### ivf::sharded_merge_mode

```cpp
enum sharded_merge_mode { ... } ;
```

**Values**

| Name | Value |
| --- | --- |
| `MERGE_ON_ROOT_RANK` | `` |
| `TREE_MERGE` | `` |

_Source: `cpp/include/cuvs/neighbors/common.hpp:924`_
