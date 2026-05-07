---
slug: api-reference/cpp-api-neighbors-dynamic-batching
---

# Dynamic Batching

_Source header: `cpp/include/cuvs/neighbors/dynamic_batching.hpp`_

## Dynamic Batching index parameters

_Doxygen group: `dynamic_batching_cpp_index_params`_

### detail::index_params

Dynamic Batching index parameters

```cpp
struct index_params : cuvs::neighbors::index_params { ... } ;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `k` | `int64_t` | The number of neighbors to search is fixed at construction time. |
| `max_batch_size` | `int64_t` | Maximum size of the batch to submit to the upstream index. |
| `n_queues` | `size_t` | The number of independent request queues. |
| `conservative_dispatch` | `bool` | By default (`conservative_dispatch = false`) the first CPU thread to commit a query to a batch |

_Source: `cpp/include/cuvs/neighbors/dynamic_batching.hpp:21`_

## Dynamic Batching search parameters

_Doxygen group: `dynamic_batching_cpp_search_params`_

### detail::search_params

Dynamic Batching search parameters

```cpp
struct search_params : cuvs::neighbors::search_params { ... } ;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `dispatch_timeout_ms` | `double` | How long a request can stay in the queue (milliseconds). |

_Source: `cpp/include/cuvs/neighbors/dynamic_batching.hpp:60`_

## Dynamic Batching index type

_Doxygen group: `dynamic_batching_cpp_index`_

### detail::index

Construct a dynamic batching index by wrapping the upstream index.

```cpp
template <typename Upstream>
index(const raft::resources& res,
const cuvs::neighbors::dynamic_batching::index_params& params,
const Upstream& upstream_index,
const typename Upstream::search_params_type& upstream_params,
const cuvs::neighbors::filtering::base_filter* sample_filter = nullptr);
```

**Template Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `Upstream` | `` | the upstream index type |

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `const raft::resources&` | raft resources |
| `params` | in | `const cuvs::neighbors::dynamic_batching::index_params&` | dynamic batching parameters |
| `upstream_index` | in | `const Upstream&` | the original index to perform the search (the reference must be alive for the lifetime of the dynamic batching index) |
| `upstream_params` | in | `const typename Upstream::search_params_type&` | the original index search parameters for all queries in a batch (the parameters are captured by value for the lifetime of the dynamic batching index) |
| `sample_filter` | in | `const cuvs::neighbors::filtering::base_filter*` | filtering function, if any, must be the same for all requests in a batch (the pointer must be alive for the lifetime of the dynamic batching index) Default: `nullptr`. |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/dynamic_batching.hpp:174`_

## Dynamic Batching search

_Doxygen group: `dynamic_batching_cpp_search`_

### detail::search

Search ANN using a dynamic batching index.

```cpp
void search(raft::resources const& res,
cuvs::neighbors::dynamic_batching::search_params const& params,
dynamic_batching::index<float, uint32_t> const& index,
raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
raft::device_matrix_view<uint32_t, int64_t, raft::row_major> neighbors,
raft::device_matrix_view<float, int64_t, raft::row_major> distances);
```

The search parameters of the upstream index and the optional filtering function are configured at the dynamic batching index construction time. Like with many other indexes, the dynamic batching search has the stream-ordered semantics: the host function may return the control before the results are ready. Synchronize with the main CUDA stream in the given resource object to wait for arrival of the search results. Dynamic batching search is thread-safe: call the search function with copies of the same index in multiple threads to increase the occupancy of the batches.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` |  |
| `params` | in | `cuvs::neighbors::dynamic_batching::search_params const&` | query-specific batching parameters, such as the maximum waiting time |
| `index` | in | `dynamic_batching::index<float, uint32_t> const&` | a dynamic batching index |
| `queries` | in | `raft::device_matrix_view<const float, int64_t, raft::row_major>` | a device matrix view to a row-major matrix [n_queries, dim] |
| `neighbors` | out | `raft::device_matrix_view<uint32_t, int64_t, raft::row_major>` | a device matrix view to the indices of the neighbors in the source dataset [n_queries, k] |
| `distances` | out | `raft::device_matrix_view<float, int64_t, raft::row_major>` | a device matrix view to the distances to the selected neighbors [n_queries, k] |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/dynamic_batching.hpp:214`_

**Additional overload:** `detail::search`

```cpp
void search(raft::resources const& res,
cuvs::neighbors::dynamic_batching::search_params const& params,
dynamic_batching::index<half, uint32_t> const& index,
raft::device_matrix_view<const half, int64_t, raft::row_major> queries,
raft::device_matrix_view<uint32_t, int64_t, raft::row_major> neighbors,
raft::device_matrix_view<float, int64_t, raft::row_major> distances);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `params` |  | `cuvs::neighbors::dynamic_batching::search_params const&` |  |
| `index` |  | `dynamic_batching::index<half, uint32_t> const&` |  |
| `queries` |  | `raft::device_matrix_view<const half, int64_t, raft::row_major>` |  |
| `neighbors` |  | `raft::device_matrix_view<uint32_t, int64_t, raft::row_major>` |  |
| `distances` |  | `raft::device_matrix_view<float, int64_t, raft::row_major>` |  |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/dynamic_batching.hpp:222`_

**Additional overload:** `detail::search`

```cpp
void search(raft::resources const& res,
cuvs::neighbors::dynamic_batching::search_params const& params,
dynamic_batching::index<int8_t, uint32_t> const& index,
raft::device_matrix_view<const int8_t, int64_t, raft::row_major> queries,
raft::device_matrix_view<uint32_t, int64_t, raft::row_major> neighbors,
raft::device_matrix_view<float, int64_t, raft::row_major> distances);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `params` |  | `cuvs::neighbors::dynamic_batching::search_params const&` |  |
| `index` |  | `dynamic_batching::index<int8_t, uint32_t> const&` |  |
| `queries` |  | `raft::device_matrix_view<const int8_t, int64_t, raft::row_major>` |  |
| `neighbors` |  | `raft::device_matrix_view<uint32_t, int64_t, raft::row_major>` |  |
| `distances` |  | `raft::device_matrix_view<float, int64_t, raft::row_major>` |  |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/dynamic_batching.hpp:230`_

**Additional overload:** `detail::search`

```cpp
void search(raft::resources const& res,
cuvs::neighbors::dynamic_batching::search_params const& params,
dynamic_batching::index<uint8_t, uint32_t> const& index,
raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> queries,
raft::device_matrix_view<uint32_t, int64_t, raft::row_major> neighbors,
raft::device_matrix_view<float, int64_t, raft::row_major> distances);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `params` |  | `cuvs::neighbors::dynamic_batching::search_params const&` |  |
| `index` |  | `dynamic_batching::index<uint8_t, uint32_t> const&` |  |
| `queries` |  | `raft::device_matrix_view<const uint8_t, int64_t, raft::row_major>` |  |
| `neighbors` |  | `raft::device_matrix_view<uint32_t, int64_t, raft::row_major>` |  |
| `distances` |  | `raft::device_matrix_view<float, int64_t, raft::row_major>` |  |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/dynamic_batching.hpp:238`_

**Additional overload:** `detail::search`

```cpp
void search(raft::resources const& res,
cuvs::neighbors::dynamic_batching::search_params const& params,
dynamic_batching::index<float, int64_t> const& index,
raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
raft::device_matrix_view<float, int64_t, raft::row_major> distances);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `params` |  | `cuvs::neighbors::dynamic_batching::search_params const&` |  |
| `index` |  | `dynamic_batching::index<float, int64_t> const&` |  |
| `queries` |  | `raft::device_matrix_view<const float, int64_t, raft::row_major>` |  |
| `neighbors` |  | `raft::device_matrix_view<int64_t, int64_t, raft::row_major>` |  |
| `distances` |  | `raft::device_matrix_view<float, int64_t, raft::row_major>` |  |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/dynamic_batching.hpp:246`_

**Additional overload:** `detail::search`

```cpp
void search(raft::resources const& res,
cuvs::neighbors::dynamic_batching::search_params const& params,
dynamic_batching::index<half, int64_t> const& index,
raft::device_matrix_view<const half, int64_t, raft::row_major> queries,
raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
raft::device_matrix_view<float, int64_t, raft::row_major> distances);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `params` |  | `cuvs::neighbors::dynamic_batching::search_params const&` |  |
| `index` |  | `dynamic_batching::index<half, int64_t> const&` |  |
| `queries` |  | `raft::device_matrix_view<const half, int64_t, raft::row_major>` |  |
| `neighbors` |  | `raft::device_matrix_view<int64_t, int64_t, raft::row_major>` |  |
| `distances` |  | `raft::device_matrix_view<float, int64_t, raft::row_major>` |  |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/dynamic_batching.hpp:254`_

**Additional overload:** `detail::search`

```cpp
void search(raft::resources const& res,
cuvs::neighbors::dynamic_batching::search_params const& params,
dynamic_batching::index<int8_t, int64_t> const& index,
raft::device_matrix_view<const int8_t, int64_t, raft::row_major> queries,
raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
raft::device_matrix_view<float, int64_t, raft::row_major> distances);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `params` |  | `cuvs::neighbors::dynamic_batching::search_params const&` |  |
| `index` |  | `dynamic_batching::index<int8_t, int64_t> const&` |  |
| `queries` |  | `raft::device_matrix_view<const int8_t, int64_t, raft::row_major>` |  |
| `neighbors` |  | `raft::device_matrix_view<int64_t, int64_t, raft::row_major>` |  |
| `distances` |  | `raft::device_matrix_view<float, int64_t, raft::row_major>` |  |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/dynamic_batching.hpp:262`_

**Additional overload:** `detail::search`

```cpp
void search(raft::resources const& res,
cuvs::neighbors::dynamic_batching::search_params const& params,
dynamic_batching::index<uint8_t, int64_t> const& index,
raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> queries,
raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
raft::device_matrix_view<float, int64_t, raft::row_major> distances);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `params` |  | `cuvs::neighbors::dynamic_batching::search_params const&` |  |
| `index` |  | `dynamic_batching::index<uint8_t, int64_t> const&` |  |
| `queries` |  | `raft::device_matrix_view<const uint8_t, int64_t, raft::row_major>` |  |
| `neighbors` |  | `raft::device_matrix_view<int64_t, int64_t, raft::row_major>` |  |
| `distances` |  | `raft::device_matrix_view<float, int64_t, raft::row_major>` |  |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/dynamic_batching.hpp:270`_
