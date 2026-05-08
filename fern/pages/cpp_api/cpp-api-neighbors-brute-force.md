---
slug: api-reference/cpp-api-neighbors-brute-force
---

# Brute Force

_Source header: `cpp/include/cuvs/neighbors/brute_force.hpp`_

## Bruteforce index

_Doxygen group: `bruteforce_cpp_index`_

### cuvs::neighbors::brute_force::index

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

_Source: `cpp/include/cuvs/neighbors/brute_force.hpp:54`_

### cuvs::neighbors::brute_force::index

Construct a brute force index from dataset

```cpp
index(raft::resources const& res,
raft::host_matrix_view<const T, int64_t, raft::row_major> dataset_view,
std::optional<raft::device_vector<DistT, int64_t>>&& norms,
cuvs::distance::DistanceType metric,
DistT metric_arg = 0.0);
```

Constructs a brute force index from a dataset. This lets us precompute norms for the dataset, providing a speed benefit over doing this at query time. This index will copy the host dataset onto the device, and take ownership of any precalculated norms.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `dataset_view` |  | `raft::host_matrix_view<const T, int64_t, raft::row_major>` |  |
| `norms` |  | `std::optional<raft::device_vector<DistT, int64_t>>&&` |  |
| `metric` |  | `cuvs::distance::DistanceType` |  |
| `metric_arg` |  | `DistT` | Default: `0.0`. |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/brute_force.hpp:63`_

### cuvs::neighbors::brute_force::index

Construct a brute force index from dataset

```cpp
index(raft::resources const& res,
raft::device_matrix_view<const T, int64_t, raft::row_major> dataset_view,
std::optional<raft::device_vector<DistT, int64_t>>&& norms,
cuvs::distance::DistanceType metric,
DistT metric_arg = 0.0);
```

Constructs a brute force index from a dataset. This lets us precompute norms for the dataset, providing a speed benefit over doing this at query time. This index will store a non-owning reference to the dataset, but will move any norms supplied.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `dataset_view` |  | `raft::device_matrix_view<const T, int64_t, raft::row_major>` |  |
| `norms` |  | `std::optional<raft::device_vector<DistT, int64_t>>&&` |  |
| `metric` |  | `cuvs::distance::DistanceType` |  |
| `metric_arg` |  | `DistT` | Default: `0.0`. |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/brute_force.hpp:76`_

### cuvs::neighbors::brute_force::index

Construct a brute force index from dataset

```cpp
index(raft::resources const& res,
raft::device_matrix_view<const T, int64_t, raft::row_major> dataset_view,
std::optional<raft::device_vector_view<const DistT, int64_t>> norms_view,
cuvs::distance::DistanceType metric,
DistT metric_arg = 0.0);
```

This class stores a non-owning reference to the dataset and norms. Having precomputed norms gives us a performance advantage at query time.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `dataset_view` |  | `raft::device_matrix_view<const T, int64_t, raft::row_major>` |  |
| `norms_view` |  | `std::optional<raft::device_vector_view<const DistT, int64_t>>` |  |
| `metric` |  | `cuvs::distance::DistanceType` |  |
| `metric_arg` |  | `DistT` | Default: `0.0`. |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/brute_force.hpp:87`_

### cuvs::neighbors::brute_force::index

Construct a brute force index from dataset

```cpp
index(raft::resources const& res,
raft::device_matrix_view<const T, int64_t, raft::col_major> dataset_view,
std::optional<raft::device_vector<DistT, int64_t>>&& norms,
cuvs::distance::DistanceType metric,
DistT metric_arg = 0.0);
```

Constructs a brute force index from a dataset. This lets us precompute norms for the dataset, providing a speed benefit over doing this at query time. This index will store a non-owning reference to the dataset, but will move any norms supplied.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `dataset_view` |  | `raft::device_matrix_view<const T, int64_t, raft::col_major>` |  |
| `norms` |  | `std::optional<raft::device_vector<DistT, int64_t>>&&` |  |
| `metric` |  | `cuvs::distance::DistanceType` |  |
| `metric_arg` |  | `DistT` | Default: `0.0`. |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/brute_force.hpp:100`_

### cuvs::neighbors::brute_force::index

Construct a brute force index from dataset

```cpp
index(raft::resources const& res,
raft::device_matrix_view<const T, int64_t, raft::col_major> dataset_view,
std::optional<raft::device_vector_view<const DistT, int64_t>> norms_view,
cuvs::distance::DistanceType metric,
DistT metric_arg = 0.0);
```

This class stores a non-owning reference to the dataset and norms, with the dataset being supplied on device in a col_major format

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `dataset_view` |  | `raft::device_matrix_view<const T, int64_t, raft::col_major>` |  |
| `norms_view` |  | `std::optional<raft::device_vector_view<const DistT, int64_t>>` |  |
| `metric` |  | `cuvs::distance::DistanceType` |  |
| `metric_arg` |  | `DistT` | Default: `0.0`. |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/brute_force.hpp:111`_

### cuvs::neighbors::brute_force::update_dataset

Replace the dataset with a new dataset.

```cpp
void update_dataset(raft::resources const& res,
raft::device_matrix_view<const T, int64_t, raft::row_major> dataset);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `dataset` |  | `raft::device_matrix_view<const T, int64_t, raft::row_major>` |  |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/brute_force.hpp:120`_

### cuvs::neighbors::brute_force::update_dataset

Replace the dataset with a new dataset.

```cpp
void update_dataset(raft::resources const& res,
raft::host_matrix_view<const T, int64_t, raft::row_major> dataset);
```

We create a copy of the dataset on the device. The index manages the lifetime of this copy.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `dataset` |  | `raft::host_matrix_view<const T, int64_t, raft::row_major>` |  |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/brute_force.hpp:128`_

### cuvs::neighbors::brute_force::metric

Distance metric used for retrieval

```cpp
cuvs::distance::DistanceType metric() const noexcept;
```

**Returns**

`cuvs::distance::DistanceType`

_Source: `cpp/include/cuvs/neighbors/brute_force.hpp:132`_

### cuvs::neighbors::brute_force::metric_arg

Metric argument

```cpp
DistT metric_arg() const noexcept;
```

**Returns**

`DistT`

_Source: `cpp/include/cuvs/neighbors/brute_force.hpp:135`_

### cuvs::neighbors::brute_force::size

Total length of the index (number of vectors).

```cpp
size_t size() const noexcept;
```

**Returns**

`size_t`

_Source: `cpp/include/cuvs/neighbors/brute_force.hpp:138`_

### cuvs::neighbors::brute_force::dim

Dimensionality of the data.

```cpp
size_t dim() const noexcept;
```

**Returns**

`size_t`

_Source: `cpp/include/cuvs/neighbors/brute_force.hpp:141`_

### cuvs::neighbors::brute_force::dataset

Dataset [size, dim]

```cpp
raft::device_matrix_view<const T, int64_t, raft::row_major> dataset() const noexcept;
```

**Returns**

`raft::device_matrix_view<const T, int64_t, raft::row_major>`

_Source: `cpp/include/cuvs/neighbors/brute_force.hpp:144`_

### cuvs::neighbors::brute_force::norms

Dataset norms

```cpp
raft::device_vector_view<const DistT, int64_t, raft::row_major> norms() const;
```

**Returns**

`raft::device_vector_view<const DistT, int64_t, raft::row_major>`

_Source: `cpp/include/cuvs/neighbors/brute_force.hpp:150`_

### cuvs::neighbors::brute_force::has_norms

Whether ot not this index has dataset norms

```cpp
inline bool has_norms() const noexcept;
```

**Returns**

`inline bool`

_Source: `cpp/include/cuvs/neighbors/brute_force.hpp:156`_

## Bruteforce index build

_Doxygen group: `bruteforce_cpp_index_build`_

### cuvs::neighbors::brute_force::build

Build the index from the dataset for efficient search.

```cpp
auto build(raft::resources const& handle,
const cuvs::neighbors::brute_force::index_params& index_params,
raft::device_matrix_view<const float, int64_t, raft::row_major> dataset)
-> cuvs::neighbors::brute_force::index<float, float>;
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::brute_force::index_params&` | parameters such as the distance metric to use |
| `dataset` | in | `raft::device_matrix_view<const float, int64_t, raft::row_major>` | a device pointer to a row-major matrix [n_rows, dim] |

**Returns**

`cuvs::neighbors::brute_force::index<float, float>`

the constructed brute-force index

_Source: `cpp/include/cuvs/neighbors/brute_force.hpp:191`_

### cuvs::neighbors::brute_force::build

Build the index from the dataset for efficient search.

```cpp
auto build(raft::resources const& handle,
const cuvs::neighbors::brute_force::index_params& index_params,
raft::host_matrix_view<const float, int64_t, raft::row_major> dataset)
-> cuvs::neighbors::brute_force::index<float, float>;
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::brute_force::index_params&` | parameters such as the distance metric to use |
| `dataset` | in | `raft::host_matrix_view<const float, int64_t, raft::row_major>` | a host pointer to a row-major matrix [n_rows, dim] |

**Returns**

`cuvs::neighbors::brute_force::index<float, float>`

the constructed brute-force index

_Source: `cpp/include/cuvs/neighbors/brute_force.hpp:205`_

### cuvs::neighbors::brute_force::build

Build the index from the dataset for efficient search.

```cpp
auto build(raft::resources const& handle,
const cuvs::neighbors::brute_force::index_params& index_params,
raft::device_matrix_view<const half, int64_t, raft::row_major> dataset)
-> cuvs::neighbors::brute_force::index<half, float>;
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::brute_force::index_params&` | parameters such as the distance metric to use |
| `dataset` | in | `raft::device_matrix_view<const half, int64_t, raft::row_major>` | a device pointer to a row-major matrix [n_rows, dim] |

**Returns**

`cuvs::neighbors::brute_force::index<half, float>`

the constructed brute force index

_Source: `cpp/include/cuvs/neighbors/brute_force.hpp:232`_

### cuvs::neighbors::brute_force::build

Build the index from the dataset for efficient search.

```cpp
auto build(raft::resources const& handle,
const cuvs::neighbors::brute_force::index_params& index_params,
raft::host_matrix_view<const half, int64_t, raft::row_major> dataset)
-> cuvs::neighbors::brute_force::index<half, float>;
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::brute_force::index_params&` | parameters such as the distance metric to use |
| `dataset` | in | `raft::host_matrix_view<const half, int64_t, raft::row_major>` | a host pointer to a row-major matrix [n_rows, dim] |

**Returns**

`cuvs::neighbors::brute_force::index<half, float>`

the constructed brute-force index

_Source: `cpp/include/cuvs/neighbors/brute_force.hpp:246`_

### cuvs::neighbors::brute_force::build

Build the index from the dataset for efficient search.

```cpp
auto build(raft::resources const& handle,
const cuvs::neighbors::brute_force::index_params& index_params,
raft::device_matrix_view<const float, int64_t, raft::col_major> dataset)
-> cuvs::neighbors::brute_force::index<float, float>;
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::brute_force::index_params&` | parameters such as the distance metric to use |
| `dataset` | in | `raft::device_matrix_view<const float, int64_t, raft::col_major>` | a device pointer to a col-major matrix [n_rows, dim] |

**Returns**

`cuvs::neighbors::brute_force::index<float, float>`

the constructed brute force index

_Source: `cpp/include/cuvs/neighbors/brute_force.hpp:272`_

### cuvs::neighbors::brute_force::build

Build the index from the dataset for efficient search.

```cpp
auto build(raft::resources const& handle,
const cuvs::neighbors::brute_force::index_params& index_params,
raft::device_matrix_view<const half, int64_t, raft::col_major> dataset)
-> cuvs::neighbors::brute_force::index<half, float>;
```

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` |  |
| `index_params` | in | `const cuvs::neighbors::brute_force::index_params&` | parameters such as the distance metric to use |
| `dataset` | in | `raft::device_matrix_view<const half, int64_t, raft::col_major>` | a device pointer to a col-major matrix [n_rows, dim] |

**Returns**

`cuvs::neighbors::brute_force::index<half, float>`

the constructed brute force index

_Source: `cpp/include/cuvs/neighbors/brute_force.hpp:298`_

## Sparse Brute Force index

_Doxygen group: `sparse_bruteforce_cpp_index`_

### cuvs::neighbors::brute_force::sparse_index

Construct a sparse brute force sparse_index from dataset

```cpp
sparse_index(raft::resources const& res,
raft::device_csr_matrix_view<const T, IdxT, IdxT, IdxT> dataset,
cuvs::distance::DistanceType metric,
T metric_arg);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `dataset` |  | `raft::device_csr_matrix_view<const T, IdxT, IdxT, IdxT>` |  |
| `metric` |  | `cuvs::distance::DistanceType` |  |
| `metric_arg` |  | `T` |  |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/brute_force.hpp:609`_

### cuvs::neighbors::brute_force::metric

Distance metric used for retrieval

```cpp
cuvs::distance::DistanceType metric() const noexcept;
```

**Returns**

`cuvs::distance::DistanceType`

_Source: `cpp/include/cuvs/neighbors/brute_force.hpp:615`_

### cuvs::neighbors::brute_force::metric_arg

Metric argument

```cpp
T metric_arg() const noexcept;
```

**Returns**

`T`

_Source: `cpp/include/cuvs/neighbors/brute_force.hpp:618`_

## Sparse Brute Force index search

_Doxygen group: `sparse_bruteforce_cpp_index_search`_

### cuvs::neighbors::brute_force::sparse_search_params

Sparse Brute Force index search

```cpp
struct sparse_search_params { ... } ;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `batch_size_index` | `int` |  |
| `batch_size_query` | `int` |  |

_Source: `cpp/include/cuvs/neighbors/brute_force.hpp:668`_

## Bruteforce index serialize functions

_Doxygen group: `bruteforce_cpp_index_serialize`_

### cuvs::neighbors::brute_force::serialize

Save the index to file.

```cpp
void serialize(raft::resources const& handle,
const std::string& filename,
const cuvs::neighbors::brute_force::index<half, float>& index,
bool include_dataset = true);
```

The serialization format can be subject to changes, therefore loading an index saved with a previous version of cuvs is not guaranteed to work.

**Template Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `T` | `` | data element type |

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `filename` | in | `const std::string&` | the file name for saving the index |
| `index` | in | `const cuvs::neighbors::brute_force::index<half, float>&` | brute force index |
| `include_dataset` | in | `bool` | whether to include the dataset in the serialized Default: `true`. |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/brute_force.hpp:724`_

### cuvs::neighbors::brute_force::serialize

Save the index to file.

```cpp
void serialize(raft::resources const& handle,
const std::string& filename,
const cuvs::neighbors::brute_force::index<float, float>& index,
bool include_dataset = true);
```

The serialization format can be subject to changes, therefore loading an index saved with a previous version of cuvs is not guaranteed to work. output

**Template Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `T` | `` | data element type |

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `filename` | in | `const std::string&` | the file name for saving the index |
| `index` | in | `const cuvs::neighbors::brute_force::index<float, float>&` | brute force index |
| `include_dataset` | in | `bool` | whether to include the dataset in the serialized Default: `true`. |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/brute_force.hpp:755`_

### cuvs::neighbors::brute_force::serialize

Write the index to an output stream

```cpp
void serialize(raft::resources const& handle,
std::ostream& os,
const cuvs::neighbors::brute_force::index<half, float>& index,
bool include_dataset = true);
```

The serialization format can be subject to changes, therefore loading an index saved with a previous version of cuvs is not guaranteed to work.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `os` | in | `std::ostream&` | output stream |
| `index` | in | `const cuvs::neighbors::brute_force::index<half, float>&` | brute force index |
| `include_dataset` | in | `bool` | Whether or not to write out the dataset to the file. Default: `true`. |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/brute_force.hpp:783`_

### cuvs::neighbors::brute_force::serialize

Write the index to an output stream

```cpp
void serialize(raft::resources const& handle,
std::ostream& os,
const cuvs::neighbors::brute_force::index<float, float>& index,
bool include_dataset = true);
```

The serialization format can be subject to changes, therefore loading an index saved with a previous version of cuvs is not guaranteed to work.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `os` | in | `std::ostream&` | output stream |
| `index` | in | `const cuvs::neighbors::brute_force::index<float, float>&` | brute force index |
| `include_dataset` | in | `bool` | Whether or not to write out the dataset to the file. Default: `true`. |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/brute_force.hpp:811`_

### cuvs::neighbors::brute_force::deserialize

Load index from file.

```cpp
void deserialize(raft::resources const& handle,
const std::string& filename,
cuvs::neighbors::brute_force::index<half, float>* index);
```

The serialization format can be subject to changes, therefore loading an index saved with a previous version of cuvs is not guaranteed to work.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `filename` | in | `const std::string&` | the name of the file that stores the index |
| `index` | out | `cuvs::neighbors::brute_force::index<half, float>*` | brute force index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/brute_force.hpp:840`_

### cuvs::neighbors::brute_force::deserialize

Load index from file.

```cpp
void deserialize(raft::resources const& handle,
const std::string& filename,
cuvs::neighbors::brute_force::index<float, float>* index);
```

The serialization format can be subject to changes, therefore loading an index saved with a previous version of cuvs is not guaranteed to work.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `filename` | in | `const std::string&` | the name of the file that stores the index |
| `index` | out | `cuvs::neighbors::brute_force::index<float, float>*` | brute force index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/brute_force.hpp:867`_

### cuvs::neighbors::brute_force::deserialize

Load index from input stream

```cpp
void deserialize(raft::resources const& handle,
std::istream& is,
cuvs::neighbors::brute_force::index<half, float>* index);
```

The serialization format can be subject to changes, therefore loading an index saved with a previous version of cuvs is not guaranteed to work.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `is` | in | `std::istream&` | input stream |
| `index` | out | `cuvs::neighbors::brute_force::index<half, float>*` | brute force index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/brute_force.hpp:894`_

### cuvs::neighbors::brute_force::deserialize

Load index from input stream

```cpp
void deserialize(raft::resources const& handle,
std::istream& is,
cuvs::neighbors::brute_force::index<float, float>* index);
```

The serialization format can be subject to changes, therefore loading an index saved with a previous version of cuvs is not guaranteed to work.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `is` | in | `std::istream&` | input stream |
| `index` | out | `cuvs::neighbors::brute_force::index<float, float>*` | brute force index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/brute_force.hpp:921`_
