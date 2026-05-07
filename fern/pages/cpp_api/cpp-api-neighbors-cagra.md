---
slug: api-reference/cpp-api-neighbors-cagra
---

# Cagra

_Source header: `cpp/include/cuvs/neighbors/cagra.hpp`_

## Types

<a id="cuvs-neighbors-graph-build-params-ace-params"></a>
### cuvs::neighbors::graph_build_params::ace_params

Specialized parameters for ACE (Augmented Core Extraction) graph build

```cpp
struct ace_params { ... } ;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `npartitions` | `size_t` | Number of partitions for ACE (Augmented Core Extraction) partitioned build. When set to 0 (default), the number of partitions is automatically derived based on available host and GPU memory to maximize partition size while ensuring the build fits in memory. Small values might improve recall but potentially degrade performance and increase memory usage. Partitions should not be too small to prevent issues in KNN graph construction. The partition size is on average 2 * (n_rows / npartitions) * dim * sizeof(T). 2 is because of the core and augmented vectors. Please account for imbalance in the partition sizes (up to 3x in our tests). If the specified number of partitions results in partitions that exceed available memory, the value will be automatically increased to fit memory constraints and a warning will be issued. |
| `ef_construction` | `size_t` | The index quality for the ACE build. Bigger values increase the index quality. At some point, increasing this will no longer improve the quality. |
| `build_dir` | `std::string` | Directory to store ACE build artifacts (e.g., KNN graph, optimized graph). Used when `use_disk` is true or when the graph does not fit in host and GPU memory. This should be the fastest disk in the system and hold enough space for twice the dataset, final graph, and label mapping. |
| `use_disk` | `bool` | Whether to use disk-based storage for ACE build. When true, enables disk-based operations for memory-efficient graph construction. |
| `max_host_memory_gb` | `double` | Maximum host memory to use for ACE build in GiB. When set to 0 (default), uses available host memory. When set to a positive value, limits host memory usage to the specified amount. Useful for testing or when running alongside other memory-intensive processes. |
| `max_gpu_memory_gb` | `double` | Maximum GPU memory to use for ACE build in GiB. When set to 0 (default), uses available GPU memory. When set to a positive value, limits GPU memory usage to the specified amount. Useful for testing or when running alongside other memory-intensive processes. |

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:37`_

## CAGRA index build parameters

_Doxygen group: `cagra_cpp_index_params`_

<a id="cuvs-neighbors-vpq-params"></a>
### cuvs::neighbors::vpq_params

Parameters for VPQ compression.

```cpp
struct vpq_params { ... } ;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `pq_bits` | `uint32_t` | The bit length of the vector element after compression by PQ. Possible values: [4, 5, 6, 7, 8]. Hint: the smaller the 'pq_bits', the smaller the index size and the better the search performance, but the lower the recall. |
| `pq_dim` | `uint32_t` | The dimensionality of the vector after compression by PQ. When zero, an optimal value is selected using a heuristic. TODO: at the moment `dim` must be a multiple `pq_dim`. |
| `vq_n_centers` | `uint32_t` | Vector Quantization (VQ) codebook size - number of "coarse cluster centers". When zero, an optimal value is selected using a heuristic. |
| `kmeans_n_iters` | `uint32_t` | The number of iterations searching for kmeans centers (both VQ & PQ phases). |
| `vq_kmeans_trainset_fraction` | `double` | The fraction of data to use during iterative kmeans building (VQ phase). When zero, an optimal value is selected using a heuristic. |
| `pq_kmeans_trainset_fraction` | `double` | The fraction of data to use during iterative kmeans building (PQ phase). When zero, an optimal value is selected using a heuristic. |
| `pq_kmeans_type` | [`cuvs::cluster::kmeans::kmeans_type`](/api-reference/cpp-api-cluster-kmeans#cuvs-cluster-kmeans-kmeans-type) | Type of k-means algorithm for PQ training. Balanced k-means tends to be faster than regular k-means for PQ training, for problem sets where the number of points per cluster are approximately equal. Regular k-means may be better for skewed cluster distributions. |
| `max_train_points_per_pq_code` | `uint32_t` | The max number of data points to use per PQ code during PQ codebook training. Using more data points per PQ code may increase the quality of PQ codebook but may also increase the build time. We will use `pq_n_centers * max_train_points_per_pq_code` training points to train each PQ codebook. |
| `max_train_points_per_vq_cluster` | `uint32_t` | The max number of data points to use per VQ cluster during training. |

_Source: `cpp/include/cuvs/neighbors/common.hpp:42`_

<a id="cuvs-neighbors-cagra-hnsw-heuristic-type"></a>
### cuvs::neighbors::cagra::hnsw_heuristic_type

A strategy for selecting the graph build parameters based on similar HNSW index

parameters.

Define how `cagra::index_params::from_hnsw_params` should construct a graph to construct a graph that is to be converted to (used by) a CPU HNSW index.

```cpp
enum class hnsw_heuristic_type : uint32_t { ... } ;
```

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:115`_

<a id="cuvs-neighbors-cagra-from-hnsw-params"></a>
### cuvs::neighbors::cagra::from_hnsw_params

Create a CAGRA index parameters compatible with HNSW index

```cpp
static cagra::index_params from_hnsw_params(
raft::matrix_extent<int64_t> dataset,
int M,
int ef_construction,
hnsw_heuristic_type heuristic       = hnsw_heuristic_type::SIMILAR_SEARCH_PERFORMANCE,
cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded);
```

* IMPORTANT NOTE *

The reference HNSW index and the corresponding from-CAGRA generated HNSW index will NOT produce exactly the same recalls and QPS for the same parameter `ef`. The graphs are different internally. Depending on the selected heuristics, the CAGRA-produced graph's QPS-Recall curve may be shifted along the curve right or left. See the heuristics descriptions for more details.

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `dataset` |  | `raft::matrix_extent<int64_t>` | The shape of the input dataset |
| `M` |  | `int` | HNSW index parameter M |
| `ef_construction` |  | `int` | HNSW index parameter ef_construction |
| `heuristic` |  | [`hnsw_heuristic_type`](/api-reference/cpp-api-neighbors-cagra#cuvs-neighbors-cagra-hnsw-heuristic-type) | The heuristic to use for selecting the graph build parameters Default: `hnsw_heuristic_type::SIMILAR_SEARCH_PERFORMANCE`. |
| `metric` |  | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#cuvs-distance-distancetype) | The distance metric to search Default: `cuvs::distance::DistanceType::L2Expanded`. |

**Returns**

`static cagra::index_params`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:245`_

## CAGRA index search parameters

_Doxygen group: `cagra_cpp_search_params`_

<a id="cuvs-neighbors-cagra-search-algo"></a>
### cuvs::neighbors::cagra::search_algo

CAGRA index search parameters

```cpp
enum class search_algo { ... } ;
```

**Values**

| Name | Value |
| --- | --- |
| `SINGLE_CTA` | `0` |
| `MULTI_CTA` | `1` |
| `MULTI_KERNEL` | `2` |
| `AUTO` | `100` |

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:262`_

## CAGRA index extend parameters

_Doxygen group: `cagra_cpp_extend_params`_

<a id="cuvs-neighbors-cagra-extend-params"></a>
### cuvs::neighbors::cagra::extend_params

CAGRA index extend parameters

```cpp
struct extend_params { ... } ;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `max_chunk_size` | `uint32_t` | The additional dataset is divided into chunks and added to the graph. This is the knob to adjust the tradeoff between the recall and operation throughput. Large chunk sizes can result in high throughput, but use more working memory (O(max_chunk_size*degree^2)). This can also degrade recall because no edges are added between the nodes in the same chunk. Auto select when 0. |

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:357`_

## CAGRA index type

_Doxygen group: `cagra_cpp_index`_

<a id="cuvs-neighbors-cagra-metric"></a>
### cuvs::neighbors::cagra::metric

Distance metric used for clustering.

```cpp
[[nodiscard]] constexpr inline auto metric() const noexcept -> cuvs::distance::DistanceType;
```

**Returns**

[`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#cuvs-distance-distancetype)

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:401`_

<a id="cuvs-neighbors-cagra-size"></a>
### cuvs::neighbors::cagra::size

Total length of the index (number of vectors).

```cpp
[[nodiscard]] constexpr inline auto size() const noexcept -> IdxT;
```

**Returns**

`IdxT`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:407`_

<a id="cuvs-neighbors-cagra-dim"></a>
### cuvs::neighbors::cagra::dim

Dimensionality of the data.

```cpp
[[nodiscard]] constexpr inline auto dim() const noexcept -> uint32_t;
```

**Returns**

`uint32_t`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:415`_

<a id="cuvs-neighbors-cagra-graph-degree"></a>
### cuvs::neighbors::cagra::graph_degree

Graph degree

```cpp
[[nodiscard]] constexpr inline auto graph_degree() const noexcept -> uint32_t;
```

**Returns**

`uint32_t`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:420`_

<a id="cuvs-neighbors-cagra-data"></a>
### cuvs::neighbors::cagra::data

Dataset [size, dim]

```cpp
[[nodiscard]] inline auto data() const noexcept -> const cuvs::neighbors::dataset<int64_t>&;
```

**Returns**

`const cuvs::neighbors::dataset<int64_t>&`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:435`_

<a id="cuvs-neighbors-cagra-graph"></a>
### cuvs::neighbors::cagra::graph

neighborhood graph [size, graph-degree]

```cpp
[[nodiscard]] inline auto graph() const noexcept
-> raft::device_matrix_view<const graph_index_type, int64_t, raft::row_major>;
```

**Returns**

`raft::device_matrix_view<const graph_index_type, int64_t, raft::row_major>`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:441`_

<a id="cuvs-neighbors-cagra-source-indices"></a>
### cuvs::neighbors::cagra::source_indices

Mapping from internal graph node indices to the original user-provided indices.

```cpp
[[nodiscard]] inline auto source_indices() const noexcept
-> std::optional<raft::device_vector_view<const index_type, int64_t>>;
```

**Returns**

`std::optional<raft::device_vector_view<const index_type, int64_t>>`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:448`_

<a id="cuvs-neighbors-cagra-dataset-fd"></a>
### cuvs::neighbors::cagra::dataset_fd

Get the dataset file descriptor (for disk-backed index)

```cpp
[[nodiscard]] inline auto dataset_fd() const noexcept
-> const std::optional<cuvs::util::file_descriptor>&;
```

**Returns**

[`const std::optional<cuvs::util::file_descriptor>&`](/api-reference/cpp-api-util-file-io#cuvs-util-file-descriptor)

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:458`_

<a id="cuvs-neighbors-cagra-graph-fd"></a>
### cuvs::neighbors::cagra::graph_fd

Get the graph file descriptor (for disk-backed index)

```cpp
[[nodiscard]] inline auto graph_fd() const noexcept
-> const std::optional<cuvs::util::file_descriptor>&;
```

**Returns**

[`const std::optional<cuvs::util::file_descriptor>&`](/api-reference/cpp-api-util-file-io#cuvs-util-file-descriptor)

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:465`_

<a id="cuvs-neighbors-cagra-mapping-fd"></a>
### cuvs::neighbors::cagra::mapping_fd

Get the mapping file descriptor (for disk-backed index)

```cpp
[[nodiscard]] inline auto mapping_fd() const noexcept
-> const std::optional<cuvs::util::file_descriptor>&;
```

**Returns**

[`const std::optional<cuvs::util::file_descriptor>&`](/api-reference/cpp-api-util-file-io#cuvs-util-file-descriptor)

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:472`_

<a id="cuvs-neighbors-cagra-dataset-norms"></a>
### cuvs::neighbors::cagra::dataset_norms

Dataset norms for cosine distance [size]

```cpp
[[nodiscard]] inline auto dataset_norms() const noexcept
-> std::optional<raft::device_vector_view<const float, int64_t>>;
```

**Returns**

`std::optional<raft::device_vector_view<const float, int64_t>>`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:479`_

<a id="cuvs-neighbors-cagra-index"></a>
### cuvs::neighbors::cagra::index

```cpp
index(const index&)                    = delete;
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `arg1` |  | `const index&` |  |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:488`_

**Additional overload:** `cuvs::neighbors::cagra::index`

Construct an empty index.

```cpp
index(raft::resources const& res,
cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded)
: cuvs::neighbors::index(),
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `metric` |  | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#cuvs-distance-distancetype) | Default: `cuvs::distance::DistanceType::L2Expanded`. |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:496`_

**Additional overload:** `cuvs::neighbors::cagra::index`

Construct an index from dataset and knn_graph arrays

```cpp
template <typename data_accessor, typename graph_accessor>
index(raft::resources const& res,
cuvs::distance::DistanceType metric,
raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, data_accessor> dataset,
raft::mdspan<const graph_index_type,
raft::matrix_extent<int64_t>,
raft::row_major,
graph_accessor> knn_graph)
: cuvs::neighbors::index(),
```

If the dataset and graph is already in GPU memory, then the index is just a thin wrapper around these that stores a non-owning a reference to the arrays.

The constructor also accepts host arrays. In that case they are copied to the device, and the device arrays will be owned by the index.

In case the dasates rows are not 16 bytes aligned, then we create a padded copy in device memory to ensure alignment for vectorized load.

Usage examples:

- Cagra index is normally created by the cagra::build In the above example, we have passed a host dataset to build. The returned index will own a device copy of the dataset and the knn_graph. In contrast, if we pass the dataset as a device_mdspan to build, then it will only store a reference to it.

- Constructing index using existing knn-graph

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `metric` |  | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#cuvs-distance-distancetype) |  |
| `dataset` |  | `raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, data_accessor>` |  |
| `knn_graph` |  | `raft::mdspan<const graph_index_type, raft::matrix_extent<int64_t>, raft::row_major, graph_accessor>` |  |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:559`_

<a id="cuvs-neighbors-cagra-update-dataset"></a>
### cuvs::neighbors::cagra::update_dataset

Replace the dataset with a new dataset.

```cpp
void update_dataset(raft::resources const& res,
raft::device_matrix_view<const T, int64_t, raft::row_major> dataset);
```

If the new dataset rows are aligned on 16 bytes, then only a reference is stored to the dataset. It is the caller's responsibility to ensure that dataset stays alive as long as the index. It is expected that the same set of vectors are used for update_dataset and index build.

Note: This will clear any precomputed dataset norms.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `dataset` |  | `raft::device_matrix_view<const T, int64_t, raft::row_major>` |  |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:597`_

**Additional overload:** `cuvs::neighbors::cagra::update_dataset`

Set the dataset reference explicitly to a device matrix view with padding.

```cpp
void update_dataset(raft::resources const& res,
raft::device_matrix_view<const T, int64_t, raft::layout_stride> dataset);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `dataset` |  | `raft::device_matrix_view<const T, int64_t, raft::layout_stride>` |  |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:609`_

**Additional overload:** `cuvs::neighbors::cagra::update_dataset`

Replace the dataset with a new dataset.

```cpp
void update_dataset(raft::resources const& res,
raft::host_matrix_view<const T, int64_t, raft::row_major> dataset);
```

We create a copy of the dataset on the device. The index manages the lifetime of this copy. It is expected that the same set of vectors are used for update_dataset and index build.

Note: This will clear any precomputed dataset norms.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `dataset` |  | `raft::host_matrix_view<const T, int64_t, raft::row_major>` |  |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:628`_

**Additional overload:** `cuvs::neighbors::cagra::update_dataset`

Replace the dataset with a new dataset. It is expected that the same set of vectors are used

```cpp
template <typename DatasetT>
auto update_dataset(raft::resources const& res, DatasetT&& dataset)
-> std::enable_if_t<std::is_base_of_v<cuvs::neighbors::dataset<dataset_index_type>, DatasetT>>;
```

for update_dataset and index build.

Note: This will clear any precomputed dataset norms.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `dataset` |  | `DatasetT&&` |  |

**Returns**

`std::enable_if_t<std::is_base_of_v<cuvs::neighbors::dataset<dataset_index_type>, DatasetT>>`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:644`_

<a id="cuvs-neighbors-cagra-update-graph"></a>
### cuvs::neighbors::cagra::update_graph

Replace the graph with a new graph.

```cpp
void update_graph(
raft::resources const& res,
raft::device_matrix_view<const graph_index_type, int64_t, raft::row_major> knn_graph);
```

Since the new graph is a device array, we store a reference to that, and it is the caller's responsibility to ensure that knn_graph stays alive as long as the index.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `knn_graph` |  | `raft::device_matrix_view<const graph_index_type, int64_t, raft::row_major>` |  |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:677`_

**Additional overload:** `cuvs::neighbors::cagra::update_graph`

Replace the graph with a new graph.

```cpp
void update_graph(
raft::resources const& res,
raft::host_matrix_view<const graph_index_type, int64_t, raft::row_major> knn_graph);
```

We create a copy of the graph on the device. The index manages the lifetime of this copy.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `knn_graph` |  | `raft::host_matrix_view<const graph_index_type, int64_t, raft::row_major>` |  |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:689`_

<a id="cuvs-neighbors-cagra-update-source-indices"></a>
### cuvs::neighbors::cagra::update_source_indices

Replace the source indices with a new source indices taking the ownership of the passed vector.

```cpp
void update_source_indices(raft::device_vector<index_type, int64_t>&& source_indices);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `source_indices` |  | `raft::device_vector<index_type, int64_t>&&` |  |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:713`_

**Additional overload:** `cuvs::neighbors::cagra::update_source_indices`

Copy the provided source indices into the index.

```cpp
template <typename Accessor>
void update_source_indices(
raft::resources const& res,
raft::mdspan<const index_type, raft::vector_extent<int64_t>, raft::row_major, Accessor>
source_indices);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `source_indices` |  | `raft::mdspan<const index_type, raft::vector_extent<int64_t>, raft::row_major, Accessor>` |  |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:723`_

**Additional overload:** `cuvs::neighbors::cagra::update_dataset`

Update the dataset from a disk file using a file descriptor.

```cpp
void update_dataset(raft::resources const& res, cuvs::util::file_descriptor&& fd);
```

This method configures the index to use a disk-based dataset. The dataset file should contain a numpy header followed by vectors in row-major format. The number of rows and dimensionality are read from the numpy header.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `fd` | in | [`cuvs::util::file_descriptor&&`](/api-reference/cpp-api-util-file-io#cuvs-util-file-descriptor) | File descriptor (will be moved into the index for lifetime management) |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:759`_

**Additional overload:** `cuvs::neighbors::cagra::update_graph`

Update the graph from a disk file using a file descriptor.

```cpp
void update_graph(raft::resources const& res, cuvs::util::file_descriptor&& fd);
```

This method configures the index to use a disk-based graph. The graph file should contain a numpy header followed by neighbor indices in row-major format. The number of rows and graph degree are read from the numpy header.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `fd` | in | [`cuvs::util::file_descriptor&&`](/api-reference/cpp-api-util-file-io#cuvs-util-file-descriptor) | File descriptor (will be moved into the index for lifetime management) |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:794`_

<a id="cuvs-neighbors-cagra-update-mapping"></a>
### cuvs::neighbors::cagra::update_mapping

Update the dataset mapping from a disk file using a file descriptor.

```cpp
void update_mapping(raft::resources const& res, cuvs::util::file_descriptor&& fd);
```

This method configures the index to use a disk-based dataset mapping. The mapping file should contain a numpy header followed by index mappings.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `fd` | in | [`cuvs::util::file_descriptor&&`](/api-reference/cpp-api-util-file-io#cuvs-util-file-descriptor) | File descriptor (will be moved into the index for lifetime management) |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:834`_

## CAGRA index build functions

_Doxygen group: `cagra_cpp_index_build`_

<a id="cuvs-neighbors-cagra-build"></a>
### cuvs::neighbors::cagra::build

Build the index from the dataset for efficient search.

```cpp
auto build(raft::resources const& res,
const cuvs::neighbors::cagra::index_params& params,
raft::device_matrix_view<const float, int64_t, raft::row_major> dataset)
-> cuvs::neighbors::cagra::index<float, uint32_t>;
```

The build consist of two steps: build an intermediate knn-graph, and optimize it to create the final graph. The index_params struct controls the node degree of these graphs.

The following distance metrics are supported:

- L2
- InnerProduct (currently only supported with IVF-PQ as the build algorithm)
- CosineExpanded
- L1 (currently only supported with NN-Descent and Iterative Search as the build algorithm)

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` |  |
| `params` | in | `const cuvs::neighbors::cagra::index_params&` | parameters for building the index |
| `dataset` | in | `raft::device_matrix_view<const float, int64_t, raft::row_major>` | a matrix view (device) to a row-major matrix [n_rows, dim] |

**Returns**

`cuvs::neighbors::cagra::index<float, uint32_t>`

the constructed cagra index

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:923`_

**Additional overload:** `cuvs::neighbors::cagra::build`

Build the index from the dataset for efficient search.

```cpp
auto build(raft::resources const& res,
const cuvs::neighbors::cagra::index_params& params,
raft::host_matrix_view<const float, int64_t, raft::row_major> dataset)
-> cuvs::neighbors::cagra::index<float, uint32_t>;
```

The build consist of two steps: build an intermediate knn-graph, and optimize it to create the final graph. The index_params struct controls the node degree of these graphs.

The following distance metrics are supported:

- L2
- InnerProduct (currently only supported with IVF-PQ as the build algorithm)
- CosineExpanded
- L1 (currently only supported with NN-Descent and Iterative Search as the build algorithm)

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` |  |
| `params` | in | `const cuvs::neighbors::cagra::index_params&` | parameters for building the index |
| `dataset` | in | `raft::host_matrix_view<const float, int64_t, raft::row_major>` | a matrix view (host) to a row-major matrix [n_rows, dim] |

**Returns**

`cuvs::neighbors::cagra::index<float, uint32_t>`

the constructed cagra index

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:962`_

**Additional overload:** `cuvs::neighbors::cagra::build`

Build the index from the dataset for efficient search.

```cpp
auto build(raft::resources const& res,
const cuvs::neighbors::cagra::index_params& params,
raft::device_matrix_view<const half, int64_t, raft::row_major> dataset)
-> cuvs::neighbors::cagra::index<half, uint32_t>;
```

The build consist of two steps: build an intermediate knn-graph, and optimize it to create the final graph. The index_params struct controls the node degree of these graphs.

The following distance metrics are supported:

- L2
- InnerProduct (currently only supported with IVF-PQ as the build algorithm)
- CosineExpanded (dataset norms are computed as float regardless of input data type)
- L1 (currently only supported with NN-Descent and Iterative Search as the build algorithm)

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` |  |
| `params` | in | `const cuvs::neighbors::cagra::index_params&` | parameters for building the index |
| `dataset` | in | `raft::device_matrix_view<const half, int64_t, raft::row_major>` | a matrix view (device) to a row-major matrix [n_rows, dim] |

**Returns**

`cuvs::neighbors::cagra::index<half, uint32_t>`

the constructed cagra index

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:1001`_

**Additional overload:** `cuvs::neighbors::cagra::build`

Build the index from the dataset for efficient search.

```cpp
auto build(raft::resources const& res,
const cuvs::neighbors::cagra::index_params& params,
raft::host_matrix_view<const half, int64_t, raft::row_major> dataset)
-> cuvs::neighbors::cagra::index<half, uint32_t>;
```

The build consist of two steps: build an intermediate knn-graph, and optimize it to create the final graph. The index_params struct controls the node degree of these graphs.

The following distance metrics are supported:

- L2
- CosineExpanded (dataset norms are computed as float regardless of input data type)
- L1 (currently only supported with NN-Descent and Iterative Search as the build algorithm)

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` |  |
| `params` | in | `const cuvs::neighbors::cagra::index_params&` | parameters for building the index |
| `dataset` | in | `raft::host_matrix_view<const half, int64_t, raft::row_major>` | a matrix view (host) to a row-major matrix [n_rows, dim] |

**Returns**

`cuvs::neighbors::cagra::index<half, uint32_t>`

the constructed cagra index

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:1039`_

**Additional overload:** `cuvs::neighbors::cagra::build`

Build the index from the dataset for efficient search.

```cpp
auto build(raft::resources const& res,
const cuvs::neighbors::cagra::index_params& params,
raft::device_matrix_view<const int8_t, int64_t, raft::row_major> dataset)
-> cuvs::neighbors::cagra::index<int8_t, uint32_t>;
```

The build consist of two steps: build an intermediate knn-graph, and optimize it to create the final graph. The index_params struct controls the node degree of these graphs.

The following distance metrics are supported:

- L2
- CosineExpanded (dataset norms are computed as float regardless of input data type)
- L1 (currently only supported with NN-Descent and Iterative Search as the build algorithm)
- BitwiseHamming (currently only supported with NN-Descent and Iterative Search as the build algorithm, and only for int8_t and uint8_t data types)

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` |  |
| `params` | in | `const cuvs::neighbors::cagra::index_params&` | parameters for building the index |
| `dataset` | in | `raft::device_matrix_view<const int8_t, int64_t, raft::row_major>` | a matrix view (device) to a row-major matrix [n_rows, dim] |

**Returns**

`cuvs::neighbors::cagra::index<int8_t, uint32_t>`

the constructed cagra index

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:1079`_

**Additional overload:** `cuvs::neighbors::cagra::build`

Build the index from the dataset for efficient search.

```cpp
auto build(raft::resources const& res,
const cuvs::neighbors::cagra::index_params& params,
raft::host_matrix_view<const int8_t, int64_t, raft::row_major> dataset)
-> cuvs::neighbors::cagra::index<int8_t, uint32_t>;
```

The build consist of two steps: build an intermediate knn-graph, and optimize it to create the final graph. The index_params struct controls the node degree of these graphs.

The following distance metrics are supported:

- L2
- InnerProduct (currently only supported with IVF-PQ as the build algorithm)
- CosineExpanded (dataset norms are computed as float regardless of input data type)
- L1 (currently only supported with NN-Descent and Iterative Search as the build algorithm)
- BitwiseHamming (currently only supported with NN-Descent and Iterative Search as the build algorithm, and only for int8_t and uint8_t data types)

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` |  |
| `params` | in | `const cuvs::neighbors::cagra::index_params&` | parameters for building the index |
| `dataset` | in | `raft::host_matrix_view<const int8_t, int64_t, raft::row_major>` | a matrix view (host) to a row-major matrix [n_rows, dim] |

**Returns**

`cuvs::neighbors::cagra::index<int8_t, uint32_t>`

the constructed cagra index

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:1120`_

**Additional overload:** `cuvs::neighbors::cagra::build`

Build the index from the dataset for efficient search.

```cpp
auto build(raft::resources const& res,
const cuvs::neighbors::cagra::index_params& params,
raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> dataset)
-> cuvs::neighbors::cagra::index<uint8_t, uint32_t>;
```

The build consist of two steps: build an intermediate knn-graph, and optimize it to create the final graph. The index_params struct controls the node degree of these graphs.

The following distance metrics are supported:

- L2
- InnerProduct (currently only supported with IVF-PQ as the build algorithm)
- CosineExpanded (dataset norms are computed as float regardless of input data type)
- L1 (currently only supported with NN-Descent and Iterative Search as the build algorithm)
- BitwiseHamming (currently only supported with NN-Descent and Iterative Search as the build algorithm, and only for int8_t and uint8_t data types)

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` |  |
| `params` | in | `const cuvs::neighbors::cagra::index_params&` | parameters for building the index |
| `dataset` | in | `raft::device_matrix_view<const uint8_t, int64_t, raft::row_major>` | a matrix view (device) to a row-major matrix [n_rows, dim] |

**Returns**

`cuvs::neighbors::cagra::index<uint8_t, uint32_t>`

the constructed cagra index

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:1161`_

**Additional overload:** `cuvs::neighbors::cagra::build`

Build the index from the dataset for efficient search.

```cpp
auto build(raft::resources const& res,
const cuvs::neighbors::cagra::index_params& params,
raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> dataset)
-> cuvs::neighbors::cagra::index<uint8_t, uint32_t>;
```

The build consist of two steps: build an intermediate knn-graph, and optimize it to create the final graph. The index_params struct controls the node degree of these graphs.

The following distance metrics are supported:

- L2
- InnerProduct (currently only supported with IVF-PQ as the build algorithm)
- CosineExpanded (dataset norms are computed as float regardless of input data type)
- L1 (currently only supported with NN-Descent and Iterative Search as the build algorithm)
- BitwiseHamming (currently only supported with NN-Descent and Iterative Search as the build algorithm, and only for int8_t and uint8_t data types)

Usage example:

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` |  |
| `params` | in | `const cuvs::neighbors::cagra::index_params&` | parameters for building the index |
| `dataset` | in | `raft::host_matrix_view<const uint8_t, int64_t, raft::row_major>` | a matrix view (host) to a row-major matrix [n_rows, dim] |

**Returns**

`cuvs::neighbors::cagra::index<uint8_t, uint32_t>`

the constructed cagra index

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:1202`_

## CAGRA extend functions

_Doxygen group: `cagra_cpp_index_extend`_

<a id="cuvs-neighbors-cagra-extend"></a>
### cuvs::neighbors::cagra::extend

Add new vectors to a CAGRA index

```cpp
void extend(
raft::resources const& handle,
const cagra::extend_params& params,
raft::device_matrix_view<const float, int64_t, raft::row_major> additional_dataset,
cuvs::neighbors::cagra::index<float, uint32_t>& idx,
std::optional<raft::device_matrix_view<float, int64_t, raft::layout_stride>>
new_dataset_buffer_view                                                        = std::nullopt,
std::optional<raft::device_matrix_view<uint32_t, int64_t>> new_graph_buffer_view = std::nullopt);
```

Usage example:

part. The data will be copied from the current index in this function. The num rows must be the sum of the original and additional datasets, cols must be the dimension of the dataset, and the stride must be the same as the original index dataset. This view will be stored in the output index. It is the caller's responsibility to ensure that dataset stays alive as long as the index. This option is useful when users want to manage the memory space for the dataset themselves. The data will be copied from the current index in this function. The num rows must be the sum of the original and additional datasets and cols must be the graph degree. This view will be stored in the output index. It is the caller's responsibility to ensure that dataset stays alive as long as the index. This option is useful when users want to manage the memory space for the graph themselves.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | raft resources |
| `params` | in | [`const cagra::extend_params&`](/api-reference/cpp-api-neighbors-cagra#cuvs-neighbors-cagra-extend-params) | extend params |
| `additional_dataset` | in | `raft::device_matrix_view<const float, int64_t, raft::row_major>` | additional dataset on device memory |
| `idx` | in,out | `cuvs::neighbors::cagra::index<float, uint32_t>&` | CAGRA index |
| `new_dataset_buffer_view` | out | `std::optional<raft::device_matrix_view<float, int64_t, raft::layout_stride>>` | memory buffer view for the dataset including the additional Default: `std::nullopt`. |
| `new_graph_buffer_view` | out | `std::optional<raft::device_matrix_view<uint32_t, int64_t>>` | memory buffer view for the graph including the additional part. Default: `std::nullopt`. |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:1244`_

**Additional overload:** `cuvs::neighbors::cagra::extend`

Add new vectors to a CAGRA index

```cpp
void extend(
raft::resources const& handle,
const cagra::extend_params& params,
raft::host_matrix_view<const float, int64_t, raft::row_major> additional_dataset,
cuvs::neighbors::cagra::index<float, uint32_t>& idx,
std::optional<raft::device_matrix_view<float, int64_t, raft::layout_stride>>
new_dataset_buffer_view                                                        = std::nullopt,
std::optional<raft::device_matrix_view<uint32_t, int64_t>> new_graph_buffer_view = std::nullopt);
```

Usage example:

part. The data will be copied from the current index in this function. The num rows must be the sum of the original and additional datasets, cols must be the dimension of the dataset, and the stride must be the same as the original index dataset. This view will be stored in the output index. It is the caller's responsibility to ensure that dataset stays alive as long as the index. This option is useful when users want to manage the memory space for the dataset themselves. The data will be copied from the current index in this function. The num rows must be the sum of the original and additional datasets and cols must be the graph degree. This view will be stored in the output index. It is the caller's responsibility to ensure that dataset stays alive as long as the index. This option is useful when users want to manage the memory space for the graph themselves.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | raft resources |
| `params` | in | [`const cagra::extend_params&`](/api-reference/cpp-api-neighbors-cagra#cuvs-neighbors-cagra-extend-params) | extend params |
| `additional_dataset` | in | `raft::host_matrix_view<const float, int64_t, raft::row_major>` | additional dataset on host memory |
| `idx` | in,out | `cuvs::neighbors::cagra::index<float, uint32_t>&` | CAGRA index |
| `new_dataset_buffer_view` | out | `std::optional<raft::device_matrix_view<float, int64_t, raft::layout_stride>>` | memory buffer view for the dataset including the additional Default: `std::nullopt`. |
| `new_graph_buffer_view` | out | `std::optional<raft::device_matrix_view<uint32_t, int64_t>>` | memory buffer view for the graph including the additional part. Default: `std::nullopt`. |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:1282`_

**Additional overload:** `cuvs::neighbors::cagra::extend`

Add new vectors to a CAGRA index

```cpp
void extend(
raft::resources const& handle,
const cagra::extend_params& params,
raft::device_matrix_view<const half, int64_t, raft::row_major> additional_dataset,
cuvs::neighbors::cagra::index<half, uint32_t>& idx,
std::optional<raft::device_matrix_view<half, int64_t, raft::layout_stride>>
new_dataset_buffer_view                                                        = std::nullopt,
std::optional<raft::device_matrix_view<uint32_t, int64_t>> new_graph_buffer_view = std::nullopt);
```

Usage example:

part. The data will be copied from the current index in this function. The num rows must be the sum of the original and additional datasets, cols must be the dimension of the dataset, and the stride must be the same as the original index dataset. This view will be stored in the output index. It is the caller's responsibility to ensure that dataset stays alive as long as the index. This option is useful when users want to manage the memory space for the dataset themselves. The data will be copied from the current index in this function. The num rows must be the sum of the original and additional datasets and cols must be the graph degree. This view will be stored in the output index. It is the caller's responsibility to ensure that dataset stays alive as long as the index. This option is useful when users want to manage the memory space for the graph themselves.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | raft resources |
| `params` | in | [`const cagra::extend_params&`](/api-reference/cpp-api-neighbors-cagra#cuvs-neighbors-cagra-extend-params) | extend params |
| `additional_dataset` | in | `raft::device_matrix_view<const half, int64_t, raft::row_major>` | additional dataset on device memory |
| `idx` | in,out | `cuvs::neighbors::cagra::index<half, uint32_t>&` | CAGRA index |
| `new_dataset_buffer_view` | out | `std::optional<raft::device_matrix_view<half, int64_t, raft::layout_stride>>` | memory buffer view for the dataset including the additional Default: `std::nullopt`. |
| `new_graph_buffer_view` | out | `std::optional<raft::device_matrix_view<uint32_t, int64_t>>` | memory buffer view for the graph including the additional part. Default: `std::nullopt`. |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:1320`_

**Additional overload:** `cuvs::neighbors::cagra::extend`

Add new vectors to a CAGRA index

```cpp
void extend(
raft::resources const& handle,
const cagra::extend_params& params,
raft::host_matrix_view<const half, int64_t, raft::row_major> additional_dataset,
cuvs::neighbors::cagra::index<half, uint32_t>& idx,
std::optional<raft::device_matrix_view<half, int64_t, raft::layout_stride>>
new_dataset_buffer_view                                                        = std::nullopt,
std::optional<raft::device_matrix_view<uint32_t, int64_t>> new_graph_buffer_view = std::nullopt);
```

Usage example:

part. The data will be copied from the current index in this function. The num rows must be the sum of the original and additional datasets, cols must be the dimension of the dataset, and the stride must be the same as the original index dataset. This view will be stored in the output index. It is the caller's responsibility to ensure that dataset stays alive as long as the index. This option is useful when users want to manage the memory space for the dataset themselves. The data will be copied from the current index in this function. The num rows must be the sum of the original and additional datasets and cols must be the graph degree. This view will be stored in the output index. It is the caller's responsibility to ensure that dataset stays alive as long as the index. This option is useful when users want to manage the memory space for the graph themselves.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | raft resources |
| `params` | in | [`const cagra::extend_params&`](/api-reference/cpp-api-neighbors-cagra#cuvs-neighbors-cagra-extend-params) | extend params |
| `additional_dataset` | in | `raft::host_matrix_view<const half, int64_t, raft::row_major>` | additional dataset on host memory |
| `idx` | in,out | `cuvs::neighbors::cagra::index<half, uint32_t>&` | CAGRA index |
| `new_dataset_buffer_view` | out | `std::optional<raft::device_matrix_view<half, int64_t, raft::layout_stride>>` | memory buffer view for the dataset including the additional Default: `std::nullopt`. |
| `new_graph_buffer_view` | out | `std::optional<raft::device_matrix_view<uint32_t, int64_t>>` | memory buffer view for the graph including the additional part. Default: `std::nullopt`. |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:1358`_

**Additional overload:** `cuvs::neighbors::cagra::extend`

Add new vectors to a CAGRA index

```cpp
void extend(
raft::resources const& handle,
const cagra::extend_params& params,
raft::device_matrix_view<const int8_t, int64_t, raft::row_major> additional_dataset,
cuvs::neighbors::cagra::index<int8_t, uint32_t>& idx,
std::optional<raft::device_matrix_view<int8_t, int64_t, raft::layout_stride>>
new_dataset_buffer_view                                                        = std::nullopt,
std::optional<raft::device_matrix_view<uint32_t, int64_t>> new_graph_buffer_view = std::nullopt);
```

Usage example:

part. The data will be copied from the current index in this function. The num rows must be the sum of the original and additional datasets, cols must be the dimension of the dataset, and the stride must be the same as the original index dataset. This view will be stored in the output index. It is the caller's responsibility to ensure that dataset stays alive as long as the index. This option is useful when users want to manage the memory space for the dataset themselves. The data will be copied from the current index in this function. The num rows must be the sum of the original and additional datasets and cols must be the graph degree. This view will be stored in the output index. It is the caller's responsibility to ensure that dataset stays alive as long as the index. This option is useful when users want to manage the memory space for the graph themselves.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | raft resources |
| `params` | in | [`const cagra::extend_params&`](/api-reference/cpp-api-neighbors-cagra#cuvs-neighbors-cagra-extend-params) | extend params |
| `additional_dataset` | in | `raft::device_matrix_view<const int8_t, int64_t, raft::row_major>` | additional dataset on device memory |
| `idx` | in,out | `cuvs::neighbors::cagra::index<int8_t, uint32_t>&` | CAGRA index |
| `new_dataset_buffer_view` | out | `std::optional<raft::device_matrix_view<int8_t, int64_t, raft::layout_stride>>` | memory buffer view for the dataset including the additional Default: `std::nullopt`. |
| `new_graph_buffer_view` | out | `std::optional<raft::device_matrix_view<uint32_t, int64_t>>` | memory buffer view for the graph including the additional part. Default: `std::nullopt`. |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:1396`_

**Additional overload:** `cuvs::neighbors::cagra::extend`

Add new vectors to a CAGRA index

```cpp
void extend(
raft::resources const& handle,
const cagra::extend_params& params,
raft::host_matrix_view<const int8_t, int64_t, raft::row_major> additional_dataset,
cuvs::neighbors::cagra::index<int8_t, uint32_t>& idx,
std::optional<raft::device_matrix_view<int8_t, int64_t, raft::layout_stride>>
new_dataset_buffer_view                                                        = std::nullopt,
std::optional<raft::device_matrix_view<uint32_t, int64_t>> new_graph_buffer_view = std::nullopt);
```

Usage example:

part. The data will be copied from the current index in this function. The num rows must be the sum of the original and additional datasets, cols must be the dimension of the dataset, and the stride must be the same as the original index dataset. This view will be stored in the output index. It is the caller's responsibility to ensure that dataset stays alive as long as the index. This option is useful when users want to manage the memory space for the dataset themselves. The data will be copied from the current index in this function. The num rows must be the sum of the original and additional datasets and cols must be the graph degree. This view will be stored in the output index. It is the caller's responsibility to ensure that dataset stays alive as long as the index. This option is useful when users want to manage the memory space for the graph themselves.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | raft resources |
| `params` | in | [`const cagra::extend_params&`](/api-reference/cpp-api-neighbors-cagra#cuvs-neighbors-cagra-extend-params) | extend params |
| `additional_dataset` | in | `raft::host_matrix_view<const int8_t, int64_t, raft::row_major>` | additional dataset on host memory |
| `idx` | in,out | `cuvs::neighbors::cagra::index<int8_t, uint32_t>&` | CAGRA index |
| `new_dataset_buffer_view` | out | `std::optional<raft::device_matrix_view<int8_t, int64_t, raft::layout_stride>>` | memory buffer view for the dataset including the additional Default: `std::nullopt`. |
| `new_graph_buffer_view` | out | `std::optional<raft::device_matrix_view<uint32_t, int64_t>>` | memory buffer view for the graph including the additional part. Default: `std::nullopt`. |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:1434`_

**Additional overload:** `cuvs::neighbors::cagra::extend`

Add new vectors to a CAGRA index

```cpp
void extend(
raft::resources const& handle,
const cagra::extend_params& params,
raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> additional_dataset,
cuvs::neighbors::cagra::index<uint8_t, uint32_t>& idx,
std::optional<raft::device_matrix_view<uint8_t, int64_t, raft::layout_stride>>
new_dataset_buffer_view                                                        = std::nullopt,
std::optional<raft::device_matrix_view<uint32_t, int64_t>> new_graph_buffer_view = std::nullopt);
```

Usage example:

part. The data will be copied from the current index in this function. The num rows must be the sum of the original and additional datasets, cols must be the dimension of the dataset, and the stride must be the same as the original index dataset. This view will be stored in the output index. It is the caller's responsibility to ensure that dataset stays alive as long as the index. This option is useful when users want to manage the memory space for the dataset themselves. The data will be copied from the current index in this function. The num rows must be the sum of the original and additional datasets and cols must be the graph degree. This view will be stored in the output index. It is the caller's responsibility to ensure that dataset stays alive as long as the index. This option is useful when users want to manage the memory space for the graph themselves.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | raft resources |
| `params` | in | [`const cagra::extend_params&`](/api-reference/cpp-api-neighbors-cagra#cuvs-neighbors-cagra-extend-params) | extend params |
| `additional_dataset` | in | `raft::device_matrix_view<const uint8_t, int64_t, raft::row_major>` | additional dataset on host memory |
| `idx` | in,out | `cuvs::neighbors::cagra::index<uint8_t, uint32_t>&` | CAGRA index |
| `new_dataset_buffer_view` | out | `std::optional<raft::device_matrix_view<uint8_t, int64_t, raft::layout_stride>>` | memory buffer view for the dataset including the additional Default: `std::nullopt`. |
| `new_graph_buffer_view` | out | `std::optional<raft::device_matrix_view<uint32_t, int64_t>>` | memory buffer view for the graph including the additional part. Default: `std::nullopt`. |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:1472`_

**Additional overload:** `cuvs::neighbors::cagra::extend`

Add new vectors to a CAGRA index

```cpp
void extend(
raft::resources const& handle,
const cagra::extend_params& params,
raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> additional_dataset,
cuvs::neighbors::cagra::index<uint8_t, uint32_t>& idx,
std::optional<raft::device_matrix_view<uint8_t, int64_t, raft::layout_stride>>
new_dataset_buffer_view                                                        = std::nullopt,
std::optional<raft::device_matrix_view<uint32_t, int64_t>> new_graph_buffer_view = std::nullopt);
```

Usage example:

part. The data will be copied from the current index in this function. The num rows must be the sum of the original and additional datasets, cols must be the dimension of the dataset, and the stride must be the same as the original index dataset. This view will be stored in the output index. It is the caller's responsibility to ensure that dataset stays alive as long as the index. This option is useful when users want to manage the memory space for the dataset themselves. The data will be copied from the current index in this function. The num rows must be the sum of the original and additional datasets and cols must be the graph degree. This view will be stored in the output index. It is the caller's responsibility to ensure that dataset stays alive as long as the index. This option is useful when users want to manage the memory space for the graph themselves.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | raft resources |
| `params` | in | [`const cagra::extend_params&`](/api-reference/cpp-api-neighbors-cagra#cuvs-neighbors-cagra-extend-params) | extend params |
| `additional_dataset` | in | `raft::host_matrix_view<const uint8_t, int64_t, raft::row_major>` | additional dataset on host memory |
| `idx` | in,out | `cuvs::neighbors::cagra::index<uint8_t, uint32_t>&` | CAGRA index |
| `new_dataset_buffer_view` | out | `std::optional<raft::device_matrix_view<uint8_t, int64_t, raft::layout_stride>>` | memory buffer view for the dataset including the additional Default: `std::nullopt`. |
| `new_graph_buffer_view` | out | `std::optional<raft::device_matrix_view<uint32_t, int64_t>>` | memory buffer view for the graph including the additional part. Default: `std::nullopt`. |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:1510`_

## CAGRA search functions

_Doxygen group: `cagra_cpp_index_search`_

<a id="none-sample-filter"></a>
### none_sample_filter

Search ANN using the constructed index.

```cpp
void search(raft::resources const& res,
cuvs::neighbors::cagra::search_params const& params,
const cuvs::neighbors::cagra::index<float, uint32_t>& index,
raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
raft::device_matrix_view<uint32_t, int64_t, raft::row_major> neighbors,
raft::device_matrix_view<float, int64_t, raft::row_major> distances,
const cuvs::neighbors::filtering::base_filter& sample_filter =
cuvs::neighbors::filtering::none_sample_filter{}
```

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:1541`_

## CAGRA serialize functions

_Doxygen group: `cagra_cpp_serialize`_

<a id="cuvs-neighbors-cagra-serialize"></a>
### cuvs::neighbors::cagra::serialize

Save the index to file.

```cpp
void serialize(raft::resources const& handle,
const std::string& filename,
const cuvs::neighbors::cagra::index<float, uint32_t>& index,
bool include_dataset = true);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `filename` | in | `const std::string&` | the file name for saving the index |
| `index` | in | `const cuvs::neighbors::cagra::index<float, uint32_t>&` | CAGRA index |
| `include_dataset` | in | `bool` | Whether or not to write out the dataset to the file. Default: `true`. |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:1758`_

<a id="cuvs-neighbors-cagra-deserialize"></a>
### cuvs::neighbors::cagra::deserialize

Load index from file.

```cpp
void deserialize(raft::resources const& handle,
const std::string& filename,
cuvs::neighbors::cagra::index<float, uint32_t>* index);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `filename` | in | `const std::string&` | the name of the file that stores the index |
| `index` | out | `cuvs::neighbors::cagra::index<float, uint32_t>*` | the cagra index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:1785`_

**Additional overload:** `cuvs::neighbors::cagra::serialize`

Write the index to an output stream

```cpp
void serialize(raft::resources const& handle,
std::ostream& os,
const cuvs::neighbors::cagra::index<float, uint32_t>& index,
bool include_dataset = true);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `os` | in | `std::ostream&` | output stream |
| `index` | in | `const cuvs::neighbors::cagra::index<float, uint32_t>&` | CAGRA index |
| `include_dataset` | in | `bool` | Whether or not to write out the dataset to the file. Default: `true`. |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:1811`_

**Additional overload:** `cuvs::neighbors::cagra::deserialize`

Load index from input stream

```cpp
void deserialize(raft::resources const& handle,
std::istream& is,
cuvs::neighbors::cagra::index<float, uint32_t>* index);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `is` | in | `std::istream&` | input stream |
| `index` | out | `cuvs::neighbors::cagra::index<float, uint32_t>*` | the cagra index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:1837`_

**Additional overload:** `cuvs::neighbors::cagra::serialize`

Save the index to file.

```cpp
void serialize(raft::resources const& handle,
const std::string& filename,
const cuvs::neighbors::cagra::index<half, uint32_t>& index,
bool include_dataset = true);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `filename` | in | `const std::string&` | the file name for saving the index |
| `index` | in | `const cuvs::neighbors::cagra::index<half, uint32_t>&` | CAGRA index |
| `include_dataset` | in | `bool` | Whether or not to write out the dataset to the file. Default: `true`. |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:1863`_

**Additional overload:** `cuvs::neighbors::cagra::deserialize`

Load index from file.

```cpp
void deserialize(raft::resources const& handle,
const std::string& filename,
cuvs::neighbors::cagra::index<half, uint32_t>* index);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `filename` | in | `const std::string&` | the name of the file that stores the index |
| `index` | out | `cuvs::neighbors::cagra::index<half, uint32_t>*` | the cagra index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:1890`_

**Additional overload:** `cuvs::neighbors::cagra::serialize`

Write the index to an output stream

```cpp
void serialize(raft::resources const& handle,
std::ostream& os,
const cuvs::neighbors::cagra::index<half, uint32_t>& index,
bool include_dataset = true);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `os` | in | `std::ostream&` | output stream |
| `index` | in | `const cuvs::neighbors::cagra::index<half, uint32_t>&` | CAGRA index |
| `include_dataset` | in | `bool` | Whether or not to write out the dataset to the file. Default: `true`. |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:1916`_

**Additional overload:** `cuvs::neighbors::cagra::deserialize`

Load index from input stream

```cpp
void deserialize(raft::resources const& handle,
std::istream& is,
cuvs::neighbors::cagra::index<half, uint32_t>* index);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `is` | in | `std::istream&` | input stream |
| `index` | out | `cuvs::neighbors::cagra::index<half, uint32_t>*` | the cagra index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:1942`_

**Additional overload:** `cuvs::neighbors::cagra::serialize`

Save the index to file.

```cpp
void serialize(raft::resources const& handle,
const std::string& filename,
const cuvs::neighbors::cagra::index<int8_t, uint32_t>& index,
bool include_dataset = true);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `filename` | in | `const std::string&` | the file name for saving the index |
| `index` | in | `const cuvs::neighbors::cagra::index<int8_t, uint32_t>&` | CAGRA index |
| `include_dataset` | in | `bool` | Whether or not to write out the dataset to the file. Default: `true`. |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:1968`_

**Additional overload:** `cuvs::neighbors::cagra::deserialize`

Load index from file.

```cpp
void deserialize(raft::resources const& handle,
const std::string& filename,
cuvs::neighbors::cagra::index<int8_t, uint32_t>* index);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `filename` | in | `const std::string&` | the name of the file that stores the index |
| `index` | out | `cuvs::neighbors::cagra::index<int8_t, uint32_t>*` | the cagra index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:1995`_

**Additional overload:** `cuvs::neighbors::cagra::serialize`

Write the index to an output stream

```cpp
void serialize(raft::resources const& handle,
std::ostream& os,
const cuvs::neighbors::cagra::index<int8_t, uint32_t>& index,
bool include_dataset = true);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `os` | in | `std::ostream&` | output stream |
| `index` | in | `const cuvs::neighbors::cagra::index<int8_t, uint32_t>&` | CAGRA index |
| `include_dataset` | in | `bool` | Whether or not to write out the dataset to the file. Default: `true`. |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:2021`_

**Additional overload:** `cuvs::neighbors::cagra::deserialize`

Load index from input stream

```cpp
void deserialize(raft::resources const& handle,
std::istream& is,
cuvs::neighbors::cagra::index<int8_t, uint32_t>* index);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `is` | in | `std::istream&` | input stream |
| `index` | out | `cuvs::neighbors::cagra::index<int8_t, uint32_t>*` | the cagra index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:2047`_

**Additional overload:** `cuvs::neighbors::cagra::serialize`

Save the index to file.

```cpp
void serialize(raft::resources const& handle,
const std::string& filename,
const cuvs::neighbors::cagra::index<uint8_t, uint32_t>& index,
bool include_dataset = true);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `filename` | in | `const std::string&` | the file name for saving the index |
| `index` | in | `const cuvs::neighbors::cagra::index<uint8_t, uint32_t>&` | CAGRA index |
| `include_dataset` | in | `bool` | Whether or not to write out the dataset to the file. Default: `true`. |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:2073`_

**Additional overload:** `cuvs::neighbors::cagra::deserialize`

Load index from file.

```cpp
void deserialize(raft::resources const& handle,
const std::string& filename,
cuvs::neighbors::cagra::index<uint8_t, uint32_t>* index);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `filename` | in | `const std::string&` | the name of the file that stores the index |
| `index` | out | `cuvs::neighbors::cagra::index<uint8_t, uint32_t>*` | the cagra index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:2100`_

**Additional overload:** `cuvs::neighbors::cagra::serialize`

Write the index to an output stream

```cpp
void serialize(raft::resources const& handle,
std::ostream& os,
const cuvs::neighbors::cagra::index<uint8_t, uint32_t>& index,
bool include_dataset = true);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `os` | in | `std::ostream&` | output stream |
| `index` | in | `const cuvs::neighbors::cagra::index<uint8_t, uint32_t>&` | CAGRA index |
| `include_dataset` | in | `bool` | Whether or not to write out the dataset to the file. Default: `true`. |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:2126`_

**Additional overload:** `cuvs::neighbors::cagra::deserialize`

Load index from input stream

```cpp
void deserialize(raft::resources const& handle,
std::istream& is,
cuvs::neighbors::cagra::index<uint8_t, uint32_t>* index);
```

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `is` | in | `std::istream&` | input stream |
| `index` | out | `cuvs::neighbors::cagra::index<uint8_t, uint32_t>*` | the cagra index |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:2152`_

<a id="cuvs-neighbors-cagra-serialize-to-hnswlib"></a>
### cuvs::neighbors::cagra::serialize_to_hnswlib

Write the CAGRA built index as a base layer HNSW index to an output stream

```cpp
void serialize_to_hnswlib(
raft::resources const& handle,
std::ostream& os,
const cuvs::neighbors::cagra::index<float, uint32_t>& index,
std::optional<raft::host_matrix_view<const float, int64_t, raft::row_major>> dataset =
std::nullopt);
```

NOTE: The saved index can only be read by the hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib.

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `os` | in | `std::ostream&` | output stream |
| `index` | in | `const cuvs::neighbors::cagra::index<float, uint32_t>&` | CAGRA index |
| `dataset` | in | `std::optional<raft::host_matrix_view<const float, int64_t, raft::row_major>>` | [optional] host array that stores the dataset, required if the index does not contain the dataset. Default: `std::nullopt`. |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:2182`_

**Additional overload:** `cuvs::neighbors::cagra::serialize_to_hnswlib`

Save a CAGRA build index in hnswlib base-layer-only serialized format

```cpp
void serialize_to_hnswlib(
raft::resources const& handle,
const std::string& filename,
const cuvs::neighbors::cagra::index<float, uint32_t>& index,
std::optional<raft::host_matrix_view<const float, int64_t, raft::row_major>> dataset =
std::nullopt);
```

NOTE: The saved index can only be read by the hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib.

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `filename` | in | `const std::string&` | the file name for saving the index |
| `index` | in | `const cuvs::neighbors::cagra::index<float, uint32_t>&` | CAGRA index |
| `dataset` | in | `std::optional<raft::host_matrix_view<const float, int64_t, raft::row_major>>` | [optional] host array that stores the dataset, required if the index does not contain the dataset. Default: `std::nullopt`. |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:2216`_

**Additional overload:** `cuvs::neighbors::cagra::serialize_to_hnswlib`

Write the CAGRA built index as a base layer HNSW index to an output stream

```cpp
void serialize_to_hnswlib(
raft::resources const& handle,
std::ostream& os,
const cuvs::neighbors::cagra::index<half, uint32_t>& index,
std::optional<raft::host_matrix_view<const half, int64_t, raft::row_major>> dataset =
std::nullopt);
```

NOTE: The saved index can only be read by the hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib.

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `os` | in | `std::ostream&` | output stream |
| `index` | in | `const cuvs::neighbors::cagra::index<half, uint32_t>&` | CAGRA index |
| `dataset` | in | `std::optional<raft::host_matrix_view<const half, int64_t, raft::row_major>>` | [optional] host array that stores the dataset, required if the index does not contain the dataset. Default: `std::nullopt`. |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:2249`_

**Additional overload:** `cuvs::neighbors::cagra::serialize_to_hnswlib`

Save a CAGRA build index in hnswlib base-layer-only serialized format

```cpp
void serialize_to_hnswlib(
raft::resources const& handle,
const std::string& filename,
const cuvs::neighbors::cagra::index<half, uint32_t>& index,
std::optional<raft::host_matrix_view<const half, int64_t, raft::row_major>> dataset =
std::nullopt);
```

NOTE: The saved index can only be read by the hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib.

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `filename` | in | `const std::string&` | the file name for saving the index |
| `index` | in | `const cuvs::neighbors::cagra::index<half, uint32_t>&` | CAGRA index |
| `dataset` | in | `std::optional<raft::host_matrix_view<const half, int64_t, raft::row_major>>` | [optional] host array that stores the dataset, required if the index does not contain the dataset. Default: `std::nullopt`. |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:2283`_

**Additional overload:** `cuvs::neighbors::cagra::serialize_to_hnswlib`

Write the CAGRA built index as a base layer HNSW index to an output stream

```cpp
void serialize_to_hnswlib(
raft::resources const& handle,
std::ostream& os,
const cuvs::neighbors::cagra::index<int8_t, uint32_t>& index,
std::optional<raft::host_matrix_view<const int8_t, int64_t, raft::row_major>> dataset =
std::nullopt);
```

NOTE: The saved index can only be read by the hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib.

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `os` | in | `std::ostream&` | output stream |
| `index` | in | `const cuvs::neighbors::cagra::index<int8_t, uint32_t>&` | CAGRA index |
| `dataset` | in | `std::optional<raft::host_matrix_view<const int8_t, int64_t, raft::row_major>>` | [optional] host array that stores the dataset, required if the index does not contain the dataset. Default: `std::nullopt`. |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:2316`_

**Additional overload:** `cuvs::neighbors::cagra::serialize_to_hnswlib`

Save a CAGRA build index in hnswlib base-layer-only serialized format

```cpp
void serialize_to_hnswlib(
raft::resources const& handle,
const std::string& filename,
const cuvs::neighbors::cagra::index<int8_t, uint32_t>& index,
std::optional<raft::host_matrix_view<const int8_t, int64_t, raft::row_major>> dataset =
std::nullopt);
```

NOTE: The saved index can only be read by the hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib.

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `filename` | in | `const std::string&` | the file name for saving the index |
| `index` | in | `const cuvs::neighbors::cagra::index<int8_t, uint32_t>&` | CAGRA index |
| `dataset` | in | `std::optional<raft::host_matrix_view<const int8_t, int64_t, raft::row_major>>` | [optional] host array that stores the dataset, required if the index does not contain the dataset. Default: `std::nullopt`. |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:2350`_

**Additional overload:** `cuvs::neighbors::cagra::serialize_to_hnswlib`

Write the CAGRA built index as a base layer HNSW index to an output stream

```cpp
void serialize_to_hnswlib(
raft::resources const& handle,
std::ostream& os,
const cuvs::neighbors::cagra::index<uint8_t, uint32_t>& index,
std::optional<raft::host_matrix_view<const uint8_t, int64_t, raft::row_major>> dataset =
std::nullopt);
```

NOTE: The saved index can only be read by the hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib.

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `os` | in | `std::ostream&` | output stream |
| `index` | in | `const cuvs::neighbors::cagra::index<uint8_t, uint32_t>&` | CAGRA index |
| `dataset` | in | `std::optional<raft::host_matrix_view<const uint8_t, int64_t, raft::row_major>>` | [optional] host array that stores the dataset, required if the index does not contain the dataset. Default: `std::nullopt`. |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:2383`_

**Additional overload:** `cuvs::neighbors::cagra::serialize_to_hnswlib`

Save a CAGRA build index in hnswlib base-layer-only serialized format

```cpp
void serialize_to_hnswlib(
raft::resources const& handle,
const std::string& filename,
const cuvs::neighbors::cagra::index<uint8_t, uint32_t>& index,
std::optional<raft::host_matrix_view<const uint8_t, int64_t, raft::row_major>> dataset =
std::nullopt);
```

NOTE: The saved index can only be read by the hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib.

Experimental, both the API and the serialization format are subject to change.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `handle` | in | `raft::resources const&` | the raft handle |
| `filename` | in | `const std::string&` | the file name for saving the index |
| `index` | in | `const cuvs::neighbors::cagra::index<uint8_t, uint32_t>&` | CAGRA index |
| `dataset` | in | `std::optional<raft::host_matrix_view<const uint8_t, int64_t, raft::row_major>>` | [optional] host array that stores the dataset, required if the index does not contain the dataset. Default: `std::nullopt`. |

**Returns**

`void`

_Source: `cpp/include/cuvs/neighbors/cagra.hpp:2417`_
