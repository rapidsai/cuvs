/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "common.hpp"
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <cuvs/neighbors/nn_descent.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resource/stream_view.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/integer_utils.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <optional>
#include <variant>

namespace cuvs::neighbors::cagra {
/**
 * @defgroup cagra_cpp_index_params CAGRA index build parameters
 * @{
 */

/**
 * @brief ANN parameters used by CAGRA to build knn graph
 *
 */
namespace graph_build_params {

/** Specialized parameters utilizing IVF-PQ to build knn graph */
struct ivf_pq_params {
  cuvs::neighbors::ivf_pq::index_params build_params;
  cuvs::neighbors::ivf_pq::search_params search_params;
  float refinement_rate;

  ivf_pq_params() = default;
  /**
   * Set default parameters based on shape of the input dataset.
   * Usage example:
   * @code{.cpp}
   *   using namespace cuvs::neighbors;
   *   raft::resources res;
   *   // create index_params for a [N. D] dataset
   *   auto dataset = raft::make_device_matrix<float, int64_t>(res, N, D);
   *   auto pq_params =
   *     cagra::graph_build_params::ivf_pq_params(dataset.extents());
   *   // modify/update index_params as needed
   *   index_params.add_data_on_build = true;
   * @endcode
   */
  ivf_pq_params(raft::matrix_extent<int64_t> dataset_extents,
                cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded);
};

using nn_descent_params = cuvs::neighbors::nn_descent::index_params;
}  // namespace graph_build_params

struct index_params : cuvs::neighbors::index_params {
  /** Degree of input graph for pruning. */
  size_t intermediate_graph_degree = 128;
  /** Degree of output graph. */
  size_t graph_degree = 64;
  /**
   * Specify compression parameters if compression is desired.
   */
  std::optional<cuvs::neighbors::vpq_params> compression = std::nullopt;

  /** Parameters for graph building.
   *
   * Set ivf_pq_params or nn_descent_params to select the graph build algorithm and control their
   * parameters. The default (std::monostate) is to use a heuristic to decide the algorithm and its
   * parameters.
   *
   * @code{.cpp}
   * cagra::index_params params;
   * // 1. Choose IVF-PQ algorithm
   * params.graph_build_params = cagra::graph_build_params::ivf_pq_params(dataset.extent,
   * params.metric);
   *
   * // 2. Choose NN Descent algorithm for kNN graph construction
   * params.graph_build_params =
   * cagra::graph_build_params::nn_descent_params(params.intermediate_graph_degree);
   * @endcode
   */
  std::variant<std::monostate,
               graph_build_params::ivf_pq_params,
               graph_build_params::nn_descent_params>
    graph_build_params;
};

/**
 * @}
 */

/**
 * @defgroup cagra_cpp_search_params CAGRA index search parameters
 * @{
 */

enum class search_algo {
  /** For large batch sizes. */
  SINGLE_CTA,
  /** For small batch sizes. */
  MULTI_CTA,
  MULTI_KERNEL,
  AUTO
};

enum class hash_mode { HASH, SMALL, AUTO };

struct search_params : cuvs::neighbors::search_params {
  /** Maximum number of queries to search at the same time (batch size). Auto select when 0.*/
  size_t max_queries = 0;

  /** Number of intermediate search results retained during the search.
   *
   *  This is the main knob to adjust trade off between accuracy and search speed.
   *  Higher values improve the search accuracy.
   */
  size_t itopk_size = 64;

  /** Upper limit of search iterations. Auto select when 0.*/
  size_t max_iterations = 0;

  // In the following we list additional search parameters for fine tuning.
  // Reasonable default values are automatically chosen.

  /** Which search implementation to use. */
  search_algo algo = search_algo::AUTO;

  /** Number of threads used to calculate a single distance. 4, 8, 16, or 32. */
  size_t team_size = 0;

  /** Number of graph nodes to select as the starting point for the search in each iteration. aka
   * search width?*/
  size_t search_width = 1;
  /** Lower limit of search iterations. */
  size_t min_iterations = 0;

  /** Thread block size. 0, 64, 128, 256, 512, 1024. Auto selection when 0. */
  size_t thread_block_size = 0;
  /** Hashmap type. Auto selection when AUTO. */
  hash_mode hashmap_mode = hash_mode::AUTO;
  /** Lower limit of hashmap bit length. More than 8. */
  size_t hashmap_min_bitlen = 0;
  /** Upper limit of hashmap fill rate. More than 0.1, less than 0.9.*/
  float hashmap_max_fill_rate = 0.5;

  /** Number of iterations of initial random seed node selection. 1 or more. */
  uint32_t num_random_samplings = 1;
  /** Bit mask used for initial random seed node selection. */
  uint64_t rand_xor_mask = 0x128394;
};

/**
 * @}
 */

/**
 * @defgroup cagra_cpp_extend_params CAGRA index extend parameters
 * @{
 */

struct extend_params {
  /** The additional dataset is divided into chunks and added to the graph. This is the knob to
   * adjust the tradeoff between the recall and operation throughput. Large chunk sizes can result
   * in high throughput, but use more working memory (O(max_chunk_size*degree^2)). This can also
   * degrade recall because no edges are added between the nodes in the same chunk. Auto select when
   * 0. */
  uint32_t max_chunk_size = 0;
};

/**
 * @}
 */

static_assert(std::is_aggregate_v<index_params>);
static_assert(std::is_aggregate_v<search_params>);

/**
 * @defgroup cagra_cpp_index CAGRA index type
 * @{
 */

/**
 * @brief CAGRA index.
 *
 * The index stores the dataset and a kNN graph in device memory.
 *
 * @tparam T data element type
 * @tparam IdxT type of the vector indices (represent dataset.extent(0))
 *
 */
template <typename T, typename IdxT>
struct index : cuvs::neighbors::index {
  static_assert(!raft::is_narrowing_v<uint32_t, IdxT>,
                "IdxT must be able to represent all values of uint32_t");

 public:
  /** Distance metric used for clustering. */
  [[nodiscard]] constexpr inline auto metric() const noexcept -> cuvs::distance::DistanceType
  {
    return metric_;
  }

  /** Total length of the index (number of vectors). */
  [[nodiscard]] constexpr inline auto size() const noexcept -> IdxT
  {
    auto data_rows = dataset_->n_rows();
    return data_rows > 0 ? data_rows : graph_view_.extent(0);
  }

  /** Dimensionality of the data. */
  [[nodiscard]] constexpr inline auto dim() const noexcept -> uint32_t { return dataset_->dim(); }
  /** Graph degree */
  [[nodiscard]] constexpr inline auto graph_degree() const noexcept -> uint32_t
  {
    return graph_view_.extent(1);
  }

  [[nodiscard]] inline auto dataset() const noexcept
    -> raft::device_matrix_view<const T, int64_t, raft::layout_stride>
  {
    auto p = dynamic_cast<strided_dataset<T, int64_t>*>(dataset_.get());
    if (p != nullptr) { return p->view(); }
    auto d = dataset_->dim();
    return raft::make_device_strided_matrix_view<const T, int64_t>(nullptr, 0, d, d);
  }

  /** Dataset [size, dim] */
  [[nodiscard]] inline auto data() const noexcept -> const cuvs::neighbors::dataset<int64_t>&
  {
    return *dataset_;
  }

  /** neighborhood graph [size, graph-degree] */
  [[nodiscard]] inline auto graph() const noexcept
    -> raft::device_matrix_view<const IdxT, int64_t, raft::row_major>
  {
    return graph_view_;
  }

  // Don't allow copying the index for performance reasons (try avoiding copying data)
  index(const index&)                    = delete;
  index(index&&)                         = default;
  auto operator=(const index&) -> index& = delete;
  auto operator=(index&&) -> index&      = default;
  ~index()                               = default;

  /** Construct an empty index. */
  index(raft::resources const& res,
        cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded)
    : cuvs::neighbors::index(),
      metric_(metric),
      graph_(raft::make_device_matrix<IdxT, int64_t>(res, 0, 0)),
      dataset_(new cuvs::neighbors::empty_dataset<int64_t>(0))
  {
  }

  /** Construct an index from dataset and knn_graph arrays
   *
   * If the dataset and graph is already in GPU memory, then the index is just a thin wrapper around
   * these that stores a non-owning a reference to the arrays.
   *
   * The constructor also accepts host arrays. In that case they are copied to the device, and the
   * device arrays will be owned by the index.
   *
   * In case the dasates rows are not 16 bytes aligned, then we create a padded copy in device
   * memory to ensure alignment for vectorized load.
   *
   * Usage examples:
   *
   * - Cagra index is normally created by the cagra::build
   * @code{.cpp}
   *   using namespace raft::neighbors::experimental;
   *   auto dataset = raft::make_host_matrix<float, int64_t>(n_rows, n_cols);
   *   load_dataset(dataset.view());
   *   // use default index parameters
   *   cagra::index_params index_params;
   *   // create and fill the index from a [N, D] dataset
   *   auto index = cagra::build(res, index_params, dataset);
   *   // use default search parameters
   *   cagra::search_params search_params;
   *   // search K nearest neighbours
   *   auto neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, n_queries, k);
   *   auto distances = raft::make_device_matrix<float, int64_t>(res, n_queries, k);
   *   cagra::search(res, search_params, index, queries, neighbors, distances);
   * @endcode
   *   In the above example, we have passed a host dataset to build. The returned index will own a
   * device copy of the dataset and the knn_graph. In contrast, if we pass the dataset as a
   * device_mdspan to build, then it will only store a reference to it.
   *
   * - Constructing index using existing knn-graph
   * @code{.cpp}
   *   using namespace raft::neighbors::experimental;
   *
   *   auto dataset = raft::make_device_matrix<float, int64_t>(res, n_rows, n_cols);
   *   auto knn_graph = raft::make_device_matrix<uint32_n, int64_t>(res, n_rows, graph_degree);
   *
   *   // custom loading and graph creation
   *   // load_dataset(dataset.view());
   *   // create_knn_graph(knn_graph.view());
   *
   *   // Wrap the existing device arrays into an index structure
   *   cagra::index<T, IdxT> index(res, metric, raft::make_const_mdspan(dataset.view()),
   *                               raft::make_const_mdspan(knn_graph.view()));
   *
   *   // Both knn_graph and dataset objects have to be in scope while the index is used because
   *   // the index only stores a reference to these.
   *   cagra::search(res, search_params, index, queries, neighbors, distances);
   * @endcode
   *
   */
  template <typename data_accessor, typename graph_accessor>
  index(raft::resources const& res,
        cuvs::distance::DistanceType metric,
        raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, data_accessor> dataset,
        raft::mdspan<const IdxT, raft::matrix_extent<int64_t>, raft::row_major, graph_accessor>
          knn_graph)
    : cuvs::neighbors::index(),
      metric_(metric),
      graph_(raft::make_device_matrix<IdxT, int64_t>(res, 0, 0)),
      dataset_(make_aligned_dataset(res, dataset, 16))
  {
    RAFT_EXPECTS(dataset.extent(0) == knn_graph.extent(0),
                 "Dataset and knn_graph must have equal number of rows");
    update_graph(res, knn_graph);

    raft::resource::sync_stream(res);
  }

  /**
   * Replace the dataset with a new dataset.
   *
   * If the new dataset rows are aligned on 16 bytes, then only a reference is stored to the
   * dataset. It is the caller's responsibility to ensure that dataset stays alive as long as the
   * index.
   */
  void update_dataset(raft::resources const& res,
                      raft::device_matrix_view<const T, int64_t, raft::row_major> dataset)
  {
    dataset_ = make_aligned_dataset(res, dataset, 16);
  }

  /** Set the dataset reference explicitly to a device matrix view with padding. */
  void update_dataset(raft::resources const& res,
                      raft::device_matrix_view<const T, int64_t, raft::layout_stride> dataset)
  {
    dataset_ = make_aligned_dataset(res, dataset, 16);
  }

  /**
   * Replace the dataset with a new dataset.
   *
   * We create a copy of the dataset on the device. The index manages the lifetime of this copy.
   */
  void update_dataset(raft::resources const& res,
                      raft::host_matrix_view<const T, int64_t, raft::row_major> dataset)
  {
    dataset_ = make_aligned_dataset(res, dataset, 16);
  }

  /** Replace the dataset with a new dataset. */
  template <typename DatasetT>
  auto update_dataset(raft::resources const& res, DatasetT&& dataset)
    -> std::enable_if_t<std::is_base_of_v<cuvs::neighbors::dataset<int64_t>, DatasetT>>
  {
    dataset_ = std::make_unique<DatasetT>(std::move(dataset));
  }

  template <typename DatasetT>
  auto update_dataset(raft::resources const& res, std::unique_ptr<DatasetT>&& dataset)
    -> std::enable_if_t<std::is_base_of_v<neighbors::dataset<int64_t>, DatasetT>>
  {
    dataset_ = std::move(dataset);
  }

  /**
   * Replace the graph with a new graph.
   *
   * Since the new graph is a device array, we store a reference to that, and it is
   * the caller's responsibility to ensure that knn_graph stays alive as long as the index.
   */
  void update_graph(raft::resources const& res,
                    raft::device_matrix_view<const IdxT, int64_t, raft::row_major> knn_graph)
  {
    graph_view_ = knn_graph;
  }

  /**
   * Replace the graph with a new graph.
   *
   * We create a copy of the graph on the device. The index manages the lifetime of this copy.
   */
  void update_graph(raft::resources const& res,
                    raft::host_matrix_view<const IdxT, int64_t, raft::row_major> knn_graph)
  {
    RAFT_LOG_DEBUG("Copying CAGRA knn graph from host to device");

    if ((graph_.extent(0) != knn_graph.extent(0)) || (graph_.extent(1) != knn_graph.extent(1))) {
      // clear existing memory before allocating to prevent OOM errors on large graphs
      if (graph_.size()) { graph_ = raft::make_device_matrix<IdxT, int64_t>(res, 0, 0); }
      graph_ =
        raft::make_device_matrix<IdxT, int64_t>(res, knn_graph.extent(0), knn_graph.extent(1));
    }
    raft::copy(graph_.data_handle(),
               knn_graph.data_handle(),
               knn_graph.size(),
               raft::resource::get_cuda_stream(res));
    graph_view_ = graph_.view();
  }

 private:
  cuvs::distance::DistanceType metric_;
  raft::device_matrix<IdxT, int64_t, raft::row_major> graph_;
  raft::device_matrix_view<const IdxT, int64_t, raft::row_major> graph_view_;
  std::unique_ptr<neighbors::dataset<int64_t>> dataset_;
};
/**
 * @}
 */

/**
 * @defgroup cagra_cpp_index_build CAGRA index build functions
 * @{
 */

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * The build consist of two steps: build an intermediate knn-graph, and optimize it to
 * create the final graph. The index_params struct controls the node degree of these
 * graphs.
 *
 * The following distance metrics are supported:
 * - L2
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   cagra::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = cagra::build(res, index_params, dataset);
 *   // use default search parameters
 *   cagra::search_params search_params;
 *   // search K nearest neighbours
 *   auto neighbors = raft::make_device_matrix<uint32_t>(res, n_queries, k);
 *   auto distances = raft::make_device_matrix<float>(res, n_queries, k);
 *   cagra::search(res, search_params, index, queries, neighbors, distances);
 * @endcode
 *
 * @param[in] res
 * @param[in] params parameters for building the index
 * @param[in] dataset a matrix view (device) to a row-major matrix [n_rows, dim]
 *
 * @return the constructed cagra index
 */
auto build(raft::resources const& res,
           const cuvs::neighbors::cagra::index_params& params,
           raft::device_matrix_view<const float, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::cagra::index<float, uint32_t>;

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * The build consist of two steps: build an intermediate knn-graph, and optimize it to
 * create the final graph. The index_params struct controls the node degree of these
 * graphs.
 *
 * The following distance metrics are supported:
 * - L2
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   cagra::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = cagra::build(res, index_params, dataset);
 *   // use default search parameters
 *   cagra::search_params search_params;
 *   // search K nearest neighbours
 *   auto neighbors = raft::make_device_matrix<uint32_t>(res, n_queries, k);
 *   auto distances = raft::make_device_matrix<float>(res, n_queries, k);
 *   cagra::search(res, search_params, index, queries, neighbors, distances);
 * @endcode
 *
 * @param[in] res
 * @param[in] params parameters for building the index
 * @param[in] dataset a matrix view (host) to a row-major matrix [n_rows, dim]
 *
 * @return the constructed cagra index
 */
auto build(raft::resources const& res,
           const cuvs::neighbors::cagra::index_params& params,
           raft::host_matrix_view<const float, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::cagra::index<float, uint32_t>;

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * The build consist of two steps: build an intermediate knn-graph, and optimize it to
 * create the final graph. The index_params struct controls the node degree of these
 * graphs.
 *
 * The following distance metrics are supported:
 * - L2
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   cagra::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = cagra::build(res, index_params, dataset);
 *   // use default search parameters
 *   cagra::search_params search_params;
 *   // search K nearest neighbours
 *   auto neighbors = raft::make_device_matrix<uint32_t>(res, n_queries, k);
 *   auto distances = raft::make_device_matrix<float>(res, n_queries, k);
 *   cagra::search(res, search_params, index, queries, neighbors, distances);
 * @endcode
 *
 * @param[in] res
 * @param[in] params parameters for building the index
 * @param[in] dataset a matrix view (device) to a row-major matrix [n_rows, dim]
 *
 * @return the constructed cagra index
 */
auto build(raft::resources const& res,
           const cuvs::neighbors::cagra::index_params& params,
           raft::device_matrix_view<const int8_t, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::cagra::index<int8_t, uint32_t>;

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * The build consist of two steps: build an intermediate knn-graph, and optimize it to
 * create the final graph. The index_params struct controls the node degree of these
 * graphs.
 *
 * The following distance metrics are supported:
 * - L2
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   cagra::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = cagra::build(res, index_params, dataset);
 *   // use default search parameters
 *   cagra::search_params search_params;
 *   // search K nearest neighbours
 *   auto neighbors = raft::make_device_matrix<uint32_t>(res, n_queries, k);
 *   auto distances = raft::make_device_matrix<float>(res, n_queries, k);
 *   cagra::search(res, search_params, index, queries, neighbors, distances);
 * @endcode
 *
 * @param[in] res
 * @param[in] params parameters for building the index
 * @param[in] dataset a matrix view (host) to a row-major matrix [n_rows, dim]
 *
 * @return the constructed cagra index
 */
auto build(raft::resources const& res,
           const cuvs::neighbors::cagra::index_params& params,
           raft::host_matrix_view<const int8_t, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::cagra::index<int8_t, uint32_t>;

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * The build consist of two steps: build an intermediate knn-graph, and optimize it to
 * create the final graph. The index_params struct controls the node degree of these
 * graphs.
 *
 * The following distance metrics are supported:
 * - L2
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   cagra::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = cagra::build(res, index_params, dataset);
 *   // use default search parameters
 *   cagra::search_params search_params;
 *   // search K nearest neighbours
 *   auto neighbors = raft::make_device_matrix<uint32_t>(res, n_queries, k);
 *   auto distances = raft::make_device_matrix<float>(res, n_queries, k);
 *   cagra::search(res, search_params, index, queries, neighbors, distances);
 * @endcode
 *
 * @param[in] res
 * @param[in] params parameters for building the index
 * @param[in] dataset a matrix view (device) to a row-major matrix [n_rows, dim]
 *
 * @return the constructed cagra index
 */
auto build(raft::resources const& res,
           const cuvs::neighbors::cagra::index_params& params,
           raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::cagra::index<uint8_t, uint32_t>;

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * The build consist of two steps: build an intermediate knn-graph, and optimize it to
 * create the final graph. The index_params struct controls the node degree of these
 * graphs.
 *
 * The following distance metrics are supported:
 * - L2
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   cagra::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = cagra::build(res, index_params, dataset);
 *   // use default search parameters
 *   cagra::search_params search_params;
 *   // search K nearest neighbours
 *   auto neighbors = raft::make_device_matrix<uint32_t>(res, n_queries, k);
 *   auto distances = raft::make_device_matrix<float>(res, n_queries, k);
 *   cagra::search(res, search_params, index, queries, neighbors, distances);
 * @endcode
 *
 * @param[in] res
 * @param[in] params parameters for building the index
 * @param[in] dataset a matrix view (host) to a row-major matrix [n_rows, dim]
 *
 * @return the constructed cagra index
 */
auto build(raft::resources const& res,
           const cuvs::neighbors::cagra::index_params& params,
           raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::cagra::index<uint8_t, uint32_t>;
/**
 * @}
 */

/**
 * @defgroup cagra_cpp_index_extend CAGRA extend functions
 * @{
 */

/** @brief Add new vectors to a CAGRA index
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace raft::neighbors;
 *   auto additional_dataset = raft::make_device_matrix<float, int64_t>(handle,add_size,dim);
 *   // set_additional_dataset(additional_dataset.view());
 *
 *   cagra::extend_params params;
 *   cagra::extend(res, params, raft::make_const_mdspan(additional_dataset.view()), index);
 * @endcode
 *
 * @param[in] handle raft resources
 * @param[in] params extend params
 * @param[in] additional_dataset additional dataset on device memory
 * @param[in,out] idx CAGRA index
 * @param[out] new_dataset_buffer_view memory buffer view for the dataset including the additional
 * part. The data will be copied from the current index in this function. The num rows must be the
 * sum of the original and additional datasets, cols must be the dimension of the dataset, and the
 * stride must be the same as the original index dataset. This view will be stored in the output
 * index. It is the caller's responsibility to ensure that dataset stays alive as long as the index.
 * This option is useful when users want to manage the memory space for the dataset themselves.
 * @param[out] new_graph_buffer_view memory buffer view for the graph including the additional part.
 * The data will be copied from the current index in this function. The num rows must be the sum of
 * the original and additional datasets and cols must be the graph degree. This view will be stored
 * in the output index. It is the caller's responsibility to ensure that dataset stays alive as long
 * as the index. This option is useful when users want to manage the memory space for the graph
 * themselves.
 */
void extend(
  raft::resources const& handle,
  const cagra::extend_params& params,
  raft::device_matrix_view<const float, int64_t, raft::row_major> additional_dataset,
  cuvs::neighbors::cagra::index<float, uint32_t>& idx,
  std::optional<raft::device_matrix_view<float, int64_t, raft::layout_stride>>
    new_dataset_buffer_view                                                        = std::nullopt,
  std::optional<raft::device_matrix_view<uint32_t, int64_t>> new_graph_buffer_view = std::nullopt);

/** @brief Add new vectors to a CAGRA index
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace raft::neighbors;
 *   auto additional_dataset = raft::make_host_matrix<float, int64_t>(handle,add_size,dim);
 *   // set_additional_dataset(additional_dataset.view());
 *
 *   cagra::extend_params params;
 *   cagra::extend(res, params, raft::make_const_mdspan(additional_dataset.view()), index);
 * @endcode
 *
 * @param[in] handle raft resources
 * @param[in] params extend params
 * @param[in] additional_dataset additional dataset on host memory
 * @param[in,out] idx CAGRA index
 * @param[out] new_dataset_buffer_view memory buffer view for the dataset including the additional
 * part. The data will be copied from the current index in this function. The num rows must be the
 * sum of the original and additional datasets, cols must be the dimension of the dataset, and the
 * stride must be the same as the original index dataset. This view will be stored in the output
 * index. It is the caller's responsibility to ensure that dataset stays alive as long as the index.
 * This option is useful when users want to manage the memory space for the dataset themselves.
 * @param[out] new_graph_buffer_view memory buffer view for the graph including the additional part.
 * The data will be copied from the current index in this function. The num rows must be the sum of
 * the original and additional datasets and cols must be the graph degree. This view will be stored
 * in the output index. It is the caller's responsibility to ensure that dataset stays alive as long
 * as the index. This option is useful when users want to manage the memory space for the graph
 * themselves.
 */
void extend(
  raft::resources const& handle,
  const cagra::extend_params& params,
  raft::host_matrix_view<const float, int64_t, raft::row_major> additional_dataset,
  cuvs::neighbors::cagra::index<float, uint32_t>& idx,
  std::optional<raft::device_matrix_view<float, int64_t, raft::layout_stride>>
    new_dataset_buffer_view                                                        = std::nullopt,
  std::optional<raft::device_matrix_view<uint32_t, int64_t>> new_graph_buffer_view = std::nullopt);

/** @brief Add new vectors to a CAGRA index
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace raft::neighbors;
 *   auto additional_dataset = raft::make_device_matrix<int8_t, int64_t>(handle,add_size,dim);
 *   // set_additional_dataset(additional_dataset.view());
 *
 *   cagra::extend_params params;
 *   cagra::extend(res, params, raft::make_const_mdspan(additional_dataset.view()), index);
 * @endcode
 *
 * @param[in] handle raft resources
 * @param[in] params extend params
 * @param[in] additional_dataset additional dataset on device memory
 * @param[in,out] idx CAGRA index
 * @param[out] new_dataset_buffer_view memory buffer view for the dataset including the additional
 * part. The data will be copied from the current index in this function. The num rows must be the
 * sum of the original and additional datasets, cols must be the dimension of the dataset, and the
 * stride must be the same as the original index dataset. This view will be stored in the output
 * index. It is the caller's responsibility to ensure that dataset stays alive as long as the index.
 * This option is useful when users want to manage the memory space for the dataset themselves.
 * @param[out] new_graph_buffer_view memory buffer view for the graph including the additional part.
 * The data will be copied from the current index in this function. The num rows must be the sum of
 * the original and additional datasets and cols must be the graph degree. This view will be stored
 * in the output index. It is the caller's responsibility to ensure that dataset stays alive as long
 * as the index. This option is useful when users want to manage the memory space for the graph
 * themselves.
 */
void extend(
  raft::resources const& handle,
  const cagra::extend_params& params,
  raft::device_matrix_view<const int8_t, int64_t, raft::row_major> additional_dataset,
  cuvs::neighbors::cagra::index<int8_t, uint32_t>& idx,
  std::optional<raft::device_matrix_view<int8_t, int64_t, raft::layout_stride>>
    new_dataset_buffer_view                                                        = std::nullopt,
  std::optional<raft::device_matrix_view<uint32_t, int64_t>> new_graph_buffer_view = std::nullopt);

/** @brief Add new vectors to a CAGRA index
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace raft::neighbors;
 *   auto additional_dataset = raft::make_host_matrix<int8_t, int64_t>(handle,add_size,dim);
 *   // set_additional_dataset(additional_dataset.view());
 *
 *   cagra::extend_params params;
 *   cagra::extend(res, params, raft::make_const_mdspan(additional_dataset.view()), index);
 * @endcode
 *
 * @param[in] handle raft resources
 * @param[in] params extend params
 * @param[in] additional_dataset additional dataset on host memory
 * @param[in,out] idx CAGRA index
 * @param[out] new_dataset_buffer_view memory buffer view for the dataset including the additional
 * part. The data will be copied from the current index in this function. The num rows must be the
 * sum of the original and additional datasets, cols must be the dimension of the dataset, and the
 * stride must be the same as the original index dataset. This view will be stored in the output
 * index. It is the caller's responsibility to ensure that dataset stays alive as long as the index.
 * This option is useful when users want to manage the memory space for the dataset themselves.
 * @param[out] new_graph_buffer_view memory buffer view for the graph including the additional part.
 * The data will be copied from the current index in this function. The num rows must be the sum of
 * the original and additional datasets and cols must be the graph degree. This view will be stored
 * in the output index. It is the caller's responsibility to ensure that dataset stays alive as long
 * as the index. This option is useful when users want to manage the memory space for the graph
 * themselves.
 */
void extend(
  raft::resources const& handle,
  const cagra::extend_params& params,
  raft::host_matrix_view<const int8_t, int64_t, raft::row_major> additional_dataset,
  cuvs::neighbors::cagra::index<int8_t, uint32_t>& idx,
  std::optional<raft::device_matrix_view<int8_t, int64_t, raft::layout_stride>>
    new_dataset_buffer_view                                                        = std::nullopt,
  std::optional<raft::device_matrix_view<uint32_t, int64_t>> new_graph_buffer_view = std::nullopt);

/** @brief Add new vectors to a CAGRA index
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace raft::neighbors;
 *   auto additional_dataset = raft::make_host_matrix<uint8_t, int64_t>(handle,add_size,dim);
 *   // set_additional_dataset(additional_dataset.view());
 *
 *   cagra::extend_params params;
 *   cagra::extend(res, params, raft::make_const_mdspan(additional_dataset.view()), index);
 * @endcode
 *
 * @param[in] handle raft resources
 * @param[in] params extend params
 * @param[in] additional_dataset additional dataset on host memory
 * @param[in,out] idx CAGRA index
 * @param[out] new_dataset_buffer_view memory buffer view for the dataset including the additional
 * part. The data will be copied from the current index in this function. The num rows must be the
 * sum of the original and additional datasets, cols must be the dimension of the dataset, and the
 * stride must be the same as the original index dataset. This view will be stored in the output
 * index. It is the caller's responsibility to ensure that dataset stays alive as long as the index.
 * This option is useful when users want to manage the memory space for the dataset themselves.
 * @param[out] new_graph_buffer_view memory buffer view for the graph including the additional part.
 * The data will be copied from the current index in this function. The num rows must be the sum of
 * the original and additional datasets and cols must be the graph degree. This view will be stored
 * in the output index. It is the caller's responsibility to ensure that dataset stays alive as long
 * as the index. This option is useful when users want to manage the memory space for the graph
 * themselves.
 */
void extend(
  raft::resources const& handle,
  const cagra::extend_params& params,
  raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> additional_dataset,
  cuvs::neighbors::cagra::index<uint8_t, uint32_t>& idx,
  std::optional<raft::device_matrix_view<uint8_t, int64_t, raft::layout_stride>>
    new_dataset_buffer_view                                                        = std::nullopt,
  std::optional<raft::device_matrix_view<uint32_t, int64_t>> new_graph_buffer_view = std::nullopt);

/** @brief Add new vectors to a CAGRA index
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace raft::neighbors;
 *   auto additional_dataset = raft::make_host_matrix<uint8_t, int64_t>(handle,add_size,dim);
 *   // set_additional_dataset(additional_dataset.view());
 *
 *   cagra::extend_params params;
 *   cagra::extend(res, params, raft::make_const_mdspan(additional_dataset.view()), index);
 * @endcode
 *
 * @param[in] handle raft resources
 * @param[in] params extend params
 * @param[in] additional_dataset additional dataset on host memory
 * @param[in,out] idx CAGRA index
 * @param[out] new_dataset_buffer_view memory buffer view for the dataset including the additional
 * part. The data will be copied from the current index in this function. The num rows must be the
 * sum of the original and additional datasets, cols must be the dimension of the dataset, and the
 * stride must be the same as the original index dataset. This view will be stored in the output
 * index. It is the caller's responsibility to ensure that dataset stays alive as long as the index.
 * This option is useful when users want to manage the memory space for the dataset themselves.
 * @param[out] new_graph_buffer_view memory buffer view for the graph including the additional part.
 * The data will be copied from the current index in this function. The num rows must be the sum of
 * the original and additional datasets and cols must be the graph degree. This view will be stored
 * in the output index. It is the caller's responsibility to ensure that dataset stays alive as long
 * as the index. This option is useful when users want to manage the memory space for the graph
 * themselves.
 */
void extend(
  raft::resources const& handle,
  const cagra::extend_params& params,
  raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> additional_dataset,
  cuvs::neighbors::cagra::index<uint8_t, uint32_t>& idx,
  std::optional<raft::device_matrix_view<uint8_t, int64_t, raft::layout_stride>>
    new_dataset_buffer_view                                                        = std::nullopt,
  std::optional<raft::device_matrix_view<uint32_t, int64_t>> new_graph_buffer_view = std::nullopt);
/**
 * @}
 */

/**
 * @defgroup cagra_cpp_index_search CAGRA search functions
 * @{
 * @brief Search ANN using the constructed index.
 *
 * See the [cagra::build](#cagra::build) documentation for a usage example.
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices
 *
 * @param[in] res raft resources
 * @param[in] params configure the search
 * @param[in] idx cagra index
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 */

void search(raft::resources const& res,
            cuvs::neighbors::cagra::search_params const& params,
            const cuvs::neighbors::cagra::index<float, uint32_t>& index,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_matrix_view<uint32_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances);

/**
 * @brief Search ANN using the constructed index.
 *
 * See the [cagra::build](#cagra::build) documentation for a usage example.
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices
 *
 * @param[in] res raft resources
 * @param[in] params configure the search
 * @param[in] index cagra index
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 */
void search(raft::resources const& res,
            cuvs::neighbors::cagra::search_params const& params,
            const cuvs::neighbors::cagra::index<int8_t, uint32_t>& index,
            raft::device_matrix_view<const int8_t, int64_t, raft::row_major> queries,
            raft::device_matrix_view<uint32_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances);

/**
 * @brief Search ANN using the constructed index.
 *
 * See the [cagra::build](#cagra::build) documentation for a usage example.
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices
 *
 * @param[in] res raft resources
 * @param[in] params configure the search
 * @param[in] index cagra index
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 */
void search(raft::resources const& res,
            cuvs::neighbors::cagra::search_params const& params,
            const cuvs::neighbors::cagra::index<uint8_t, uint32_t>& index,
            raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> queries,
            raft::device_matrix_view<uint32_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances);
/**
 * @}
 */

/**
 * @defgroup cagra_cpp_serialize CAGRA serialize functions
 * @{
 */

/**
 * Save the index to file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/cagra.hpp>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");
 * // create an index with `auto index = cuvs::neighbors::cagra::build(...);`
 * cuvs::neighbors::cagra::serialize(handle, filename, index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] filename the file name for saving the index
 * @param[in] index CAGRA index
 * @param[in] include_dataset Whether or not to write out the dataset to the file.
 *
 */
void serialize(raft::resources const& handle,
               const std::string& filename,
               const cuvs::neighbors::cagra::index<float, uint32_t>& index,
               bool include_dataset = true);

/**
 * Load index from file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/cagra.hpp>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");

 * cuvs::neighbors::cagra::index<float, uint32_t> index;
 * cuvs::neighbors::cagra::deserialize(handle, filename, &index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] filename the name of the file that stores the index
 * @param[out] index the cagra index
 */
void deserialize(raft::resources const& handle,
                 const std::string& filename,
                 cuvs::neighbors::cagra::index<float, uint32_t>* index);

/**
 * Write the index to an output stream
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/cagra.hpp>
 *
 * raft::resources handle;
 *
 * // create an output stream
 * std::ostream os(std::cout.rdbuf());
 * // create an index with `auto index = cuvs::neighbors::cagra::build(...);`
 * cuvs::neighbors::cagra::serialize(handle, os, index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] os output stream
 * @param[in] index CAGRA index
 * @param[in] include_dataset Whether or not to write out the dataset to the file.
 */
void serialize(raft::resources const& handle,
               std::ostream& os,
               const cuvs::neighbors::cagra::index<float, uint32_t>& index,
               bool include_dataset = true);

/**
 * Load index from input stream
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/cagra.hpp>
 *
 * raft::resources handle;
 *
 * // create an input stream
 * std::istream is(std::cin.rdbuf());
 * cuvs::neighbors::cagra::index<float, uint32_t> index;
 * cuvs::neighbors::cagra::deserialize(handle, is, &index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] is input stream
 * @param[out] index the cagra index
 */
void deserialize(raft::resources const& handle,
                 std::istream& is,
                 cuvs::neighbors::cagra::index<float, uint32_t>* index);

/**
 * Save the index to file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/cagra.hpp>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");
 * // create an index with `auto index = cuvs::neighbors::cagra::build(...);`
 * cuvs::neighbors::cagra::serialize(handle, filename, index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] filename the file name for saving the index
 * @param[in] index CAGRA index
 * @param[in] include_dataset Whether or not to write out the dataset to the file.
 */
void serialize(raft::resources const& handle,
               const std::string& filename,
               const cuvs::neighbors::cagra::index<int8_t, uint32_t>& index,
               bool include_dataset = true);

/**
 * Load index from file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/cagra.hpp>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");

 * cuvs::neighbors::cagra::index<int8_t, uint32_t> index;
 * cuvs::neighbors::cagra::deserialize(handle, filename, &index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] filename the name of the file that stores the index
 * @param[out] index the cagra index
 */
void deserialize(raft::resources const& handle,
                 const std::string& filename,
                 cuvs::neighbors::cagra::index<int8_t, uint32_t>* index);

/**
 * Write the index to an output stream
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/cagra.hpp>
 *
 * raft::resources handle;
 *
 * // create an output stream
 * std::ostream os(std::cout.rdbuf());
 * // create an index with `auto index = cuvs::neighbors::cagra::build(...);`
 * cuvs::neighbors::cagra::serialize(handle, os, index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] os output stream
 * @param[in] index CAGRA index
 * @param[in] include_dataset Whether or not to write out the dataset to the file.
 */
void serialize(raft::resources const& handle,
               std::ostream& os,
               const cuvs::neighbors::cagra::index<int8_t, uint32_t>& index,
               bool include_dataset = true);

/**
 * Load index from input stream
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/cagra.hpp>
 *
 * raft::resources handle;
 *
 * // create an input stream
 * std::istream is(std::cin.rdbuf());
 * cuvs::neighbors::cagra::index<int8_t, uint32_t> index;
 * cuvs::neighbors::cagra::deserialize(handle, is, &index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] is input stream
 * @param[out] index the cagra index
 */
void deserialize(raft::resources const& handle,
                 std::istream& is,
                 cuvs::neighbors::cagra::index<int8_t, uint32_t>* index);

/**
 * Save the index to file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/cagra.hpp>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");
 * // create an index with `auto index = cuvs::neighbors::cagra::build(...);`
 * cuvs::neighbors::cagra::serialize(handle, filename, index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] filename the file name for saving the index
 * @param[in] index CAGRA index
 * @param[in] include_dataset Whether or not to write out the dataset to the file.
 */
void serialize(raft::resources const& handle,
               const std::string& filename,
               const cuvs::neighbors::cagra::index<uint8_t, uint32_t>& index,
               bool include_dataset = true);

/**
 * Load index from file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/cagra.hpp>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");

 * cuvs::neighbors::cagra::index<uint8_t, uint32_t> index;
 * cuvs::neighbors::cagra::deserialize(handle, filename, &index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] filename the name of the file that stores the index
 * @param[out] index the cagra index
 */
void deserialize(raft::resources const& handle,
                 const std::string& filename,
                 cuvs::neighbors::cagra::index<uint8_t, uint32_t>* index);

/**
 * Write the index to an output stream
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/cagra.hpp>
 *
 * raft::resources handle;
 *
 * // create an output stream
 * std::ostream os(std::cout.rdbuf());
 * // create an index with `auto index = cuvs::neighbors::cagra::build(...);`
 * cuvs::neighbors::cagra::serialize(handle, os, index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] os output stream
 * @param[in] index CAGRA index
 * @param[in] include_dataset Whether or not to write out the dataset to the file.
 */
void serialize(raft::resources const& handle,
               std::ostream& os,
               const cuvs::neighbors::cagra::index<uint8_t, uint32_t>& index,
               bool include_dataset = true);

/**
 * Load index from input stream
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/cagra.hpp>
 *
 * raft::resources handle;
 *
 * // create an input stream
 * std::istream is(std::cin.rdbuf());
 * cuvs::neighbors::cagra::index<uint8_t, uint32_t> index;
 * cuvs::neighbors::cagra::deserialize(handle, is, &index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] is input stream
 * @param[out] index the cagra index
 */
void deserialize(raft::resources const& handle,
                 std::istream& is,
                 cuvs::neighbors::cagra::index<uint8_t, uint32_t>* index);

/**
 * Write the CAGRA built index as a base layer HNSW index to an output stream
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/cagra_serialize.hpp>
 *
 * raft::resources handle;
 *
 * // create an output stream
 * std::ostream os(std::cout.rdbuf());
 * // create an index with `auto index = raft::cagra::build(...);`
 * cuvs::neighbors::cagra::serialize_to_hnswlib(handle, os, index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] os output stream
 * @param[in] index CAGRA index
 *
 */
void serialize_to_hnswlib(raft::resources const& handle,
                          std::ostream& os,
                          const cuvs::neighbors::cagra::index<float, uint32_t>& index);

/**
 * Save a CAGRA build index in hnswlib base-layer-only serialized format
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/cagra_serialize.hpp>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");
 * // create an index with `auto index = raft::cagra::build(...);`
 * cuvs::neighbors::cagra::serialize_to_hnswlib(handle, filename, index);
 * @endcode
 *
 *
 * @param[in] handle the raft handle
 * @param[in] filename the file name for saving the index
 * @param[in] index CAGRA index
 *
 */
void serialize_to_hnswlib(raft::resources const& handle,
                          const std::string& filename,
                          const cuvs::neighbors::cagra::index<float, uint32_t>& index);

/**
 * Write the CAGRA built index as a base layer HNSW index to an output stream
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/cagra_serialize.hpp>
 *
 * raft::resources handle;
 *
 * // create an output stream
 * std::ostream os(std::cout.rdbuf());
 * // create an index with `auto index = raft::cagra::build(...);`
 * cuvs::neighbors::cagra::serialize_to_hnswlib(handle, os, index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] os output stream
 * @param[in] index CAGRA index
 *
 */
void serialize_to_hnswlib(raft::resources const& handle,
                          std::ostream& os,
                          const cuvs::neighbors::cagra::index<int8_t, uint32_t>& index);

/**
 * Save a CAGRA build index in hnswlib base-layer-only serialized format
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/cagra_serialize.hpp>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");
 * // create an index with `auto index = raft::cagra::build(...);`
 * cuvs::neighbors::cagra::serialize_to_hnswlib(handle, filename, index);
 * @endcode
 *
 *
 * @param[in] handle the raft handle
 * @param[in] filename the file name for saving the index
 * @param[in] index CAGRA index
 *
 */
void serialize_to_hnswlib(raft::resources const& handle,
                          const std::string& filename,
                          const cuvs::neighbors::cagra::index<int8_t, uint32_t>& index);

/**
 * Write the CAGRA built index as a base layer HNSW index to an output stream
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/cagra_serialize.hpp>
 *
 * raft::resources handle;
 *
 * // create an output stream
 * std::ostream os(std::cout.rdbuf());
 * // create an index with `auto index = raft::cagra::build(...);`
 * cuvs::neighbors::cagra::serialize_to_hnswlib(handle, os, index);
 * @endcode
 *
 * @param[in] handle the raft handle
 * @param[in] os output stream
 * @param[in] index CAGRA index
 *
 */
void serialize_to_hnswlib(raft::resources const& handle,
                          std::ostream& os,
                          const cuvs::neighbors::cagra::index<uint8_t, uint32_t>& index);

/**
 * Save a CAGRA build index in hnswlib base-layer-only serialized format
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <cuvs/neighbors/cagra_serialize.hpp>
 *
 * raft::resources handle;
 *
 * // create a string with a filepath
 * std::string filename("/path/to/index");
 * // create an index with `auto index = raft::cagra::build(...);`
 * cuvs::neighbors::cagra::serialize_to_hnswlib(handle, filename, index);
 * @endcode
 *
 *
 * @param[in] handle the raft handle
 * @param[in] filename the file name for saving the index
 * @param[in] index CAGRA index
 *
 */
void serialize_to_hnswlib(raft::resources const& handle,
                          const std::string& filename,
                          const cuvs::neighbors::cagra::index<uint8_t, uint32_t>& index);

/**
 * @}
 */

}  // namespace cuvs::neighbors::cagra
