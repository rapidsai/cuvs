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

#include <cuvs/neighbors/common.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#include <cuvs/distance/distance.hpp>

#include <cuda_fp16.h>

namespace cuvs::neighbors::nn_descent {
/**
 * @defgroup nn_descent_cpp_index_params The nn-descent algorithm parameters.
 * @{
 */

/**
 * @brief Parameters used to build an nn-descent index
 *
 * `graph_degree`: For an input dataset of dimensions (N, D),
 * determines the final dimensions of the all-neighbors knn graph
 * which turns out to be of dimensions (N, graph_degree)
 * `intermediate_graph_degree`: Internally, nn-descent builds an
 * all-neighbors knn graph of dimensions (N, intermediate_graph_degree)
 * before selecting the final `graph_degree` neighbors. It's recommended
 * that `intermediate_graph_degree` >= 1.5 * graph_degree
 * `max_iterations`: The number of iterations that nn-descent will refine
 * the graph for. More iterations produce a better quality graph at cost of performance
 * `termination_threshold`: The delta at which nn-descent will terminate its iterations
 *
 */
struct index_params : cuvs::neighbors::index_params {
  size_t graph_degree              = 64;      // Degree of output graph.
  size_t intermediate_graph_degree = 128;     // Degree of input graph for pruning.
  size_t max_iterations            = 20;      // Number of nn-descent iterations.
  float termination_threshold      = 0.0001;  // Termination threshold of nn-descent.

  /** @brief Construct NN descent parameters for a specific kNN graph degree
   *
   * @param graph_degree output graph degree
   */
  index_params(size_t graph_degree = 64)
    : graph_degree(graph_degree), intermediate_graph_degree(1.5 * graph_degree)
  {
  }
};

/**
 * @}
 */

/**
 * @defgroup nn_descent_cpp_index nn-descent index
 * @{
 */
/**
 * @brief nn-descent Build an nn-descent index
 * The index contains an all-neighbors graph of the input dataset
 * stored in host memory of dimensions (n_rows, n_cols)
 *
 * Reference:
 * Hui Wang, Wan-Lei Zhao, Xiangxiang Zeng, and Jianye Yang. 2021.
 * Fast k-NN Graph Construction by GPU based NN-Descent. In Proceedings of the 30th ACM
 * International Conference on Information & Knowledge Management (CIKM '21). Association for
 * Computing Machinery, New York, NY, USA, 1929–1938. https://doi.org/10.1145/3459637.3482344
 *
 * @tparam IdxT dtype to be used for constructing knn-graph
 */
template <typename IdxT>
struct index : cuvs::neighbors::index {
 public:
  /**
   * @brief Construct a new index object
   *
   * This constructor creates an nn-descent index which is a knn-graph in host memory.
   * The type of the knn-graph is a dense raft::host_matrix and dimensions are
   * (n_rows, n_cols).
   *
   * @param res raft::resources is an object mangaging resources
   * @param n_rows number of rows in knn-graph
   * @param n_cols number of cols in knn-graph
   */
  index(raft::resources const& res, int64_t n_rows, int64_t n_cols)
    : cuvs::neighbors::index(),
      res_{res},
      metric_{cuvs::distance::DistanceType::L2Expanded},
      graph_{raft::make_host_matrix<IdxT, int64_t, raft::row_major>(n_rows, n_cols)},
      graph_view_{graph_.view()}
  {
  }

  /**
   * @brief Construct a new index object
   *
   * This constructor creates an nn-descent index using a user allocated host memory knn-graph.
   * The type of the knn-graph is a dense raft::host_matrix and dimensions are
   * (n_rows, n_cols).
   *
   * @param res raft::resources is an object mangaging resources
   * @param graph_view raft::host_matrix_view<IdxT, int64_t, raft::row_major> for storing knn-graph
   */
  index(raft::resources const& res,
        raft::host_matrix_view<IdxT, int64_t, raft::row_major> graph_view)
    : cuvs::neighbors::index(),
      res_{res},
      metric_{cuvs::distance::DistanceType::L2Expanded},
      graph_{raft::make_host_matrix<IdxT, int64_t, raft::row_major>(0, 0)},
      graph_view_{graph_view}
  {
  }

  /** Distance metric used for clustering. */
  [[nodiscard]] constexpr inline auto metric() const noexcept -> cuvs::distance::DistanceType
  {
    return metric_;
  }

  // /** Total length of the index (number of vectors). */
  [[nodiscard]] constexpr inline auto size() const noexcept -> IdxT
  {
    return graph_view_.extent(0);
  }

  /** Graph degree */
  [[nodiscard]] constexpr inline auto graph_degree() const noexcept -> uint32_t
  {
    return graph_view_.extent(1);
  }

  /** neighborhood graph [size, graph-degree] */
  [[nodiscard]] inline auto graph() noexcept
    -> raft::host_matrix_view<IdxT, int64_t, raft::row_major>
  {
    return graph_view_;
  }

  // Don't allow copying the index for performance reasons (try avoiding copying data)
  index(const index&)                    = delete;
  index(index&&)                         = default;
  auto operator=(const index&) -> index& = delete;
  auto operator=(index&&) -> index&      = default;
  ~index()                               = default;

 private:
  raft::resources const& res_;
  cuvs::distance::DistanceType metric_;
  raft::host_matrix<IdxT, int64_t, raft::row_major> graph_;  // graph to return for non-int IdxT
  raft::host_matrix_view<IdxT, int64_t, raft::row_major>
    graph_view_;  // view of graph for user provided matrix
};

/** @} */

/**
 * @defgroup nn_descent_cpp_index_build nn-descent index build
 * @{
 */

/**
 * @brief Build nn-descent Index with dataset in device memory
 *
 * The following distance metrics are supported:
 * - L2
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   nn_descent::index_params index_params;
 *   // create and fill the index from a [N, D] raft::device_matrix_view dataset
 *   auto index = nn_descent::build(res, index_params, dataset);
 *   // index.graph() provides a raft::host_matrix_view of an
 *   // all-neighbors knn graph of dimensions [N, k] of the input
 *   // dataset
 * @endcode
 *
 * @param[in] res raft::resources is an object mangaging resources
 * @param[in] params an instance of nn_descent::index_params that are parameters
 *               to run the nn-descent algorithm
 * @param[in] dataset raft::device_matrix_view input dataset expected to be located
 *                in device memory
 * @return index<IdxT> index containing all-neighbors knn graph in host memory
 */
auto build(raft::resources const& res,
           index_params const& params,
           raft::device_matrix_view<const float, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::nn_descent::index<uint32_t>;

/**
 * @brief Build nn-descent Index with dataset in host memory
 *
 * The following distance metrics are supported:
 * - L2
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors::experimental;
 *   // use default index parameters
 *   nn_descent::index_params index_params;
 *   // create and fill the index from a [N, D] raft::host_matrix_view dataset
 *   auto index = cagra::build(res, index_params, dataset);
 *   // index.graph() provides a raft::host_matrix_view of an
 *   // all-neighbors knn graph of dimensions [N, k] of the input
 *   // dataset
 * @endcode
 *
 * @tparam T data-type of the input dataset
 * @tparam IdxT data-type for the output index
 * @param res raft::resources is an object mangaging resources
 * @param[in] params an instance of nn_descent::index_params that are parameters
 *               to run the nn-descent algorithm
 * @param[in] dataset raft::host_matrix_view input dataset expected to be located
 *                in host memory
 * @return index<IdxT> index containing all-neighbors knn graph in host memory
 */
auto build(raft::resources const& res,
           index_params const& params,
           raft::host_matrix_view<const float, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::nn_descent::index<uint32_t>;

/**
 * @brief Build nn-descent Index with dataset in device memory
 *
 * The following distance metrics are supported:
 * - L2
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   nn_descent::index_params index_params;
 *   // create and fill the index from a [N, D] raft::device_matrix_view dataset
 *   auto index = nn_descent::build(res, index_params, dataset);
 *   // index.graph() provides a raft::host_matrix_view of an
 *   // all-neighbors knn graph of dimensions [N, k] of the input
 *   // dataset
 * @endcode
 *
 * @param[in] res raft::resources is an object mangaging resources
 * @param[in] params an instance of nn_descent::index_params that are parameters
 *               to run the nn-descent algorithm
 * @param[in] dataset raft::device_matrix_view input dataset expected to be located
 *                in device memory
 * @return index<IdxT> index containing all-neighbors knn graph in host memory
 */
auto build(raft::resources const& res,
           index_params const& params,
           raft::device_matrix_view<const half, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::nn_descent::index<uint32_t>;

/**
 * @brief Build nn-descent Index with dataset in host memory
 *
 * The following distance metrics are supported:
 * - L2
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors::experimental;
 *   // use default index parameters
 *   nn_descent::index_params index_params;
 *   // create and fill the index from a [N, D] raft::host_matrix_view dataset
 *   auto index = cagra::build(res, index_params, dataset);
 *   // index.graph() provides a raft::host_matrix_view of an
 *   // all-neighbors knn graph of dimensions [N, k] of the input
 *   // dataset
 * @endcode
 *
 * @tparam T data-type of the input dataset
 * @tparam IdxT data-type for the output index
 * @param res raft::resources is an object mangaging resources
 * @param[in] params an instance of nn_descent::index_params that are parameters
 *               to run the nn-descent algorithm
 * @param[in] dataset raft::host_matrix_view input dataset expected to be located
 *                in host memory
 * @return index<IdxT> index containing all-neighbors knn graph in host memory
 */
auto build(raft::resources const& res,
           index_params const& params,
           raft::host_matrix_view<const half, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::nn_descent::index<uint32_t>;

/**
 * @brief Build nn-descent Index with dataset in device memory
 *
 * The following distance metrics are supported:
 * - L2
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   nn_descent::index_params index_params;
 *   // create and fill the index from a [N, D] raft::device_matrix_view dataset
 *   auto index = nn_descent::build(res, index_params, dataset);
 *   // index.graph() provides a raft::host_matrix_view of an
 *   // all-neighbors knn graph of dimensions [N, k] of the input
 *   // dataset
 * @endcode
 *
 * @param[in] res raft::resources is an object mangaging resources
 * @param[in] params an instance of nn_descent::index_params that are parameters
 *               to run the nn-descent algorithm
 * @param[in] dataset raft::device_matrix_view input dataset expected to be located
 *                in device memory
 * @return index<IdxT> index containing all-neighbors knn graph in host memory
 */
auto build(raft::resources const& res,
           index_params const& params,
           raft::device_matrix_view<const int8_t, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::nn_descent::index<uint32_t>;

/**
 * @brief Build nn-descent Index with dataset in host memory
 *
 * The following distance metrics are supported:
 * - L2
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors::experimental;
 *   // use default index parameters
 *   nn_descent::index_params index_params;
 *   // create and fill the index from a [N, D] raft::host_matrix_view dataset
 *   auto index = cagra::build(res, index_params, dataset);
 *   // index.graph() provides a raft::host_matrix_view of an
 *   // all-neighbors knn graph of dimensions [N, k] of the input
 *   // dataset
 * @endcode
 *
 * @tparam T data-type of the input dataset
 * @tparam IdxT data-type for the output index
 * @param res raft::resources is an object mangaging resources
 * @param[in] params an instance of nn_descent::index_params that are parameters
 *               to run the nn-descent algorithm
 * @param[in] dataset raft::host_matrix_view input dataset expected to be located
 *                in host memory
 * @return index<IdxT> index containing all-neighbors knn graph in host memory
 */
auto build(raft::resources const& res,
           index_params const& params,
           raft::host_matrix_view<const int8_t, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::nn_descent::index<uint32_t>;

/**
 * @brief Build nn-descent Index with dataset in device memory
 *
 * The following distance metrics are supported:
 * - L2
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   nn_descent::index_params index_params;
 *   // create and fill the index from a [N, D] raft::device_matrix_view dataset
 *   auto index = nn_descent::build(res, index_params, dataset);
 *   // index.graph() provides a raft::host_matrix_view of an
 *   // all-neighbors knn graph of dimensions [N, k] of the input
 *   // dataset
 * @endcode
 *
 * @param[in] res raft::resources is an object mangaging resources
 * @param[in] params an instance of nn_descent::index_params that are parameters
 *               to run the nn-descent algorithm
 * @param[in] dataset raft::device_matrix_view input dataset expected to be located
 *                in device memory
 * @return index<IdxT> index containing all-neighbors knn graph in host memory
 */
auto build(raft::resources const& res,
           index_params const& params,
           raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::nn_descent::index<uint32_t>;

/** @} */

/**
 * @brief Build nn-descent Index with dataset in host memory
 *
 * The following distance metrics are supported:
 * - L2
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors::experimental;
 *   // use default index parameters
 *   nn_descent::index_params index_params;
 *   // create and fill the index from a [N, D] raft::host_matrix_view dataset
 *   auto index = cagra::build(res, index_params, dataset);
 *   // index.graph() provides a raft::host_matrix_view of an
 *   // all-neighbors knn graph of dimensions [N, k] of the input
 *   // dataset
 * @endcode
 *
 * @tparam T data-type of the input dataset
 * @tparam IdxT data-type for the output index
 * @param res raft::resources is an object mangaging resources
 * @param[in] params an instance of nn_descent::index_params that are parameters
 *               to run the nn-descent algorithm
 * @param[in] dataset raft::host_matrix_view input dataset expected to be located
 *                in host memory
 * @return index<IdxT> index containing all-neighbors knn graph in host memory
 */
auto build(raft::resources const& res,
           index_params const& params,
           raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::nn_descent::index<uint32_t>;

/**
 * @brief Test if we have enough GPU memory to run NN descent algorithm.
 *
 * @param res
 * @param dataset shape of the dataset
 * @param idx_size the size of index type in bytes
 * @return true if enough GPU memory can be allocated
 * @return false otherwise
 */
bool has_enough_device_memory(raft::resources const& res,
                              raft::matrix_extent<int64_t> dataset,
                              size_t idx_size = 4);

}  // namespace cuvs::neighbors::nn_descent