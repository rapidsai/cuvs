/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#ifdef CUVS_BUILD_CAGRA_HNSWLIB

#pragma once

#include "common.hpp"

#include <cuvs/distance/distance.hpp>

#include "cagra.hpp"
#include <raft/core/host_mdspan.hpp>

#include <sys/types.h>

#include <cstdint>
#include <memory>
#include <type_traits>

namespace cuvs::neighbors::hnsw {

/**
 * @defgroup hnsw_cpp_search_params Build CAGRA index and search with hnswlib
 * @{
 */

struct search_params : cuvs::neighbors::search_params {
  int ef;               // size of the candidate list
  int num_threads = 0;  // number of host threads to use for concurrent searches. Value of 0
                        // automatically maximizes parallelism
};

/**@}*/

/**
 * @defgroup hnsw_cpp_index hnswlib index wrapper
 * @{
 */

template <typename T>
struct index : cuvs::neighbors::index {
 public:
  /**
   * @brief load a base-layer-only hnswlib index originally saved from a built CAGRA index.
   *  This is a virtual class and it cannot be used directly. To create an index, use the factory
   *  function `cuvs::neighbors::hnsw::from_cagra` from the header
   *  `cuvs/neighbors/hnsw.hpp`
   *
   * @param[in] dim dimensions of the training dataset
   * @param[in] metric distance metric to search. Supported metrics ("L2Expanded", "InnerProduct")
   */
  index(int dim, cuvs::distance::DistanceType metric) : dim_{dim}, metric_{metric} {}

  virtual ~index() {}

  /**
  @brief Get underlying index
  */
  virtual auto get_index() const -> void const* = 0;

  auto dim() const -> int const { return dim_; }

  auto metric() const -> cuvs::distance::DistanceType { return metric_; }

  /**
  @brief Set ef for search
  */
  virtual void set_ef(int ef) const;

 private:
  int dim_;
  cuvs::distance::DistanceType metric_;
};

/**@}*/

/**
 * @defgroup hnsw_cpp_index_load Load CAGRA index as hnswlib index
 * @{
 */

/**
 * @brief Construct an immutable hnswlib base-layer-only index from a CAGRA index
 * NOTE: This method uses the filesystem to write the CAGRA index in `/tmp/<random_number>.bin`
 * before reading it as an hnswlib index, then deleting the temporary file. The returned index
 * is immutable and can only be searched by the hnswlib wrapper in cuVS, as the format is not
 * compatible with the original hnswlib.
 *
 * @param[in] res raft resources
 * @param[in] cagra_index cagra index
 *
 * Usage example:
 * @code{.cpp}
 *   // Build a CAGRA index
 *   using namespace raft::neighbors;
 *   // use default index parameters
 *   cagra::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = cagra::build<float, uint32_t>(res, index_params, dataset);
 *
 *   // Load CAGRA index as base-layer-only hnswlib index
 *   auto hnsw_index = hnsw::from_cagra(res, index);
 * @endcode
 */
std::unique_ptr<index<float>> from_cagra(
  raft::resources const& res, const cuvs::neighbors::cagra::index<float, uint32_t>& cagra_index);

/**
 * @brief Construct an immutable hnswlib base-layer-only index from a CAGRA index
 * NOTE: This method uses the filesystem to write the CAGRA index in `/tmp/<random_number>.bin`
 * before reading it as an hnswlib index, then deleting the temporary file.  The returned index
 * is immutable and can only be searched by the hnswlib wrapper in cuVS, as the format is not
 * compatible with the original hnswlib.
 *
 * @param[in] res raft resources
 * @param[in] cagra_index cagra index
 *
 * Usage example:
 * @code{.cpp}
 *   // Build a CAGRA index
 *   using namespace raft::neighbors;
 *   // use default index parameters
 *   cagra::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = cagra::build<uint8_t, uint32_t>(res, index_params, dataset);
 *
 *   // Load CAGRA index as base-layer-only hnswlib index
 *   auto hnsw_index = hnsw::from_cagra(res, index);
 * @endcode
 */
std::unique_ptr<index<uint8_t>> from_cagra(
  raft::resources const& res, const cuvs::neighbors::cagra::index<uint8_t, uint32_t>& cagra_index);

/**
 * @brief Construct an immutable hnswlib base-layer-only index from a CAGRA index
 * NOTE: This method uses the filesystem to write the CAGRA index in `/tmp/<random_number>.bin`
 * before reading it as an hnswlib index, then deleting the temporary file.  The returned index
 * is immutable and can only be searched by the hnswlib wrapper in cuVS, as the format is not
 * compatible with the original hnswlib.
 *
 * @param[in] res raft resources
 * @param[in] cagra_index cagra index
 *
 * Usage example:
 * @code{.cpp}
 *   // Build a CAGRA index
 *   using namespace raft::neighbors;
 *   // use default index parameters
 *   cagra::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = cagra::build<int8_t, uint32_t>(res, index_params, dataset);
 *
 *   // Load CAGRA index as base-layer-only hnswlib index
 *   auto hnsw_index = hnsw::from_cagra(res, index);
 * @endcode
 */
std::unique_ptr<index<int8_t>> from_cagra(
  raft::resources const& res, const cuvs::neighbors::cagra::index<int8_t, uint32_t>& cagra_index);

/**@}*/

/**
 * @defgroup hnsw_cpp_index_search Search hnswlib index
 * @{
 */

/**
 * @brief Search hnswlib base-layer-only index constructed from a CAGRA index
 * NOTE: The HNSW index can only be searched by the hnswlib wrapper in cuVS,
 *       as the format is not compatible with the original hnswlib.
 *
 * @param[in] res raft resources
 * @param[in] params configure the search
 * @param[in] idx cagra index
 * @param[in] queries a host matrix view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors a host matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a host matrix view to the distances to the selected neighbors [n_queries,
 * k]
 *
 * Usage example:
 * @code{.cpp}
 *   // Build a CAGRA index
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   cagra::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = cagra::build<float, uint32_t>(res, index_params, dataset);
 *
 *   // Load CAGRA index as a base-layer HNSW index using the filesystem
 *   auto hnsw_index = hnsw::from_cagra(res, index);
 *
 *   // Search K nearest neighbors as an hnswlib index
 *   // using host threads for concurrency
 *   hnsw::search_params search_params;
 *   search_params.ef = 50 // ef >= K;
 *   search_params.num_threads = 10;
 *   auto neighbors = raft::make_host_matrix<uint64_t>(res, n_queries, k);
 *   auto distances = raft::make_host_matrix<float>(res, n_queries, k);
 *   hnsw::search(res, search_params, *index.get(), queries, neighbors, distances);
 * @endcode
 */
void search(raft::resources const& res,
            const search_params& params,
            const index<float>& idx,
            raft::host_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::host_matrix_view<uint64_t, int64_t, raft::row_major> neighbors,
            raft::host_matrix_view<float, int64_t, raft::row_major> distances);

/**
 * @brief Search hnswlib base-layer-only index constructed from a CAGRA index
 * NOTE: The HNSW index can only be searched by the hnswlib wrapper in cuVS,
 *       as the format is not compatible with the original hnswlib.
 *
 * @param[in] res raft resources
 * @param[in] params configure the search
 * @param[in] idx cagra index
 * @param[in] queries a host matrix view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors a host matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a host matrix view to the distances to the selected neighbors [n_queries,
 * k]
 *
 * Usage example:
 * @code{.cpp}
 *   // Build a CAGRA index
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   cagra::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = cagra::build<uint8_t, uint32_t>(res, index_params, dataset);
 *
 *   // Load CAGRA index as a base-layer HNSW index using the filesystem
 *   auto hnsw_index = hnsw::from_cagra(res, index);
 *
 *   // Search K nearest neighbors as an hnswlib index
 *   // using host threads for concurrency
 *   hnsw::search_params search_params;
 *   search_params.ef = 50 // ef >= K;
 *   search_params.num_threads = 10;
 *   auto neighbors = raft::make_host_matrix<uint64_t>(res, n_queries, k);
 *   auto distances = raft::make_host_matrix<float>(res, n_queries, k);
 *   hnsw::search(res, search_params, *index.get(), queries, neighbors, distances);
 * @endcode
 */
void search(raft::resources const& res,
            const search_params& params,
            const index<uint8_t>& idx,
            raft::host_matrix_view<const int, int64_t, raft::row_major> queries,
            raft::host_matrix_view<uint64_t, int64_t, raft::row_major> neighbors,
            raft::host_matrix_view<float, int64_t, raft::row_major> distances);

/**
 * @brief Search hnswlib base-layer-only index constructed from a CAGRA index
 * NOTE: The HNSW index can only be searched by the hnswlib wrapper in cuVS,
 *       as the format is not compatible with the original hnswlib.
 *
 * @param[in] res raft resources
 * @param[in] params configure the search
 * @param[in] idx cagra index
 * @param[in] queries a host matrix view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors a host matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a host matrix view to the distances to the selected neighbors [n_queries,
 * k]
 *
 * Usage example:
 * @code{.cpp}
 *   // Build a CAGRA index
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   cagra::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = cagra::build<int8_t, uint32_t>(res, index_params, dataset);
 *
 *   // Load CAGRA index as a base-layer HNSW index using the filesystem
 *   auto hnsw_index = hnsw::from_cagra(res, index);
 *
 *   // Search K nearest neighbors as an hnswlib index
 *   // using host threads for concurrency
 *   hnsw::search_params search_params;
 *   search_params.ef = 50 // ef >= K;
 *   search_params.num_threads = 10;
 *   auto neighbors = raft::make_host_matrix<uint64_t>(res, n_queries, k);
 *   auto distances = raft::make_host_matrix<float>(res, n_queries, k);
 *   hnsw::search(res, search_params, *index.get(), queries, neighbors, distances);
 * @endcode
 */
void search(raft::resources const& res,
            const search_params& params,
            const index<int8_t>& idx,
            raft::host_matrix_view<const int, int64_t, raft::row_major> queries,
            raft::host_matrix_view<uint64_t, int64_t, raft::row_major> neighbors,
            raft::host_matrix_view<float, int64_t, raft::row_major> distances);

/**@}*/

/**
 * @defgroup hnsw_cpp_index_deserialize Deserialize CAGRA index as hnswlib index
 * @{
 */

/**
 * @brief De-serialize a CAGRA index saved to a file as an hnswlib index
 * NOTE: The loaded hnswlib index is immutable, and only be read by the
 * hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib.
 *
 * @param[in] res raft resources
 * @param[in] filename path to the file containing the serialized CAGRA index
 * @param[in] dim dimensions of the training dataset
 * @param[in] metric distance metric to search. Supported metrics ("L2Expanded", "InnerProduct")
 * @param[out] index hnsw index
 *
 * Usage example:
 * @code{.cpp}
 *   // Build a CAGRA index
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   cagra::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = cagra::build<float, uint32_t>(res, index_params, dataset);
 *
 *   // save a CAGRA index to a file
 *   cagra::serialize(res, index, "index.bin");
 *   // De-serialize a CAGRA index as a base-layer HNSW index using the filesystem
 *   index<float>* hnsw_index = nullptr;
 *   hnsw::deserialize(res, "index.bin", index->dim(), index->metric(), &hnsw_index);
 *
 *   // Delete index after use
 *   delete hnsw_index;
 * @endcode
 */
void deserialize(raft::resources const& res,
                 const std::string& filename,
                 int dim,
                 cuvs::distance::DistanceType metric,
                 index<float>** index);

/**
 * @brief De-serialize a CAGRA index saved to a file as an hnswlib index
 * NOTE: The loaded hnswlib index is immutable, and only be read by the
 * hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib.
 *
 * @param[in] res raft resources
 * @param[in] filename path to the file containing the serialized CAGRA index
 * @param[in] dim dimensions of the training dataset
 * @param[in] metric distance metric to search. Supported metrics ("L2Expanded", "InnerProduct")
 * @param[out] index hnsw index
 *
 * Usage example:
 * @code{.cpp}
 *   // Build a CAGRA index
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   cagra::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = cagra::build<uint8_t, uint32_t>(res, index_params, dataset);
 *
 *   // save a CAGRA index to a file
 *   cagra::serialize(res, index, "index.bin");
 *   // De-serialize a CAGRA index as a base-layer HNSW index using the filesystem
 *   index<uint8_t>* hnsw_index = nullptr;
 *   hnsw::deserialize(res, "index.bin", index->dim(), index->metric(), &hnsw_index);
 *
 *   // Delete index after use
 *   delete hnsw_index;
 * @endcode
 */
void deserialize(raft::resources const& res,
                 const std::string& filename,
                 int dim,
                 cuvs::distance::DistanceType metric,
                 index<uint8_t>** index);

/**
 * @brief De-serialize a CAGRA index saved to a file as an hnswlib index
 * NOTE: The loaded hnswlib index is immutable, and only be read by the
 * hnswlib wrapper in cuVS, as the serialization format is not compatible with the original hnswlib.
 *
 * @param[in] res raft resources
 * @param[in] filename path to the file containing the serialized CAGRA index
 * @param[in] dim dimensions of the training dataset
 * @param[in] metric distance metric to search. Supported metrics ("L2Expanded", "InnerProduct")
 * @param[out] index hnsw index
 *
 * Usage example:
 * @code{.cpp}
 *   // Build a CAGRA index
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   cagra::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = cagra::build<int8_t, uint32_t>(res, index_params, dataset);
 *
 *   // save a CAGRA index to a file
 *   cagra::serialize(res, index, "index.bin");
 *   // De-serialize a CAGRA index as a base-layer HNSW index using the filesystem
 *   index<int8_t>* hnsw_index = nullptr;
 *   hnsw::deserialize(res, "index.bin", index->dim(), index->metric(), &hnsw_index);
 *
 *   // Delete index after use
 *   delete hnsw_index;
 * @endcode
 */
void deserialize(raft::resources const& res,
                 const std::string& filename,
                 int dim,
                 cuvs::distance::DistanceType metric,
                 index<int8_t>** index);

/**@}*/

}  // namespace cuvs::neighbors::hnsw

#else
#error "This header is only available if cuVS CMake option `BUILD_CAGRA_HNSWLIB=ON"
#endif