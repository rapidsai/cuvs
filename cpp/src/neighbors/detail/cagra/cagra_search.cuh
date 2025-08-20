/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

#include <cstdio>
#include <iostream>
#include "../../../core/nvtx.hpp"
#include "factory.cuh"
#include "sample_filter_utils.cuh"
#include "search_plan.cuh"
#include "search_single_cta_inst.cuh"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>
// #include <raft/util/print.hpp>

#include <cuvs/distance/distance.hpp>

#include <cuvs/neighbors/cagra.hpp>

// TODO: Fix these when ivf methods are moved over
#include "../../ivf_common.cuh"
#include "../../ivf_pq/ivf_pq_search.cuh"
#include <cuvs/neighbors/common.hpp>

// TODO: This shouldn't be calling spatial/knn apis
#include "../ann_utils.cuh"

#include <raft/linalg/map.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/util/integer_utils.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

namespace cuvs::neighbors::cagra::detail {

template <typename DataT,
          typename IndexT,
          typename DistanceT,
          typename CagraSampleFilterT,
          typename OutputIdxT = IndexT>
void search_main_core(raft::resources const& res,
                      search_params params,
                      const dataset_descriptor_host<DataT, IndexT, DistanceT>& dataset_desc,
                      raft::device_matrix_view<const IndexT, int64_t, raft::row_major> graph,
                      raft::device_matrix_view<const DataT, int64_t, raft::row_major> queries,
                      raft::device_matrix_view<OutputIdxT, int64_t, raft::row_major> neighbors,
                      raft::device_matrix_view<DistanceT, int64_t, raft::row_major> distances,
                      CagraSampleFilterT sample_filter = CagraSampleFilterT())
{
  RAFT_LOG_INFO("[CAGRA DEBUG] Entering search_main_core: dataset_size=%ld, graph_degree=%ld, max_queries=%u",
                graph.extent(0), graph.extent(1), params.max_queries);
  RAFT_LOG_DEBUG("# dataset size = %lu, dim = %lu\n",
                 static_cast<size_t>(graph.extent(0)),
                 static_cast<size_t>(queries.extent(1)));
  RAFT_LOG_DEBUG("# query size = %lu, dim = %lu\n",
                 static_cast<size_t>(queries.extent(0)),
                 static_cast<size_t>(queries.extent(1)));
  const uint32_t topk = neighbors.extent(1);

  cudaDeviceProp deviceProp = raft::resource::get_device_properties(res);
  if (params.max_queries == 0) {
    params.max_queries = std::min<size_t>(queries.extent(0), deviceProp.maxGridSize[1]);
  }

  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "cagra::search(max_queries = %u, k = %u, dim = %zu)",
    params.max_queries,
    topk,
    queries.extent(1));

  using CagraSampleFilterT_s = typename CagraSampleFilterT_Selector<CagraSampleFilterT>::type;
  RAFT_LOG_INFO("[CAGRA DEBUG] Creating search plan: topk=%u, max_queries=%u", topk, params.max_queries);
  std::unique_ptr<search_plan_impl<DataT, IndexT, DistanceT, CagraSampleFilterT_s, OutputIdxT>>
    plan = factory<DataT, IndexT, DistanceT, CagraSampleFilterT_s, OutputIdxT>::create(
      res, params, dataset_desc, queries.extent(1), graph.extent(0), graph.extent(1), topk);
  RAFT_LOG_INFO("[CAGRA DEBUG] Search plan created successfully");

  plan->check(topk);

  RAFT_LOG_DEBUG("Cagra search");
  const uint32_t max_queries = plan->max_queries;
  const uint32_t query_dim   = queries.extent(1);

  RAFT_LOG_INFO("[CAGRA DEBUG] Starting query processing loop: total_queries=%ld, max_queries=%u", queries.extent(0), max_queries);
  for (unsigned qid = 0; qid < queries.extent(0); qid += max_queries) {
    const uint32_t n_queries = std::min<std::size_t>(max_queries, queries.extent(0) - qid);
    RAFT_LOG_INFO("[CAGRA DEBUG] Processing batch: qid=%u, n_queries=%u", qid, n_queries);
    auto _topk_indices_ptr   = neighbors.data_handle() + (topk * qid);
    auto _topk_distances_ptr = distances.data_handle() + (topk * qid);
    // todo(tfeher): one could keep distances optional and pass nullptr
    const auto* _query_ptr = queries.data_handle() + (query_dim * qid);
    const auto* _seed_ptr =
      plan->num_seeds > 0
        ? reinterpret_cast<const IndexT*>(plan->dev_seed.data()) + (plan->num_seeds * qid)
        : nullptr;
    uint32_t* _num_executed_iterations = nullptr;

    RAFT_LOG_INFO("[CAGRA DEBUG] Executing search plan for batch qid=%u", qid);
    (*plan)(res,
            graph,
            _topk_indices_ptr,
            _topk_distances_ptr,
            _query_ptr,
            n_queries,
            _seed_ptr,
            _num_executed_iterations,
            topk,
            set_offset(sample_filter, qid));
    RAFT_LOG_INFO("[CAGRA DEBUG] Batch qid=%u completed", qid);
    
    // Print some neighbors and distances after batch completion (useful for all metrics, especially cosine)
    if (qid == 0) {  // Only print for first batch to avoid too much output
      RAFT_LOG_INFO("[CAGRA DEBUG] Sample results after first batch completion:");
      raft::print_device_vector("batch_queries[0:8]", _query_ptr, std::min(8, (int)query_dim), std::cout);
      raft::print_device_vector("batch_neighbors[0:8]", _topk_indices_ptr, std::min(8, (int)topk), std::cout);
      raft::print_device_vector("batch_distances[0:8]", _topk_distances_ptr, std::min(8, (int)topk), std::cout);
    }
  }
  
  // Print final summary of results from core search (before postprocessing)
  RAFT_LOG_INFO("[CAGRA DEBUG] Core search completed. Final summary:");
  RAFT_LOG_INFO("[CAGRA DEBUG] Total queries processed: %ld, neighbors per query: %ld", queries.extent(0), neighbors.extent(1));
  // Print a sample of the final neighbors and distances from the last batch
  RAFT_LOG_INFO("[CAGRA DEBUG] Sample final core results (useful for cosine metric analysis):");
  raft::print_device_vector("final_core_neighbors[0:6]", neighbors.data_handle(), std::min(6, (int)neighbors.extent(1)), std::cout);
  raft::print_device_vector("final_core_distances[0:6]", distances.data_handle(), std::min(6, (int)distances.extent(1)), std::cout);
}

/**
 * @brief Search ANN using the constructed index.
 *
 * See the [build](#build) documentation for a usage example.
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices in the CAGRA graph
 * @tparam OutputIdxT type of the returned indices
 *
 * @param[in] handle
 * @param[in] params configure the search
 * @param[in] idx ivf-pq constructed index
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 */
template <typename T,
          typename OutputIdxT,
          typename CagraSampleFilterT,
          typename IdxT      = uint32_t,
          typename DistanceT = float>
void search_main(raft::resources const& res,
                 search_params params,
                 const index<T, IdxT>& index,
                 raft::device_matrix_view<const T, int64_t, raft::row_major> queries,
                 raft::device_matrix_view<OutputIdxT, int64_t, raft::row_major> neighbors,
                 raft::device_matrix_view<DistanceT, int64_t, raft::row_major> distances,
                 CagraSampleFilterT sample_filter = CagraSampleFilterT())
{
  RAFT_LOG_INFO("[CAGRA DEBUG] Entering search_main: queries=%ldx%ld, neighbors=%ldx%ld, k=%ld",
                queries.extent(0), queries.extent(1), neighbors.extent(0), neighbors.extent(1), neighbors.extent(1));
  // n_rows has the same type as the dataset index (the array extents type)
  using ds_idx_type = decltype(index.data().n_rows());
  // Dispatch search parameters based on the dataset kind.
  RAFT_LOG_INFO("[CAGRA DEBUG] Detecting dataset type...");
  if (auto* strided_dset = dynamic_cast<const strided_dataset<T, ds_idx_type>*>(&index.data());
      strided_dset != nullptr) {
    // Search using a plain (strided) row-major dataset
    RAFT_LOG_INFO("[CAGRA DEBUG] Using strided dataset");
    RAFT_EXPECTS(index.metric() != cuvs::distance::DistanceType::CosineExpanded ||
                   index.dataset_norms().has_value(),
                 "Dataset norms must be provided for CosineExpanded metric");

    const float* dataset_norms_ptr = nullptr;
    if (index.dataset_norms().has_value()) {
      RAFT_LOG_INFO("[CAGRA DEBUG] Dataset norms found");
      dataset_norms_ptr = index.dataset_norms().value().data_handle();
      raft::print_device_vector("dataset_norms", dataset_norms_ptr, 10, std::cout);
    } else {
      RAFT_LOG_INFO("[CAGRA DEBUG] No dataset norms found");
    }
    auto desc = dataset_descriptor_init_with_cache<T, IdxT, DistanceT>(
      res, params, *strided_dset, index.metric(), dataset_norms_ptr);
    search_main_core<T, IdxT, DistanceT, CagraSampleFilterT, OutputIdxT>(
      res, params, desc, index.graph(), queries, neighbors, distances, sample_filter);
    RAFT_LOG_INFO("[CAGRA DEBUG] Strided dataset search completed");
    
    // Print neighbors and distances for cosine metric debugging
    if (index.metric() == cuvs::distance::DistanceType::CosineExpanded) {
      RAFT_LOG_INFO("[CAGRA DEBUG] Printing first few neighbors and distances for cosine metric:");
      raft::print_device_vector("neighbors[0:5]", neighbors.data_handle(), std::min(5, (int)neighbors.extent(1)), std::cout);
      raft::print_device_vector("distances[0:5]", distances.data_handle(), std::min(5, (int)distances.extent(1)), std::cout);
    }
  } else if (auto* vpq_dset = dynamic_cast<const vpq_dataset<float, ds_idx_type>*>(&index.data());
             vpq_dset != nullptr) {
    // Search using a compressed dataset
    RAFT_LOG_INFO("[CAGRA DEBUG] Detected FP32 VPQ dataset");
    RAFT_FAIL("FP32 VPQ dataset support is coming soon");
  } else if (auto* vpq_dset = dynamic_cast<const vpq_dataset<half, ds_idx_type>*>(&index.data());
             vpq_dset != nullptr) {
    auto desc = dataset_descriptor_init_with_cache<T, IdxT, DistanceT>(
      res, params, *vpq_dset, index.metric(), nullptr);
    search_main_core<T, IdxT, DistanceT, CagraSampleFilterT, OutputIdxT>(
      res, params, desc, index.graph(), queries, neighbors, distances, sample_filter);
    RAFT_LOG_INFO("[CAGRA DEBUG] VPQ dataset search completed");
    
    // Print neighbors and distances for cosine metric debugging
    if (index.metric() == cuvs::distance::DistanceType::CosineExpanded) {
      RAFT_LOG_INFO("[CAGRA DEBUG] Printing first few neighbors and distances for cosine metric (VPQ):");
      raft::print_device_vector("neighbors[0:5]", neighbors.data_handle(), std::min(5, (int)neighbors.extent(1)), std::cout);
      raft::print_device_vector("distances[0:5]", distances.data_handle(), std::min(5, (int)distances.extent(1)), std::cout);
    }
  } else if (auto* empty_dset = dynamic_cast<const empty_dataset<ds_idx_type>*>(&index.data());
             empty_dset != nullptr) {
    // Forgot to add a dataset.
    RAFT_LOG_INFO("[CAGRA DEBUG] Error: Empty dataset detected");
    RAFT_FAIL(
      "Attempted to search without a dataset. Please call index.update_dataset(...) first.");
  } else {
    // This is a logic error.
    RAFT_LOG_INFO("[CAGRA DEBUG] Error: Unrecognized dataset format");
    RAFT_FAIL("Unrecognized dataset format");
  }

  static_assert(std::is_same_v<DistanceT, float>,
                "only float distances are supported at the moment");
  RAFT_LOG_INFO("[CAGRA DEBUG] Starting distance postprocessing, metric=%d", static_cast<int>(index.metric()));
  float* dist_out          = distances.data_handle();
  const DistanceT* dist_in = distances.data_handle();
  // We're converting the data from T to DistanceT during distance computation
  // and divide the values by kDivisor. Here we restore the original scale.
  constexpr float kScale = cuvs::spatial::knn::detail::utils::config<T>::kDivisor /
                           cuvs::spatial::knn::detail::utils::config<DistanceT>::kDivisor;

  if (index.metric() == cuvs::distance::DistanceType::CosineExpanded) {
    RAFT_LOG_INFO("[CAGRA DEBUG] Applying CosineExpanded metric postprocessing");
    
    // Print distances before cosine postprocessing
    RAFT_LOG_INFO("[CAGRA DEBUG] Distances before cosine postprocessing:");
    raft::print_device_vector("distances_before[0:10]", distances.data_handle(), std::min(10, (int)(distances.extent(0) * distances.extent(1))), std::cout);
    
    auto stream      = raft::resource::get_cuda_stream(res);
    auto query_norms = raft::make_device_vector<DistanceT, int64_t>(res, queries.extent(0));

    raft::linalg::rowNorm<raft::linalg::L2Norm, true>(query_norms.data_handle(),
                                                      queries.data_handle(),
                                                      queries.extent(1),
                                                      queries.extent(0),
                                                      stream,
                                                      raft::sqrt_op{});

    const auto n_queries = distances.extent(0);
    const auto k         = distances.extent(1);
    auto query_norms_ptr = query_norms.data_handle();

    raft::linalg::matrix_vector_op<raft::Apply::ALONG_COLUMNS>(
      res,
      raft::make_const_mdspan(distances),
      raft::make_const_mdspan(query_norms.view()),
      distances,
      raft::compose_op(raft::add_const_op<DistanceT>{DistanceT(1)}, raft::div_op{}));
    
    // Print query norms and distances after cosine processing
    RAFT_LOG_INFO("[CAGRA DEBUG] Query norms:");
    raft::print_device_vector("query_norms[0:5]", query_norms.data_handle(), std::min(5, (int)query_norms.extent(0)), std::cout);
    RAFT_LOG_INFO("[CAGRA DEBUG] Distances after cosine processing:");
    raft::print_device_vector("distances_after[0:10]", distances.data_handle(), std::min(10, (int)(distances.extent(0) * distances.extent(1))), std::cout);
  }
  RAFT_LOG_INFO("[CAGRA DEBUG] Final distance postprocessing with scale=%f", kScale);
  cuvs::neighbors::ivf::detail::postprocess_distances(
    dist_out,
    dist_in,
    index.metric(),
    distances.extent(0),
    distances.extent(1),
    kScale,
    index.metric() != distance::DistanceType::CosineExpanded,
    raft::resource::get_cuda_stream(res));
  
  // Print final results for cosine metric
  if (index.metric() == cuvs::distance::DistanceType::CosineExpanded) {
    RAFT_LOG_INFO("[CAGRA DEBUG] Final results after all postprocessing (cosine metric):");
    raft::print_device_vector("final_neighbors[0:10]", neighbors.data_handle(), std::min(10, (int)(neighbors.extent(0) * neighbors.extent(1))), std::cout);
    raft::print_device_vector("final_distances[0:10]", distances.data_handle(), std::min(10, (int)(distances.extent(0) * distances.extent(1))), std::cout);
  }
  
  RAFT_LOG_INFO("[CAGRA DEBUG] CAGRA search completed successfully");
}
/** @} */  // end group cagra

}  // namespace cuvs::neighbors::cagra::detail
