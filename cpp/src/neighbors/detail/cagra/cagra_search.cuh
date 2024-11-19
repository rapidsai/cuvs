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

#include "../../../core/nvtx.hpp"
#include "factory.cuh"
#include "sample_filter_utils.cuh"
#include "search_plan.cuh"
#include "search_single_cta_inst.cuh"
#include "utils.hpp"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/unary_op.cuh>

#include <cuvs/distance/distance.hpp>

#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/cagra.hpp>

// TODO: Fix these when ivf methods are moved over
#include "../../ivf_common.cuh"
#include "../../ivf_pq/ivf_pq_search.cuh"
#include <cuvs/neighbors/common.hpp>

// TODO: This shouldn't be calling spatial/knn apis
#include "../ann_utils.cuh"

#include <rmm/cuda_stream_view.hpp>

namespace cuvs::neighbors::cagra::detail {

template <typename DataT, typename IndexT, typename DistanceT, typename CagraSampleFilterT>
void search_main_core(raft::resources const& res,
                      search_params params,
                      const dataset_descriptor_host<DataT, IndexT, DistanceT>& dataset_desc,
                      raft::device_matrix_view<const IndexT, int64_t, raft::row_major> graph,
                      raft::device_matrix_view<const DataT, int64_t, raft::row_major> queries,
                      raft::device_matrix_view<IndexT, int64_t, raft::row_major> neighbors,
                      raft::device_matrix_view<DistanceT, int64_t, raft::row_major> distances,
                      CagraSampleFilterT sample_filter = CagraSampleFilterT())
{
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
  std::unique_ptr<search_plan_impl<DataT, IndexT, DistanceT, CagraSampleFilterT_s>> plan =
    factory<DataT, IndexT, DistanceT, CagraSampleFilterT_s>::create(
      res, params, dataset_desc, queries.extent(1), graph.extent(1), topk);

  plan->check(topk);

  RAFT_LOG_DEBUG("Cagra search");
  const uint32_t max_queries = plan->max_queries;
  const uint32_t query_dim   = queries.extent(1);

  for (unsigned qid = 0; qid < queries.extent(0); qid += max_queries) {
    const uint32_t n_queries = std::min<std::size_t>(max_queries, queries.extent(0) - qid);
    auto _topk_indices_ptr   = reinterpret_cast<IndexT*>(neighbors.data_handle()) + (topk * qid);
    auto _topk_distances_ptr = distances.data_handle() + (topk * qid);
    // todo(tfeher): one could keep distances optional and pass nullptr
    const auto* _query_ptr = queries.data_handle() + (query_dim * qid);
    const auto* _seed_ptr =
      plan->num_seeds > 0
        ? reinterpret_cast<const IndexT*>(plan->dev_seed.data()) + (plan->num_seeds * qid)
        : nullptr;
    uint32_t* _num_executed_iterations = nullptr;

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
  }
}

/**
 * @brief Performs ANN search using brute force when filter sparsity exceeds a specified threshold.
 *
 * This function switches to a brute force search approach to improve recall rate when the
 * `sample_filter` function filters out a high proportion of samples, resulting in a sparsity level
 * (proportion of unfiltered samples) exceeding the specified `params.threshold_to_bf`.
 *
 * @tparam T data element type
 * @tparam IdxT type of database vector indices
 * @tparam internal_IdxT during search we map IdxT to internal_IdxT, this way we do not need
 * separate kernels for int/uint.
 *
 * @param[in] handle
 * @param[in] params configure the search
 * @param[in] strided_dataset CAGRA strided dataset
 * @param[in] metric distance type
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 * @param[in] sample_filter a device filter function that greenlights samples for a given query
 *
 * @return true If the brute force search was applied successfully.
 * @return false If the brute force search was not applied.
 */
template <typename T,
          typename InternalIdxT,
          typename CagraSampleFilterT,
          typename IdxT      = uint32_t,
          typename DistanceT = float>
bool search_using_brute_force(
  raft::resources const& res,
  const search_params& params,
  const strided_dataset<T, IdxT>& strided_dataset,
  cuvs::distance::DistanceType metric,
  raft::device_matrix_view<const T, int64_t, raft::row_major> queries,
  raft::device_matrix_view<InternalIdxT, int64_t, raft::row_major> neighbors,
  raft::device_matrix_view<DistanceT, int64_t, raft::row_major> distances,
  CagraSampleFilterT& sample_filter)
{
  auto n_queries = queries.extent(0);
  auto n_dataset = strided_dataset.n_rows();

  auto bitset_filter_view = sample_filter.bitset_view_;
  auto sparsity           = bitset_filter_view.sparsity(res);

  if (sparsity < params.threshold_to_bf) { return false; }

  // TODO: Support host dataset in `brute_force::build`
  RAFT_LOG_DEBUG("CAGRA is switching to brute force with sparsity:%f", sparsity);
  using bitmap_view_t = cuvs::core::bitmap_view<const uint32_t, int64_t>;

  auto stream            = raft::resource::get_cuda_stream(res);
  auto bitmap_n_elements = bitmap_view_t::eval_n_elements(bitset_filter_view.size() * n_queries);

  rmm::device_uvector<uint32_t> raw_bitmap(bitmap_n_elements, stream);
  rmm::device_uvector<int64_t> raw_neighbors(neighbors.size(), stream);

  bitset_filter_view.repeat(res, n_queries, raw_bitmap.data());

  auto brute_force_filter = bitmap_view_t(raw_bitmap.data(), n_queries, n_dataset);

  auto brute_force_neighbors = raft::make_device_matrix_view<int64_t, int64_t, raft::row_major>(
    raw_neighbors.data(), neighbors.extent(0), neighbors.extent(1));
  auto brute_force_dataset = raft::make_device_matrix_view<const T, int64_t, raft::row_major>(
    strided_dataset.view().data_handle(), strided_dataset.n_rows(), strided_dataset.stride());

  auto brute_force_idx = cuvs::neighbors::brute_force::build(res, brute_force_dataset, metric);

  auto brute_force_queries = queries;
  auto padding_queries     = raft::make_device_matrix<T, int64_t>(res, 0, 0);

  // Happens when the original dataset is a strided matrix.
  if (brute_force_dataset.extent(1) != queries.extent(1)) {
    padding_queries = raft::make_device_mdarray<T, int64_t>(
      res,
      raft::resource::get_workspace_resource(res),
      raft::make_extents<int64_t>(n_queries, brute_force_dataset.extent(1)));
    // Copy the queries and fill the padded elements with zeros
    raft::linalg::map_offset(
      res,
      padding_queries.view(),
      [queries, stride = brute_force_dataset.extent(1)] __device__(int64_t i) {
        auto row_ix = i / stride;
        auto el_ix  = i % stride;
        return el_ix < queries.extent(1) ? queries(row_ix, el_ix) : T{0};
      });
    brute_force_queries = raft::make_device_matrix_view<const T, int64_t, raft::row_major>(
      padding_queries.data_handle(), padding_queries.extent(0), padding_queries.extent(1));
  }
  cuvs::neighbors::brute_force::search(
    res,
    brute_force_idx,
    brute_force_queries,
    brute_force_neighbors,
    distances,
    cuvs::neighbors::filtering::bitmap_filter(brute_force_filter));
  raft::linalg::unaryOp(neighbors.data_handle(),
                        brute_force_neighbors.data_handle(),
                        neighbors.size(),
                        raft::cast_op<InternalIdxT>(),
                        raft::resource::get_cuda_stream(res));
  return true;
}

/**
 * @brief Search ANN using the constructed index.
 *
 * See the [build](#build) documentation for a usage example.
 *
 * @tparam T data element type
 * @tparam IdxT type of database vector indices
 * @tparam internal_IdxT during search we map IdxT to internal_IdxT, this way we do not need
 * separate kernels for int/uint.
 *
 * @param[in] handle
 * @param[in] params configure the search
 * @param[in] idx ivf-pq constructed index
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 * @param[in] sample_filter a device filter function that greenlights samples for a given query
 */
template <typename T,
          typename InternalIdxT,
          typename CagraSampleFilterT,
          typename IdxT      = uint32_t,
          typename DistanceT = float>
void search_main(raft::resources const& res,
                 search_params params,
                 const index<T, IdxT>& index,
                 raft::device_matrix_view<const T, int64_t, raft::row_major> queries,
                 raft::device_matrix_view<InternalIdxT, int64_t, raft::row_major> neighbors,
                 raft::device_matrix_view<DistanceT, int64_t, raft::row_major> distances,
                 CagraSampleFilterT sample_filter = CagraSampleFilterT())
{
  auto stream         = raft::resource::get_cuda_stream(res);
  const auto& graph   = index.graph();
  auto graph_internal = raft::make_device_matrix_view<const InternalIdxT, int64_t, raft::row_major>(
    reinterpret_cast<const InternalIdxT*>(graph.data_handle()), graph.extent(0), graph.extent(1));

  // n_rows has the same type as the dataset index (the array extents type)
  using ds_idx_type = decltype(index.data().n_rows());
  // Dispatch search parameters based on the dataset kind.
  if (auto* strided_dset = dynamic_cast<const strided_dataset<T, ds_idx_type>*>(&index.data());
      strided_dset != nullptr) {
    if constexpr (!std::is_same_v<CagraSampleFilterT,
                                  cuvs::neighbors::filtering::none_sample_filter> &&
                  (std::is_same_v<T, float> || std::is_same_v<T, half>)) {
      bool bf_search_done = search_using_brute_force(
        res, params, *strided_dset, index.metric(), queries, neighbors, distances, sample_filter);
      if (bf_search_done) return;
    }

    // Search using a plain (strided) row-major dataset
    auto desc = dataset_descriptor_init_with_cache<T, InternalIdxT, DistanceT>(
      res, params, *strided_dset, index.metric());
    search_main_core<T, InternalIdxT, DistanceT, CagraSampleFilterT>(
      res, params, desc, graph_internal, queries, neighbors, distances, sample_filter);
  } else if (auto* vpq_dset = dynamic_cast<const vpq_dataset<float, ds_idx_type>*>(&index.data());
             vpq_dset != nullptr) {
    // Search using a compressed dataset
    RAFT_FAIL("FP32 VPQ dataset support is coming soon");
  } else if (auto* vpq_dset = dynamic_cast<const vpq_dataset<half, ds_idx_type>*>(&index.data());
             vpq_dset != nullptr) {
    auto desc = dataset_descriptor_init_with_cache<T, InternalIdxT, DistanceT>(
      res, params, *vpq_dset, index.metric());
    search_main_core<T, InternalIdxT, DistanceT, CagraSampleFilterT>(
      res, params, desc, graph_internal, queries, neighbors, distances, sample_filter);
  } else if (auto* empty_dset = dynamic_cast<const empty_dataset<ds_idx_type>*>(&index.data());
             empty_dset != nullptr) {
    // Forgot to add a dataset.
    RAFT_FAIL(
      "Attempted to search without a dataset. Please call index.update_dataset(...) first.");
  } else {
    // This is a logic error.
    RAFT_FAIL("Unrecognized dataset format");
  }

  static_assert(std::is_same_v<DistanceT, float>,
                "only float distances are supported at the moment");
  float* dist_out          = distances.data_handle();
  const DistanceT* dist_in = distances.data_handle();
  // We're converting the data from T to DistanceT during distance computation
  // and divide the values by kDivisor. Here we restore the original scale.
  constexpr float kScale = cuvs::spatial::knn::detail::utils::config<T>::kDivisor /
                           cuvs::spatial::knn::detail::utils::config<DistanceT>::kDivisor;
  cuvs::neighbors::ivf::detail::postprocess_distances(dist_out,
                                                      dist_in,
                                                      index.metric(),
                                                      distances.extent(0),
                                                      distances.extent(1),
                                                      kScale,
                                                      true,
                                                      raft::resource::get_cuda_stream(res));
}
/** @} */  // end group cagra

}  // namespace cuvs::neighbors::cagra::detail
