/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "../detail/nn_descent.cuh"
#include "cuvs/neighbors/batch_ann.hpp"
#include "cuvs/neighbors/ivf_pq.hpp"
#include "cuvs/neighbors/nn_descent.hpp"
#include "cuvs/neighbors/refine.hpp"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/kvp.hpp>
#include <raft/core/managed_mdspan.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

namespace cuvs::neighbors::batch_ann::detail {
using namespace cuvs::neighbors;
using align32 = raft::Pow2<32>;

template <typename KeyType, typename ValueType>
struct CustomComparator {
  __device__ bool operator()(const raft::KeyValuePair<KeyType, ValueType>& a,
                             const raft::KeyValuePair<KeyType, ValueType>& b) const
  {
    return a < b;
  }
};

template <typename IdxT, int BLOCK_SIZE, int ITEMS_PER_THREAD>
RAFT_KERNEL merge_subgraphs_kernel(IdxT* cluster_data_indices,
                                   size_t graph_degree,
                                   size_t num_cluster_in_batch,
                                   float* global_distances,
                                   float* batch_distances,
                                   IdxT* global_indices,
                                   IdxT* batch_indices)
{
  size_t batch_row = blockIdx.x;
  typedef cub::BlockMergeSort<raft::KeyValuePair<float, IdxT>, BLOCK_SIZE, ITEMS_PER_THREAD>
    BlockMergeSortType;
  __shared__ typename cub::BlockMergeSort<raft::KeyValuePair<float, IdxT>,
                                          BLOCK_SIZE,
                                          ITEMS_PER_THREAD>::TempStorage tmpSmem;

  extern __shared__ char sharedMem[];
  float* blockKeys  = reinterpret_cast<float*>(sharedMem);
  IdxT* blockValues = reinterpret_cast<IdxT*>(&sharedMem[graph_degree * 2 * sizeof(float)]);
  int16_t* uniqueMask =
    reinterpret_cast<int16_t*>(&sharedMem[graph_degree * 2 * (sizeof(float) + sizeof(IdxT))]);

  if (batch_row < num_cluster_in_batch) {
    // load batch or global depending on threadIdx
    size_t global_row = cluster_data_indices[batch_row];

    raft::KeyValuePair<float, IdxT> threadKeyValuePair[ITEMS_PER_THREAD];

    size_t halfway   = BLOCK_SIZE / 2;
    size_t do_global = threadIdx.x < halfway;

    float* distances;
    IdxT* indices;

    if (do_global) {
      distances = global_distances;
      indices   = global_indices;
    } else {
      distances = batch_distances;
      indices   = batch_indices;
    }

    size_t idxBase = (threadIdx.x * do_global + (threadIdx.x - halfway) * (1lu - do_global)) *
                     static_cast<size_t>(ITEMS_PER_THREAD);
    size_t arrIdxBase = (global_row * do_global + batch_row * (1lu - do_global)) * graph_degree;
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
      size_t colId = idxBase + i;
      if (colId < graph_degree) {
        threadKeyValuePair[i].key   = distances[arrIdxBase + colId];
        threadKeyValuePair[i].value = indices[arrIdxBase + colId];
      } else {
        threadKeyValuePair[i].key   = std::numeric_limits<float>::max();
        threadKeyValuePair[i].value = std::numeric_limits<IdxT>::max();
      }
    }

    __syncthreads();

    BlockMergeSortType(tmpSmem).Sort(threadKeyValuePair, CustomComparator<float, IdxT>{});

    // load sorted result into shared memory to get unique values
    idxBase = threadIdx.x * ITEMS_PER_THREAD;
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
      size_t colId = idxBase + i;
      if (colId < 2 * graph_degree) {
        blockKeys[colId]   = threadKeyValuePair[i].key;
        blockValues[colId] = threadKeyValuePair[i].value;
      }
    }

    __syncthreads();

    // get unique mask
    if (threadIdx.x == 0) { uniqueMask[0] = 1; }
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
      size_t colId = idxBase + i;
      if (colId > 0 && colId < 2 * graph_degree) {
        uniqueMask[colId] = static_cast<int16_t>(blockValues[colId] != blockValues[colId - 1]);
      }
    }

    __syncthreads();

    // prefix sum
    if (threadIdx.x == 0) {
      for (int i = 1; i < 2 * graph_degree; i++) {
        uniqueMask[i] += uniqueMask[i - 1];
      }
    }

    __syncthreads();
    // load unique values to global memory
    if (threadIdx.x == 0) {
      global_distances[global_row * graph_degree] = blockKeys[0];
      global_indices[global_row * graph_degree]   = blockValues[0];
    }

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
      size_t colId = idxBase + i;
      if (colId > 0 && colId < 2 * graph_degree) {
        bool is_unique       = uniqueMask[colId] != uniqueMask[colId - 1];
        int16_t global_colId = uniqueMask[colId] - 1;
        if (is_unique && static_cast<size_t>(global_colId) < graph_degree) {
          global_distances[global_row * graph_degree + global_colId] = blockKeys[colId];
          global_indices[global_row * graph_degree + global_colId]   = blockValues[colId];
        }
      }
    }
  }
}

template <typename T,
          typename IdxT = int64_t,
          typename Accessor =
            host_device_accessor<std::experimental::default_accessor<T>, memory_type::host>>
void merge_subgraphs(raft::resources const& res,
                     size_t k,
                     size_t num_data_in_cluster,
                     IdxT* inverted_indices_d,
                     T* global_distances,
                     T* batch_distances_d,
                     IdxT* global_neighbors,
                     IdxT* batch_neighbors_d)
{
  size_t num_elems     = k * 2;
  size_t sharedMemSize = num_elems * (sizeof(float) + sizeof(IdxT) + sizeof(int16_t));

#pragma omp critical  // for omp-using multi-gpu purposes
  {
    if (num_elems <= 128) {
      merge_subgraphs_kernel<IdxT, 32, 4>
        <<<num_data_in_cluster, 32, sharedMemSize, raft::resource::get_cuda_stream(res)>>>(
          inverted_indices_d,
          k,
          num_data_in_cluster,
          global_distances,
          batch_distances_d,
          global_neighbors,
          batch_neighbors_d);
    } else if (num_elems <= 512) {
      merge_subgraphs_kernel<IdxT, 128, 4>
        <<<num_data_in_cluster, 128, sharedMemSize, raft::resource::get_cuda_stream(res)>>>(
          inverted_indices_d,
          k,
          num_data_in_cluster,
          global_distances,
          batch_distances_d,
          global_neighbors,
          batch_neighbors_d);
    } else if (num_elems <= 1024) {
      merge_subgraphs_kernel<IdxT, 128, 8>
        <<<num_data_in_cluster, 128, sharedMemSize, raft::resource::get_cuda_stream(res)>>>(
          inverted_indices_d,
          k,
          num_data_in_cluster,
          global_distances,
          batch_distances_d,
          global_neighbors,
          batch_neighbors_d);
    } else if (num_elems <= 2048) {
      merge_subgraphs_kernel<IdxT, 256, 8>
        <<<num_data_in_cluster, 256, sharedMemSize, raft::resource::get_cuda_stream(res)>>>(
          inverted_indices_d,
          k,
          num_data_in_cluster,
          global_distances,
          batch_distances_d,
          global_neighbors,
          batch_neighbors_d);
    } else {
      // this is as far as we can get due to the shared mem usage of cub::BlockMergeSort
      RAFT_FAIL("The degree of knn is too large (%lu). It must be smaller than 1024", k);
    }
    raft::resource::sync_stream(res);
  }
}

template <typename T, typename IdxT = int64_t, typename BeforeRemapT = int64_t>
void remap_and_merge_subgraphs(
  raft::resources const& res,
  raft::device_vector_view<IdxT, IdxT> inverted_indices_d,
  raft::host_vector_view<IdxT, IdxT> inverted_indices,
  raft::host_matrix_view<BeforeRemapT, IdxT, row_major> indices_for_remap_h,
  raft::host_matrix_view<IdxT, IdxT, row_major> batch_neighbors_h,
  raft::device_matrix_view<IdxT, IdxT, row_major> batch_neighbors_d,
  raft::device_matrix_view<T, IdxT, row_major> batch_distances_d,
  raft::managed_matrix_view<IdxT, IdxT> global_neighbors,
  raft::managed_matrix_view<T, IdxT> global_distances,
  size_t num_data_in_cluster,
  size_t k)
{
  // remap indices
#pragma omp parallel for
  for (size_t i = 0; i < num_data_in_cluster; i++) {
    for (size_t j = 0; j < k; j++) {
      size_t local_idx        = indices_for_remap_h(i, j);
      batch_neighbors_h(i, j) = inverted_indices(local_idx);
    }
  }

  raft::copy(inverted_indices_d.data_handle(),
             inverted_indices.data_handle(),
             num_data_in_cluster,
             raft::resource::get_cuda_stream(res));

  raft::copy(batch_neighbors_d.data_handle(),
             batch_neighbors_h.data_handle(),
             num_data_in_cluster * k,
             raft::resource::get_cuda_stream(res));

  merge_subgraphs(res,
                  k,
                  num_data_in_cluster,
                  inverted_indices_d.data_handle(),
                  global_distances.data_handle(),
                  batch_distances_d.data_handle(),
                  global_neighbors.data_handle(),
                  batch_neighbors_d.data_handle());
}

template <typename T, typename IdxT = int64_t>
struct batch_ann_builder {
  batch_ann_builder(raft::resources const& res,
                    size_t n_clusters,
                    size_t min_cluster_size,
                    size_t max_cluster_size,
                    size_t k,
                    cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded)
    : res{res},
      k{k},
      n_clusters{n_clusters},
      min_cluster_size{min_cluster_size},
      max_cluster_size{max_cluster_size},
      metric{metric},
      inverted_indices_d(raft::make_device_vector<IdxT, IdxT>(res, max_cluster_size)),
      batch_neighbors_h(raft::make_host_matrix<IdxT, IdxT>(max_cluster_size, k)),
      batch_neighbors_d(raft::make_device_matrix<IdxT, IdxT>(res, max_cluster_size, k)),
      batch_distances_d(raft::make_device_matrix<T, IdxT>(res, max_cluster_size, k))
  {
  }

  /**
   * Some memory-heavy allocations that can be used over multiple clusters should be allocated here
   * Arguments:
   * - [in] dataset: host_matrix_view of the the ENTIRE dataset
   */
  virtual void prepare_build(raft::host_matrix_view<const T, IdxT, raft::row_major> dataset) {}

  /**
   * Running the ann algorithm on the given cluster, and merging it into the global result
   * Arguments:
   * - [in] res: raft resource
   * - [in] params: batch_ann::index_params
   * - [in] dataset: host_matrix_view of the cluster dataset
   * - [in] inverted_indices: global data indices for the data points in the current cluster of size
   * (num_data_in_cluster)
   * - [out] global_neighbors: raft::managed_matrix_view type of (total_num_rows, k) for final
   * all-neighbors graph indices
   * - [out] global_distances: raft::managed_matrix_view type of (total_num_rows, k) for final
   * all-neighbors graph distances
   */
  virtual void build_knn(raft::resources const& res,
                         const index_params& params,
                         raft::host_matrix_view<const T, IdxT, row_major> dataset,
                         raft::host_vector_view<IdxT, IdxT> inverted_indices,
                         raft::managed_matrix_view<IdxT, IdxT> global_neighbors,
                         raft::managed_matrix_view<T, IdxT> global_distances)
  {
  }

  raft::resources const& res;
  size_t n_clusters, min_cluster_size, max_cluster_size, k;
  cuvs::distance::DistanceType metric;

  raft::device_vector<IdxT, IdxT> inverted_indices_d;
  raft::host_matrix<IdxT, IdxT> batch_neighbors_h;
  raft::device_matrix<IdxT, IdxT> batch_neighbors_d;
  raft::device_matrix<T, IdxT> batch_distances_d;
};

template <typename T, typename IdxT = int64_t>
struct batch_ann_builder_ivfpq : public batch_ann_builder<T, IdxT> {
  batch_ann_builder_ivfpq(raft::resources const& res,
                          size_t n_clusters,
                          size_t min_cluster_size,
                          size_t max_cluster_size,
                          size_t k,
                          cuvs::distance::DistanceType metric,
                          batch_ann::graph_build_params::ivf_pq_params& params)
    : batch_ann_builder<T, IdxT>(res, n_clusters, min_cluster_size, max_cluster_size, k, metric),
      all_ivf_pq_params{params}
  {
    if (all_ivf_pq_params.build_params.metric != metric) {
      RAFT_LOG_WARN("Setting ivfpq_params metric to metric given for batching algorithm");
      all_ivf_pq_params.build_params.metric = metric;
    }
  }

  void prepare_build(raft::host_matrix_view<const T, IdxT, row_major> dataset) override
  {
    size_t num_cols = static_cast<size_t>(dataset.extent(1));
    candidate_k     = std::min<IdxT>(
      std::max(static_cast<size_t>(this->k * all_ivf_pq_params.refinement_rate), this->k),
      this->min_cluster_size);

    data_d.emplace(
      raft::make_device_matrix<T, IdxT, row_major>(this->res, this->max_cluster_size, num_cols));

    distances_candidate_d.emplace(
      raft::make_device_matrix<T, IdxT, row_major>(this->res, this->max_cluster_size, candidate_k));
    neighbors_candidate_d.emplace(raft::make_device_matrix<IdxT, IdxT, row_major>(
      this->res, this->max_cluster_size, candidate_k));
    neighbors_candidate_h.emplace(
      raft::make_host_matrix<IdxT, IdxT, row_major>(this->max_cluster_size, candidate_k));

    // for host refining
    refined_neighbors_h.emplace(
      raft::make_host_matrix<IdxT, IdxT, row_major>(this->max_cluster_size, this->k));
    refined_distances_h.emplace(
      raft::make_host_matrix<T, IdxT, row_major>(this->max_cluster_size, this->k));
  }

  void build_knn(raft::resources const& res,
                 const index_params& params,
                 raft::host_matrix_view<const T, IdxT, row_major> dataset,
                 raft::host_vector_view<IdxT, IdxT> inverted_indices,
                 raft::managed_matrix_view<IdxT, IdxT> global_neighbors,
                 raft::managed_matrix_view<T, IdxT> global_distances) override
  {
    size_t num_data_in_cluster = dataset.extent(0);
    size_t num_cols            = dataset.extent(1);

    // we need data on device for ivfpq build and search.
    // num_data_in_cluster is always <= max_cluster_size
    raft::copy(data_d.value().data_handle(),
               dataset.data_handle(),
               num_data_in_cluster * num_cols,
               raft::resource::get_cuda_stream(this->res));

    auto data_view = raft::make_device_matrix_view<const T, IdxT>(
      data_d.value().data_handle(), num_data_in_cluster, num_cols);

    auto index = ivf_pq::build(this->res, all_ivf_pq_params.build_params, data_view);

    auto distances_candidate_view = raft::make_device_matrix_view<T, IdxT>(
      distances_candidate_d.value().data_handle(), num_data_in_cluster, candidate_k);
    auto neighbors_candidate_view = raft::make_device_matrix_view<IdxT, IdxT>(
      neighbors_candidate_d.value().data_handle(), num_data_in_cluster, candidate_k);
    cuvs::neighbors::ivf_pq::search(this->res,
                                    all_ivf_pq_params.search_params,
                                    index,
                                    data_view,
                                    neighbors_candidate_view,
                                    distances_candidate_view);
    raft::copy(neighbors_candidate_h.value().data_handle(),
               neighbors_candidate_view.data_handle(),
               num_data_in_cluster * candidate_k,
               raft::resource::get_cuda_stream(this->res));

    auto neighbors_candidate_h_view = raft::make_host_matrix_view<IdxT, IdxT>(
      neighbors_candidate_h.value().data_handle(), num_data_in_cluster, candidate_k);
    auto refined_distances_h_view = raft::make_host_matrix_view<T, IdxT>(
      refined_distances_h.value().data_handle(), num_data_in_cluster, this->k);
    auto refined_neighbors_h_view = raft::make_host_matrix_view<IdxT, IdxT>(
      refined_neighbors_h.value().data_handle(), num_data_in_cluster, this->k);
    refine(this->res,
           dataset,
           dataset,
           raft::make_const_mdspan(neighbors_candidate_h_view),
           refined_neighbors_h_view,
           refined_distances_h_view,
           params.metric);
    raft::copy(this->batch_distances_d.data_handle(),
               refined_distances_h_view.data_handle(),
               num_data_in_cluster * this->k,
               raft::resource::get_cuda_stream(this->res));

    remap_and_merge_subgraphs<T, IdxT, IdxT>(this->res,
                                             this->inverted_indices_d.view(),
                                             inverted_indices,
                                             refined_neighbors_h.value().view(),
                                             this->batch_neighbors_h.view(),
                                             this->batch_neighbors_d.view(),
                                             this->batch_distances_d.view(),
                                             global_neighbors,
                                             global_distances,
                                             num_data_in_cluster,
                                             this->k);
  }

  batch_ann::graph_build_params::ivf_pq_params all_ivf_pq_params;
  size_t candidate_k;

  std::optional<raft::device_matrix<T, IdxT>> data_d;
  std::optional<raft::device_matrix<T, IdxT>> distances_candidate_d;
  std::optional<raft::device_matrix<IdxT, IdxT>> neighbors_candidate_d;
  std::optional<raft::host_matrix<IdxT, IdxT>> neighbors_candidate_h;
  std::optional<raft::host_matrix<IdxT, IdxT>> refined_neighbors_h;
  std::optional<raft::host_matrix<T, IdxT>> refined_distances_h;
};

template <typename T, typename IdxT = int64_t>
struct batch_ann_builder_nn_descent : public batch_ann_builder<T, IdxT> {
  batch_ann_builder_nn_descent(raft::resources const& res,
                               size_t n_clusters,
                               size_t min_cluster_size,
                               size_t max_cluster_size,
                               size_t k,
                               cuvs::distance::DistanceType metric,
                               batch_ann::graph_build_params::nn_descent_params& params)
    : batch_ann_builder<T, IdxT>(res, n_clusters, min_cluster_size, max_cluster_size, k, metric),
      nnd_params{params}
  {
    auto allowed_metrics = metric == cuvs::distance::DistanceType::L2Expanded ||
                           metric == cuvs::distance::DistanceType::CosineExpanded ||
                           metric == cuvs::distance::DistanceType::InnerProduct;
    RAFT_EXPECTS(allowed_metrics,
                 "The metric for NN Descent should be L2Expanded, CosineExpanded or InnerProduct");
    if (nnd_params.metric != metric) {
      RAFT_LOG_WARN("Setting nnd_params metric to metric given for batching algorithm");
      nnd_params.metric = metric;
    }
  }

  void prepare_build(raft::host_matrix_view<const T, IdxT, row_major> dataset) override
  {
    size_t intermediate_degree = nnd_params.intermediate_graph_degree;
    size_t graph_degree        = nnd_params.graph_degree;

    if (graph_degree < this->k) {
      RAFT_LOG_WARN(
        "NN Descent's graph degree (%lu) has to be larger than or equal to k. Setting graph_degree "
        "to k (%lu).",
        graph_degree,
        this->k);
      graph_degree = this->k;
    }
    if (intermediate_degree < graph_degree) {
      RAFT_LOG_WARN(
        "Intermediate graph degree (%lu) cannot be smaller than graph degree (%lu), increasing "
        "intermediate_degree.",
        intermediate_degree,
        graph_degree);
      intermediate_degree = 1.5 * graph_degree;
    }

    size_t extended_graph_degree =
      align32::roundUp(static_cast<size_t>(graph_degree * (graph_degree <= 32 ? 1.0 : 1.3)));
    size_t extended_intermediate_degree = align32::roundUp(
      static_cast<size_t>(intermediate_degree * (intermediate_degree <= 32 ? 1.0 : 1.3)));

    build_config.max_dataset_size      = this->max_cluster_size;
    build_config.dataset_dim           = dataset.extent(1);
    build_config.node_degree           = extended_graph_degree;
    build_config.internal_node_degree  = extended_intermediate_degree;
    build_config.max_iterations        = nnd_params.max_iterations;
    build_config.termination_threshold = nnd_params.termination_threshold;
    build_config.output_graph_degree   = this->k;
    build_config.metric                = this->metric;

    nnd_builder.emplace(this->res, build_config);
    int_graph.emplace(raft::make_host_matrix<int, IdxT, row_major>(
      this->max_cluster_size, static_cast<IdxT>(extended_graph_degree)));
  }

  void build_knn(raft::resources const& res,
                 const index_params& params,
                 raft::host_matrix_view<const T, IdxT> dataset,
                 raft::host_vector_view<IdxT, IdxT> inverted_indices,
                 raft::managed_matrix_view<IdxT, IdxT> global_neighbors,
                 raft::managed_matrix_view<T, IdxT> global_distances) override
  {
    size_t num_data_in_cluster = dataset.extent(0);
    bool return_distances      = true;
    nnd_builder.value().build(dataset.data_handle(),
                              static_cast<int>(num_data_in_cluster),
                              int_graph.value().data_handle(),
                              return_distances,
                              this->batch_distances_d.data_handle());

    remap_and_merge_subgraphs<T, IdxT, int>(res,
                                            this->inverted_indices_d.view(),
                                            inverted_indices,
                                            int_graph.value().view(),
                                            this->batch_neighbors_h.view(),
                                            this->batch_neighbors_d.view(),
                                            this->batch_distances_d.view(),
                                            global_neighbors,
                                            global_distances,
                                            num_data_in_cluster,
                                            this->k);
  }

  nn_descent::index_params nnd_params;
  nn_descent::detail::BuildConfig build_config;

  std::optional<nn_descent::detail::GNND<const T, int>> nnd_builder;
  std::optional<raft::host_matrix<int, IdxT>> int_graph;
};

}  // namespace cuvs::neighbors::batch_ann::detail
