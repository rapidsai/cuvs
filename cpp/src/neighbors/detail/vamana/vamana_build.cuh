/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include "../../../sparse/neighbors/cross_component_nn.cuh"
#include "../../detail/vpq_dataset.cuh"
#include "greedy_search.cuh"
#include "robust_prune.cuh"
#include "vamana_structs.cuh"
#include <cuvs/neighbors/vamana.hpp>

#include <raft/cluster/kmeans.cuh>
#include <raft/cluster/kmeans_types.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/gemm.hpp>
#include <raft/matrix/copy.cuh>
#include <raft/matrix/init.cuh>
#include <raft/matrix/slice.cuh>
#include <raft/random/make_blobs.cuh>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/unique.h>

#include <cuvs/distance/distance.hpp>

#include <chrono>
#include <cstdio>
#include <vector>

namespace cuvs::neighbors::vamana::detail {

/* @defgroup vamana_build_detail vamana build
 * @{
 */

static const int blockD    = 32;
static const int maxBlocks = 10000;

// generate random permutation of inserts - TODO do this on GPU / faster
template <typename IdxT>
void create_insert_permutation(std::vector<IdxT>& insert_order, uint32_t N)
{
  insert_order.resize(N);
  for (uint32_t i = 0; i < N; i++) {
    insert_order[i] = (IdxT)i;
  }
  for (uint32_t i = 0; i < N; i++) {
    uint32_t temp;
    uint32_t rand_idx      = rand() % N;
    temp                   = insert_order[i];
    insert_order[i]        = insert_order[rand_idx];
    insert_order[rand_idx] = temp;
  }
}

/********************************************************************************************
 * Main Vamana building function - insert vectors into empty graph in batches
 * Pre - dataset contains the vector data, host matrix allocated to store the graph
 * Post - graph matrix contains the graph edges of the final Vamana graph
 *******************************************************************************************/
template <typename T,
          typename accT,
          typename IdxT     = uint32_t,
          typename Accessor = raft::host_device_accessor<std::experimental::default_accessor<T>,
                                                         raft::memory_type::host>>
void batched_insert_vamana(
  raft::resources const& res,
  const index_params& params,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset,
  raft::host_matrix_view<IdxT, int64_t> graph,
  IdxT* medoid_id,
  cuvs::distance::DistanceType metric)
//  int dim)
{
  auto stream = raft::resource::get_cuda_stream(res);
  int N       = dataset.extent(0);
  int dim     = dataset.extent(1);
  int degree  = graph.extent(1);

  // Algorithm params
  int max_batchsize = (int)(params.max_fraction * (float)N);
  if (max_batchsize > (int)dataset.extent(0)) {
    RAFT_LOG_WARN(
      "Max fraction is the fraction of the total dataset, so it cannot be larger 1.0, reducing it "
      "to 1.0");
    max_batchsize = (int)dataset.extent(0);
  }
  int insert_iters  = (int)(params.vamana_iters);
  double base       = (double)(params.batch_base);
  float alpha       = (float)(params.alpha);
  int visited_size  = params.visited_size;
  int queue_size    = params.queue_size;
  int reverse_batch = params.reverse_batchsize;

  if ((visited_size & (visited_size - 1)) != 0) {
    RAFT_LOG_WARN("visited_size must be a power of 2, rounding up.");
    int power = params.graph_degree;
    while (power < visited_size)
      power <<= 1;
    visited_size = power;
  }

  // create gpu graph and set to all -1s
  auto d_graph = raft::make_device_matrix<IdxT, int64_t>(res, graph.extent(0), graph.extent(1));
  raft::linalg::map(res, d_graph.view(), raft::const_op<IdxT>{raft::upper_bound<IdxT>()});

  // Temp storage about each batch of inserts being performed
  auto query_ids      = raft::make_device_vector<IdxT>(res, max_batchsize);
  auto query_list_ptr = raft::make_device_mdarray<QueryCandidates<IdxT, accT>>(
    res,
    raft::resource::get_large_workspace_resource(res),
    raft::make_extents<int64_t>(max_batchsize + 1));
  QueryCandidates<IdxT, accT>* query_list =
    static_cast<QueryCandidates<IdxT, accT>*>(query_list_ptr.data_handle());

  // Results of each batch of inserts during build - Memory is used by query_list structure
  auto visited_ids =
    raft::make_device_mdarray<IdxT>(res,
                                    raft::resource::get_large_workspace_resource(res),
                                    raft::make_extents<int64_t>(max_batchsize, visited_size));
  auto visited_dists =
    raft::make_device_mdarray<accT>(res,
                                    raft::resource::get_large_workspace_resource(res),
                                    raft::make_extents<int64_t>(max_batchsize, visited_size));

  // Assign memory to query_list structures and initiailize
  init_query_candidate_list<IdxT, accT><<<256, blockD, 0, stream>>>(query_list,
                                                                    visited_ids.data_handle(),
                                                                    visited_dists.data_handle(),
                                                                    (int)max_batchsize,
                                                                    visited_size);

  // Create random permutation for order of node inserts into graph
  std::vector<IdxT> insert_order;
  create_insert_permutation<IdxT>(insert_order, (uint32_t)N);

  // Calculate the shared memory sizes of each kernel
  int search_smem_sort_size = 0;
  int prune_smem_sort_size  = 0;
  SELECT_SMEM_SIZES(degree, visited_size);  // Sets above 2 variables to appropriate sizes

  // Total dynamic shared memory used by GreedySearch
  int align_padding          = raft::alignTo(dim, 16) - dim;
  int search_smem_total_size = static_cast<int>(
    search_smem_sort_size + (dim + align_padding) * sizeof(T) + visited_size * sizeof(Node<accT>) +
    degree * sizeof(int) + queue_size * sizeof(DistPair<IdxT, accT>));

  // Total dynamic shared memory size needed by both RobustPrune calls
  int prune_smem_total_size = prune_smem_sort_size + (dim + align_padding) * sizeof(T) +
                              (degree + visited_size) * sizeof(DistPair<IdxT, accT>);

  RAFT_LOG_DEBUG("Dynamic shared memory usage (bytes): GreedySearch: %d, RobustPrune: %d",
                 search_smem_total_size,
                 prune_smem_total_size);

  if (prune_smem_sort_size == 0) {  // If sizes not supported, smem sizes will be 0
    RAFT_FAIL("Vamana graph parameters not supported: graph_degree=%d, visited_size:%d\n",
              degree,
              visited_size);
  }

  // Random medoid has minor impact on recall
  // TODO: use heuristic for better medoid selection, issue:
  // https://github.com/rapidsai/cuvs/issues/355
  *medoid_id = rand() % N;

  // size of current batch of inserts, increases logarithmically until max_batchsize
  int step_size = 1;
  // Number of passes over dataset (default 1)
  for (int iter = 0; iter < insert_iters; iter++) {
    // Loop through batches and call the insert and prune kernels
    for (int start = 0; start < N;) {
      if (start + step_size > N) {
        int new_size = N - start;
        step_size    = new_size;
      }
      RAFT_LOG_DEBUG("Starting batch of inserts indices_start:%d, batch_size:%d", start, step_size);

      int num_blocks = min(maxBlocks, step_size);

      // Copy ids to be inserted for this batch
      raft::copy(query_ids.data_handle(), &insert_order.data()[start], step_size, stream);
      set_query_ids<IdxT, accT><<<num_blocks, blockD, 0, stream>>>(
        query_list_ptr.data_handle(), query_ids.data_handle(), step_size);

      // Call greedy search to get candidates for every vector being inserted
      GreedySearchKernel<T, accT, IdxT, Accessor>
        <<<num_blocks, blockD, search_smem_total_size, stream>>>(d_graph.view(),
                                                                 dataset,
                                                                 query_list_ptr.data_handle(),
                                                                 step_size,
                                                                 *medoid_id,
                                                                 visited_size,
                                                                 metric,
                                                                 queue_size,
                                                                 search_smem_sort_size);
      // Run on candidates of vectors being inserted
      RobustPruneKernel<T, accT, IdxT>
        <<<num_blocks, blockD, prune_smem_total_size, stream>>>(d_graph.view(),
                                                                dataset,
                                                                query_list_ptr.data_handle(),
                                                                step_size,
                                                                visited_size,
                                                                metric,
                                                                alpha,
                                                                prune_smem_sort_size);

      // Write results from first prune to graph edge list
      write_graph_edges_kernel<accT, IdxT><<<num_blocks, blockD, 0, stream>>>(
        d_graph.view(), query_list_ptr.data_handle(), degree, step_size);

      // compute prefix sums of query_list sizes - TODO parallelize prefix sums
      auto d_total_edges = raft::make_device_mdarray<int>(
        res, raft::resource::get_workspace_resource(res), raft::make_extents<int64_t>(1));
      prefix_sums_sizes<accT, IdxT>
        <<<1, 1, 0, stream>>>(query_list, step_size, d_total_edges.data_handle());

      int total_edges;
      raft::copy(&total_edges, d_total_edges.data_handle(), 1, stream);
      RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

      auto edge_dest =
        raft::make_device_mdarray<IdxT>(res,
                                        raft::resource::get_large_workspace_resource(res),
                                        raft::make_extents<int64_t>(total_edges));
      auto edge_src =
        raft::make_device_mdarray<IdxT>(res,
                                        raft::resource::get_large_workspace_resource(res),
                                        raft::make_extents<int64_t>(total_edges));

      // Create reverse edge list
      create_reverse_edge_list<accT, IdxT>
        <<<num_blocks, blockD, 0, stream>>>(query_list_ptr.data_handle(),
                                            step_size,
                                            degree,
                                            edge_src.data_handle(),
                                            edge_dest.data_handle());

      void* d_temp_storage      = nullptr;
      size_t temp_storage_bytes = 0;

      cub::DeviceMergeSort::SortPairs(d_temp_storage,
                                      temp_storage_bytes,
                                      edge_dest.data_handle(),
                                      edge_src.data_handle(),
                                      total_edges,
                                      CmpEdge<IdxT>(),
                                      stream);

      RAFT_LOG_DEBUG("Temp storage needed for sorting (bytes): %lu", temp_storage_bytes);

      auto temp_sort_storage = raft::make_device_mdarray<IdxT>(
        res,
        raft::resource::get_large_workspace_resource(res),
        raft::make_extents<int64_t>(temp_storage_bytes / sizeof(IdxT)));

      // Sort to group reverse edges by destination
      cub::DeviceMergeSort::SortPairs(temp_sort_storage.data_handle(),
                                      temp_storage_bytes,
                                      edge_dest.data_handle(),
                                      edge_src.data_handle(),
                                      total_edges,
                                      CmpEdge<IdxT>(),
                                      stream);

      // Get number of unique node destinations
      IdxT unique_dests =
        cuvs::sparse::neighbors::get_n_components(edge_dest.data_handle(), total_edges, stream);

      // Find which node IDs have reverse edges and their indices in the reverse edge list
      thrust::device_vector<IdxT> edge_dest_vec(edge_dest.data_handle(),
                                                edge_dest.data_handle() + total_edges);
      auto unique_indices = raft::make_device_vector<int>(res, total_edges);
      raft::linalg::map_offset(res, unique_indices.view(), raft::identity_op{});

      thrust::unique_by_key(
        edge_dest_vec.begin(), edge_dest_vec.end(), unique_indices.data_handle());

      edge_dest_vec.clear();
      edge_dest_vec.shrink_to_fit();

      // Batch execution of reverse edge creation/application
      reverse_batch = params.reverse_batchsize;
      for (int rev_start = 0; rev_start < (int)unique_dests; rev_start += reverse_batch) {
        if (rev_start + reverse_batch > (int)unique_dests) {
          reverse_batch = (int)unique_dests - rev_start;
        }

        // Allocate reverse QueryCandidate list based on number of unique destinations
        auto reverse_list_ptr = raft::make_device_mdarray<QueryCandidates<IdxT, accT>>(
          res,
          raft::resource::get_large_workspace_resource(res),
          raft::make_extents<int64_t>(reverse_batch));
        auto rev_ids =
          raft::make_device_mdarray<IdxT>(res,
                                          raft::resource::get_large_workspace_resource(res),
                                          raft::make_extents<int64_t>(reverse_batch, visited_size));
        auto rev_dists =
          raft::make_device_mdarray<accT>(res,
                                          raft::resource::get_large_workspace_resource(res),
                                          raft::make_extents<int64_t>(reverse_batch, visited_size));

        QueryCandidates<IdxT, accT>* reverse_list =
          static_cast<QueryCandidates<IdxT, accT>*>(reverse_list_ptr.data_handle());

        init_query_candidate_list<IdxT, accT><<<256, blockD, 0, stream>>>(reverse_list,
                                                                          rev_ids.data_handle(),
                                                                          rev_dists.data_handle(),
                                                                          (int)reverse_batch,
                                                                          visited_size);

        // May need more blocks for reverse list
        num_blocks = min(maxBlocks, reverse_batch);

        // Populate reverse list ids and candidate lists from edge_src and edge_dest
        populate_reverse_list_struct<T, accT, IdxT>
          <<<num_blocks, blockD, 0, stream>>>(reverse_list,
                                              edge_src.data_handle(),
                                              edge_dest.data_handle(),
                                              unique_indices.data_handle(),
                                              unique_dests,
                                              total_edges,
                                              dataset.extent(0),
                                              rev_start,
                                              reverse_batch);

        // Recompute distances (avoided keeping it during sorting)
        recompute_reverse_dists<T, accT, IdxT>
          <<<num_blocks, blockD, 0, stream>>>(reverse_list, dataset, reverse_batch, metric);

        // Call 2nd RobustPrune on reverse query_list
        RobustPruneKernel<T, accT, IdxT>
          <<<num_blocks, blockD, prune_smem_total_size, stream>>>(d_graph.view(),
                                                                  raft::make_const_mdspan(dataset),
                                                                  reverse_list_ptr.data_handle(),
                                                                  reverse_batch,
                                                                  visited_size,
                                                                  metric,
                                                                  alpha,
                                                                  prune_smem_sort_size);

        // Write new edge lists to graph
        write_graph_edges_kernel<accT, IdxT><<<num_blocks, blockD, 0, stream>>>(
          d_graph.view(), reverse_list_ptr.data_handle(), degree, reverse_batch);
      }

      start += step_size;
      step_size *= base;
      if (step_size > max_batchsize) step_size = max_batchsize;

    }  // Batch of inserts

  }  // insert iterations

  raft::copy(graph.data_handle(), d_graph.data_handle(), d_graph.size(), stream);

  RAFT_CHECK_CUDA(stream);
}

template <typename T>
auto quantize_all_vectors(raft::resources const& res,
                          raft::device_matrix_view<const T, int64_t> residuals,
                          raft::device_matrix_view<float, uint32_t, raft::row_major> pq_codebook,
                          cuvs::neighbors::vpq_params ps)
  -> raft::device_matrix<uint8_t, int64_t, raft::row_major>
{
  auto dim         = residuals.extent(1);
  auto vq_codebook = raft::make_device_matrix<float, uint32_t, raft::row_major>(res, 1, dim);

  auto codes = cuvs::neighbors::detail::process_and_fill_codes_subspaces<float, int64_t>(
    res, ps, residuals, raft::make_const_mdspan(vq_codebook.view()), pq_codebook);
  return codes;
}

template <typename T,
          typename IdxT     = uint64_t,
          typename Accessor = raft::host_device_accessor<std::experimental::default_accessor<T>,
                                                         raft::memory_type::host>>
index<T, IdxT> build(
  raft::resources const& res,
  const index_params& params,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset)
{
  uint32_t graph_degree = params.graph_degree;

  RAFT_EXPECTS(params.metric == cuvs::distance::DistanceType::L2Expanded,
               "Currently only L2Expanded metric is supported");

  const int* deg_size = std::find(std::begin(DEGREE_SIZES), std::end(DEGREE_SIZES), graph_degree);
  RAFT_EXPECTS(deg_size != std::end(DEGREE_SIZES), "Provided graph_degree not currently supported");

  RAFT_EXPECTS(params.visited_size > graph_degree, "visited_size must be > graph_degree");

  int dim = dataset.extent(1);

  RAFT_LOG_DEBUG("Creating empty graph structure");
  auto vamana_graph = raft::make_host_matrix<IdxT, int64_t>(dataset.extent(0), graph_degree);

  RAFT_LOG_DEBUG("Running Vamana batched insert algorithm");

  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded;

  IdxT medoid_id;
  batched_insert_vamana<T, float, IdxT, Accessor>(
    res, params, dataset, vamana_graph.view(), &medoid_id, metric);

  std::optional<raft::device_matrix<uint8_t, int64_t, raft::row_major>> quantized_vectors;
  if (params.codebooks) {
    // Full codebook should be a raft::matrix of dimension [2^PQ_BITS * PQ_DIM, VEC_DIM / PQ_DIM]
    // Every row is (VEC_DIM/PQ_DIM) floats representing a group of cluster centroids.
    // Every consecutive [PQ_DIM] rows is a set.

    // short-hand
    auto& codebook_params = params.codebooks.value();
    int pq_codebook_size  = codebook_params.pq_codebook_size;
    int pq_dim            = codebook_params.pq_dim;

    cuvs::neighbors::vpq_params pq_params;
    pq_params.pq_bits = raft::log2(pq_codebook_size);
    pq_params.pq_dim  = pq_dim;

    // transform pq_encoding_table (dimensions: pq_codebook_size x dim_per_subspace * pq_dim ) to
    // pq_codebook (dimensions: pq_codebook_size * pq_dim, dim_per_subspace)
    auto pq_encoding_table_device_vec = raft::make_device_vector<float, uint32_t>(
      res,
      codebook_params.pq_encoding_table.size());  // logically a 2D matrix with dimensions
                                                  // pq_codebook_size x dim_per_subspace * pq_dim
    raft::copy(pq_encoding_table_device_vec.data_handle(),
               codebook_params.pq_encoding_table.data(),
               codebook_params.pq_encoding_table.size(),
               raft::resource::get_cuda_stream(res));
    int dim_per_subspace = dim / pq_dim;
    auto pq_codebook =
      raft::make_device_matrix<float, uint32_t>(res, pq_codebook_size * pq_dim, dim_per_subspace);
    auto pq_encoding_table_device_vec_view = pq_encoding_table_device_vec.view();
    raft::linalg::map_offset(
      res,
      pq_codebook.view(),
      [pq_encoding_table_device_vec_view,
       pq_dim,
       pq_codebook_size,
       dim_per_subspace,
       dim] __device__(size_t i) {
        int row_idx        = i / dim_per_subspace;
        int subspace_id    = row_idx / pq_codebook_size;  // idx_pq_dim
        int codebook_id    = row_idx % pq_codebook_size;  // idx_pq_codebook_size
        int id_in_subspace = i % dim_per_subspace;        // idx_dim_per_subspace

        return pq_encoding_table_device_vec_view[codebook_id * pq_dim * dim_per_subspace +
                                                 subspace_id * dim_per_subspace + id_in_subspace];
      });

    // prepare rotation matrix
    auto rotation_matrix_device = raft::make_device_matrix<float, int64_t>(res, dim, dim);
    raft::copy(rotation_matrix_device.data_handle(),
               codebook_params.rotation_matrix.data(),
               codebook_params.rotation_matrix.size(),
               raft::resource::get_cuda_stream(res));

    // process in batches
    const uint32_t n_rows = dataset.extent(0);
    // codes_rowlen defined as in cuvs::neighbors::detail::process_and_fill_codes_subspaces()
    const int64_t codes_rowlen =
      sizeof(uint32_t) *
      (1 + raft::div_rounding_up_safe<int64_t>(pq_dim * pq_params.pq_bits, 8 * sizeof(uint32_t)));
    quantized_vectors = raft::make_device_matrix<uint8_t, int64_t, raft::row_major>(
      res,
      n_rows,
      codes_rowlen - 4);  // first 4 columns of output from quantize_all_vectors() to be discarded
    // TODO: with scaling workspace we could choose the batch size dynamically
    constexpr uint32_t kReasonableMaxBatchSize = 65536;
    const uint32_t max_batch_size              = std::min(n_rows, kReasonableMaxBatchSize);
    for (const auto& batch : cuvs::spatial::knn::detail::utils::batch_load_iterator<T>(
           dataset.data_handle(),
           n_rows,
           dim,
           max_batch_size,
           raft::resource::get_cuda_stream(res),
           raft::resource::get_workspace_resource(res))) {
      // perform rotation
      auto dataset_rotated = raft::make_device_matrix<float, int64_t>(res, batch.size(), dim);
      if constexpr (std::is_same_v<T, float>) {
        auto dataset_view = raft::make_device_matrix_view(const_cast<T*>(batch.data()),
                                                          static_cast<int64_t>(batch.size()),
                                                          static_cast<int64_t>(dim));
        raft::linalg::gemm(
          res, dataset_view, rotation_matrix_device.view(), dataset_rotated.view());
      } else {
        // convert dataset to float
        auto dataset_float = raft::make_device_matrix<float, int64_t>(res, batch.size(), dim);
        auto dataset_view  = raft::make_device_matrix_view(
          batch.data(), static_cast<int64_t>(batch.size()), static_cast<int64_t>(dim));
        raft::linalg::map_offset(
          res, dataset_float.view(), [dataset_view, dim] __device__(size_t i) {
            int row_idx = i / dim;
            int col_idx = i % dim;
            return static_cast<float>(dataset_view(row_idx, col_idx));
          });
        raft::linalg::gemm(
          res, dataset_float.view(), rotation_matrix_device.view(), dataset_rotated.view());
      }

      // quantize rotated vectors using codebook
      auto temp_vectors =
        quantize_all_vectors<float>(res, dataset_rotated.view(), pq_codebook.view(), pq_params);

      // Remove the vector quantization header values
      raft::matrix::slice_coordinates<int64_t> slice_coords(
        0, 4, temp_vectors.extent(0), temp_vectors.extent(1));

      raft::matrix::slice(res,
                          raft::make_const_mdspan(temp_vectors.view()),
                          raft::make_device_matrix_view<uint8_t, int64_t>(
                            quantized_vectors.value().data_handle() +
                              batch.offset() * quantized_vectors.value().extent(1),
                            batch.size(),
                            quantized_vectors.value().extent(1)),
                          slice_coords);
    }
  }

  try {
    auto idx = index<T, IdxT>(
      res, params.metric, dataset, raft::make_const_mdspan(vamana_graph.view()), medoid_id);
    if (quantized_vectors) idx.update_quantized_dataset(res, std::move(quantized_vectors.value()));
    return idx;
  } catch (std::bad_alloc& e) {
    RAFT_LOG_DEBUG("Insufficient GPU memory to construct VAMANA index with dataset on GPU");
    // We just add the graph. User is expected to update dataset separately (e.g allocating in
    // managed memory).
  } catch (raft::logic_error& e) {
    // The memory error can also manifest as logic_error.
    RAFT_LOG_DEBUG("Insufficient GPU memory to construct VAMANA index with dataset on GPU");
  }
  index<T, IdxT> idx(res, params.metric);
  RAFT_LOG_WARN("Constructor not called, returning empty index");
  return idx;
}

/**
 * @}
 */

}  // namespace cuvs::neighbors::vamana::detail
