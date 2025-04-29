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

#pragma once

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
#include <raft/matrix/copy.cuh>
#include <raft/matrix/init.cuh>
#include <raft/random/make_blobs.cuh>
#include <raft/sparse/neighbors/cross_component_nn.cuh>

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
//  int max_batchsize = (int)(params.max_fraction * (float)N);
  int max_batchsize = 1;
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

  auto start_t = std::chrono::system_clock::now();


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

  auto topk_pq_mem =
        raft::make_device_mdarray<Node<accT>>(res,
             raft::resource::get_large_workspace_resource(res),
             raft::make_extents<int64_t>(max_batchsize, visited_size));

  auto s_coords_mem =
	  raft::make_device_mdarray<T>(res,
			  raft::resource::get_large_workspace_resource(res),
			  raft::make_extents<int64_t>(max_batchsize, dim));



  // Create random permutation for order of node inserts into graph
  std::vector<IdxT> insert_order;
  create_insert_permutation<IdxT>(insert_order, (uint32_t)N);

  // Calculate the shared memory sizes of each kernel
  int search_smem_sort_size = 0;
  int prune_smem_sort_size  = 0;
  SELECT_SMEM_SIZES(degree, visited_size);  // Sets above 2 variables to appropriate sizes
  prune_smem_sort_size=0;

  // Total dynamic shared memory used by GreedySearch
  int align_padding          = raft::alignTo(dim, 16) - dim;
  int search_smem_total_size = static_cast<int>(
   // search_smem_sort_size + 
    (dim + align_padding) * sizeof(T) + //visited_size * sizeof(Node<accT>) +
    degree * sizeof(int) + queue_size * sizeof(DistPair<IdxT, accT>));

  // Total dynamic shared memory size needed by both RobustPrune calls
  int prune_smem_total_size = prune_smem_sort_size + //(dim + align_padding) * sizeof(T) +
//                              (degree) * sizeof(DistPair<IdxT, accT>);
	                      (degree + visited_size) * sizeof(float) + // Occlusion list
                              (degree + visited_size) * sizeof(DistPair<IdxT, accT>);

  RAFT_LOG_DEBUG("Dynamic shared memory usage (bytes): GreedySearch: %d, RobustPrune: %d",
                 search_smem_total_size,
                 prune_smem_total_size);

  printf("Dynamic shared memory usage (bytes): GreedySearch: %d, segment sort: %d, RobustPrune: %d\n",
                 search_smem_total_size,
		 search_smem_sort_size,
                 prune_smem_total_size);

  /*
  if (prune_smem_sort_size == 0) {  // If sizes not supported, smem sizes will be 0
    RAFT_FAIL("Vamana graph parameters not supported: graph_degree=%d, visited_size:%d\n",
              degree,
              visited_size);
  }
  */


    auto end_t   = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end_t - start_t;
  double alloc_time = elapsed_seconds.count();

  double search_time = 0.0;
  double segment_sort_time = 0.0;
  double prune1_time = 0.0;
  double write1_time = 0.0;
  double rev_time =0.0;
  double batch_prune = 0.0;

  // Random medoid has minor impact on recall
  // TODO: use heuristic for better medoid selection, issue:
  // https://github.com/rapidsai/cuvs/issues/355
  *medoid_id = rand() % N;

  // size of current batch of inserts, increases logarithmically until max_batchsize
  int step_size = 1;
  // Number of passes over dataset (default 1)
  for (int iter = 0; iter < insert_iters; iter++) {
    // Loop through batches and call the insert and prune kernels
//    for (int start = 0; start < N;) {
    for (int start = 0; start < 40;) {
start_t = std::chrono::system_clock::now();


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
								 topk_pq_mem.data_handle());
                                                                 //search_smem_sort_size);
cudaDeviceSynchronize();
end_t   = std::chrono::system_clock::now();
elapsed_seconds = end_t - start_t;
search_time += elapsed_seconds.count();
start_t = std::chrono::system_clock::now();

// SEGMENTED SORT ON QUERY_LIST
TestSortKernel<T, accT, IdxT, Accessor>
	<<<num_blocks, blockD, search_smem_sort_size, stream>>>(query_list_ptr.data_handle(), step_size, visited_size);
cudaDeviceSynchronize();

end_t   = std::chrono::system_clock::now();
elapsed_seconds = end_t - start_t;
segment_sort_time += elapsed_seconds.count();
start_t = std::chrono::system_clock::now();

printf("------------ PRUNE 1 ------------\n");
      // Run on candidates of vectors being inserted
      RobustPruneKernel<T, accT, IdxT>
        <<<num_blocks, blockD, prune_smem_total_size, stream>>>(d_graph.view(),
                                                                dataset,
                                                                query_list_ptr.data_handle(),
                                                                step_size,
                                                                visited_size,
                                                                metric,
                                                                alpha,
                                                                prune_smem_sort_size,
								s_coords_mem.data_handle());
// SEGMENTED SORT ON QUERY_LIST
TestSortKernel<T, accT, IdxT, Accessor>
	<<<num_blocks, blockD, search_smem_sort_size, stream>>>(query_list_ptr.data_handle(), step_size, degree);

cudaDeviceSynchronize();
end_t   = std::chrono::system_clock::now();
elapsed_seconds = end_t - start_t;
prune1_time += elapsed_seconds.count();
start_t = std::chrono::system_clock::now();

      // Write results from first prune to graph edge list
      write_graph_edges_kernel<accT, IdxT><<<num_blocks, blockD, 0, stream>>>(
        d_graph.view(), query_list_ptr.data_handle(), degree, step_size);

      cudaDeviceSynchronize();
end_t   = std::chrono::system_clock::now();
elapsed_seconds = end_t - start_t;
write1_time += elapsed_seconds.count();
start_t = std::chrono::system_clock::now();

      // compute prefix sums of query_list sizes - TODO parallelize prefix sums
      auto d_total_edges = raft::make_device_mdarray<int>(
        res, raft::resource::get_workspace_resource(res), raft::make_extents<int64_t>(1));
      prefix_sums_sizes<accT, IdxT>
        <<<1, 1, 0, stream>>>(query_list, step_size, d_total_edges.data_handle());

      int total_edges;
      raft::copy(&total_edges, d_total_edges.data_handle(), 1, stream);
      RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

      /*
      auto edge_dist_pair = raft::make_device_mdarray<DistPair<IdxT, accT>>(
        res,
        raft::resource::get_large_workspace_resource(res),
        raft::make_extents<int64_t>(total_edges));
	*/
      DistPair<IdxT,accT>* edge_dist_pair;
      cudaMalloc(&edge_dist_pair, total_edges*sizeof(DistPair<IdxT,accT>));
//      custom_memory_resource mr;
//      rmm::device_buffer edge_dist_pair(total_edges*sizeof(int64_t), stream, &s);

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
                                            edge_dist_pair);
                                            //edge_dist_pair.data_handle());


      {
      // Sort by dists first so final edge lists are each sorted by dist
      void* d_temp_storage      = nullptr;
      size_t temp_storage_bytes = 0;

      cub::DeviceMergeSort::SortPairs(d_temp_storage,
                                      temp_storage_bytes,
                                      edge_dist_pair,
                                      //edge_dist_pair.data_handle(),
                                      edge_src.data_handle(),
                                      total_edges,
                                      CmpDist<IdxT,accT>(),
                                      stream);

      RAFT_LOG_DEBUG("Temp storage needed for sorting dist (bytes): %lu", temp_storage_bytes);

      auto temp_sort_storage = raft::make_device_mdarray<IdxT>(
        res,
        raft::resource::get_large_workspace_resource(res),
        raft::make_extents<int64_t>(temp_storage_bytes / sizeof(IdxT)));

      // Sort to group reverse edges by destination
      cub::DeviceMergeSort::SortPairs(temp_sort_storage.data_handle(),
                                      temp_storage_bytes,
                                      edge_dist_pair,
                                      //edge_dist_pair.data_handle(),
                                      edge_src.data_handle(),
                                      total_edges,
                                      CmpDist<IdxT,accT>(),
                                      stream);

      }

      DistPair<IdxT,accT>* temp_ptr = edge_dist_pair;
      //DistPair<IdxT,accT>* temp_ptr = edge_dist_pair.data_handle();
      raft::linalg::map_offset(res, edge_dest.view(), [temp_ptr] __device__(size_t i) {
                      return temp_ptr[i].idx;
      });


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
        raft::sparse::neighbors::get_n_components(edge_dest.data_handle(), total_edges, stream);

      // Find which node IDs have reverse edges and their indices in the reverse edge list
      thrust::device_vector<IdxT> edge_dest_vec(edge_dest.data_handle(),
                                                edge_dest.data_handle() + total_edges);
      auto unique_indices = raft::make_device_vector<int>(res, total_edges);
      raft::linalg::map_offset(res, unique_indices.view(), raft::identity_op{});

      thrust::unique_by_key(
        edge_dest_vec.begin(), edge_dest_vec.end(), unique_indices.data_handle());

      edge_dest_vec.clear();
      edge_dest_vec.shrink_to_fit();

      cudaDeviceSynchronize();
end_t   = std::chrono::system_clock::now();
elapsed_seconds = end_t - start_t;
rev_time += elapsed_seconds.count();
start_t = std::chrono::system_clock::now();

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

printf("------------ PRUNE 2 ------------\n");
        // Call 2nd RobustPrune on reverse query_list
        RobustPruneKernel<T, accT, IdxT>
          <<<num_blocks, blockD, prune_smem_total_size, stream>>>(d_graph.view(),
                                                                  raft::make_const_mdspan(dataset),
                                                                  reverse_list_ptr.data_handle(),
                                                                  reverse_batch,
                                                                  visited_size,
                                                                  metric,
                                                                  alpha,
                                                                  prune_smem_sort_size,
								  s_coords_mem.data_handle());
	printf("After prune2: %d\n", cudaDeviceSynchronize());

// SEGMENTED SORT ON QUERY_LIST
TestSortKernel<T, accT, IdxT, Accessor>
	<<<num_blocks, blockD, search_smem_sort_size, stream>>>(reverse_list_ptr.data_handle(), reverse_batch, degree);

        // Write new edge lists to graph
        write_graph_edges_kernel<accT, IdxT><<<num_blocks, blockD, 0, stream>>>(
          d_graph.view(), reverse_list_ptr.data_handle(), degree, reverse_batch);
      }

cudaDeviceSynchronize();
end_t   = std::chrono::system_clock::now();
elapsed_seconds = end_t - start_t;
batch_prune += elapsed_seconds.count();      

      start += step_size;
      step_size *= base;
      if (step_size > max_batchsize) step_size = max_batchsize;

    }  // Batch of inserts

  }  // insert iterations

  printf("intro:%lf\ngreedy:%lf\nseg_sort:%lf\nprune1:%lf\nwrite1:%lf\nrev:%lf\nbatch_prune:%lf\n", alloc_time, search_time, segment_sort_time, prune1_time, write1_time, rev_time, batch_prune);

  raft::copy(graph.data_handle(), d_graph.data_handle(), d_graph.size(), stream);

  RAFT_CHECK_CUDA(stream);
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

  try {
    return index<T, IdxT>(
      res, params.metric, dataset, raft::make_const_mdspan(vamana_graph.view()), medoid_id);
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
