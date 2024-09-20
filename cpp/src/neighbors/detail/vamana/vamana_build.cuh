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

//#include "../../vpq_dataset.cuh"
//#include "graph_core.cuh"
#include "vamana_structs.cuh"
#include "vamana_search.cuh"
#include "robust_prune.cuh"
#include <cuvs/neighbors/vamana.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/logger-ext.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/matrix/init.cuh>
#include <raft/sparse/neighbors/cross_component_nn.cuh>
#include <raft/random/make_blobs.cuh>
#include <raft/cluster/kmeans.cuh>
#include <raft/cluster/kmeans_types.hpp>
#include <raft/matrix/copy.cuh>


#include <thrust/device_vector.h>
#include <thrust/unique.h>
#include <thrust/sequence.h>

#include <cuvs/distance/distance.hpp>

#include <rmm/resource_ref.hpp>

#include <chrono>
#include <cstdio>
#include <vector>

namespace cuvs::neighbors::vamana::detail {

/* @defgroup vamana_build_detail vamana build
 * @{
 */

static const std::string RAFT_NAME = "raft";

static const int blockD = 32;
static const int maxBlocks = 10000;


// generate random permutation of inserts - TODO do this on GPU / faster
template<typename IdxT>
void create_insert_permutation(std::vector<IdxT>& insert_order, uint32_t N)
{
  insert_order.resize(N);
  for(uint32_t i=0; i<N; i++) {
    insert_order[i] = (IdxT)i;
  }
  for(uint32_t i=0; i<N; i++) {
    uint32_t temp;
    uint32_t rand_idx = rand()%N;
    temp = insert_order[i];
    insert_order[i] = insert_order[rand_idx];
    insert_order[rand_idx] = temp;
  }
}

// Initialize empty graph memory
template<typename IdxT>
__global__ void memset_graph(raft::device_matrix_view<IdxT, int64_t> graph) {
  for(int i = blockIdx.x; i<graph.extent(0); i+=gridDim.x) {
    for(int j=threadIdx.x; j<graph.extent(1); j+=blockDim.x) {
      graph(i,j) = INFTY<IdxT>();
    }
  }
}

// TODO - fix true approximate medoid selection below
/*
template<typename T, typename accT, typename IdxT = uint32_t,
         typename Accessor = raft::host_device_accessor<std::experimental::default_accessor<T>,
                                                         raft::memory_type::host>>
IdxT select_medoid(raft::resources const& dev_resources,
	  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset,
          float fraction) {

  auto stream = raft::resource::get_cuda_stream(dev_resources);
  IdxT medoid;
  int64_t n = dataset.extent(0);
  int64_t dim = dataset.extent(1);
//  int64_t n_sample = n * fraction;
  int64_t n_samples = 1024;

  const int num_reduc_threads=256;
  if(n_samples < num_reduc_threads) n_samples = num_reduc_threads;

  int seed = 137;
  raft::random::RngState rng(seed);

  auto random_indices = raft::make_device_vector<int>(dev_resources, n_samples);
  raft::random::uniformInt(dev_resources, rng,
		  random_indices.view(), (int)0, (int)n);


  auto trainset     = raft::make_device_matrix<T, int64_t>(dev_resources, n_samples, dim);
//  auto train_indices = raft::make_device_vector<int64_t>(dev_resources, n_samples);
//  raft::random::sample_without_replacement(dev_resources, rng, data
  raft::matrix::copy_rows(dev_resources, dataset, trainset.view(), raft::make_const_mdspan(random_indices.view()));

  raft::cluster::KMeansParams params;
  params.n_clusters = 1;
  int inertia, n_iter;
  auto centroids = raft::make_device_matrix<T, int64_t>(dev_resources, 1, dim);

  raft::cluster::kmeans::fit(dev_resources,
		  	params,
			trainset.view(),
			std::nullopt,
			centroids.view(),
			raft::make_host_scalar_view(&inertia),
			raft::make_host_scalar_view(&n_iter));
			

  auto h_centroid = raft::make_host_matrix<T,int64_t>(1, dim);
  raft::copy(h_centroid.data_handle(), centroids.data_handle(), centroids.size(), stream);

  printf("centroid\n");
  for(int i=0; i<dim; i++){
    printf("%f, ", h_centroid(0,i));
  }
  printf("\n");

//  rmm::device_uvector<DistPair<IdxT,accT>> d_medoid(
//        num_reduc_threads, stream, raft::resource::get_large_workspace_resource(res));


  IdxT final_medoid=0;
  return final_medoid;
 
}
*/

/********************************************************************************************
 * Main Vamana building function - insert vectors into empty graph in batches
 * Pre - dataset contains the vector data, host matrix allocated to store the graph
 * Post - graph matrix contains the graph edges of the final Vamana graph
 *******************************************************************************************/
template<typename T,
         typename accT,
         typename IdxT = uint32_t,
         typename Accessor = raft::host_device_accessor<std::experimental::default_accessor<T>,
                                                         raft::memory_type::host>>
void batched_insert_vamana(
  raft::resources const& res,
  const index_params& params,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset,
  raft::host_matrix_view<IdxT, int64_t> graph,
  cuvs::distance::DistanceType metric, int dim)
{
  auto stream = raft::resource::get_cuda_stream(res);
  int N = dataset.extent(0);
  int degree = graph.extent(1);

  // Algorithm params
  int max_batchsize = (int)(params.max_batchsize*(float)N);
  if (max_batchsize > (int)dataset.extent(0)) {
    RAFT_LOG_WARN(
      "Max batch insert size cannot be larger 1.0, reducing it to 1");
    max_batchsize = (int)dataset.extent(0);
  }
  int insert_iters = (int)(params.vamana_iters);
  double base = (double)(params.batch_base);
  float alpha = (float)(params.alpha);
  int visited_size = params.visited_size;
  int queue_size = params.queue_size;

  // create gpu graph and set to all -1s
  auto d_graph = raft::make_device_matrix<IdxT, int64_t>(res, graph.extent(0), graph.extent(1));
  memset_graph<IdxT><<<256,blockD>>>(d_graph.view());

  // Temp storage about each batch of inserts being performed
  auto query_ids = raft::make_device_vector<IdxT>(res, max_batchsize);
  rmm::device_buffer query_list_ptr{
      (max_batchsize+1)*sizeof(QueryCandidates<IdxT, accT>), stream, raft::resource::get_large_workspace_resource(res)};
  QueryCandidates<IdxT,accT>* query_list = static_cast<QueryCandidates<IdxT,accT>*>(query_list_ptr.data());

  
  // Results of each batch of inserts during build - Memory is used by query_list structure
  rmm::device_uvector<IdxT> visited_ids(
       max_batchsize*visited_size, stream, raft::resource::get_large_workspace_resource(res));
  rmm::device_uvector<accT> visited_dists(
       max_batchsize*visited_size, stream, raft::resource::get_large_workspace_resource(res));

  // Assign memory to query_list stuctures and initiailize
  init_query_candidate_list<IdxT, accT><<<256, blockD>>>(query_list, visited_ids.data(), visited_dists.data(), (int)max_batchsize, visited_size);

  // Create random permutation for order of node inserts into graph
  std::vector<IdxT> insert_order;
  create_insert_permutation<IdxT>(insert_order, (uint32_t)N);

  // Memory needed to sort reverse edges - potentially large memory footprint
  rmm::device_uvector<IdxT> edge_dest(
    max_batchsize*degree, stream, raft::resource::get_large_workspace_resource(res));
  rmm::device_uvector<IdxT> edge_src(
    max_batchsize*degree, stream, raft::resource::get_large_workspace_resource(res));

  size_t temp_storage_bytes = max_batchsize*degree*(2*sizeof(int));
  RAFT_LOG_DEBUG("Temp storage needed for sorting (bytes): %lu", temp_storage_bytes);
  rmm::device_buffer temp_sort_storage{
      temp_storage_bytes, stream, raft::resource::get_large_workspace_resource(res)};
  
  // Calcualte the shared memory sizes of each kernel
  int search_smem_sort_size=0;
  int prune_smem_sort_size=0;
  SELECT_SMEM_SIZES(degree, visited_size); // Sets above 2 variables to appropriate sizes

  // Total dynamic shared memory used by GreedySearch
  int search_smem_total_size = static_cast<int>(search_smem_sort_size + 
                             dim * sizeof(T) +
			     visited_size * sizeof(Node<accT>) + 
			     degree * sizeof(int) +
			     queue_size * sizeof(DistPair<IdxT,accT>));


  // Total dynamic shared memory size needed by both RobustPrune calls
  int prune_smem_total_size = prune_smem_sort_size +
			       dim * sizeof(T) +
			       (degree + visited_size) * sizeof(DistPair<IdxT,accT>); 

  RAFT_LOG_DEBUG("Dynamic shared memory usage (bytes): GreedySearch: %d, RobustPrune: %d", search_smem_total_size, prune_smem_total_size);

  if(prune_smem_sort_size == 0) { // If sizes not supported, smem sizes will be 0
    RAFT_FAIL("Vamana graph parameters not supported: graph_degree=%d, visited_size:%d\n", 
		    degree, visited_size);
  }

  size_t free, total;
  cudaMemGetInfo(&free, &total);
  RAFT_LOG_DEBUG("Device memory - free:%ld, total:%ld", free, total);

// TODO - fix medoid selection
//  IdxT medoid_test = select_medoid<T, accT>(res, dataset, 0.01);

// Random medoid has minor impact on recall - TODO compute actual approximate medoid
    int medoid_id = rand()%N; 

  // size of current batch of inserts, increases logarithmically until max_batchsize
  int step_size=1;
  // Number of passes over dataset (default 1)
  for(int iter=0; iter < insert_iters; iter++) {
  
  // Loop through batches and call the insert and prune kernels
    for(int start=0; start < N; ) {

      if(start+step_size > N) {
        int new_size = N - start;
        step_size = new_size;
      }
      RAFT_LOG_DEBUG("Starting batch of inserts indices_start:%d, batch_size:%d", start, step_size);

      int num_blocks = min(maxBlocks, step_size);

      // Copy ids to be inserted for this batch
      raft::copy(query_ids.data_handle(), &insert_order.data()[start], step_size, stream);
      set_query_ids<IdxT,accT><<<num_blocks,blockD>>>(query_list_ptr.data(), query_ids.data_handle(), step_size);

      // Call greedy search to get candidates for every vector being inserted
      GreedySearchKernel<T, accT, IdxT, Accessor>
          <<<num_blocks, blockD, search_smem_total_size>>>(
                     d_graph.view(), 
                     dataset, 
                     query_list_ptr.data(), 
                     step_size, 
                     medoid_id,
                     degree, 
                     dataset.extent(0), 
                     visited_size,
                     metric, 
		     dim,
		     queue_size,
		     search_smem_sort_size);


        // Run on candidates of vectors being inserted
      RobustPruneKernel<T,accT, IdxT>
          <<<num_blocks, blockD, prune_smem_total_size>>>(
                     d_graph.view(), 
                     dataset,
                     query_list_ptr.data(),
                      step_size,
                     degree,
		     visited_size,
                     dataset.extent(0),
                     metric,
                     alpha,
		     dim,
		     prune_smem_sort_size);
		     

// Write results from first prune to graph edge list
      write_graph_edges_kernel<accT, IdxT><<<num_blocks, blockD>>>(
                     d_graph.view(), 
                     query_list_ptr.data(), 
                     degree, 
                     step_size);

// compute prefix sums of query_list sizes - TODO parallelize prefix sums
      rmm::device_uvector<int> d_total_edges(
          1, stream, raft::resource::get_large_workspace_resource(res));
      prefix_sums_sizes<accT,IdxT><<<1,1>>>(query_list, step_size, d_total_edges.data());
//      cudaDeviceSynchronize(); // TODO -remove?

      int total_edges;
      raft::copy(&total_edges, d_total_edges.data(), 1, stream);
  
// Create reverse edge list
      create_reverse_edge_list<accT,IdxT><<<num_blocks,blockD>>>(
                     query_list_ptr.data(), 
                     step_size, 
                     degree, 
                     edge_src.data(), 
                     edge_dest.data());

      // Sort to group reverse edges by destination
      cub::DeviceMergeSort::SortPairs(
                     temp_sort_storage.data(),
                     temp_storage_bytes,
                     edge_dest.data(),
                     edge_src.data(),
                     total_edges,
                     CmpEdge<IdxT>());

// Get number of unique node destinations
     IdxT unique_dests = raft::sparse::neighbors::get_n_components(edge_dest.data(), total_edges, stream);

     // Find which node IDs have reverse edges and their indices in the reverse edge list
     thrust::device_vector<IdxT> edge_dest_vec(edge_dest.data(), edge_dest.data()+total_edges);
     thrust::device_vector<int> unique_indices(total_edges);
     thrust::sequence(unique_indices.begin(), unique_indices.end());
     thrust::unique_by_key(edge_dest_vec.begin(), edge_dest_vec.end(), unique_indices.begin());


// Allocate reverse QueryCandidate list based on number of unique destinations
// TODO - Do this in batches to reduce memory footprint / support larger datasets
      rmm::device_buffer reverse_list_ptr{
          unique_dests*sizeof(QueryCandidates<IdxT, accT>), stream, raft::resource::get_large_workspace_resource(res)};
      rmm::device_uvector<IdxT> rev_ids(
           unique_dests*visited_size, stream, raft::resource::get_large_workspace_resource(res));
      rmm::device_uvector<accT> rev_dists(
           unique_dests*visited_size, stream, raft::resource::get_large_workspace_resource(res));

  QueryCandidates<IdxT,accT>* reverse_list = static_cast<QueryCandidates<IdxT,accT>*>(reverse_list_ptr.data());

      init_query_candidate_list<IdxT, accT><<<256, blockD>>>(reverse_list, rev_ids.data(), rev_dists.data(), (int)unique_dests, visited_size);


      // May need more blocks for reverse list
      num_blocks = min(maxBlocks, unique_dests);

// Populate reverse list ids and candidate lists from edge_src and edge_dest
      int* unique_indices_ptr = thrust::raw_pointer_cast(unique_indices.data());
      populate_reverse_list_struct<T,accT,IdxT><<<num_blocks,blockD>>>(
                     reverse_list,
                     edge_src.data(),
                     edge_dest.data(),
		     unique_indices_ptr,
		     unique_dests,
                     total_edges,
                     dataset.extent(0));
           
      // Recompute distances (avoided keeping it during sorting)
      recompute_reverse_dists<T,accT,IdxT><<<num_blocks, blockD>>>(
                      reverse_list,
                      dataset,
                      unique_dests,
		      dim,
		      metric);
                       
// Call 2nd RobustPrune on reverse query_list
      RobustPruneKernel<T,accT,IdxT>
            <<<num_blocks, blockD, prune_smem_total_size>>>(
                      d_graph.view(),
                      raft::make_const_mdspan(dataset),
                      reverse_list_ptr.data(),
                      unique_dests,
                      degree,
		      visited_size,
                      dataset.extent(0),
                      metric,
                      alpha,
		      dim,
		      prune_smem_sort_size);
		      
// Write new edge lists to graph
      write_graph_edges_kernel<accT, IdxT><<<num_blocks, blockD>>>(
                      d_graph.view(),
                      reverse_list_ptr.data(),
                      degree,
                      unique_dests);

      start += step_size;
      step_size *= base;
      if(step_size > max_batchsize) step_size = max_batchsize;

    } // Batch of inserts

  } // insert iterations

  raft::copy(graph.data_handle(), d_graph.data_handle(), d_graph.size(), stream);

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
  size_t graph_degree = params.graph_degree;
 
  RAFT_EXPECTS(params.metric == cuvs::distance::DistanceType::L2Expanded, 
              "Currently only L2Expanded metric is supported");

  RAFT_LOG_DEBUG("Creating empty graph structure");
  auto vamana_graph = raft::make_host_matrix<IdxT, int64_t>(dataset.extent(0), graph_degree);

  RAFT_LOG_DEBUG("Running Vamana batched insert algorithm");

  int dim = dataset.extent(1);

  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded;

  batched_insert_vamana<T, float, IdxT, Accessor>(res, params, dataset, vamana_graph.view(), metric, dim);

  try {
    return index<T, IdxT>(res, params.metric, dataset, raft::make_const_mdspan(vamana_graph.view()));
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
