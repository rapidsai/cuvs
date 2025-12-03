/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Created by Stardust on 8/18/25.
//

#include "query_gatherer.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cuda_rt_essentials.hpp>

#include <cstdio>

namespace cuvs::neighbors::ivf_rabitq::detail {

// ============================
// Kernel definitions
// ============================

// Optimized kernel with coalesced memory access
__global__ void gather_queries_kernel(const float* __restrict__ d_rotated_queries,
                                      const int* __restrict__ d_gather_indices,
                                      float* __restrict__ d_gathered_queries,
                                      int num_queries,
                                      int D,
                                      int batch_size)
{
  int tid            = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = num_queries * D;

  if (tid >= total_elements) return;

  int query_idx = tid / D;
  int dim_idx   = tid % D;

  int original_query_idx = d_gather_indices[query_idx];

  if (original_query_idx >= 0 && original_query_idx < batch_size) {
    d_gathered_queries[tid] = d_rotated_queries[original_query_idx * D + dim_idx];
  }
}

// Alternative: Vectorized kernel for better performance
__global__ void gather_queries_vectorized_kernel(const float* __restrict__ d_rotated_queries,
                                                 const int* __restrict__ d_gather_indices,
                                                 float* __restrict__ d_gathered_queries,
                                                 int num_queries,
                                                 int D)
{
  int query_idx = blockIdx.x;
  if (query_idx >= num_queries) return;

  int original_query_idx = d_gather_indices[query_idx];

  const float* src = d_rotated_queries + original_query_idx * D;
  float* dst       = d_gathered_queries + query_idx * D;

  int tid    = threadIdx.x;
  int stride = blockDim.x;

  for (int i = tid; i < D; i += stride) {
    dst[i] = src[i];
  }
}

// ============================
// BatchedQueryGatherer methods
// ============================

BatchedQueryGatherer::BatchedQueryGatherer(raft::resources const& handle,
                                           int dim,
                                           int max_batch_size,
                                           int max_clusters)
  : D(dim),
    inner_batch_size(max_batch_size),
    max_clusters_per_batch(max_clusters),
    handle_(handle),
    stream_(raft::resource::get_cuda_stream(handle_))
{
  // Allocate device memory
  RAFT_CUDA_TRY(cudaMallocAsync(
    &d_gathered_queries, static_cast<size_t>(inner_batch_size) * D * sizeof(float), stream_));
  RAFT_CUDA_TRY(cudaMallocAsync(
    &d_cluster_offsets, static_cast<size_t>(max_clusters_per_batch + 1) * sizeof(int), stream_));
  RAFT_CUDA_TRY(cudaMallocAsync(
    &d_cluster_ids, static_cast<size_t>(max_clusters_per_batch) * sizeof(int), stream_));
  RAFT_CUDA_TRY(cudaMallocAsync(
    &d_gather_indices, static_cast<size_t>(inner_batch_size) * sizeof(int), stream_));

  // Allocate pinned host memory for faster transfers
  RAFT_CUDA_TRY(
    cudaMallocHost(&h_gather_indices, static_cast<size_t>(inner_batch_size) * sizeof(int)));
  RAFT_CUDA_TRY(cudaMallocHost(&h_cluster_offsets,
                               static_cast<size_t>(max_clusters_per_batch + 1) * sizeof(int)));
  RAFT_CUDA_TRY(
    cudaMallocHost(&h_cluster_ids, static_cast<size_t>(max_clusters_per_batch) * sizeof(int)));

  //    // Create CUDA stream
  //    cudaStreamCreate(&stream);

  reset_batch();
  raft::resource::sync_stream(handle_);
}

BatchedQueryGatherer::~BatchedQueryGatherer()
{
  // Free device memory
  RAFT_CUDA_TRY(cudaFreeAsync(d_gathered_queries, stream_));
  RAFT_CUDA_TRY(cudaFreeAsync(d_cluster_offsets, stream_));
  RAFT_CUDA_TRY(cudaFreeAsync(d_cluster_ids, stream_));
  RAFT_CUDA_TRY(cudaFreeAsync(d_gather_indices, stream_));

  // Free pinned host memory
  RAFT_CUDA_TRY(cudaFreeHost(h_gather_indices));
  RAFT_CUDA_TRY(cudaFreeHost(h_cluster_offsets));
  RAFT_CUDA_TRY(cudaFreeHost(h_cluster_ids));

  //    // Destroy stream
  //    cudaStreamDestroy(stream);
}

void BatchedQueryGatherer::reset_batch()
{
  current_batch_queries  = 0;
  current_batch_clusters = 0;
  if (h_cluster_offsets) h_cluster_offsets[0] = 0;
}

void BatchedQueryGatherer::add_cluster_to_batch(int cluster_id,
                                                const std::vector<int>& query_indices)
{
  h_cluster_ids[current_batch_clusters] = cluster_id;

  // Add query indices
  for (size_t i = 0; i < query_indices.size(); ++i) {
    h_gather_indices[current_batch_queries + static_cast<int>(i)] = query_indices[i];
  }

  current_batch_queries += static_cast<int>(query_indices.size());
  current_batch_clusters += 1;
  h_cluster_offsets[current_batch_clusters] = current_batch_queries;
}

void BatchedQueryGatherer::execute_batch(const float* d_rotated_queries, int batch_size)
{
  if (current_batch_queries == 0) return;

  // Copy metadata to GPU (async)
  RAFT_CUDA_TRY(cudaMemcpyAsync(d_gather_indices,
                                h_gather_indices,
                                static_cast<size_t>(current_batch_queries) * sizeof(int),
                                cudaMemcpyHostToDevice,
                                stream_));

  RAFT_CUDA_TRY(cudaMemcpyAsync(d_cluster_offsets,
                                h_cluster_offsets,
                                static_cast<size_t>(current_batch_clusters + 1) * sizeof(int),
                                cudaMemcpyHostToDevice,
                                stream_));

  RAFT_CUDA_TRY(cudaMemcpyAsync(d_cluster_ids,
                                h_cluster_ids,
                                static_cast<size_t>(current_batch_clusters) * sizeof(int),
                                cudaMemcpyHostToDevice,
                                stream_));

  // Launch kernel
  int threads     = 256;
  long long total = static_cast<long long>(current_batch_queries) * D;  // guard overflow
  int blocks      = static_cast<int>((total + threads - 1) / threads);

  gather_queries_kernel<<<blocks, threads, 0, stream_>>>(
    d_rotated_queries, d_gather_indices, d_gathered_queries, current_batch_queries, D, batch_size);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // Ensure completion before handing buffers to the callback
  raft::resource::sync_stream(handle_);
}

}  // namespace cuvs::neighbors::ivf_rabitq::detail
