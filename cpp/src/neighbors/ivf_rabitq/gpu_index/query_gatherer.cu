/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Created by Stardust on 8/18/25.
//

#include <cstdio>
#include <cuvs/neighbors/ivf_rabitq/gpu_index/query_gatherer.cuh>

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

BatchedQueryGatherer::BatchedQueryGatherer(int dim,
                                           int max_batch_size,
                                           int max_clusters,
                                           cudaStream_t stream_input)
  : D(dim),
    inner_batch_size(max_batch_size),
    max_clusters_per_batch(max_clusters),
    stream(stream_input)
{
  // Allocate device memory
  cudaMalloc(&d_gathered_queries, static_cast<size_t>(inner_batch_size) * D * sizeof(float));
  cudaMalloc(&d_cluster_offsets, static_cast<size_t>(max_clusters_per_batch + 1) * sizeof(int));
  cudaMalloc(&d_cluster_ids, static_cast<size_t>(max_clusters_per_batch) * sizeof(int));
  cudaMalloc(&d_gather_indices, static_cast<size_t>(inner_batch_size) * sizeof(int));

  // Allocate pinned host memory for faster transfers
  cudaMallocHost(&h_gather_indices, static_cast<size_t>(inner_batch_size) * sizeof(int));
  cudaMallocHost(&h_cluster_offsets, static_cast<size_t>(max_clusters_per_batch + 1) * sizeof(int));
  cudaMallocHost(&h_cluster_ids, static_cast<size_t>(max_clusters_per_batch) * sizeof(int));

  //    // Create CUDA stream
  //    cudaStreamCreate(&stream);

  reset_batch();
}

BatchedQueryGatherer::~BatchedQueryGatherer()
{
  // Free device memory
  cudaFree(d_gathered_queries);
  cudaFree(d_cluster_offsets);
  cudaFree(d_cluster_ids);
  cudaFree(d_gather_indices);

  // Free pinned host memory
  cudaFreeHost(h_gather_indices);
  cudaFreeHost(h_cluster_offsets);
  cudaFreeHost(h_cluster_ids);

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
  cudaMemcpyAsync(d_gather_indices,
                  h_gather_indices,
                  static_cast<size_t>(current_batch_queries) * sizeof(int),
                  cudaMemcpyHostToDevice,
                  stream);

  cudaMemcpyAsync(d_cluster_offsets,
                  h_cluster_offsets,
                  static_cast<size_t>(current_batch_clusters + 1) * sizeof(int),
                  cudaMemcpyHostToDevice,
                  stream);

  cudaMemcpyAsync(d_cluster_ids,
                  h_cluster_ids,
                  static_cast<size_t>(current_batch_clusters) * sizeof(int),
                  cudaMemcpyHostToDevice,
                  stream);

  // Launch kernel
  int threads     = 256;
  long long total = static_cast<long long>(current_batch_queries) * D;  // guard overflow
  int blocks      = static_cast<int>((total + threads - 1) / threads);

  gather_queries_kernel<<<blocks, threads, 0, stream>>>(
    d_rotated_queries, d_gather_indices, d_gathered_queries, current_batch_queries, D, batch_size);

  // Ensure completion before handing buffers to the callback
  cudaStreamSynchronize(stream);
}
