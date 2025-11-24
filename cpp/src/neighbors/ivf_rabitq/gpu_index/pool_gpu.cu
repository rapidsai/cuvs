/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Created by Stardust on 4/14/25.
//

#include <cuvs/neighbors/ivf_rabitq/gpu_index/pool_gpu.cuh>

#include <raft/util/cuda_rt_essentials.hpp>

namespace cuvs::neighbors::ivf_rabitq::detail {

// Host function to create and initialize a DeviceResultPool.
DeviceResultPool* createDeviceResultPool(int capacity, cudaStream_t stream)
{
  // Allocate device memory for the DeviceResultPool structure.
  // In each pool, only two pointers pointed space is on the device
  //    DeviceResultPool* d_pool = nullptr;
  //    CUDA_CHECK(cudaMalloc((void**)&d_pool, sizeof(DeviceResultPool)));
  DeviceResultPool* h_pool = new DeviceResultPool;

  // Allocate device memory for the arrays.
  uint32_t* d_ids    = nullptr;
  float* d_distances = nullptr;
  RAFT_CUDA_TRY(cudaMallocAsync((void**)&d_ids, (capacity + 1) * sizeof(uint32_t), stream));
  RAFT_CUDA_TRY(cudaMallocAsync((void**)&d_distances, (capacity + 1) * sizeof(float), stream));

  // Initialize a host instance of the pool.
  h_pool->ids       = d_ids;
  h_pool->distances = d_distances;
  h_pool->capacity  = capacity;
  h_pool->size      = 0;
  // Optionally, you might initialize the distances to max float.
  //    h_pool.distances = (float*)malloc(sizeof(float) * capacity);
  //    for (int i = 0; i < capacity; i++) {
  //        h_pool.distances[i] = std::numeric_limits<float>::max();
  //    }

  // Copy the initialized structure to device memory.
  //    CUDA_CHECK(cudaMemcpy(d_pool, &h_pool, sizeof(DeviceResultPool), cudaMemcpyHostToDevice));
  // instead
  return h_pool;
}

void copy_results_from_pool(const DeviceResultPool* d_pool, uint32_t* host_results)
{
  // First, copy the DeviceResultPool struct from device to host.
  DeviceResultPool h_pool;
  RAFT_CUDA_TRY(cudaMemcpy(&h_pool, d_pool, sizeof(DeviceResultPool), cudaMemcpyDeviceToHost));

  // Now, h_pool.size tells us how many candidates were inserted.
  size_t num_candidates = h_pool.size;
  std::cout << "Number of candidates in pool: " << num_candidates << std::endl;

  // Copy the candidate IDs array from device (h_pool.ids) to host.
  RAFT_CUDA_TRY(cudaMemcpy(
    host_results, h_pool.ids, num_candidates * sizeof(uint32_t), cudaMemcpyDeviceToHost));
}

}  // namespace cuvs::neighbors::ivf_rabitq::detail
