/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Created by Stardust on 4/14/25.
//

#include <cuvs/neighbors/ivf_rabitq/gpu_index/pool_gpu.cuh>

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cuda_rt_essentials.hpp>

namespace cuvs::neighbors::ivf_rabitq::detail {

// Host function to create and initialize a DeviceResultPool.
DeviceResultPool createDeviceResultPool(raft::resources const& handle, const int capacity)
{
  // Allocate device memory for the DeviceResultPool structure.
  // In each pool, only two vectors are on the device
  return DeviceResultPool{raft::make_device_vector<uint32_t, int64_t>(handle, capacity + 1),
                          raft::make_device_vector<float, int64_t>(handle, capacity + 1),
                          capacity,
                          0};
}

void copy_results_from_pool(raft::resources const& handle,
                            const DeviceResultPool* d_pool,
                            uint32_t* host_results)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  // First, copy the DeviceResultPool struct from device to host.
  DeviceResultPool h_pool;
  raft::copy(&h_pool, d_pool, 1, stream);

  // Now, h_pool.size tells us how many candidates were inserted.
  size_t num_candidates = h_pool.size;
  std::cout << "Number of candidates in pool: " << num_candidates << std::endl;

  // Copy the candidate IDs array from device (h_pool.ids) to host.
  raft::copy(host_results, h_pool.ids.data_handle(), num_candidates, stream);
  raft::resource::sync_stream(handle);
}

}  // namespace cuvs::neighbors::ivf_rabitq::detail
