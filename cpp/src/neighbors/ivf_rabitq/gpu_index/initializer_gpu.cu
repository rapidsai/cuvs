/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Created by Stardust on 3/3/25.
//

#include "initializer_gpu.cuh"

#include <raft/core/host_mdarray.hpp>

namespace cuvs::neighbors::ivf_rabitq::detail {

FlatInitializerGPU::FlatInitializerGPU(raft::resources const& handle, size_t d, size_t k)
  : InitializerGPU(handle, d, k),
    centroids_(raft::make_device_matrix<float, int64_t, raft::row_major>(handle_, K, D))
{
}

__host__ __device__ float* FlatInitializerGPU::GetCentroid(PID id) const
{
  return const_cast<float*>(centroids_.data_handle()) + id * D;
}

void FlatInitializerGPU::AddVectors(const float* cent)
{
  raft::copy(centroids_.data_handle(), cent, data_elements(), stream_);
}

// Loads the centroids from a file.
// The centroids are read into a temporary host buffer and then copied to the
// device memory.
void FlatInitializerGPU::LoadCentroids(std::ifstream& input, const char* filename)
{
  // Allocate temporary host buffer for centroids.
  auto host_buf = raft::make_host_vector<float, int64_t>(K * D);
  // Read the raw data from the file.
  input.read(reinterpret_cast<char*>(host_buf.data_handle()), data_bytes());
  // Copy the data from host to device.
  raft::copy(centroids_.data_handle(), host_buf.data_handle(), data_elements(), stream_);
  raft::resource::sync_stream(handle_);
}

// Saves the centroids to a file.
// The centroids are first copied from device memory to a temporary host buffer,
// and then the entire block of data is written to the output stream.
void FlatInitializerGPU::SaveCentroids(std::ofstream& output, const char* filename) const
{
  // Allocate temporary host buffer for centroids.
  auto host_buf = raft::make_host_vector<float, int64_t>(K * D);
  // Copy centroids from device to host.
  raft::copy(host_buf.data_handle(), centroids_.data_handle(), data_elements(), stream_);
  raft::resource::sync_stream(handle_);
  // Write the raw data to the file.
  output.write(reinterpret_cast<char*>(host_buf.data_handle()), data_bytes());
}

}  // namespace cuvs::neighbors::ivf_rabitq::detail
