/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Created by Stardust on 3/3/25.
//

#include "initializer_gpu.cuh"

#include <raft/core/host_mdarray.hpp>

#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/version.h>

namespace cuvs::neighbors::ivf_rabitq::detail {

// A simple L2-squared function (provided for compatibility).
__host__ __device__ float L2SqrGPU(const float* a, const float* b, size_t d)
{
  float dist = 0.0f;
  for (size_t i = 0; i < d; ++i) {
    float diff = a[i] - b[i];
    dist += diff * diff;
  }
  return dist;
}

// D must still be divisible by 4 ⇒ D_vec = D/4
// Each warp handles ONE centroid
__global__ void ComputeDistancesKernelWarp(const float4* __restrict__ centroids,
                                           const float4* __restrict__ d_query,
                                           float* __restrict__ d_distances,
                                           size_t K,
                                           size_t D_vec)  // #float4’s per vector
{
  constexpr unsigned FULL_MASK = 0xffffffff;
  const int globalTid          = blockIdx.x * blockDim.x + threadIdx.x;
  const int lane               = globalTid & 31;  // 0 … 31 inside the warp
  const int warpId             = globalTid >> 5;  // one warp ⇒ one centroid

  if (warpId >= K) return;  // guard

  // ------------------------------------------------------------------
  // 1. Each lane accumulates a strided slice of the dimension space
  // ------------------------------------------------------------------
  float sum = 0.0f;
  for (int d = lane; d < D_vec; d += warpSize)  // stride by 32
  {
    const float4 q = d_query[d];  // same for the whole warp
    const float4 c = centroids[warpId * D_vec + d];

    const float4 diff = make_float4(q.x - c.x, q.y - c.y, q.z - c.z, q.w - c.w);

    sum += diff.x * diff.x + diff.y * diff.y + diff.z * diff.z + diff.w * diff.w;
  }

  // ------------------------------------------------------------------
  // 2. Warp-wide reduction (lane 0 ends up with the full distance)
  // ------------------------------------------------------------------
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    sum += __shfl_down_sync(FULL_MASK, sum, offset);

  // ------------------------------------------------------------------
  // 3. Lane 0 writes the result
  // ------------------------------------------------------------------
  if (lane == 0) d_distances[warpId] = sum;
}

// Kernel to fill the output Candidate array from sorted distances and IDs.
__global__ void FillCandidatesKernel(const float* d_distances,
                                     const PID* d_candidate_ids,
                                     Candidate* output,
                                     size_t nprobe)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < nprobe) {
    output[idx].id       = d_candidate_ids[idx];
    output[idx].distance = d_distances[idx];
  }
}

FlatInitializerGPU::FlatInitializerGPU(raft::resources const& handle, size_t d, size_t k)
  : InitializerGPU(handle, d, k),
    centroids_(raft::make_device_matrix<float, int64_t, raft::row_major>(handle_, K, D))
{
  dist_func = L2SqrGPU;
  raft::resource::sync_stream(handle_);
}

__host__ __device__ float* FlatInitializerGPU::GetCentroid(PID id) const
{
  return const_cast<float*>(centroids_.data_handle()) + id * D;
}

void FlatInitializerGPU::AddVectors(const float* cent)
{
  raft::copy(centroids_.data_handle(), cent, data_elements(), stream_);
}

// Kernel to initialize sequence
__global__ void init_sequence_kernel(PID* candidate_ids, int K)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < K) { candidate_ids[idx] = static_cast<PID>(idx); }
}

void FlatInitializerGPU::ComputeCentroidsDistances(const float* query,
                                                   size_t nprobe,
                                                   Candidate* candidates,
                                                   size_t num_candidates) const
{
  // Allocate device memory for the query vector.
  float* d_query = nullptr;
  RAFT_CUDA_TRY(cudaMallocAsync((void**)&d_query, sizeof(float) * D, stream_));
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(d_query, query, sizeof(float) * D, cudaMemcpyHostToDevice, stream_));

  // Allocate device memory for distances (one per centroid).
  float* d_distances = nullptr;
  RAFT_CUDA_TRY(cudaMallocAsync((void**)&d_distances, sizeof(float) * K, stream_));
  // Launch kernel: each thread computes distance for one centroid.
  int warpSize              = 32;
  const int WARPS_PER_BLOCK = 8;  // 4 × 32 = 128 threads
  dim3 block(WARPS_PER_BLOCK * warpSize);
  dim3 grid((K + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
  ComputeDistancesKernelWarp<<<grid, block, 0, stream_>>>(
    reinterpret_cast<const float4*>(centroids_.data_handle()),
    reinterpret_cast<const float4*>(d_query),
    d_distances,
    K,
    D / 4);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // Allocate device memory for candidate IDs.
  PID* d_candidate_ids = nullptr;
  RAFT_CUDA_TRY(cudaMallocAsync((void**)&d_candidate_ids, sizeof(PID) * K, stream_));

  // Another option instead of using thrust:
  // Replace the first block with:
  {
    // Initialize candidate IDs as a sequence 0,1,...,K-1 using a custom kernel
    int block_size = 256;
    int grid_size  = (K + block_size - 1) / block_size;
    init_sequence_kernel<<<grid_size, block_size, 0, stream_>>>(d_candidate_ids, K);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    raft::resource::sync_stream(handle_);  // Wait for kernel completion
  }

  // Replace the second block with:
  {
    // Sort distances and candidate IDs by distance using CUB

    // Determine temporary device storage requirements
    void* d_temp_storage      = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp_storage,
                                    temp_storage_bytes,
                                    d_distances,
                                    d_distances,  // keys (in-place sort)
                                    d_candidate_ids,
                                    d_candidate_ids,  // values (in-place sort)
                                    K,
                                    0,
                                    32,
                                    stream_);

    // Allocate temporary storage
    RAFT_CUDA_TRY(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream_));

    // Run sorting operation
    cub::DeviceRadixSort::SortPairs(d_temp_storage,
                                    temp_storage_bytes,
                                    d_distances,
                                    d_distances,  // keys (in-place sort)
                                    d_candidate_ids,
                                    d_candidate_ids,  // values (in-place sort)
                                    K,
                                    0,
                                    32,
                                    stream_);

    // Clean up temporary storage
    RAFT_CUDA_TRY(cudaFreeAsync(d_temp_storage, stream_));
  }

  // Fill the output Candidate array with the top nprobe candidates.
  int blockSize2 = 512;
  int gridSize2  = (nprobe + blockSize2 - 1) / blockSize2;
  FillCandidatesKernel<<<gridSize2, blockSize2, 0, stream_>>>(
    d_distances, d_candidate_ids, candidates, nprobe);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // Free temporary device memory.
  RAFT_CUDA_TRY(cudaFreeAsync(d_query, stream_));
  RAFT_CUDA_TRY(cudaFreeAsync(d_distances, stream_));
  RAFT_CUDA_TRY(cudaFreeAsync(d_candidate_ids, stream_));
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
