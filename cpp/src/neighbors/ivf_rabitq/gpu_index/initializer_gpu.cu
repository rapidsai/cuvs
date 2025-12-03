/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Created by Stardust on 3/3/25.
//

#include "initializer_gpu.cuh"

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

// Kernel for normal layout (row-major)
// Each thread computes the full distance for one GetCentroid.
__global__ void ComputeDistancesKernelNormal(const float* __restrict__ centroids,
                                             const float* __restrict__ d_query,
                                             float* d_distances,
                                             size_t K,
                                             size_t D)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < K) {
    float sum = 0.0f;
    // For normal layout, getCentroidbyId i is stored contiguously.
    for (int d = 0; d < D; d++) {
      float diff = d_query[d] - centroids[i * D + d];
      sum += diff * diff;
    }
    d_distances[i] = sum;
    //        printf("Centroid %d, distances %f\n", i, d_distances[i]);
  }
}

// No vectorized but shared memory version
__global__ void ComputeDistancesKernelOptimized(const float* __restrict__ centroids,
                                                const float* __restrict__ d_query,
                                                float* d_distances,
                                                size_t K,
                                                size_t D)
{
  extern __shared__ float shared_data[];
  float* shared_centroids = shared_data;
  float* shared_query     = shared_data + blockDim.x * D;

  int tid = threadIdx.x;
  int i   = blockIdx.x * blockDim.x + threadIdx.x;

  // Cooperatively load query into shared memory (broadcast optimization)
  for (int d = tid; d < D; d += blockDim.x) {
    shared_query[d] = d_query[d];
  }

  __syncthreads();

  if (i < K) {
    float sum = 0.0f;

    // Process centroids in chunks to fit in shared memory
    const int CHUNK_SIZE = blockDim.x;

    for (int chunk_start = 0; chunk_start < D; chunk_start += CHUNK_SIZE) {
      int chunk_end  = min(chunk_start + CHUNK_SIZE, (int)D);
      int chunk_size = chunk_end - chunk_start;

      // Cooperatively load centroid data with coalesced access
      for (int d_offset = tid; d_offset < chunk_size * blockDim.x; d_offset += blockDim.x) {
        int local_d            = d_offset % chunk_size;
        int local_thread       = d_offset / chunk_size;
        int global_centroid_id = blockIdx.x * blockDim.x + local_thread;
        int global_d           = chunk_start + local_d;

        if (global_centroid_id < K && global_d < D) {
          shared_centroids[local_thread * chunk_size + local_d] =
            centroids[global_centroid_id * D + global_d];
        }
      }

      __syncthreads();

      // Compute partial distance for this chunk
      if (i < K) {
        for (int d = 0; d < chunk_size; d++) {
          float diff = shared_query[chunk_start + d] - shared_centroids[tid * chunk_size + d];
          sum += diff * diff;
        }
      }

      __syncthreads();
    }

    d_distances[i] = sum;
  }
}

// D must be divisible by 4
__global__ void ComputeDistancesKernelVectorized(const float4* __restrict__ centroids,
                                                 const float4* __restrict__ d_query,
                                                 float* d_distances,
                                                 size_t K,
                                                 size_t D_vec)
{  // D_vec = D/4
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < K) {
    float sum = 0.0f;

    for (int d = 0; d < D_vec; d++) {
      float4 q = d_query[d];
      float4 c = centroids[i * D_vec + d];

      float4 diff = make_float4(q.x - c.x, q.y - c.y, q.z - c.z, q.w - c.w);
      sum += diff.x * diff.x + diff.y * diff.y + diff.z * diff.z + diff.w * diff.w;
    }

    d_distances[i] = sum;
  }
}

// Vectorized warp-cooperative version for better memory throughput
__global__ void ComputeDistancesKernelWarpCoopVec(const float4* __restrict__ centroids,
                                                  const float4* __restrict__ d_query,
                                                  float* d_distances,
                                                  size_t K,
                                                  size_t D_vec)
{  // D_vec = D/4
  int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  int lane_id = threadIdx.x % 32;

  if (warp_id < K) {
    float sum = 0.0f;

    // Each thread in the warp processes D_vec/32 float4 vectors
    for (int d = lane_id; d < D_vec; d += 32) {
      float4 q = d_query[d];
      float4 c = centroids[warp_id * D_vec + d];

      float4 diff = make_float4(q.x - c.x, q.y - c.y, q.z - c.z, q.w - c.w);
      sum += diff.x * diff.x + diff.y * diff.y + diff.z * diff.z + diff.w * diff.w;
    }

    // Warp-level reduction
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Only lane 0 writes the result
    if (lane_id == 0) { d_distances[warp_id] = sum; }
  }
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
// Kernel for transposed layout (column-major)
// Each block is assigned one getCentroidbyId; threads compute partial sums over dimensions.
//__global__ void ComputeDistancesKernelTranspose(const float* __restrict__ centroids,
//                                                const float* __restrict__ d_query,
//                                                float* d_distances,
//                                                size_t K,
//                                                size_t D) {
//    int centroid_id = blockIdx.x;
//    int tid = threadIdx.x;
//    extern __shared__ float sdata[];
//    float partial = 0.0f;
//    // For transposed layout, the d-th component of getCentroidbyId (blockIdx.x) is at
//    centroids[d*K + centroid_id]. for (int d = tid; d < D; d += blockDim.x) {
//        float diff = d_query[d] - centroids[d * K + centroid_id];
//        partial += diff * diff;
//    }
//    sdata[tid] = partial;
//    __syncthreads();
//    // Parallel reduction in shared memory.
//    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
//        if (tid < s)
//            sdata[tid] += sdata[tid + s];
//        __syncthreads();
//    }
//    if (tid == 0)
//        d_distances[centroid_id] = sdata[0];
//}

// Kernel 1: Compute partial distances per dimension.
// For each dimension d and each centroid i, compute:
//    partial = (query[d] - centroids[d*K + i])^2
// and store it at d_partial[d*K + i].
__global__ void computePartialDistancesKernel(const float* __restrict__ centroids,
                                              const float* __restrict__ d_query,
                                              float* d_partial,
                                              size_t K,
                                              size_t D)
{
  // 2D grid: blockIdx.y indexes the dimension.
  int centroid_idx = blockIdx.x * blockDim.x + threadIdx.x;  // centroid index
  int d            = blockIdx.y;  // dimension index (each block row handles one dimension)
  if (centroid_idx < K && d < D) {
    float diff                      = d_query[d] - centroids[d * K + centroid_idx];
    d_partial[d * K + centroid_idx] = diff * diff;
  }
}

// Kernel 2: For each centroid, reduce over dimensions to compute the full distance.
__global__ void reduceDistancesKernel(const float* d_partial,
                                      float* d_distances,
                                      size_t K,
                                      size_t D)
{
  int centroid_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (centroid_idx < K) {
    float sum = 0.0f;
    // Loop over all dimensions.
    for (int d = 0; d < D; d++) {
      sum += d_partial[d * K + centroid_idx];
    }
    d_distances[centroid_idx] = sum;
  }
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
  int blockSize = 256;
  int gridSize  = (K + blockSize - 1) / blockSize;
  ////    size_t sharedMemSize = blockSize * D * sizeof(float) + D * sizeof(float);
  //    ComputeDistancesKernelVectorized<<<gridSize, blockSize>>>(
  //            reinterpret_cast<const float4*>(centroids_.data_handle()),
  //            reinterpret_cast<const float4*>(d_query),
  //            d_distances, K, D/4);
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
  //    ComputeDistancesKernelNormal<<<gridSize, blockSize>>>(centroids_.data_handle(), d_query,
  //    d_distances, K, D); ComputeDistancesKernelOptimized<<<gridSize, blockSize, sharedMemSize>>>(
  //            centroids_.data_handle(), d_query, d_distances, K, D);

  // Allocate device memory for candidate IDs.
  PID* d_candidate_ids = nullptr;
  RAFT_CUDA_TRY(cudaMallocAsync((void**)&d_candidate_ids, sizeof(PID) * K, stream_));
  //    auto policy = thrust::cuda::par.on(stream);
  //    {
  //        // Initialize candidate IDs as a sequence 0,1,...,K-1 using Thrust.
  //        thrust::device_ptr<PID> dev_ptr = thrust::device_pointer_cast(d_candidate_ids);
  //        thrust::sequence(policy, dev_ptr, dev_ptr + K);
  //    }
  //    {
  //        // Sort distances and candidate IDs by distance using Thrust.
  //        thrust::device_ptr<float> dist_ptr = thrust::device_pointer_cast(d_distances);
  //        thrust::device_ptr<PID> id_ptr = thrust::device_pointer_cast(d_candidate_ids);
  //        thrust::sort_by_key(policy, dist_ptr, dist_ptr + K, id_ptr);
  //    }

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
  //    RAFT_CUDA_TRY(cudaGetLastError());
  //    RAFT_CUDA_TRY(cudaDeviceSynchronize());

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

__host__ __device__ float* FlatInitializerGPU::GetCentroidTranspose(PID id) const
{
  return const_cast<float*>(centroids_.data_handle()) +
         id;  // Note: access is via: Centroids[d*K + id]
}

// Copy centroids from host to device.
// The input 'cent' is assumed to be in row-major order (each getCentroidbyId stored contiguously).
// We transpose the data into column-major order before copying to device.
void FlatInitializerGPU::AddVectorsTranspose(const float* cent)
{
  // Allocate a temporary host buffer for the transposed data.
  auto host_buf = raft::make_host_vector<float>(K * D);
  // Transpose: for each getCentroidbyId i and dimension d,
  // place cent[i*D + d] into hostTransposed[d*K + i].
  for (size_t i = 0; i < K; i++) {
    for (size_t d = 0; d < D; d++) {
      host_buf(d * K + i) = cent[i * D + d];
    }
  }
  // Copy the transposed data to device memory.
  raft::copy(centroids_.data_handle(), host_buf.data_handle(), data_elements(), stream_);
  raft::resource::sync_stream(handle_);
}

void FlatInitializerGPU::ComputeCentroidsDistancesTranspose(const float* query,
                                                            size_t nprobe,
                                                            Candidate* candidates,
                                                            size_t num_candidates) const
{
  // Allocate device memory for the query vector.
  float* d_query = nullptr;
  RAFT_CUDA_TRY(cudaMallocAsync((void**)&d_query, sizeof(float) * D, stream_));
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(d_query, query, sizeof(float) * D, cudaMemcpyHostToDevice, stream_));

  // Allocate device memory for final distances (one per centroid).
  float* d_distances = nullptr;
  RAFT_CUDA_TRY(cudaMallocAsync((void**)&d_distances, sizeof(float) * K, stream_));

  // Allocate a temporary device array for partial results:
  // one float for each dimension for each centroid.
  float* d_partial = nullptr;
  RAFT_CUDA_TRY(cudaMallocAsync((void**)&d_partial, sizeof(float) * K * D, stream_));

  // Launch Kernel 1: Compute partial distances.
  // We'll use a 2D grid where:
  //  - gridDim.x = ceil(K / blockSize1) (for centroids)
  //  - gridDim.y = D (each block row corresponds to one dimension)
  int blockSize1 = 512;
  int gridSizeX  = (K + blockSize1 - 1) / blockSize1;
  dim3 gridDim1(gridSizeX, D);
  computePartialDistancesKernel<<<gridDim1, blockSize1, 0, stream_>>>(
    centroids_.data_handle(), d_query, d_partial, K, D);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // Launch Kernel 2: Reduce over dimensions for each centroid.
  int blockSize2 = 512;
  int gridSize2  = (K + blockSize2 - 1) / blockSize2;
  reduceDistancesKernel<<<gridSize2, blockSize2, 0, stream_>>>(d_partial, d_distances, K, D);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // Now, as before, sort the distances (with candidate IDs) using Thrust.
  PID* d_candidate_ids = nullptr;
  RAFT_CUDA_TRY(cudaMallocAsync((void**)&d_candidate_ids, sizeof(PID) * K, stream_));
  {
    thrust::device_ptr<PID> dev_ptr = thrust::device_pointer_cast(d_candidate_ids);
    thrust::sequence(thrust::cuda::par.on(stream_), dev_ptr, dev_ptr + K);
  }
  {
    thrust::device_ptr<float> dist_ptr = thrust::device_pointer_cast(d_distances);
    thrust::device_ptr<PID> id_ptr     = thrust::device_pointer_cast(d_candidate_ids);
    thrust::sort_by_key(thrust::cuda::par.on(stream_), dist_ptr, dist_ptr + K, id_ptr);
  }
  // Use the same kernel to fill the output Candidate array.
  int blockSize3 = 512;
  int gridSize3  = (nprobe + blockSize3 - 1) / blockSize3;
  FillCandidatesKernel<<<gridSize3, blockSize3, 0, stream_>>>(
    d_distances, d_candidate_ids, candidates, nprobe);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // Free temporary device memory.
  RAFT_CUDA_TRY(cudaFreeAsync(d_query, stream_));
  RAFT_CUDA_TRY(cudaFreeAsync(d_distances, stream_));
  RAFT_CUDA_TRY(cudaFreeAsync(d_partial, stream_));
  RAFT_CUDA_TRY(cudaFreeAsync(d_candidate_ids, stream_));

  raft::resource::sync_stream(handle_);
}

// LoadCentroids centroids from file.
// Since the file stores centroids in row-major order, this function transpose it to column-major,
// and then copy into device memory.
void FlatInitializerGPU::LoadCentroidsTranspose(std::ifstream& input, const char* filename)
{
  // Allocate temporary host buffers.
  auto host_buf_row_major       = raft::make_host_vector<float, int64_t>(K * D);
  auto host_buf_transposed      = raft::make_host_vector<float, int64_t>(K * D);
  auto* hostCentroidsRowMajor   = new float[K * D];
  auto* hostCentroidsTransposed = new float[K * D];

  // Read row-major data from file.
  input.read(reinterpret_cast<char*>(host_buf_row_major.data_handle()), data_bytes());

  // Transpose from row-major to column-major.
  for (size_t i = 0; i < K; i++) {
    for (size_t d = 0; d < D; d++) {
      host_buf_transposed(d * K + i) = host_buf_row_major(i * D + d);
    }
  }
  // Copy transposed data to device memory.
  raft::copy(centroids_.data_handle(), host_buf_transposed.data_handle(), data_elements(), stream_);
  raft::resource::sync_stream(handle_);
}

// SaveCentroids centroids to file. First copy them to host memory, transpose to row-major order,
// then write.
void FlatInitializerGPU::SaveCentroidsTranspose(std::ofstream& output, const char* filename) const
{
  // Allocate temporary host buffers.
  auto host_buf_row_major  = raft::make_host_vector<float, int64_t>(K * D);
  auto host_buf_transposed = raft::make_host_vector<float, int64_t>(K * D);

  // Copy centroids from device (transposed) to host.
  raft::copy(host_buf_transposed.data_handle(), centroids_.data_handle(), data_elements(), stream_);
  raft::resource::sync_stream(handle_);

  // Transpose from column-major to row-major.
  for (size_t i = 0; i < K; i++) {
    for (size_t d = 0; d < D; d++) {
      host_buf_row_major(i * D + d) = host_buf_transposed(d * K + i);
    }
  }
  // Write row-major data to file.
  output.write(reinterpret_cast<char*>(host_buf_row_major.data_handle()), data_bytes());
}

}  // namespace cuvs::neighbors::ivf_rabitq::detail
