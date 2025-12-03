/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Created by Stardust on 3/10/25.
//

#include "quantizer_gpu.cuh"

#include <raft/core/resource/cuda_stream.hpp>

#include <atomic>
#include <thread>

namespace cuvs::neighbors::ivf_rabitq::detail {

#define MAX_D 2048

////---------------------------------------------------------------------------
//// Kernel: pack_binary_kernel
////
//// For each data point (row) and for each block of 64 bits,
//// this kernel packs 64 0/1 int values into one uint64_t.
//// Input:
////    d_bin_XP: pointer to an int array of size (num_points x D)
////              (stored in row-major order)
//// Output:
////    d_binary: pointer to an array of uint64_t values of length num_points * (D/64)
////---------------------------------------------------------------------------
//__global__ void pack_binary_kernel(const int* __restrict__ d_bin_XP,
//                                   uint64_t* d_binary,
//                                   size_t num_points,
//                                   size_t D) {
//    // Each block is responsible for one data point (row).
//    size_t row = blockIdx.x;
//    // Each thread in the block processes one block of 64 bits.
//    size_t block_id = threadIdx.x; // range: 0 to (D/64 - 1)
//    if (row < num_points && block_id < D / 64) {
//        uint64_t cur = 0;
//        // Process 64 bits.
//        for (int i = 0; i < 64; i++) {
//            // Read bit from row 'row' at column (block_id*64 + i)
//            int bit = d_bin_XP[row * D + block_id * 64 + i];
//            cur |= ((uint64_t)bit << (63 - i));
//        }
//        // Write the packed uint64_t value.
//        d_binary[row * (D / 64) + block_id] = cur;
//    }
//}

//---------------------------------------------------------------------------
// Kernel: pack_binary_kernel
//
// For each data point (row) and for each block of 32 bits,
// this kernel packs 32 0/1 int values into one uint32_t.
// Input:
//    d_bin_XP: pointer to an int array of size (num_points x D)
//              (stored in row-major order)
// Output:
//    d_binary: pointer to an array of uint32_t values of length num_points * (D/32)
__global__ void pack_binary_kernel(const int* __restrict__ d_bin_XP,
                                   uint32_t* d_binary,
                                   size_t num_points,
                                   size_t D)
{
  // Each block is responsible for one data point (row).
  size_t row = blockIdx.x;
  // Each thread processes one block of 32 bits.
  size_t block_id = threadIdx.x;  // range: 0 to (D/32 - 1)
  if (row < num_points && block_id < D / 32) {
    uint32_t cur = 0;
    // Process 32 bits.
    for (int i = 0; i < 32; i++) {
      int bit = d_bin_XP[row * D + block_id * 32 + i];
      cur |= ((uint32_t)bit << (31 - i));
    }
    d_binary[row * (D / 32) + block_id] = cur;
  }
}

//---------------------------------------------------------------------------
// Kernel: pack_codes_kernel
//
// A simplified version that converts each uint64_t element from the
// binary code into 8 uint8_t values (big-endian order). In the original
// CPU code, pack_codes performs more sophisticated reordering, but here
// we illustrate the overall idea.
// Input:
//    B: The original dimension (D)
//    d_binary: pointer to an array of uint64_t elements
//    ncode: number of data points (rows)
// Output:
//    d_packed_code: pointer to output uint8_t array; here, each uint64_t yields 8 bytes.
//---------------------------------------------------------------------------

__global__ void pack_codes_kernel(size_t B,
                                  const uint64_t* d_binary,
                                  size_t ncode,
                                  uint8_t* d_packed_code)
{
  // Total number of uint64_t elements.
  size_t total = ncode * (B / 64);
  size_t idx   = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total) {
    uint64_t val = d_binary[idx];
    // Write the 8 bytes in big-endian order.
    for (int i = 0; i < 8; i++) {
      d_packed_code[idx * 8 + i] = (uint8_t)(val >> (56 - 8 * i));
    }
  }
}

//---------------------------------------------------------------------------
// GPU version of rabitq_codes.
// This function converts a binary matrix (d_bin_XP) stored as ints (0/1)
// into a packed code stored in d_packed_code. The process is:
// 1. Launch pack_binary_kernel to convert each row into an array of uint64_t.
// 2. Launch pack_codes_kernel to convert the uint64_t array into packed uint8_t codes (now
// disabled).
//---------------------------------------------------------------------------
void DataQuantizerGPU::rabitq_codes(const int* d_bin_XP,
                                    uint32_t* d_packed_code,
                                    size_t num_points) const
{
  // Number of uint32_t values per data point.
  size_t blocks_per_point = D / 32;
  size_t total_uint64     = num_points * blocks_per_point;

  // Allocate device memory for the intermediate binary representation.
  //    uint64_t* d_binary;
  //    RAFT_CUDA_TRY(cudaMalloc((void**)&d_binary, total_uint64 * sizeof(uint64_t)));

  // Launch kernel: one block per data point, each with (D/64) threads.
  dim3 grid(num_points);
  dim3 block(blocks_per_point);
  pack_binary_kernel<<<grid, block, 0, stream_>>>(d_bin_XP, d_packed_code, num_points, D);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // Launch kernel to pack the binary codes into 8-bit packed codes.
  //    // Here, we assume a simple scheme where each uint64_t is converted to 8 bytes.
  int threads  = 256;
  int gridSize = (total_uint64 + threads - 1) / threads;
  //    pack_codes_kernel<<<gridSize, threads>>>(D, d_binary, num_points, d_packed_code);
  // RAFT_CUDA_TRY(cudaGetLastError());
  // RAFT_CUDA_TRY(cudaDeviceSynchronize());

  // Free the intermediate binary array.
  //    cudaFree(d_binary);
  raft::resource::sync_stream(handle_);
}

//---------------------------------------------------------------------------
// Kernel: gatherKernel
// For each data point in the cluster (indexed by i), copy the first DIM
// elements from the corresponding row in d_data (located at d_IDs[i]*DIM)
// into row i of d_X_pad (which has D columns), padding with zeros from DIM to D.
__global__ void gatherKernel(const float* __restrict__ d_data,
                             const PID* __restrict__ d_IDs,
                             float* d_X_pad,
                             size_t num_points,
                             size_t DIM,
                             size_t D)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_points) {
    // Source row: starting at d_data[d_IDs[i] * DIM]
    const float* src = d_data + d_IDs[i] * DIM;
    // Destination row: d_X_pad[i * D]
    float* dst = d_X_pad + i * D;
    // Copy available elements.
    for (int j = 0; j < DIM; j++) {
      dst[j] = src[j];
    }
    // Zero-pad remaining elements.
    for (int j = DIM; j < D; j++) {
      dst[j] = 0.0f;
    }
  }
}

//---------------------------------------------------------------------------
// Kernel: copyCentroidKernel
// Copy the centroid (of length DIM) into the padded row vector d_C_pad (length D).
__global__ void copyCentroidKernel(const float* __restrict__ d_centroid,
                                   float* d_C_pad,
                                   size_t DIM,
                                   size_t D)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < D) {
    if (j < DIM)
      d_C_pad[j] = d_centroid[j];
    else
      d_C_pad[j] = 0.0f;
  }
}

//---------------------------------------------------------------------------
// Kernel: subtractKernel
// For each element in the matrix d_XP (num_points x D), subtract the corresponding
// element from d_CP (a 1 x D vector). d_XP and d_CP are stored in row-major order.
__global__ void subtractKernel(float* d_XP,
                               const float* __restrict__ d_CP,
                               size_t num_points,
                               size_t D)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_points * D) {
    int col   = idx % D;
    d_XP[idx] = d_XP[idx] - d_CP[col];
  }
}

//---------------------------------------------------------------------------
// Kernel: normalizeKernel
// For each row in d_XP (num_points x D), compute its L2 norm and divide each element,
// storing the result in d_XP_norm.
__global__ void normalizeKernel(const float* __restrict__ d_XP,
                                float* d_XP_norm,
                                size_t num_points,
                                size_t D)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_points) {
    float norm = 0.0f;
    // Compute L2 norm.
    for (int j = 0; j < D; j++) {
      float val = d_XP[i * D + j];
      norm += val * val;
    }
    norm = sqrtf(norm);
    // Normalize the row.
    for (int j = 0; j < D; j++) {
      d_XP_norm[i * D + j] = (norm > 0) ? d_XP[i * D + j] / norm : 0.0f;
    }
  }
}

//---------------------------------------------------------------------------
// Kernel: binarizeKernel
// For each element in d_XP (num_points x D), output 1 if the value > 0, else 0,
// storing the result in d_bin_XP.
__global__ void binarizeKernel(const float* __restrict__ d_XP,
                               int* d_bin_XP,
                               size_t num_points,
                               size_t D)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_points * D) { d_bin_XP[idx] = (d_XP[idx] > 0.0f) ? 1 : 0; }
}

//__global__ void gatherKernel(const float* __restrict__ d_data,
//                             const PID*   __restrict__ d_IDs,
//                             float*       __restrict__ d_X_pad,
//                             size_t N, uint32_t DIM, uint32_t D) {
//    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
//    if (idx >= N) return;
//    const float* src = d_data + static_cast<size_t>(d_IDs[idx]) * DIM;
//    float* dst = d_X_pad + idx * D;
//    // copy DIM floats (could be optimized via vectorized loads)
//    for (uint32_t k = 0; k < DIM; ++k) dst[k] = src[k];
//    // pad zeros automatically thanks to cudaMalloc initial value? we must pad manually
//    for (uint32_t k = DIM; k < D; ++k) dst[k] = 0.0f;
//}
//
//__global__ void copyCentroidKernel(const float* __restrict__ d_centroid,
//                                   float*       __restrict__ d_C_pad,
//                                   uint32_t DIM, uint32_t D) {
//    uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
//    if (k < DIM)  d_C_pad[k] = d_centroid[k];
//    else if (k < D) d_C_pad[k] = 0.f;
//}
//
//__global__ void subtractKernel(float* __restrict__ d_XP,
//                               const float* __restrict__ d_CP,
//                               size_t N, uint32_t D) {
//    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
//    if (idx >= N*D) return;
//    uint32_t col = idx % D;
//    d_XP[idx] -= d_CP[col];
//}
//
//__device__ __forceinline__ float wreduce_sum(float v) {
//    // warp‑level reduction (assumes 32‑thread warp)
//    for (int offset = 16; offset > 0; offset >>= 1)
//        v += __shfl_down_sync(0xffffffff, v, offset);
//    return v;
//}
//
//__global__ void normalizeKernel(const float* __restrict__ d_in,
//                                float*       __restrict__ d_out,
//                                size_t N, uint32_t D) {
//    // one warp handles one vector row for simplicity
//    const uint32_t warpId = (blockIdx.x * blockDim.x + threadIdx.x) >> 5; // /32
//    if (warpId >= N) return;
//    const uint32_t lane = threadIdx.x & 31;
//    const float* src = d_in + warpId * D;
//    float* dst = d_out + warpId * D;
//
//    // compute squared norm
//    float local = 0.f;
//    for (uint32_t k = lane; k < D; k += 32) {
//        float val = src[k];
//        local += val * val;
//    }
//    float norm2 = wreduce_sum(local);
//    float inv_norm = rsqrtf(norm2 + 1e-12f);
//
//    // write normalized values
//    for (uint32_t k = lane; k < D; k += 32) {
//        dst[k] = src[k] * inv_norm;
//    }
//}
//
//__global__ void binarizeKernel(const float* __restrict__ d_in,
//                               int*         __restrict__ d_out,
//                               size_t N, uint32_t D) {
//    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
//    if (idx >= N*D) return;
//    d_out[idx] = d_in[idx] > 0.f;
//}

//---------------------------------------------------------------------------
// DataQuantizerGPU::data_transformation (GPU version)
// This function transforms data for quantization as follows:
// 1. Gather and pad the data points from d_data using d_IDs into d_X_pad.
// 2. Copy and pad the centroid into d_C_pad.
// 3. Rotate data: XP = X_pad * P and CP = C_pad * P via rotator.rotate().
// 4. Subtract CP from every row of XP to get residuals.
// 5. Save the rotated centroid CP into d_rotated_c.
// 6. Normalize XP rowwise to produce XP_norm.
// 7. Binarize XP to produce bin_XP.
void DataQuantizerGPU::data_transformation(const float* d_data,
                                           const float* d_centroid,
                                           const PID* d_IDs,
                                           size_t num_points,
                                           const RotatorGPU& rotator,
                                           float* d_rotated_c,
                                           float* d_XP_norm,
                                           int* d_bin_XP) const
{
  // Allocate temporary matrix X_pad (num_points x D) on GPU.
  float* d_X_pad;
  RAFT_CUDA_TRY(cudaMallocAsync((void**)&d_X_pad, num_points * D * sizeof(float), stream_));

  // Launch kernel to gather and pad data.
  int blockSize = 256;
  int gridSize  = (num_points + blockSize - 1) / blockSize;
  gatherKernel<<<gridSize, blockSize, 0, stream_>>>(d_data, d_IDs, d_X_pad, num_points, DIM, D);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // Allocate temporary matrix C_pad (1 x D) on GPU.
  float* d_C_pad;
  RAFT_CUDA_TRY(cudaMallocAsync((void**)&d_C_pad, D * sizeof(float), stream_));
  int gridSizeC = (D + blockSize - 1) / blockSize;
  copyCentroidKernel<<<gridSizeC, blockSize, 0, stream_>>>(d_centroid, d_C_pad, DIM, D);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // Rotate X_pad -> XP. Allocate XP (num_points x D).
  float* d_XP;
  RAFT_CUDA_TRY(cudaMallocAsync((void**)&d_XP, num_points * D * sizeof(float), stream_));
  rotator.rotate(d_X_pad, d_XP, num_points);

  // Rotate C_pad -> CP. Allocate CP (1 x D).
  float* d_CP;
  RAFT_CUDA_TRY(cudaMallocAsync((void**)&d_CP, D * sizeof(float), stream_));
  rotator.rotate(d_C_pad, d_CP, 1);

  // Subtract CP from each row of XP: XP = XP - CP.
  int totalElements = num_points * D;
  gridSize          = (totalElements + blockSize - 1) / blockSize;
  subtractKernel<<<gridSize, blockSize, 0, stream_>>>(d_XP, d_CP, num_points, D);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // Save the rotated centroid: copy CP into d_rotated_c (assumed size D).
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(d_rotated_c, d_CP, D * sizeof(float), cudaMemcpyDeviceToDevice, stream_));

  // Normalize XP rowwise: compute XP_norm = XP / norm(XP).
  gridSize = (num_points + blockSize - 1) / blockSize;
  normalizeKernel<<<gridSize, blockSize, 0, stream_>>>(d_XP, d_XP_norm, num_points, D);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // Generate binary representation: bin_XP = (XP > 0).
  totalElements = num_points * D;
  gridSize      = (totalElements + blockSize - 1) / blockSize;
  binarizeKernel<<<gridSize, blockSize, 0, stream_>>>(d_XP_norm, d_bin_XP, num_points, D);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // Free temporary buffers.
  RAFT_CUDA_TRY(cudaFreeAsync(d_X_pad, stream_));
  RAFT_CUDA_TRY(cudaFreeAsync(d_C_pad, stream_));
  RAFT_CUDA_TRY(cudaFreeAsync(d_XP, stream_));
  RAFT_CUDA_TRY(cudaFreeAsync(d_CP, stream_));

  raft::resource::sync_stream(handle_);
}

//---------------------------------------------------------------------------
// DataQuantizerGPU::data_transformation (GPU version)
// This function transforms data for quantization as follows:
// 1. Gather and pad the data points from d_data using d_IDs into d_X_pad.
// 2. Copy and pad the centroid into d_C_pad.
// 3. Rotate data: XP = X_pad * P and CP = C_pad * P via rotator.rotate().
// 4. Subtract CP from every row of XP to get residuals.
// 5. Save the rotated centroid CP into d_rotated_c.
// 6. Normalize XP rowwise to produce XP_norm.
// 7. Binarize XP to produce bin_XP.
// slightly modified for batch data;
void DataQuantizerGPU::data_transformation_batch(const float* d_data,
                                                 const float* d_centroid,
                                                 const PID* d_IDs,
                                                 size_t num_points,
                                                 const RotatorGPU& rotator,
                                                 float* d_rotated_c,
                                                 float* d_XP_norm,
                                                 int* d_bin_XP,
                                                 float* d_XP) const
{
  // Allocate temporary matrix X_pad (num_points x D) on GPU.
  float* d_X_pad;
  RAFT_CUDA_TRY(cudaMallocAsync((void**)&d_X_pad, num_points * D * sizeof(float), stream_));

  // Launch kernel to gather and pad data.
  int blockSize = 256;
  int gridSize  = (num_points + blockSize - 1) / blockSize;
  gatherKernel<<<gridSize, blockSize, 0, stream_>>>(d_data, d_IDs, d_X_pad, num_points, DIM, D);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // Allocate temporary matrix C_pad (1 x D) on GPU.
  float* d_C_pad;
  RAFT_CUDA_TRY(cudaMallocAsync((void**)&d_C_pad, D * sizeof(float), stream_));
  int gridSizeC = (D + blockSize - 1) / blockSize;
  copyCentroidKernel<<<gridSizeC, blockSize, 0, stream_>>>(d_centroid, d_C_pad, DIM, D);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // Rotate X_pad -> XP. Allocate XP (num_points x D).
  rotator.rotate(d_X_pad, d_XP, num_points);

  // Rotate C_pad -> CP. Allocate CP (1 x D).
  float* d_CP;
  RAFT_CUDA_TRY(cudaMallocAsync((void**)&d_CP, D * sizeof(float), stream_));
  rotator.rotate(d_C_pad, d_CP, 1);

  // Subtract CP from each row of XP: XP = XP - CP.
  int totalElements = num_points * D;
  gridSize          = (totalElements + blockSize - 1) / blockSize;
  subtractKernel<<<gridSize, blockSize, 0, stream_>>>(d_XP, d_CP, num_points, D);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // Save the rotated centroid: copy CP into d_rotated_c (assumed size D).
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(d_rotated_c, d_CP, D * sizeof(float), cudaMemcpyDeviceToDevice, stream_));

  // Normalize XP rowwise: compute XP_norm = XP / norm(XP).
  gridSize = (num_points + blockSize - 1) / blockSize;
  normalizeKernel<<<gridSize, blockSize, 0, stream_>>>(d_XP, d_XP_norm, num_points, D);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // Generate binary representation: bin_XP = (XP > 0).
  totalElements = num_points * D;
  gridSize      = (totalElements + blockSize - 1) / blockSize;
  binarizeKernel<<<gridSize, blockSize, 0, stream_>>>(d_XP_norm, d_bin_XP, num_points, D);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // Free temporary buffers.
  RAFT_CUDA_TRY(cudaFreeAsync(d_X_pad, stream_));
  RAFT_CUDA_TRY(cudaFreeAsync(d_C_pad, stream_));
  RAFT_CUDA_TRY(cudaFreeAsync(d_CP, stream_));

  raft::resource::sync_stream(handle_);
}

// void DataQuantizerGPU::data_transformation(const float* d_data,
//                          const float* d_centroid,
//                          const PID* d_IDs,
//                          size_t N,
//                          const RotatorGPU& rotator,
//                          float* d_rotated_c,
//                          float* d_XP_norm,
//                          int* d_bin_XP) const {
//     const int block = 256;
//     const int gridN = (N + block - 1)/block;
//     const int gridD = (D + block - 1)/block;
//     const int gridND= (N*D + block - 1)/block;
//
//     // 1) gather
//     float* d_X_pad; RAFT_CUDA_TRY(cudaMalloc(&d_X_pad, N*D*sizeof(float)));
//     gatherKernel<<<gridN, block>>>(d_data, d_IDs, d_X_pad, N, DIM, D);
//     RAFT_CUDA_TRY(cudaGetLastError());
//
//     // 2) centroid pad
//     float* d_C_pad; RAFT_CUDA_TRY(cudaMalloc(&d_C_pad, D*sizeof(float)));
//     copyCentroidKernel<<<gridD, block>>>(d_centroid, d_C_pad, DIM, D);
//
//
//
//
//     // 3) rotate
//     float* d_XP; RAFT_CUDA_TRY(cudaMalloc(&d_XP, N*D*sizeof(float)));
//     rotator.rotate(d_X_pad, d_XP, N);
//     float* d_CP; RAFT_CUDA_TRY(cudaMalloc(&d_CP, D*sizeof(float)));
//     rotator.rotate(d_C_pad, d_CP, 1);
//
//     //debug
////    float* h_C_pad = (float*)malloc(D * sizeof(float));
////    if (h_C_pad == nullptr) {
////        fprintf(stderr, "Host malloc failed!\n");
////        exit(EXIT_FAILURE);
////    }
////    RAFT_CUDA_TRY(cudaMemcpy(h_C_pad, d_CP, D * sizeof(float), cudaMemcpyDeviceToHost));
////    printf("d_CP values on GPU:\n");
////    for (int i = 0; i < D; ++i) {
////        printf("h_CP[%d] = %f\n", i, h_C_pad[i]);
////    }
////    free(h_C_pad);
//
//    // 4) residuals
//    subtractKernel<<<gridND, block>>>(d_XP, d_CP, N, D);
//
//    // 5) save rotated centroid
//    RAFT_CUDA_TRY(cudaMemcpy(d_rotated_c, d_CP, D*sizeof(float), cudaMemcpyDeviceToDevice));
//
//    // 6) normalize
//    int warpsPerBlock = block/32;
//    int gridWarp = (N + warpsPerBlock - 1)/warpsPerBlock;
//    normalizeKernel<<<gridWarp, block>>>(d_XP, d_XP_norm, N, D);
//
//    // 7) binarize
//    binarizeKernel<<<gridND, block>>>(d_XP_norm, d_bin_XP, N, D);
//    RAFT_CUDA_TRY(cudaGetLastError());
//    RAFT_CUDA_TRY(cudaDeviceSynchronize());
//
//    cudaFree(d_X_pad);
//    cudaFree(d_C_pad);
//    cudaFree(d_XP);
//    cudaFree(d_CP);
//}

//---------------------------------------------------------------------------
// Kernel: rabitq_factor_kernel
//
// For each data point (indexed by i), compute the following:
//   - fac_x2[i] = L2Sqr( data[d_IDs[i]*DIM], centroid, DIM ) [and then possibly sqrt]
//   - dist2c = sqrt( fac_x2[i] )
//   - coded_vec[j] = (2 * bin_XP[i * D + j] - 1) * FAC_NORM  for j in [0, D)
//   - X0 = sum_{j=0}^{D-1} (XP_norm[i * D + j] * coded_vec[j])
//   - ip = sum_{j=0}^{D-1} abs(0.5 * XP_norm[i * D + j])
//   - fac_ip[i] = (1/ip) * 2 * dist2c
//   - fac_sumxb[i] = sum_{j=0}^{D-1} bin_XP[i * D + j]
//   - fac_err[i] = sqrt((1 - o_obar^2) / (o_obar^2)) * FAC_ERR * 2 * dist2c
//
// Parameters:
//    d_data      : pointer to original data, with each vector of length DIM
//    d_centroid  : pointer to centroid (length DIM)
//    d_IDs       : pointer to an array of data point IDs (each is an index)
//    d_bin_XP    : pointer to binary representation, as int (size: num_points x D)
//    d_XP_norm   : pointer to normalized rotated data, as float (size: num_points x D)
//    fac_x2, fac_ip, fac_sumxb, fac_err : output arrays (size: num_points)
//    num_points  : number of data points in the cluster
//    DIM         : original dimension
//    D           : padded dimension (multiple of 64)
//    FAC_NORM    : normalization constant
//    FAC_ERR     : error factor constant
//---------------------------------------------------------------------------
__global__ void rabitq_factor_kernel(const float* __restrict__ d_data,
                                     const float* __restrict__ d_centroid,
                                     const PID* __restrict__ d_IDs,
                                     const int* __restrict__ d_bin_XP,
                                     const float* __restrict__ d_XP_norm,
                                     float* fac_x2,
                                     float* fac_ip,
                                     float* fac_sumxb,
                                     float* fac_err,
                                     size_t num_points,
                                     size_t DIM,
                                     size_t D,
                                     double FAC_NORM,
                                     double FAC_ERR)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_points) {
    // Compute L2Sqr between the data vector and centroid.
    PID id                = d_IDs[i];
    const float* cur_data = d_data + id * DIM;
    float sum             = 0.0f;
    for (int k = 0; k < DIM; k++) {
      float diff = cur_data[k] - d_centroid[k];
      sum += diff * diff;
    }
    fac_x2[i]    = sum;
    float dist2c = sqrtf(sum);
#ifdef HIGH_ACC_FAST_SCAN
    fac_x2[i] = dist2c;
#endif
    // Compute X0 and ip.
    float X0  = 0.0f;
    float ip  = 0.0f;
    int sumxb = 0;
    for (int j = 0; j < D; j++) {
      // Retrieve binary value (assumed to be 0 or 1).
      int bin_val = d_bin_XP[i * D + j];
      // coded_vec element: (2*bin - 1) * FAC_NORM.
      float coded = (2.0f * (float)bin_val - 1.0f) * FAC_NORM;
      X0 += d_XP_norm[i * D + j] * coded;
      ip += fabsf(0.5f * d_XP_norm[i * D + j]);
      sumxb += bin_val;
    }
    // Handle o_obar.
    float o_obar = X0;
    if (!isfinite(o_obar)) { o_obar = 0.8f; }
    fac_ip[i]    = (1.0f / ip) * 2.0f * dist2c;
    fac_sumxb[i] = (float)sumxb;
    fac_err[i]   = sqrtf((1.0f - o_obar * o_obar) / (o_obar * o_obar)) * FAC_ERR * 2.0f * dist2c;
  }
}

//---------------------------------------------------------------------------
// DataQuantizerGPU::rabitq_factor (GPU version)
//---------------------------------------------------------------------------
void DataQuantizerGPU::rabitq_factor(const float* d_data,
                                     const float* d_centroid,
                                     const PID* d_IDs,
                                     const int* d_bin_XP,
                                     const float* d_XP_norm,
                                     float* fac_x2,
                                     float* fac_ip,
                                     float* fac_sumxb,
                                     float* fac_err,
                                     size_t num_points) const
{
  // Launch one thread per data point.
  int blockSize = 256;
  int gridSize  = (num_points + blockSize - 1) / blockSize;
  rabitq_factor_kernel<<<gridSize, blockSize, 0, stream_>>>(d_data,
                                                            d_centroid,
                                                            d_IDs,
                                                            d_bin_XP,
                                                            d_XP_norm,
                                                            fac_x2,
                                                            fac_ip,
                                                            fac_sumxb,
                                                            fac_err,
                                                            num_points,
                                                            this->DIM,
                                                            this->D,
                                                            FAC_NORM,
                                                            FAC_ERR);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  raft::resource::sync_stream(handle_);
}

//==============================================================================
// Kernel for EX_BITS == 4.
// Each thread processes one block of 32 bytes, outputting 16 bytes.
__global__ void store_compacted_code_kernel_4(const uint8_t* o_raw,
                                              uint8_t* o_compact,
                                              size_t num_blocks)
{
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_blocks) {
    const uint8_t* in_ptr = o_raw + idx * 32;
    uint8_t* out_ptr      = o_compact + idx * 16;
#pragma unroll
    for (int i = 0; i < 16; i++) {
      uint8_t a  = in_ptr[i];
      uint8_t b  = in_ptr[i + 16];
      out_ptr[i] = a | (b << 4);
    }
  }
}

//==============================================================================
// Kernel for EX_BITS == 6.
// Each thread processes one block of 64 bytes, outputting 48 bytes.
__global__ void store_compacted_code_kernel_6(const uint8_t* o_raw,
                                              uint8_t* o_compact,
                                              size_t num_blocks)
{
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_blocks) {
    const uint8_t* in_ptr = o_raw + idx * 64;
    uint8_t* out_ptr      = o_compact + idx * 48;
    // First 16 bytes: combine bytes [0..15] with shifted [32..47]
    for (int i = 0; i < 16; i++) {
      uint8_t a       = in_ptr[i];
      uint8_t b       = in_ptr[i + 32];
      uint8_t shifted = (b << 2) & 0xC0;  // mask2 = 0xC0
      out_ptr[i]      = a | shifted;
    }
    // Next 16 bytes: combine bytes [16..31] with shifted [48..63]
    for (int i = 0; i < 16; i++) {
      uint8_t a       = in_ptr[i + 16];
      uint8_t b       = in_ptr[i + 48];
      uint8_t shifted = (b << 2) & 0xC0;
      out_ptr[i + 16] = a | shifted;
    }
    // Last 16 bytes: combine lower 4 bits of [32..47] and [48..63]
    for (int i = 0; i < 16; i++) {
      uint8_t a       = in_ptr[i + 32] & 0x0F;  // mask4 = 0x0F
      uint8_t b       = (in_ptr[i + 48] & 0x0F) << 4;
      out_ptr[i + 32] = a | b;
    }
  }
}

//==============================================================================
// Kernel for EX_BITS == 2.
// Each thread processes one block of 64 bytes, outputting 16 bytes.
__global__ void store_compacted_code_kernel_2(const uint8_t* o_raw,
                                              uint8_t* o_compact,
                                              size_t num_blocks)
{
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_blocks) {
    const uint8_t* in_ptr = o_raw + idx * 64;
    uint8_t* out_ptr      = o_compact + idx * 16;
    for (int i = 0; i < 16; i++) {
      uint8_t a  = in_ptr[i] & 0x03;  // mask = 0b11
      uint8_t b  = (in_ptr[i + 16] & 0x03) << 2;
      uint8_t c  = (in_ptr[i + 32] & 0x03) << 4;
      uint8_t d  = (in_ptr[i + 48] & 0x03) << 6;
      out_ptr[i] = a | b | c | d;
    }
  }
}

//==============================================================================
// Kernel for EX_BITS == 3.
// Each thread processes one block of 64 bytes, outputting 24 bytes (16+8).
__global__ void store_compacted_code_kernel_3(const uint8_t* o_raw,
                                              uint8_t* o_compact,
                                              size_t num_blocks)
{
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_blocks) {
    const uint8_t* in_ptr = o_raw + idx * 64;
    uint8_t* out_ptr      = o_compact + idx * 24;  // 16 bytes then 8 bytes
    // First 16 bytes: similar to EX_BITS==2 but for 3 bits, mask = 0b11.
    for (int i = 0; i < 16; i++) {
      uint8_t a  = in_ptr[i] & 0x07;  // use lower 3 bits, but CPU code uses mask = 0b11?
      uint8_t b  = ((in_ptr[i + 16] & 0x07) << 2);
      uint8_t c  = ((in_ptr[i + 32] & 0x07) << 4);
      uint8_t d  = ((in_ptr[i + 48] & 0x07) << 6);
      out_ptr[i] = a | b | c | d;
    }
    // Next 8 bytes: compute top_bit.
    int64_t top_bit  = 0;
    int64_t top_mask = 0x0101010101010101LL;
    for (int i = 0; i < 64; i += 8) {
      int64_t cur_codes = *(const int64_t*)(in_ptr + i);
      top_bit |= ((cur_codes >> 2) & top_mask) << (i / 8);
    }
    *(int64_t*)(out_ptr + 16) = top_bit;
  }
}

//==============================================================================
// Kernel for EX_BITS == 7.
// Each thread processes one block of 64 bytes, outputting 56 bytes (48+8).
__global__ void store_compacted_code_kernel_7(const uint8_t* o_raw,
                                              uint8_t* o_compact,
                                              size_t num_blocks)
{
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_blocks) {
    const uint8_t* in_ptr = o_raw + idx * 64;
    uint8_t* out_ptr      = o_compact + idx * 56;  // 48 bytes then 8 bytes
    // First 16 bytes: using mask6 = 0x3F and mask2 = 0xC0.
    for (int i = 0; i < 16; i++) {
      uint8_t a  = in_ptr[i] & 0x3F;
      uint8_t b  = (in_ptr[i + 32] << 2) & 0xC0;
      out_ptr[i] = a | b;
    }
    // Next 16 bytes:
    for (int i = 0; i < 16; i++) {
      uint8_t a       = in_ptr[i + 16] & 0x3F;
      uint8_t b       = (in_ptr[i + 48] << 2) & 0xC0;
      out_ptr[i + 16] = a | b;
    }
    // Next 16 bytes:
    for (int i = 0; i < 16; i++) {
      uint8_t a       = in_ptr[i + 32] & 0x0F;  // mask4 = 0x0F
      uint8_t b       = (in_ptr[i + 48] & 0x0F) << 4;
      out_ptr[i + 32] = a | b;
    }
    // Then, process top_bit: next 8 bytes.
    out_ptr += 48;
    int64_t top_bit  = 0;
    int64_t top_mask = 0x0101010101010101LL;
    for (int i = 0; i < 64; i += 8) {
      int64_t cur_codes = *(const int64_t*)(in_ptr + i);
      top_bit |= ((cur_codes >> 6) & top_mask) << (i / 8);
    }
    *(int64_t*)(out_ptr) = top_bit;
  }
}

//==============================================================================
// Host function: store_compacted_code
// Dispatches based on EX_BITS.
void DataQuantizerGPU::store_compacted_code(uint8_t* o_raw, uint8_t* o_compact) const
{
  if (EX_BITS == 8) {
    RAFT_CUDA_TRY(
      cudaMemcpyAsync(o_compact, o_raw, sizeof(uint8_t) * D, cudaMemcpyDeviceToDevice, stream_));
  } else if (EX_BITS == 4) {
    size_t num_blocks = D / 32;  // Each block processes 32 bytes.
    int blockSize     = 256;
    int gridSize      = (num_blocks + blockSize - 1) / blockSize;
    store_compacted_code_kernel_4<<<gridSize, blockSize, 0, stream_>>>(
      o_raw, o_compact, num_blocks);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  } else if (EX_BITS == 6) {
    size_t num_blocks = D / 64;  // Each block processes 64 bytes.
    int blockSize     = 256;
    int gridSize      = (num_blocks + blockSize - 1) / blockSize;
    store_compacted_code_kernel_6<<<gridSize, blockSize, 0, stream_>>>(
      o_raw, o_compact, num_blocks);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  } else if (EX_BITS == 2) {
    size_t num_blocks = D / 64;
    int blockSize     = 256;
    int gridSize      = (num_blocks + blockSize - 1) / blockSize;
    store_compacted_code_kernel_2<<<gridSize, blockSize, 0, stream_>>>(
      o_raw, o_compact, num_blocks);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  } else if (EX_BITS == 3) {
    size_t num_blocks = D / 64;
    int blockSize     = 256;
    int gridSize      = (num_blocks + blockSize - 1) / blockSize;
    store_compacted_code_kernel_3<<<gridSize, blockSize, 0, stream_>>>(
      o_raw, o_compact, num_blocks);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  } else if (EX_BITS == 7) {
    size_t num_blocks = D / 64;
    int blockSize     = 256;
    int gridSize      = (num_blocks + blockSize - 1) / blockSize;
    store_compacted_code_kernel_7<<<gridSize, blockSize, 0, stream_>>>(
      o_raw, o_compact, num_blocks);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  } else {
    std::cerr << "store_compacted_code: EX_BITS value " << EX_BITS << " not implemented."
              << std::endl;
    exit(EXIT_FAILURE);
  }
  raft::resource::sync_stream(handle_);
}

//---------------------------------------------------------------------------
// Device function: fast_quantize_device
//
// This function computes a quantization code for one vector.
// It mimics the CPU version of fast_quantize.
// Parameters:
//   o_prime: pointer to input rotated absolute values (length D)
//   code: output array of length D (each element is a uint8_t code)
//   ip_norm: pointer to output scalar (1/(ip*norm))
//   D: dimension (padded)
//   EX_BITS: quantization bits
__device__ void fast_quantize_device(
  const float* o_prime, uint8_t* code, float* ip_norm, size_t D, size_t EX_BITS)
{
#ifdef DEBUG_EXRABITQ_TIME
  unsigned long long t0 = clock64();
#endif

  // First loop: compute max_o.
  const double eps = 1e-5;
  const int n_enum = 10;
  double max_o     = -1.0;
  for (int i = 0; i < D; i++) {
    double val = o_prime[i];
    if (val > max_o) max_o = val;
  }
#ifdef DEBUG_EXRABITQ_TIME
  unsigned long long t1 = clock64();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("Thread %d: Loop 1 (max_o computation) took %llu cycles\n", threadIdx.x, t1 - t0);
  }
#endif

  double t_start = (((1 << EX_BITS) - 1) / 3.0) / max_o;
  double t_end   = ((((1 << EX_BITS) - 1) + n_enum)) / max_o;

  int cur_o_bar[MAX_D];  // assume D <= MAX_D
  double sqr_denom = D * 0.25;
  double numerator = 0.0;

#ifdef DEBUG_EXRABITQ_TIME
  unsigned long long t2 = clock64();
#endif

  // Second loop: initial computation of cur_o_bar, updating sqr_denom and numerator.
  for (int i = 0; i < D; i++) {
    cur_o_bar[i] = int(t_start * o_prime[i] + eps);
    sqr_denom += cur_o_bar[i] * cur_o_bar[i] + cur_o_bar[i];
    numerator += (cur_o_bar[i] + 0.5) * o_prime[i];
  }
#ifdef DEBUG_EXRABITQ_TIME
  unsigned long long t3 = clock64();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("Thread %d: Loop 2 (cur_o_bar init) took %llu cycles\n", threadIdx.x, t3 - t2);
  }
#endif

  double max_ip    = 0.0;
  double t_val     = 0.0;
  bool improvement = true;

#ifdef DEBUG_EXRABITQ_TIME
  unsigned long long t4 = clock64();
#endif

  // While loop: iterative improvement.
  while (improvement) {
    improvement = false;
    for (int i = 0; i < D; i++) {
      if (cur_o_bar[i] < ((1 << EX_BITS) - 1)) {
        double t_next = (cur_o_bar[i] + 1.0) / o_prime[i];
        if (t_next < t_end) {
          int new_val    = cur_o_bar[i] + 1;
          double new_sqr = sqr_denom + 2 * new_val;
          double new_num = numerator + o_prime[i];
          double cur_ip  = new_num / sqrt(new_sqr);
          if (cur_ip > max_ip) {
            max_ip       = cur_ip;
            t_val        = t_next;
            cur_o_bar[i] = new_val;
            sqr_denom    = new_sqr;
            numerator    = new_num;
            improvement  = true;
          }
        }
      }
    }
  }
#ifdef DEBUG_EXRABITQ_TIME
  unsigned long long t5 = clock64();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("Thread %d: While loop took %llu cycles\n", threadIdx.x, t5 - t4);
  }
#endif

  sqr_denom = D * 0.25;
  numerator = 0.0;
  int o_bar[MAX_D];

#ifdef DEBUG_EXRABITQ_TIME
  unsigned long long t6 = clock64();
#endif

  // Third loop: compute final o_bar, update sqr_denom and numerator.
  for (int i = 0; i < D; i++) {
    o_bar[i] = int(t_val * o_prime[i] + eps);
    if (o_bar[i] >= (1 << EX_BITS)) o_bar[i] = (1 << EX_BITS) - 1;
    sqr_denom += o_bar[i] * o_bar[i] + o_bar[i];
    numerator += (o_bar[i] + 0.5) * o_prime[i];
  }
#ifdef DEBUG_EXRABITQ_TIME
  unsigned long long t7 = clock64();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("Thread %d: Loop 3 (final o_bar computation) took %llu cycles\n", threadIdx.x, t7 - t6);
  }
#endif

  *ip_norm = 1.0f / (float)numerator;
  if (!isfinite(*ip_norm)) { *ip_norm = 1.0f; }

  // Fourth loop: assign codes.
  for (int i = 0; i < D; i++) {
    code[i] = (uint8_t)o_bar[i];
  }
#ifdef DEBUG_EXRABITQ_TIME
  unsigned long long t8 = clock64();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("Thread %d: Loop 4 (code assignment) took %llu cycles\n", threadIdx.x, t8 - t7);
  }
#endif
}

//---------------------------------------------------------------------------
// Device function: fast_quantize_device
//
// batch version: return inv_norm for factor computation
// This function computes a quantization code for one vector.
// It mimics the CPU version of fast_quantize.
// Parameters:
//   o_prime: pointer to input rotated absolute values (length D)
//   code: output array of length D (each element is a uint8_t code)
//   ip_norm: pointer to output scalar (1/(ip*norm))
//   D: dimension (padded)
//   EX_BITS: quantization bits
__device__ float fast_quantize_device_batch(
  const float* o_prime, uint8_t* code, float* ip_norm, size_t D, size_t EX_BITS)
{
#ifdef DEBUG_EXRABITQ_TIME
  unsigned long long t0 = clock64();
#endif

  // First loop: compute max_o.
  const double eps = 1e-5;
  const int n_enum = 10;
  double max_o     = -1.0;
  for (int i = 0; i < D; i++) {
    double val = o_prime[i];
    if (val > max_o) max_o = val;
  }
#ifdef DEBUG_EXRABITQ_TIME
  unsigned long long t1 = clock64();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("Thread %d: Loop 1 (max_o computation) took %llu cycles\n", threadIdx.x, t1 - t0);
  }
#endif

  double t_start = (((1 << EX_BITS) - 1) / 3.0) / max_o;
  double t_end   = ((((1 << EX_BITS) - 1) + n_enum)) / max_o;

  int cur_o_bar[MAX_D];  // assume D <= MAX_D
  double sqr_denom = D * 0.25;
  double numerator = 0.0;

#ifdef DEBUG_EXRABITQ_TIME
  unsigned long long t2 = clock64();
#endif

  // Second loop: initial computation of cur_o_bar, updating sqr_denom and numerator.
  for (int i = 0; i < D; i++) {
    cur_o_bar[i] = int(t_start * o_prime[i] + eps);
    sqr_denom += cur_o_bar[i] * cur_o_bar[i] + cur_o_bar[i];
    numerator += (cur_o_bar[i] + 0.5) * o_prime[i];
  }
#ifdef DEBUG_EXRABITQ_TIME
  unsigned long long t3 = clock64();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("Thread %d: Loop 2 (cur_o_bar init) took %llu cycles\n", threadIdx.x, t3 - t2);
  }
#endif

  double max_ip    = 0.0;
  double t_val     = 0.0;
  bool improvement = true;

#ifdef DEBUG_EXRABITQ_TIME
  unsigned long long t4 = clock64();
#endif

  // While loop: iterative improvement.
  while (improvement) {
    improvement = false;
    for (int i = 0; i < D; i++) {
      if (cur_o_bar[i] < ((1 << EX_BITS) - 1)) {
        double t_next = (cur_o_bar[i] + 1.0) / o_prime[i];
        if (t_next < t_end) {
          int new_val    = cur_o_bar[i] + 1;
          double new_sqr = sqr_denom + 2 * new_val;
          double new_num = numerator + o_prime[i];
          double cur_ip  = new_num / sqrt(new_sqr);
          if (cur_ip > max_ip) {
            max_ip       = cur_ip;
            t_val        = t_next;
            cur_o_bar[i] = new_val;
            sqr_denom    = new_sqr;
            numerator    = new_num;
            improvement  = true;
          }
        }
      }
    }
  }
#ifdef DEBUG_EXRABITQ_TIME
  unsigned long long t5 = clock64();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("Thread %d: While loop took %llu cycles\n", threadIdx.x, t5 - t4);
  }
#endif

  sqr_denom = D * 0.25;
  numerator = 0.0;
  int o_bar[MAX_D];

#ifdef DEBUG_EXRABITQ_TIME
  unsigned long long t6 = clock64();
#endif

  // Third loop: compute final o_bar, update sqr_denom and numerator.
  for (int i = 0; i < D; i++) {
    o_bar[i] = int(t_val * o_prime[i] + eps);
    if (o_bar[i] >= (1 << EX_BITS)) o_bar[i] = (1 << EX_BITS) - 1;
    sqr_denom += o_bar[i] * o_bar[i] + o_bar[i];
    numerator += (o_bar[i] + 0.5) * o_prime[i];
  }
#ifdef DEBUG_EXRABITQ_TIME
  unsigned long long t7 = clock64();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("Thread %d: Loop 3 (final o_bar computation) took %llu cycles\n", threadIdx.x, t7 - t6);
  }
#endif

  *ip_norm = 1.0f / (float)numerator;
  if (!isfinite(*ip_norm)) { *ip_norm = 1.0f; }

  // Fourth loop: assign codes.
  for (int i = 0; i < D; i++) {
    code[i] = (uint8_t)o_bar[i];
  }
#ifdef DEBUG_EXRABITQ_TIME
  unsigned long long t8 = clock64();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("Thread %d: Loop 4 (code assignment) took %llu cycles\n", threadIdx.x, t8 - t7);
  }
#endif
  return *ip_norm;
}

//---------------------------------------------------------------------------
// Device function: pack_codes_generic
//
// This function packs D quantization codes (each stored in one byte, but only the lower EX_BITS are
// valid) into an output byte array of length L = (D*EX_BITS + 7) / 8. It writes bits in big-endian
// order.
__device__ void pack_codes_generic(const uint8_t* in, uint8_t* out, size_t D, size_t EX_BITS)
{
  size_t total_bits = D * EX_BITS;
  size_t L          = (total_bits + 7) / 8;
  // Zero output.
  for (size_t i = 0; i < L; i++) {
    out[i] = 0;
  }
  size_t bit_pos = 0;
  for (size_t j = 0; j < D; j++) {
    uint32_t code     = in[j] & ((1 << EX_BITS) - 1);
    size_t byte_idx   = bit_pos / 8;
    size_t bit_offset = bit_pos % 8;
    if (bit_offset + EX_BITS <= 8) {
      out[byte_idx] |= code << (8 - bit_offset - EX_BITS);
    } else {
      int first_part = 8 - bit_offset;
      out[byte_idx] |= code >> (EX_BITS - first_part);
      out[byte_idx + 1] |= code << (16 - bit_offset - EX_BITS);
    }
    bit_pos += EX_BITS;
  }
}

// CPU-side fast_quantize function (same as before but included for completeness)
void fast_quantize_cpu_single(
  const float* o_prime, uint8_t* code, float* ip_norm, size_t D, size_t EX_BITS)
{
  const double eps = 1e-5;
  const int n_enum = 10;
  double max_o     = -1.0;

  for (size_t j = 0; j < D; j++) {
    double val = fabs(o_prime[j]);
    if (val > max_o) max_o = val;
  }

  double t_start = (((1 << EX_BITS) - 1) / 3.0) / max_o;
  double t_end   = ((((1 << EX_BITS) - 1) + n_enum)) / max_o;

  std::vector<int> cur_o_bar(D);
  double sqr_denom = D * 0.25;
  double numerator = 0.0;

  for (size_t j = 0; j < D; j++) {
    cur_o_bar[j] = int(t_start * fabs(o_prime[j]) + eps);
    sqr_denom += cur_o_bar[j] * cur_o_bar[j] + cur_o_bar[j];
    numerator += (cur_o_bar[j] + 0.5) * fabs(o_prime[j]);
  }

  double max_ip    = 0.0;
  double t_val     = t_start;
  bool improvement = true;

  while (improvement) {
    improvement = false;
    for (size_t j = 0; j < D; j++) {
      if (cur_o_bar[j] < ((1 << EX_BITS) - 1)) {
        double t_next = (cur_o_bar[j] + 1.0) / fabs(o_prime[j]);
        if (t_next < t_end) {
          int new_val    = cur_o_bar[j] + 1;
          double new_sqr = sqr_denom + 2 * new_val;
          double new_num = numerator + fabs(o_prime[j]);
          double cur_ip  = new_num / sqrt(new_sqr);
          if (cur_ip > max_ip) {
            max_ip       = cur_ip;
            t_val        = t_next;
            cur_o_bar[j] = new_val;
            sqr_denom    = new_sqr;
            numerator    = new_num;
            improvement  = true;
          }
        }
      }
    }
  }

  sqr_denom = D * 0.25;
  numerator = 0.0;
  for (size_t j = 0; j < D; j++) {
    int o_bar = int(t_val * fabs(o_prime[j]) + eps);
    if (o_bar >= (1 << EX_BITS)) o_bar = (1 << EX_BITS) - 1;
    sqr_denom += o_bar * o_bar + o_bar;
    numerator += (o_bar + 0.5) * fabs(o_prime[j]);
    code[j] = (uint8_t)o_bar;
  }

  *ip_norm = 1.0f / (float)numerator;
  if (!std::isfinite(*ip_norm)) { *ip_norm = 1.0f; }
}

// Batch CPU computation with OpenMP
void fast_quantize_cpu_batch(const float* h_XP_norm,
                             uint8_t* h_tmp_codes,
                             float* h_ip_norms,
                             size_t num_points,
                             size_t D,
                             size_t EX_BITS)
{
#pragma omp parallel for schedule(dynamic, 16)
  for (size_t i = 0; i < num_points; i++) {
    fast_quantize_cpu_single(h_XP_norm + i * D, h_tmp_codes + i * D, h_ip_norms + i, D, EX_BITS);
  }
}

// Simplified GPU kernel (same as before)
__global__ void exrabitq_pack_kernel(const int* d_bin_XP,
                                     const uint8_t* d_tmp_codes,
                                     const float* d_ip_norms,
                                     uint8_t* d_long_code,
                                     float* d_ex_factor,
                                     const float* d_fac_x2,
                                     size_t num_points,
                                     size_t D,
                                     size_t EX_BITS,
                                     size_t long_code_stride)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_points) {
    const int mask = (1 << EX_BITS) - 1;

    float ip_norm = d_ip_norms[i];
#ifdef HIGH_ACC_FAST_SCAN
    d_ex_factor[i] = ip_norm * 2.0f * d_fac_x2[i];
#else
    d_ex_factor[i] = ip_norm * 2.0f * sqrtf(d_fac_x2[i]);
#endif

    uint8_t flipped_code[MAX_D];

    for (size_t j = 0; j < D; j++) {
      uint8_t code = d_tmp_codes[i * D + j];
      if (d_bin_XP[i * D + j] == 0) {
        flipped_code[j] = (~code) & mask;
      } else {
        flipped_code[j] = code;
      }
    }

    uint8_t* out_ptr = d_long_code + i * long_code_stride;

    if (EX_BITS == 8) {
      for (size_t j = 0; j < D; j++) {
        out_ptr[j] = flipped_code[j];
      }
    } else {
      pack_codes_generic(flipped_code, out_ptr, D, EX_BITS);
    }
  }
}

// Structure for batch processing state
struct BatchState {
  size_t batch_start;
  size_t batch_size;
  std::atomic<bool> cpu_done;
  std::atomic<bool> gpu_done;

  BatchState() : batch_start(0), batch_size(0), cpu_done(false), gpu_done(false) {}

  void reset(size_t start, size_t size)
  {
    batch_start = start;
    batch_size  = size;
    cpu_done    = false;
    gpu_done    = false;
  }
};

// Main advanced hybrid function with triple buffering
void DataQuantizerGPU::exrabitq_codes_hybrid_advanced(const int* d_bin_XP,
                                                      const float* d_XP_norm,
                                                      uint8_t* d_long_code,
                                                      float* d_ex_factor,
                                                      const float* d_fac_x2,
                                                      size_t num_points) const
{
  // Configuration
  const size_t batch_size       = 10000;  // Adjust based on your GPU memory and D
  const int num_buffers         = 3;      // Triple buffering
  const size_t long_code_stride = (D * EX_BITS + 7) / 8;
  const size_t num_batches      = (num_points + batch_size - 1) / batch_size;

  // Buffer structure for triple buffering
  struct BatchBuffers {
    // Host buffers (pinned memory)
    float* h_XP_norm;
    uint8_t* h_tmp_codes;
    float* h_ip_norms;

    // Device staging buffers
    uint8_t* d_tmp_codes;
    float* d_ip_norms;

    // CUDA stream and events
    cudaStream_t stream;
    cudaEvent_t h2d_done;
    cudaEvent_t d2h_done;
    cudaEvent_t gpu_compute_done;

    // Batch state
    BatchState state;
  };

  // Allocate triple buffers
  std::vector<BatchBuffers> buffers(num_buffers);

  for (int i = 0; i < num_buffers; i++) {
    // Allocate pinned host memory
    RAFT_CUDA_TRY(cudaMallocHost(&buffers[i].h_XP_norm, batch_size * D * sizeof(float)));
    RAFT_CUDA_TRY(cudaMallocHost(&buffers[i].h_tmp_codes, batch_size * D * sizeof(uint8_t)));
    RAFT_CUDA_TRY(cudaMallocHost(&buffers[i].h_ip_norms, batch_size * sizeof(float)));

    // Allocate device staging buffers
    RAFT_CUDA_TRY(
      cudaMallocAsync(&buffers[i].d_tmp_codes, batch_size * D * sizeof(uint8_t), stream_));
    RAFT_CUDA_TRY(cudaMallocAsync(&buffers[i].d_ip_norms, batch_size * sizeof(float), stream_));

    // Create stream and events
    RAFT_CUDA_TRY(cudaStreamCreate(&buffers[i].stream));
    RAFT_CUDA_TRY(cudaEventCreate(&buffers[i].h2d_done));
    RAFT_CUDA_TRY(cudaEventCreate(&buffers[i].d2h_done));
    RAFT_CUDA_TRY(cudaEventCreate(&buffers[i].gpu_compute_done));
  }
  raft::resource::sync_stream(handle_);

  // Thread pool for CPU computation
  std::vector<std::thread> cpu_threads;

  // Lambda for CPU computation task
  auto cpu_compute_task = [&](int buf_id, size_t batch_start, size_t batch_size_actual) {
    auto& buffer = buffers[buf_id];

    // Wait for D2H transfer to complete
    RAFT_CUDA_TRY(cudaEventSynchronize(buffer.d2h_done));

    // Perform CPU computation
    fast_quantize_cpu_batch(
      buffer.h_XP_norm, buffer.h_tmp_codes, buffer.h_ip_norms, batch_size_actual, D, EX_BITS);

    // Mark CPU computation as done
    buffer.state.cpu_done = true;
  };

  // Lambda for GPU task
  auto gpu_process_task = [&](int buf_id) {
    auto& buffer = buffers[buf_id];

    // Wait for CPU computation to complete
    while (!buffer.state.cpu_done) {
      std::this_thread::yield();
    }

    size_t batch_start       = buffer.state.batch_start;
    size_t batch_size_actual = buffer.state.batch_size;

    // Transfer results H2D
    RAFT_CUDA_TRY(cudaMemcpyAsync(buffer.d_tmp_codes,
                                  buffer.h_tmp_codes,
                                  batch_size_actual * D * sizeof(uint8_t),
                                  cudaMemcpyHostToDevice,
                                  buffer.stream));

    RAFT_CUDA_TRY(cudaMemcpyAsync(buffer.d_ip_norms,
                                  buffer.h_ip_norms,
                                  batch_size_actual * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  buffer.stream));

    RAFT_CUDA_TRY(cudaEventRecord(buffer.h2d_done, buffer.stream));

    // Launch GPU kernel
    int blockSize = 256;
    int gridSize  = (batch_size_actual + blockSize - 1) / blockSize;

    exrabitq_pack_kernel<<<gridSize, blockSize, 0, buffer.stream>>>(
      d_bin_XP + batch_start * D,
      buffer.d_tmp_codes,
      buffer.d_ip_norms,
      d_long_code + batch_start * long_code_stride,
      d_ex_factor + batch_start,
      d_fac_x2 + batch_start,
      batch_size_actual,
      D,
      EX_BITS,
      long_code_stride);
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    RAFT_CUDA_TRY(cudaEventRecord(buffer.gpu_compute_done, buffer.stream));
    buffer.state.gpu_done = true;
  };

  // Main pipeline loop
  size_t batch_idx         = 0;
  size_t completed_batches = 0;

  // Initial pipeline fill
  for (int i = 0; i < std::min(num_buffers, (int)num_batches); i++) {
    size_t batch_start       = batch_idx * batch_size;
    size_t batch_size_actual = std::min(batch_size, num_points - batch_start);

    auto& buffer = buffers[i];
    buffer.state.reset(batch_start, batch_size_actual);

    // Start D2H transfer
    RAFT_CUDA_TRY(cudaMemcpyAsync(buffer.h_XP_norm,
                                  d_XP_norm + batch_start * D,
                                  batch_size_actual * D * sizeof(float),
                                  cudaMemcpyDeviceToHost,
                                  buffer.stream));

    RAFT_CUDA_TRY(cudaEventRecord(buffer.d2h_done, buffer.stream));

    // Launch CPU computation thread
    cpu_threads.emplace_back(cpu_compute_task, i, batch_start, batch_size_actual);

    batch_idx++;
  }

  // Process remaining batches
  while (completed_batches < num_batches) {
    for (int buf_id = 0; buf_id < num_buffers; buf_id++) {
      auto& buffer = buffers[buf_id];

      // Check if this buffer's GPU work is done
      if (buffer.state.batch_size > 0 && buffer.state.cpu_done && !buffer.state.gpu_done) {
        // Launch GPU processing in a separate thread to avoid blocking
        std::thread gpu_thread(gpu_process_task, buf_id);
        gpu_thread.detach();
      }

      // Check if this buffer is completely done and can be reused
      if (buffer.state.batch_size > 0 && buffer.state.gpu_done) {
        // Wait for GPU work to actually complete
        RAFT_CUDA_TRY(cudaEventSynchronize(buffer.gpu_compute_done));

        completed_batches++;

        // Reuse buffer for next batch if available
        if (batch_idx < num_batches) {
          size_t batch_start       = batch_idx * batch_size;
          size_t batch_size_actual = std::min(batch_size, num_points - batch_start);

          buffer.state.reset(batch_start, batch_size_actual);

          // Start D2H transfer for new batch
          RAFT_CUDA_TRY(cudaMemcpyAsync(buffer.h_XP_norm,
                                        d_XP_norm + batch_start * D,
                                        batch_size_actual * D * sizeof(float),
                                        cudaMemcpyDeviceToHost,
                                        buffer.stream));

          RAFT_CUDA_TRY(cudaEventRecord(buffer.d2h_done, buffer.stream));

          // Launch CPU computation thread
          cpu_threads.emplace_back(cpu_compute_task, buf_id, batch_start, batch_size_actual);

          batch_idx++;
        } else {
          buffer.state.batch_size = 0;  // Mark as unused
        }
      }
    }

    // Small sleep to prevent busy waiting
    std::this_thread::sleep_for(std::chrono::microseconds(10));
  }

  // Wait for all CPU threads to complete
  for (auto& thread : cpu_threads) {
    if (thread.joinable()) { thread.join(); }
  }

  // Synchronize all streams
  for (int i = 0; i < num_buffers; i++) {
    RAFT_CUDA_TRY(cudaStreamSynchronize(buffers[i].stream));
  }

  // Cleanup
  for (int i = 0; i < num_buffers; i++) {
    // Free host memory
    RAFT_CUDA_TRY(cudaFreeHost(buffers[i].h_XP_norm));
    RAFT_CUDA_TRY(cudaFreeHost(buffers[i].h_tmp_codes));
    RAFT_CUDA_TRY(cudaFreeHost(buffers[i].h_ip_norms));

    // Free device memory
    RAFT_CUDA_TRY(cudaFreeAsync(buffers[i].d_tmp_codes, stream_));
    RAFT_CUDA_TRY(cudaFreeAsync(buffers[i].d_ip_norms, stream_));

    // Destroy stream and events
    RAFT_CUDA_TRY(cudaStreamDestroy(buffers[i].stream));
    RAFT_CUDA_TRY(cudaEventDestroy(buffers[i].h2d_done));
    RAFT_CUDA_TRY(cudaEventDestroy(buffers[i].d2h_done));
    RAFT_CUDA_TRY(cudaEventDestroy(buffers[i].gpu_compute_done));
  }
}

//---------------------------------------------------------------------------
// Kernel: exrabitq_codes_kernel
//
// Each thread processes one data point (row).
// It uses fast_quantize_device to compute a temporary code (stored in tmp_code),
// then for each dimension j, if d_bin_XP[i*D+j]==0, it flips tmp_code[j] using mask,
// and finally packs tmp_code into a compact code (of length (D*EX_BITS+7)/8)
// which is written to d_long_code.
__global__ void exrabitq_codes_kernel(const int* d_bin_XP,
                                      const float* d_XP_norm,
                                      uint8_t* d_long_code,
                                      float* d_ex_factor,
                                      const float* d_fac_x2,
                                      size_t num_points,
                                      size_t D,
                                      size_t EX_BITS)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_points) {
    const int mask = (1 << EX_BITS) - 1;
    // Allocate temporary code buffer in local memory.
    // We assume D does not exceed MAX_D.
    uint8_t tmp_code[MAX_D];
    // Prepare a local copy of absolute values of XP_norm for this data point.
    float local_abs[MAX_D];
    for (size_t j = 0; j < D; j++) {
      float val    = d_XP_norm[i * D + j];
      local_abs[j] = fabsf(val);
    }
    float ip_norm;
    fast_quantize_device(local_abs, tmp_code, &ip_norm, D, EX_BITS);
#ifdef HIGH_ACC_FAST_SCAN
    d_ex_factor[i] = ip_norm * 2.0f * d_fac_x2[i];
#else
    d_ex_factor[i] = ip_norm * 2.0f * sqrtf(d_fac_x2[i]);
#endif
    // For each dimension, if binary value is 0 then flip code.
    for (size_t j = 0; j < D; j++) {
      if (d_bin_XP[i * D + j] == 0) { tmp_code[j] = (~tmp_code[j]) & mask; }
    }
    // Pack tmp_code into compact code.
    // Compute output length in bytes.
    int long_code_length = (int)((D * EX_BITS + 7) / 8);
    uint8_t* out_ptr     = d_long_code + i * long_code_length;
    if (EX_BITS == 8) {
      // For 8 bits, simply copy.
      for (size_t j = 0; j < D; j++) {
        out_ptr[j] = tmp_code[j];
      }
    } else {
      // Use generic pack function.
      pack_codes_generic(tmp_code, out_ptr, D, EX_BITS);
    }
  }
}
void compute_factors_packed_batch(const float* d_centroid,  // [D]
                                  const int* d_xu,          // [N*D]
                                  const float* d_XP,        // [N*D]
                                  const float* ipnorm_inv,  // [D]
                                  size_t N,
                                  size_t D,
                                  float kConstEpsilon,  // e.g., 1.9f
                                  float* d_out,         // device array size = 2*N
                                  size_t ex_bits,
                                  cudaStream_t stream,
                                  int threads_per_block = 256);

__constant__ float d_kTightStart[9] = {
  0.0f,
  0.15f,
  0.20f,
  0.52f,
  0.59f,
  0.71f,
  0.75f,
  0.77f,
  0.81f,
};

struct HeapItem {
  float key;
  int value;
};

__device__ __forceinline__ void heap_push(HeapItem* heap, int& heap_size, float key, int value)
{
  int idx         = heap_size++;
  heap[idx].key   = key;
  heap[idx].value = value;

  while (idx > 0) {
    int parent = (idx - 1) / 2;
    if (heap[parent].key > heap[idx].key) {
      HeapItem temp = heap[parent];
      heap[parent]  = heap[idx];
      heap[idx]     = temp;
      idx           = parent;
    } else {
      break;
    }
  }
}

__device__ __forceinline__ HeapItem heap_pop(HeapItem* heap, int& heap_size)
{
  HeapItem result = heap[0];
  heap[0]         = heap[--heap_size];

  int idx = 0;
  while (true) {
    int left     = 2 * idx + 1;
    int right    = 2 * idx + 2;
    int smallest = idx;

    if (left < heap_size && heap[left].key < heap[smallest].key) { smallest = left; }
    if (right < heap_size && heap[right].key < heap[smallest].key) { smallest = right; }

    if (smallest != idx) {
      HeapItem temp  = heap[idx];
      heap[idx]      = heap[smallest];
      heap[smallest] = temp;
      idx            = smallest;
    } else {
      break;
    }
  }

  return result;
}

__device__ float best_rescale_factor_device(
  const float* o_abs, int dim, int ex_bits, int* cur_o_bar, HeapItem* heap_buffer)
{
  constexpr float kEps = 1e-5f;
  constexpr int kNEnum = 10;

  // Find max element
  float max_o = 0.0f;
  for (int i = 0; i < dim; ++i) {
    max_o = fmaxf(max_o, o_abs[i]);
  }

  //    // Handle edge case where all values are zero
  //    if (max_o < kEps) {
  //        return 1.0f;
  //    }

  float t_end   = static_cast<float>(((1 << ex_bits) - 1) + kNEnum) / max_o;
  float t_start = t_end * d_kTightStart[ex_bits];

  float sqr_denominator = static_cast<float>(dim) * 0.25f;
  float numerator       = 0.0f;

  // Initialize cur_o_bar and compute initial values
  for (int i = 0; i < dim; ++i) {
    int cur      = static_cast<int>((t_start * o_abs[i]) + kEps);
    cur_o_bar[i] = cur;
    sqr_denominator += cur * cur + cur;
    numerator += (cur + 0.5f) * o_abs[i];
  }

  // Initialize priority queue
  int heap_size = 0;
  for (int i = 0; i < dim; ++i) {
    //        if (o_abs[i] > kEps) {  // Only add non-zero elements
    float next_t = static_cast<float>(cur_o_bar[i] + 1) / o_abs[i];
    heap_push(heap_buffer, heap_size, next_t, i);
    //        }
  }

  float max_ip = 0.0f;
  float t      = t_start;  // Initialize with t_start instead of 0

  while (heap_size > 0) {
    HeapItem item = heap_pop(heap_buffer, heap_size);
    float cur_t   = item.key;
    int update_id = item.value;

    cur_o_bar[update_id]++;
    int update_o_bar = cur_o_bar[update_id];
    sqr_denominator += 2 * update_o_bar;
    numerator += o_abs[update_id];

    float cur_ip = numerator / sqrtf(sqr_denominator);
    if (cur_ip > max_ip) {
      max_ip = cur_ip;
      t      = cur_t;
    }

    if (update_o_bar < (1 << ex_bits) - 1) {
      float t_next = static_cast<float>(update_o_bar + 1) / o_abs[update_id];
      if (t_next < t_end) { heap_push(heap_buffer, heap_size, t_next, update_id); }
    }
  }

  return t;
}

__device__ float quantize_ex_device(
  const float* o_abs, uint8_t* code, int dim, int ex_bits, int* workspace)
{
  constexpr float kEps = 1e-5f;

  // Partition workspace
  int* tmp_code         = workspace;
  int* cur_o_bar        = tmp_code + dim;
  HeapItem* heap_buffer = (HeapItem*)(cur_o_bar + dim);

  // IMPORTANT: Initialize arrays to zero
  for (int i = 0; i < dim; ++i) {
    tmp_code[i]  = 0;
    cur_o_bar[i] = 0;
  }

  float t      = best_rescale_factor_device(o_abs, dim, ex_bits, cur_o_bar, heap_buffer);
  float ipnorm = 0.0f;

  for (int i = 0; i < dim; i++) {
    tmp_code[i] = static_cast<int>((t * o_abs[i]) + kEps);
    if (tmp_code[i] >= (1 << ex_bits)) { tmp_code[i] = (1 << ex_bits) - 1; }
    code[i] = static_cast<uint8_t>(tmp_code[i]);
    ipnorm += (tmp_code[i] + 0.5f) * o_abs[i];
  }

  float ipnorm_inv = 1.0f / ipnorm;
  if (!isfinite(ipnorm_inv) /*|| fabsf(ipnorm) < kEps*/) { ipnorm_inv = 1.0f; }

  return ipnorm_inv;
}

struct HeapItem_fp64 {
  double key;
  size_t value;
};

__device__ float quantize_ex_device_fp64(
  const float* o_abs, uint8_t* code, int dim, int ex_bits, int* workspace)
{
  constexpr double kEps = 1e-5f;

  // Partition workspace
  int* tmp_code         = workspace;
  int* cur_o_bar        = tmp_code + dim;
  HeapItem* heap_buffer = (HeapItem*)(cur_o_bar + dim);

  // IMPORTANT: Initialize arrays to zero
  for (int i = 0; i < dim; ++i) {
    tmp_code[i]  = 0;
    cur_o_bar[i] = 0;
  }

  double t      = best_rescale_factor_device(o_abs, dim, ex_bits, cur_o_bar, heap_buffer);
  double ipnorm = 0.0f;

  for (int i = 0; i < dim; i++) {
    tmp_code[i] = static_cast<int>((t * o_abs[i]) + kEps);
    if (tmp_code[i] >= (1 << ex_bits)) { tmp_code[i] = (1 << ex_bits) - 1; }
    code[i] = static_cast<uint8_t>(tmp_code[i]);
    ipnorm += (tmp_code[i] + 0.5f) * o_abs[i];
  }

  float ipnorm_inv = static_cast<double>(1.0f / ipnorm);
  if (!isfinite(ipnorm_inv) /*|| fabsf(ipnorm) < kEps*/) { ipnorm_inv = 1.0f; }

  return ipnorm_inv;
}

__device__ double best_rescale_factor_device_fp64(
  const float* o_abs, int dim, int ex_bits, int* cur_o_bar, HeapItem* heap_buffer)
{
  constexpr double kEps = 1e-5f;
  constexpr int kNEnum  = 10;

  // Find max element
  double max_o = 0.0f;
  for (int i = 0; i < dim; ++i) {
    max_o = fmaxf(max_o, o_abs[i]);
  }

  //    // Handle edge case where all values are zero
  //    if (max_o < kEps) {
  //        return 1.0f;
  //    }

  double t_end   = static_cast<float>(((1 << ex_bits) - 1) + kNEnum) / max_o;
  double t_start = t_end * d_kTightStart[ex_bits];

  double sqr_denominator = static_cast<float>(dim) * 0.25f;
  double numerator       = 0.0f;

  // Initialize cur_o_bar and compute initial values
  for (int i = 0; i < dim; ++i) {
    int cur      = static_cast<int>((t_start * o_abs[i]) + kEps);
    cur_o_bar[i] = cur;
    sqr_denominator += cur * cur + cur;
    numerator += (cur + 0.5f) * o_abs[i];
  }

  // Initialize priority queue
  int heap_size = 0;
  for (int i = 0; i < dim; ++i) {
    //        if (o_abs[i] > kEps) {  // Only add non-zero elements
    float next_t = static_cast<double>(cur_o_bar[i] + 1) / o_abs[i];
    heap_push(heap_buffer, heap_size, next_t, i);
    //        }
  }

  double max_ip = 0.0f;
  double t      = t_start;  // Initialize with t_start instead of 0

  while (heap_size > 0) {
    HeapItem item    = heap_pop(heap_buffer, heap_size);
    double cur_t     = item.key;
    size_t update_id = item.value;

    cur_o_bar[update_id]++;
    int update_o_bar = cur_o_bar[update_id];
    sqr_denominator += 2 * update_o_bar;
    numerator += o_abs[update_id];

    double cur_ip = numerator / sqrtf(sqr_denominator);
    if (cur_ip > max_ip) {
      max_ip = cur_ip;
      t      = cur_t;
    }

    if (update_o_bar < (1 << ex_bits) - 1) {
      double t_next = static_cast<double>(update_o_bar + 1) / o_abs[update_id];
      if (t_next < t_end) { heap_push(heap_buffer, heap_size, t_next, update_id); }
    }
  }

  return t;
}

//---------------------------------------------------------------------------
// Kernel: exrabitq_codes_kernel_batch
//
// output temp code for factor computation
// Each thread processes one data point (row).
// It uses fast_quantize_device to compute a temporary code (stored in tmp_code),
// then for each dimension j, if d_bin_XP[i*D+j]==0, it flips tmp_code[j] using mask,
// and finally packs tmp_code into a compact code (of length (D*EX_BITS+7)/8)
// which is written to d_long_code.
__global__ void exrabitq_codes_kernel_batch(const int* d_bin_XP,
                                            const float* d_XP_norm,
                                            uint8_t* d_long_code,
                                            size_t num_points,
                                            size_t D,
                                            size_t EX_BITS,
                                            float* ip_norm_inv,
                                            int* d_temp_codes,
                                            int* d_workspace)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_points) {
    const int mask = (1 << EX_BITS) - 1;
    // Allocate temporary code buffer in local memory.
    // We assume D does not exceed MAX_D.
    uint8_t tmp_code[MAX_D];
    // Prepare a local copy of absolute values of XP_norm for this data point.
    float local_abs[MAX_D];
    for (size_t j = 0; j < D; j++) {
      float val    = d_XP_norm[i * D + j];
      local_abs[j] = fabsf(val);
    }
    //        float ip_norm;  // ip norm env
    //        fast_quantize_device(local_abs, tmp_code, &ip_norm, D, EX_BITS);
    // Allocate shared memory or use dynamic allocation
    // Each thread gets its own workspace in global memory
    int workspace_size = D * 2 + D * 2;  // tmp_code + cur_o_bar + heap_buffer
    // If use fp64 version, workspace should be int workspace_size = D * 2 + (D * 2) * 2;
    int* my_workspace = d_workspace + i * workspace_size;
    float ip_norm     = quantize_ex_device(local_abs, tmp_code, D, EX_BITS, my_workspace);
    // For each dimension, if binary value is 0 then flip code.
    for (size_t j = 0; j < D; j++) {
      if (d_bin_XP[i * D + j] == 0) { tmp_code[j] = (~tmp_code[j]) & mask; }
    }
    // Pack tmp_code into compact code.
    // Compute output length in bytes.
    int long_code_length = (int)((D * EX_BITS + 7) / 8);
    uint8_t* out_ptr     = d_long_code + i * long_code_length;
    int* out_temp_ptr    = d_temp_codes + i * D;
    if (EX_BITS == 8) {
      // For 8 bits, simply copy.
      for (size_t j = 0; j < D; j++) {
        out_ptr[j] = tmp_code[j];
      }
    } else {
      // Use generic pack function.
      pack_codes_generic(tmp_code, out_ptr, D, EX_BITS);
    }
    for (size_t j = 0; j < D; j++) {
      out_temp_ptr[j] = (int)(tmp_code[j]);
    }
    // Then save ip_norm_inv for factors computation
    ip_norm_inv[i] = ip_norm;
  }
}

//---------------------------------------------------------------------------
// Host function: DataQuantizerGPU::exrabitq_codes
// Launches exrabitq_codes_kernel over all data points.
void DataQuantizerGPU::exrabitq_codes(const int* d_bin_XP,
                                      const float* d_XP_norm,
                                      uint8_t* d_long_code,
                                      float* d_ex_factor,
                                      const float* d_fac_x2,
                                      size_t num_points) const
{
  int blockSize = 256;
  int gridSize  = (num_points + blockSize - 1) / blockSize;
  exrabitq_codes_kernel<<<gridSize, blockSize, 0, stream_>>>(
    d_bin_XP, d_XP_norm, d_long_code, d_ex_factor, d_fac_x2, num_points, D, EX_BITS);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  raft::resource::sync_stream(handle_);
}
#ifdef DEBUG_BATCH_CONSTRUCT
int debug_first_cluster_count = 0;
#endif

//---------------------------------------------------------------------------
// Host function: DataQuantizerGPU::exrabitq_codes_batch
// First compute exrabitq codes then compute related factors
void DataQuantizerGPU::exrabitq_codes_batch(const int* d_bin_XP,
                                            const float* d_XP_norm,
                                            float* d_XP,
                                            uint8_t* d_long_code,
                                            float* d_ex_factor,
                                            const float* d_centroid,
                                            size_t num_points) const
{
  int blockSize      = 256;
  int gridSize       = (num_points + blockSize - 1) / blockSize;
  float* ip_norm_inv = nullptr;
  int* d_temp_codes  = nullptr;
  RAFT_CUDA_TRY(cudaMallocAsync((void**)&ip_norm_inv, num_points * sizeof(float), stream_));
  RAFT_CUDA_TRY(cudaMallocAsync((void**)&d_temp_codes, num_points * sizeof(int) * D, stream_));
  // Allocate workspace in global memory
  int workspace_per_vector = D * 2 + D * 2 * sizeof(HeapItem) / sizeof(int);
  int* d_workspace;
  RAFT_CUDA_TRY(
    cudaMallocAsync(&d_workspace, num_points * workspace_per_vector * sizeof(int), stream_));

  // Initialize workspace to zero (IMPORTANT!)
  RAFT_CUDA_TRY(
    cudaMemsetAsync(d_workspace, 0, num_points * workspace_per_vector * sizeof(int), stream_));
  exrabitq_codes_kernel_batch<<<gridSize, blockSize, 0, stream_>>>(d_bin_XP,
                                                                   d_XP_norm,
                                                                   d_long_code,
                                                                   num_points,
                                                                   D,
                                                                   EX_BITS,
                                                                   ip_norm_inv,
                                                                   d_temp_codes,
                                                                   d_workspace);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

#ifdef DEBUG_BATCH_CONSTRUCT
  if (debug_first_cluster_count < 1) {
    debug_first_cluster_count++;
    std::cout << "No." << debug_first_cluster_count
              << " vector of the first cluster's first 20 long codes:\n";
    int h_bin_XP[20];
    RAFT_CUDA_TRY(
      cudaMemcpyAsync(h_bin_XP, d_temp_codes, 20 * sizeof(int), cudaMemcpyDeviceToHost, stream_));
    raft::resource::sync_stream(*handle);

    // Print them
    for (int i = 0; i < 20; i++) {
      std::cout << "d_long[" << i << "] = " << h_bin_XP[i] << std::endl;
    }
  }
#endif
  // Then compute factors
  compute_factors_packed_batch(
    d_centroid, d_temp_codes, d_XP, ip_norm_inv, num_points, D, 1.9, d_ex_factor, EX_BITS, stream_);
  RAFT_CUDA_TRY(cudaFreeAsync(d_workspace, stream_));
  RAFT_CUDA_TRY(cudaFreeAsync(ip_norm_inv, stream_));
  RAFT_CUDA_TRY(cudaFreeAsync(d_temp_codes, stream_));
  RAFT_CUDA_TRY(cudaGetLastError());
  raft::resource::sync_stream(handle_);
}

//-----------------------------------------------------------------------------
// Host function: DataQuantizerGPU::quantize
// All pointers below are device pointers.
//   d_data       : raw data matrix (N x DIM), row-major
//   d_centroid   : centroid vector (1 x DIM)
//   d_IDs        : device array of IDs (indices into d_data) for the current cluster
//   num_points   : number of points in this cluster
//   rotator      : a RotatorGPU instance
//   d_short_data : output buffer for RaBitQ code and factors (pre-allocated on device)
//   d_long_code  : output buffer for ExRaBitQ code (pre-allocated on device)
//   d_ex_factor  : output buffer for ExRaBitQ factors (pre-allocated on device)
//   d_rotated_c  : output rotated centroid (size: D floats) on device
//-----------------------------------------------------------------------------
void DataQuantizerGPU::quantize(const float* d_data,
                                const float* d_centroid,
                                const PID* d_IDs,
                                size_t num_points,
                                const RotatorGPU& rotator,
                                uint32_t* d_short_data,
                                uint8_t* d_long_code,
                                float* d_ex_factor,
                                float* d_rotated_c) const
{
#ifdef DEBUG_TIME
  cudaEvent_t start, stop;
  float elapsed;
  RAFT_CUDA_TRY(cudaEventCreate(&start));
  RAFT_CUDA_TRY(cudaEventCreate(&stop));
#endif

  // 1. Data Transformation:
#ifdef DEBUG_TIME
  RAFT_CUDA_TRY(cudaEventRecord(start));
#endif
  float* d_XP_norm = nullptr;
  int* d_bin_XP    = nullptr;
  RAFT_CUDA_TRY(cudaMallocAsync((void**)&d_XP_norm, num_points * D * sizeof(float), stream_));
  RAFT_CUDA_TRY(cudaMallocAsync((void**)&d_bin_XP, num_points * D * sizeof(int), stream_));
  raft::resource::sync_stream(handle_);
  data_transformation(
    d_data, d_centroid, d_IDs, num_points, rotator, d_rotated_c, d_XP_norm, d_bin_XP);
#ifdef DEBUG_TIME
  RAFT_CUDA_TRY(cudaEventRecord(stop));
  RAFT_CUDA_TRY(cudaEventSynchronize(stop));
  RAFT_CUDA_TRY(cudaEventElapsedTime(&elapsed, start, stop));
  printf("Step 1 (Data Transformation): %f seconds\n", elapsed / 1000.0f);
#endif

  // 2. Compute total blocks for factors and short codes.
#ifdef DEBUG_TIME
  RAFT_CUDA_TRY(cudaEventRecord(start));
#endif
//    size_t total_blocks = div_rd_up(num_points, FAST_SIZE);
#ifdef DEBUG_TIME
  RAFT_CUDA_TRY(cudaEventRecord(stop));
  RAFT_CUDA_TRY(cudaEventSynchronize(stop));
  RAFT_CUDA_TRY(cudaEventElapsedTime(&elapsed, start, stop));
  printf("Step 2 (Compute total blocks): %f seconds\n", elapsed / 1000.0f);
#endif

  // 3. Allocate intermediate buffers on device.
#ifdef DEBUG_TIME
  RAFT_CUDA_TRY(cudaEventRecord(start));
#endif
  size_t code_len             = short_code_length();  // from quantizer parameters.
  size_t short_codes_bytes    = code_len * num_points * sizeof(uint32_t);
  uint32_t* d_all_short_codes = nullptr;
  RAFT_CUDA_TRY(cudaMallocAsync((void**)&d_all_short_codes, short_codes_bytes, stream_));

  size_t factor_bytes       = num_points * sizeof(float);
  float* d_all_factor_x2    = nullptr;
  float* d_all_factor_ip    = nullptr;
  float* d_all_factor_sumxb = nullptr;
  float* d_all_factor_err   = nullptr;
  RAFT_CUDA_TRY(cudaMallocAsync((void**)&d_all_factor_x2, factor_bytes, stream_));
  RAFT_CUDA_TRY(cudaMallocAsync((void**)&d_all_factor_ip, factor_bytes, stream_));
  RAFT_CUDA_TRY(cudaMallocAsync((void**)&d_all_factor_sumxb, factor_bytes, stream_));
  RAFT_CUDA_TRY(cudaMallocAsync((void**)&d_all_factor_err, factor_bytes, stream_));
#ifdef DEBUG_TIME
  raft::resource::sync_stream(handle_);
  RAFT_CUDA_TRY(cudaEventRecord(stop));
  RAFT_CUDA_TRY(cudaEventSynchronize(stop));
  RAFT_CUDA_TRY(cudaEventElapsedTime(&elapsed, start, stop));
  printf("Step 3 (Allocate intermediate buffers): %f seconds\n", elapsed / 1000.0f);
#endif

  // 4. Compute RaBitQ quantization codes.
#ifdef DEBUG_TIME
  RAFT_CUDA_TRY(cudaEventRecord(start));
#endif
  rabitq_codes(d_bin_XP, d_all_short_codes, num_points);
#ifdef DEBUG_TIME
  RAFT_CUDA_TRY(cudaEventRecord(stop));
  RAFT_CUDA_TRY(cudaEventSynchronize(stop));
  RAFT_CUDA_TRY(cudaEventElapsedTime(&elapsed, start, stop));
  printf("Step 4 (Compute RaBitQ codes): %f seconds\n", elapsed / 1000.0f);
#endif

  // 5. Compute re-ranking factors.
#ifdef DEBUG_TIME
  RAFT_CUDA_TRY(cudaEventRecord(start));
#endif
  rabitq_factor(d_data,
                d_centroid,
                d_IDs,
                d_bin_XP,
                d_XP_norm,
                d_all_factor_x2,
                d_all_factor_ip,
                d_all_factor_sumxb,
                d_all_factor_err,
                num_points);
#ifdef DEBUG_TIME
  RAFT_CUDA_TRY(cudaEventRecord(stop));
  RAFT_CUDA_TRY(cudaEventSynchronize(stop));
  RAFT_CUDA_TRY(cudaEventElapsedTime(&elapsed, start, stop));
  printf("Step 5 (Compute re-ranking factors): %f seconds\n", elapsed / 1000.0f);
#endif

  // 6. Compute ExRaBitQ quantization codes.
#ifdef DEBUG_TIME
  RAFT_CUDA_TRY(cudaEventRecord(start));
#endif
  exrabitq_codes(d_bin_XP, d_XP_norm, d_long_code, d_ex_factor, d_all_factor_x2, num_points);
#ifdef DEBUG_TIME
  RAFT_CUDA_TRY(cudaEventRecord(stop));
  RAFT_CUDA_TRY(cudaEventSynchronize(stop));
  RAFT_CUDA_TRY(cudaEventElapsedTime(&elapsed, start, stop));
  printf("Step 6 (Compute ExRaBitQ codes): %f seconds\n", elapsed / 1000.0f);
#endif

  // 7. Copy short codes and factor blocks into final output d_short_data.
#ifdef DEBUG_TIME
  RAFT_CUDA_TRY(cudaEventRecord(start));
#endif
  uint32_t* cur_block = d_short_data;
  // copy point by point (follows point-factor data layout)
  for (size_t i = 0; i < num_points; i++) {
    size_t block_code_bytes = code_len * sizeof(uint32_t);
    //        printf("short code Len %d in uint32_t", code_len);
    RAFT_CUDA_TRY(cudaMemcpyAsync(cur_block,
                                  d_all_short_codes + i * code_len,
                                  block_code_bytes,
                                  cudaMemcpyDeviceToDevice,
                                  stream_));
    raft::resource::sync_stream(handle_);
#if defined(HIGH_ACC_FAST_SCAN)
    float* block_fac = (float*)block_factor(cur_block, D);
    RAFT_CUDA_TRY(cudaMemcpyAsync(
      block_fac, d_all_factor_x2 + i, sizeof(float), cudaMemcpyDeviceToDevice, stream_));

    // debug
    //        float temp_float;
    //        RAFT_CUDA_TRY(cudaMemcpy(&temp_float,
    //                              d_all_factor_x2 + i,
    //                              sizeof(float), cudaMemcpyDeviceToHost));
    //        printf("factors: %f\n", temp_float);
    //        if (temp_float > 89623303555.0) {
    //            printf("What's wrong ????? \n");
    //        }

#else
    uint8_t* block_fac = block_factor(cur_block, D);
    float* cur_x2      = factor_x2(block_fac);
    float* cur_ip      = factor_ip(block_fac, FAST_SIZE);
    float* cur_sumxb   = factor_sumxb(block_fac, FAST_SIZE);
    float* cur_err     = factor_err(block_fac, FAST_SIZE);
    RAFT_CUDA_TRY(cudaMemcpyAsync(cur_x2,
                                  d_all_factor_x2 + i * FAST_SIZE,
                                  FAST_SIZE * sizeof(float),
                                  cudaMemcpyDeviceToDevice,
                                  stream_));
    RAFT_CUDA_TRY(cudaMemcpyAsync(cur_ip,
                                  d_all_factor_ip + i * FAST_SIZE,
                                  FAST_SIZE * sizeof(float),
                                  cudaMemcpyDeviceToDevice,
                                  stream_));
    RAFT_CUDA_TRY(cudaMemcpyAsync(cur_sumxb,
                                  d_all_factor_sumxb + i * FAST_SIZE,
                                  FAST_SIZE * sizeof(float),
                                  cudaMemcpyDeviceToDevice,
                                  stream_));
    RAFT_CUDA_TRY(cudaMemcpyAsync(cur_err,
                                  d_all_factor_err + i * FAST_SIZE,
                                  FAST_SIZE * sizeof(float),
                                  cudaMemcpyDeviceToDevice,
                                  stream_));
#endif
    raft::resource::sync_stream(handle_);
    cur_block = next_block(cur_block, code_len, FAST_SIZE, NUM_SHORT_FACTORS);
  }
#ifdef DEBUG_TIME
  RAFT_CUDA_TRY(cudaEventRecord(stop));
  RAFT_CUDA_TRY(cudaEventSynchronize(stop));
  RAFT_CUDA_TRY(cudaEventElapsedTime(&elapsed, start, stop));
  printf("Step 7 (Copy short codes and factor blocks): %f seconds\n", elapsed / 1000.0f);
#endif

  // 8. Free intermediate buffers.
#ifdef DEBUG_TIME
  RAFT_CUDA_TRY(cudaEventRecord(start));
#endif
  RAFT_CUDA_TRY(cudaFreeAsync(d_all_short_codes, stream_));
  RAFT_CUDA_TRY(cudaFreeAsync(d_all_factor_x2, stream_));
  RAFT_CUDA_TRY(cudaFreeAsync(d_all_factor_ip, stream_));
  RAFT_CUDA_TRY(cudaFreeAsync(d_all_factor_sumxb, stream_));
  RAFT_CUDA_TRY(cudaFreeAsync(d_all_factor_err, stream_));
  RAFT_CUDA_TRY(cudaFreeAsync(d_XP_norm, stream_));
  RAFT_CUDA_TRY(cudaFreeAsync(d_bin_XP, stream_));

  raft::resource::sync_stream(handle_);
#ifdef DEBUG_TIME
  RAFT_CUDA_TRY(cudaEventRecord(stop));
  RAFT_CUDA_TRY(cudaEventSynchronize(stop));
  RAFT_CUDA_TRY(cudaEventElapsedTime(&elapsed, start, stop));
  printf("Step 8 (Free intermediate buffers): %f seconds\n", elapsed / 1000.0f);
#endif

#ifdef DEBUG_TIME
  RAFT_CUDA_TRY(cudaEventDestroy(start));
  RAFT_CUDA_TRY(cudaEventDestroy(stop));
#endif
}

__inline__ __device__ float warpReduceSum(float v)
{
  for (int offset = 16; offset > 0; offset >>= 1)
    v += __shfl_down_sync(0xffffffff, v, offset);
  return v;
}

__inline__ __device__ float blockReduceSum(float v)
{
  __shared__ float shared[32];  // up to 1024 threads -> 32 warps
  int lane = threadIdx.x & 31;
  int wid  = threadIdx.x >> 5;

  v = warpReduceSum(v);
  if (lane == 0) shared[wid] = v;
  __syncthreads();

  float out = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.f;
  if (wid == 0) out = warpReduceSum(out);
  return out;
}

// One block per row; reduces across D
__global__ void RowwisePackedKernel(const float* __restrict__ d_centroid,  // [D]
                                    const int* __restrict__ d_bin_XP,      // [N*D], values in {0,1}
                                    const float* __restrict__ d_XP,        // [N*D], residuals
                                    size_t N,
                                    size_t D,
                                    float* __restrict__ d_out,  // [3*N], packed results
                                    float kConstEpsilon         // e.g. 1.9f
)
{
  size_t i = blockIdx.x;
  if (i >= N) return;

  float l2_sqr       = 0.f;
  float ip_resi_xucb = 0.f;
  float ip_cent_xucb = 0.f;
  float xu_sq        = 0.f;

  for (size_t d = threadIdx.x; d < D; d += blockDim.x) {
    float res = d_XP[i * D + d];
    float xu  = float(d_bin_XP[i * D + d]) - 0.5f;  // ((1<<1)-1)/2 = 0.5
    float c   = d_centroid[d];

    l2_sqr += res * res;
    ip_resi_xucb += res * xu;
    ip_cent_xucb += c * xu;
    xu_sq += xu * xu;
  }

  l2_sqr       = blockReduceSum(l2_sqr);
  ip_resi_xucb = blockReduceSum(ip_resi_xucb);
  ip_cent_xucb = blockReduceSum(ip_cent_xucb);
  xu_sq        = blockReduceSum(xu_sq);

  if (threadIdx.x == 0) {
    // If denom is exactly 0, set to +inf (requested)
    if (ip_resi_xucb == 0.0f) ip_resi_xucb = INFINITY;

    float l2_norm = sqrtf(fmaxf(l2_sqr, 0.f));
    float denom   = ip_resi_xucb;

    float fadd     = l2_sqr + 2.f * l2_sqr * (ip_cent_xucb / denom);
    float frescale = -2.f * l2_sqr / denom;

    float ratio     = (l2_sqr * xu_sq) / (denom * denom);
    float inner     = (ratio - 1.f) / fmaxf(float(D - 1), 1.f);
    inner           = fmaxf(inner, 0.f);
    float tmp_error = l2_norm * kConstEpsilon * sqrtf(inner);
    float ferr      = 2.f * tmp_error;

    // pack: [f_add, f_rescale, f_error]
    size_t base     = 3 * i;
    d_out[base + 0] = fadd;
    d_out[base + 1] = frescale;
    d_out[base + 2] = ferr;
  }
}
#ifdef DEBUG_BATCH_CONSTRUCT
int debug_first_cluster_count_4 = 0;
#endif
// compute factors for batch data
void compute_factors_packed(const float* d_centroid,  // [D]
                            const int* d_bin_XP,      // [N*D]
                            const float* d_XP,        // [N*D]
                            size_t N,
                            size_t D,
                            float kConstEpsilon,  // e.g., 1.9f
                            float* d_out,         // device array size = 3*N
                            cudaStream_t stream,
                            int threads_per_block = 256)
{
  dim3 grid((unsigned)N);
  dim3 block(threads_per_block);
  RowwisePackedKernel<<<grid, block, 0, stream>>>(
    d_centroid, d_bin_XP, d_XP, N, D, d_out, kConstEpsilon);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
#ifdef DEBUG_BATCH_CONSTRUCT
  if (debug_first_cluster_count_4 == 0) {
    debug_first_cluster_count_4++;
    std::cout << "First vector of the first cluster's short factors:\n";
    float h_bin_XP[3];
    RAFT_CUDA_TRY(
      cudaMemcpyAsync(h_bin_XP, d_out, 3 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

    std::cout << "f_add = " << h_bin_XP[0] << std::endl;
    std::cout << "f_rescale = " << h_bin_XP[1] << std::endl;
    std::cout << "f_error = " << h_bin_XP[2] << std::endl;
  }
#endif
}

// One block per row; reduces across D
// This is for ex codes
__global__ void RowwisePackedKernelBatch(const float* __restrict__ d_centroid,  // [D]
                                         const int* __restrict__ d_xu,  // [N*D], values in {0,1}??
                                         const float* __restrict__ d_XP,  // [N*D], residuals
                                         const float* __restrict__ d_ipnorm_inv,  //
                                         size_t N,
                                         size_t D,
                                         float* __restrict__ d_out,  // [2*N], packed results
                                         float kConstEpsilon,        // e.g. 1.9f
                                         size_t ex_bits)
{
  size_t i = blockIdx.x;
  if (i >= N) return;

  float l2_sqr       = 0.f;
  float ip_resi_xucb = 0.f;
  float ip_cent_xucb = 0.f;
  float xu_sq        = 0.f;

  for (size_t d = threadIdx.x; d < D; d += blockDim.x) {
    float res  = d_XP[i * D + d];
    int xu_pre = d_xu[i * D + d];
    xu_pre += static_cast<int>(res >= 0) << ex_bits;
    float xu = float(xu_pre) - (static_cast<float>(1 << ex_bits) - 0.5F);
    float c  = d_centroid[d];

    l2_sqr += res * res;
    ip_resi_xucb += res * xu;
    ip_cent_xucb += c * xu;
    xu_sq += xu * xu;
  }

  l2_sqr       = blockReduceSum(l2_sqr);
  ip_resi_xucb = blockReduceSum(ip_resi_xucb);
  ip_cent_xucb = blockReduceSum(ip_cent_xucb);
  xu_sq        = blockReduceSum(xu_sq);

  if (threadIdx.x == 0) {
    // If denom is exactly 0, set to +inf (requested)
    if (ip_resi_xucb == 0.0f) ip_resi_xucb = INFINITY;

    float l2_norm = sqrtf(fmaxf(l2_sqr, 0.f));
    float denom   = ip_resi_xucb;

    float fadd_ex = l2_sqr + 2.f * l2_sqr * ip_cent_xucb / denom;
#ifdef DEBUG_BATCH_CONSTRUCT
    if (fadd_ex < -40) {
      printf("f_add_ex < 0! f_add_ex: %f, l2_sqr: %f, ip_cent_xucb: %f, denom: %f\n",
             fadd_ex,
             l2_sqr,
             ip_cent_xucb,
             denom);
    }
#endif
    float frescale_ex = -2.f * l2_norm * d_ipnorm_inv[i];

    float ratio     = (l2_sqr * xu_sq) / (denom * denom);
    float inner     = (ratio - 1.f) / fmaxf(float(D - 1), 1.f);
    inner           = fmaxf(inner, 0.f);
    float tmp_error = l2_norm * kConstEpsilon * sqrtf(inner);
    //        float ferr_ex = 2.f * tmp_error;

    // pack: [f_add, f_rescale, f_error]
    size_t base     = 2 * i;
    d_out[base + 0] = fadd_ex;
    d_out[base + 1] = frescale_ex;
    //        d_out[base + 2] = ferr_ex;
    // ferr_ex not used
  }
}

// compute factors for batch data
// This is used for ex_codes
#ifdef DEBUG_BATCH_CONSTRUCT

int debug_first_cluster_count_3 = 0;
#endif

void compute_factors_packed_batch(const float* d_centroid,  // [D]
                                  const int* d_xu,          // [N*D]
                                  const float* d_XP,        // [N*D]
                                  const float* ipnorm_inv,  // [D]
                                  size_t N,
                                  size_t D,
                                  float kConstEpsilon,  // e.g., 1.9f
                                  float* d_out,         // device array size = 2*N
                                  size_t ex_bits,
                                  cudaStream_t stream,
                                  int threads_per_block)
{
  dim3 grid((unsigned)N);
  dim3 block(threads_per_block);
  RowwisePackedKernelBatch<<<grid, block, 0, stream>>>(
    d_centroid, d_xu, d_XP, ipnorm_inv, N, D, d_out, kConstEpsilon, ex_bits);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
#ifdef DEBUG_BATCH_CONSTRUCT
  if (debug_first_cluster_count_3 == 0) {
    debug_first_cluster_count_3++;
    std::cout << "First vector of the first cluster's ex factors:\n";
    float h_bin_XP[2];
    RAFT_CUDA_TRY(
      cudaMemcpyAsync(h_bin_XP, d_out, 2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

    std::cout << "f_ex_add = " << h_bin_XP[0] << std::endl;
    std::cout << "f_ex_rescale = " << h_bin_XP[1] << std::endl;

    RAFT_CUDA_TRY(
      cudaMemcpyAsync(h_bin_XP, d_out + 2, 2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

    std::cout << "Second vector of the first cluster's ex factors:\n";
    std::cout << "f_ex_add = " << h_bin_XP[0] << std::endl;
    std::cout << "f_ex_rescale = " << h_bin_XP[1] << std::endl;
  }
#endif
}

#ifdef DEBUG_BATCH_CONSTRUCT
int debug_first_cluster_count_2 = 0;
#endif

void DataQuantizerGPU::quantize_batch(const float* d_data,
                                      const float* d_centroid,
                                      const PID* d_IDs,
                                      size_t num_points,
                                      const RotatorGPU& rotator,
                                      uint32_t* d_short_data,
                                      float* d_short_data_factors,
                                      uint8_t* d_long_code,
                                      float* d_ex_factor,
                                      float* d_rotated_c) const
{
#ifdef DEBUG_TIME
  cudaEvent_t start, stop;
  float elapsed;
  RAFT_CUDA_TRY(cudaEventCreate(&start));
  RAFT_CUDA_TRY(cudaEventCreate(&stop));
#endif

  // 1. Data Transformation:
#ifdef DEBUG_TIME
  RAFT_CUDA_TRY(cudaEventRecord(start));
#endif
  float* d_XP_norm = nullptr;
  int* d_bin_XP    = nullptr;
  float* d_XP;
  RAFT_CUDA_TRY(cudaMallocAsync((void**)&d_XP_norm, num_points * D * sizeof(float), stream_));
  RAFT_CUDA_TRY(cudaMallocAsync((void**)&d_bin_XP, num_points * D * sizeof(int), stream_));
  RAFT_CUDA_TRY(cudaMallocAsync((void**)&d_XP, num_points * D * sizeof(float), stream_));
  data_transformation_batch(
    d_data, d_centroid, d_IDs, num_points, rotator, d_rotated_c, d_XP_norm, d_bin_XP, d_XP);
#ifdef DEBUG_TIME
  RAFT_CUDA_TRY(cudaEventRecord(stop));
  RAFT_CUDA_TRY(cudaEventSynchronize(stop));
  RAFT_CUDA_TRY(cudaEventElapsedTime(&elapsed, start, stop));
  printf("Step 1 (Data Transformation): %f seconds\n", elapsed / 1000.0f);
#endif

#ifdef DEBUG_BATCH_CONSTRUCT
  if (debug_first_cluster_count_2 == 0) {
    debug_first_cluster_count_2++;
    std::cout << "First vector of the first cluster's first 20 short codes:\n";
    int h_bin_XP[20];
    RAFT_CUDA_TRY(
      cudaMemcpyAsync(h_bin_XP, d_bin_XP, 20 * sizeof(int), cudaMemcpyDeviceToHost, stream_));
    raft::resource::sync_stream(handle_);

    // Print them
    for (int i = 0; i < 20; i++) {
      std::cout << "d_bin_XP[" << i << "] = " << h_bin_XP[i] << std::endl;
    }
  }
#endif

  // 2. Compute total blocks for factors and short codes.
#ifdef DEBUG_TIME
  RAFT_CUDA_TRY(cudaEventRecord(start));
#endif
//    size_t total_blocks = div_rd_up(num_points, FAST_SIZE);
#ifdef DEBUG_TIME
  RAFT_CUDA_TRY(cudaEventRecord(stop));
  RAFT_CUDA_TRY(cudaEventSynchronize(stop));
  RAFT_CUDA_TRY(cudaEventElapsedTime(&elapsed, start, stop));
  printf("Step 2 (Compute total blocks): %f seconds\n", elapsed / 1000.0f);
#endif

  // 3. Allocate intermediate buffers on device.
#ifdef DEBUG_TIME
  RAFT_CUDA_TRY(cudaEventRecord(start));
#endif
  size_t code_len          = short_code_length();  // from quantizer parameters.
  size_t short_codes_bytes = code_len * num_points * sizeof(uint32_t);
  //    uint32_t* d_all_short_codes = nullptr;
  //    RAFT_CUDA_TRY(cudaMalloc((void**) &d_all_short_codes, short_codes_bytes));

  size_t factor_bytes = num_points * sizeof(float) * 3;  // we have 3 factors for batch data
//    float* d_all_data_factors = nullptr;
//    cudaMalloc((void**) &d_all_data_factors, factor_bytes);
//    float* d_all_factor_x2 = nullptr;
//    float* d_all_factor_ip = nullptr;
//    float* d_all_factor_sumxb = nullptr;
//    float* d_all_factor_err = nullptr;
//    RAFT_CUDA_TRY(cudaMalloc((void**) &d_all_factor_x2, factor_bytes));
//    RAFT_CUDA_TRY(cudaMalloc((void**) &d_all_factor_ip, factor_bytes));
//    RAFT_CUDA_TRY(cudaMalloc((void**) &d_all_factor_sumxb, factor_bytes));
//    RAFT_CUDA_TRY(cudaMalloc((void**) &d_all_factor_err, factor_bytes));
#ifdef DEBUG_TIME
  RAFT_CUDA_TRY(cudaEventRecord(stop));
  RAFT_CUDA_TRY(cudaEventSynchronize(stop));
  RAFT_CUDA_TRY(cudaEventElapsedTime(&elapsed, start, stop));
  printf("Step 3 (Allocate intermediate buffers): %f seconds\n", elapsed / 1000.0f);
#endif

  // 4. Compute RaBitQ quantization codes.
#ifdef DEBUG_TIME
  RAFT_CUDA_TRY(cudaEventRecord(start));
#endif
  rabitq_codes(d_bin_XP, d_short_data, num_points);
#ifdef DEBUG_TIME
  RAFT_CUDA_TRY(cudaEventRecord(stop));
  RAFT_CUDA_TRY(cudaEventSynchronize(stop));
  RAFT_CUDA_TRY(cudaEventElapsedTime(&elapsed, start, stop));
  printf("Step 4 (Compute RaBitQ codes): %f seconds\n", elapsed / 1000.0f);
#endif

  // 5. Compute RaBitQ factors.
#ifdef DEBUG_TIME
  RAFT_CUDA_TRY(cudaEventRecord(start));
#endif
  compute_factors_packed(
    d_rotated_c, d_bin_XP, d_XP, num_points, D, 1.9, d_short_data_factors, stream_);
#ifdef DEBUG_TIME
  RAFT_CUDA_TRY(cudaEventRecord(stop));
  RAFT_CUDA_TRY(cudaEventSynchronize(stop));
  RAFT_CUDA_TRY(cudaEventElapsedTime(&elapsed, start, stop));
  printf("Step 5 (Compute re-ranking factors): %f seconds\n", elapsed / 1000.0f);
#endif

  // 6. Compute ExRaBitQ quantization codes.
#ifdef DEBUG_TIME
  RAFT_CUDA_TRY(cudaEventRecord(start));
#endif
  exrabitq_codes_batch(
    d_bin_XP, d_XP_norm, d_XP, d_long_code, d_ex_factor, d_rotated_c, num_points);
#ifdef DEBUG_TIME
  RAFT_CUDA_TRY(cudaEventRecord(stop));
  RAFT_CUDA_TRY(cudaEventSynchronize(stop));
  RAFT_CUDA_TRY(cudaEventElapsedTime(&elapsed, start, stop));
  printf("Step 6 (Compute ExRaBitQ codes): %f seconds\n", elapsed / 1000.0f);
#endif

  // 7. Copy short codes and factor blocks into final output d_short_data.
#ifdef DEBUG_TIME
  RAFT_CUDA_TRY(cudaEventRecord(start));
#endif
  uint32_t* cur_block = d_short_data;
  // copy point by point (follows point-factor data layout)
  for (size_t i = 0; i < num_points; i++) {
    size_t block_code_bytes = code_len * sizeof(uint32_t);
//        RAFT_CUDA_TRY(cudaMemcpy(cur_block,
//                              d_all_short_codes + i * code_len,
//                              block_code_bytes, cudaMemcpyDeviceToDevice));
#if defined(HIGH_ACC_FAST_SCAN)
    float* block_fac = (float*)block_factor(cur_block, D);
    //        RAFT_CUDA_TRY(cudaMemcpy(block_fac,
    //                              d_all_factor_x2 + i,
    //                              sizeof(float), cudaMemcpyDeviceToDevice));

#else
    uint8_t* block_fac = block_factor(cur_block, D);
    float* cur_x2      = factor_x2(block_fac);
    float* cur_ip      = factor_ip(block_fac, FAST_SIZE);
    float* cur_sumxb   = factor_sumxb(block_fac, FAST_SIZE);
    float* cur_err     = factor_err(block_fac, FAST_SIZE);
    RAFT_CUDA_TRY(cudaMemcpyAsync(cur_x2,
                                  d_all_factor_x2 + i * FAST_SIZE,
                                  FAST_SIZE * sizeof(float),
                                  cudaMemcpyDeviceToDevice,
                                  stream_));
    RAFT_CUDA_TRY(cudaMemcpyAsync(cur_ip,
                                  d_all_factor_ip + i * FAST_SIZE,
                                  FAST_SIZE * sizeof(float),
                                  cudaMemcpyDeviceToDevice,
                                  stream_));
    RAFT_CUDA_TRY(cudaMemcpyAsync(cur_sumxb,
                                  d_all_factor_sumxb + i * FAST_SIZE,
                                  FAST_SIZE * sizeof(float),
                                  cudaMemcpyDeviceToDevice,
                                  stream_));
    RAFT_CUDA_TRY(cudaMemcpyAsync(cur_err,
                                  d_all_factor_err + i * FAST_SIZE,
                                  FAST_SIZE * sizeof(float),
                                  cudaMemcpyDeviceToDevice,
                                  stream_));
#endif
    raft::resource::sync_stream(handle_);
    cur_block = next_block(cur_block, code_len, FAST_SIZE, NUM_SHORT_FACTORS);
  }
#ifdef DEBUG_TIME
  RAFT_CUDA_TRY(cudaEventRecord(stop));
  RAFT_CUDA_TRY(cudaEventSynchronize(stop));
  RAFT_CUDA_TRY(cudaEventElapsedTime(&elapsed, start, stop));
  printf("Step 7 (Copy short codes and factor blocks): %f seconds\n", elapsed / 1000.0f);
#endif

  // 8. Free intermediate buffers.
#ifdef DEBUG_TIME
  RAFT_CUDA_TRY(cudaEventRecord(start));
#endif
  RAFT_CUDA_TRY(cudaFreeAsync(d_XP_norm, stream_));
  RAFT_CUDA_TRY(cudaFreeAsync(d_bin_XP, stream_));
#ifdef DEBUG_TIME
  RAFT_CUDA_TRY(cudaEventRecord(stop));
  RAFT_CUDA_TRY(cudaEventSynchronize(stop));
  RAFT_CUDA_TRY(cudaEventElapsedTime(&elapsed, start, stop));
  printf("Step 8 (Free intermediate buffers): %f seconds\n", elapsed / 1000.0f);
#endif

#ifdef DEBUG_TIME
  RAFT_CUDA_TRY(cudaEventDestroy(start));
  RAFT_CUDA_TRY(cudaEventDestroy(stop));
#endif
}
  void DataQuantizerGPU::alloc_buffers(size_t num_points) {

  const int64_t size_norm = static_cast<int64_t>(num_points) * D;
  const int64_t size_bin  = static_cast<int64_t>(num_points) * D;
  const int64_t size_xp   = static_cast<int64_t>(num_points + 1) * D;

  // Overwrite RAFT device vectors with new allocations
  d_XP_norm      = raft::make_device_vector<float, int64_t>(handle_, size_norm);
  d_bin_XP       = raft::make_device_vector<int, int64_t>(handle_, size_bin);
  d_XP           = raft::make_device_vector<float, int64_t>(handle_, size_xp);
  d_X_and_C_pad  = raft::make_device_vector<float, int64_t>(handle_, size_xp);

}


}  // namespace cuvs::neighbors::ivf_rabitq::detail


