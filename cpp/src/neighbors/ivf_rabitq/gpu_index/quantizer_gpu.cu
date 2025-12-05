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

  // Launch kernel: one block per data point, each with (D/64) threads.
  dim3 grid(num_points);
  dim3 block(blocks_per_point);
  pack_binary_kernel<<<grid, block, 0, stream_>>>(d_bin_XP, d_packed_code, num_points, D);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

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
  // First loop: compute max_o.
  const double eps = 1e-5;
  const int n_enum = 10;
  double max_o     = -1.0;
  for (int i = 0; i < D; i++) {
    double val = o_prime[i];
    if (val > max_o) max_o = val;
  }

  double t_start = (((1 << EX_BITS) - 1) / 3.0) / max_o;
  double t_end   = ((((1 << EX_BITS) - 1) + n_enum)) / max_o;

  int cur_o_bar[MAX_D];  // assume D <= MAX_D
  double sqr_denom = D * 0.25;
  double numerator = 0.0;

  // Second loop: initial computation of cur_o_bar, updating sqr_denom and numerator.
  for (int i = 0; i < D; i++) {
    cur_o_bar[i] = int(t_start * o_prime[i] + eps);
    sqr_denom += cur_o_bar[i] * cur_o_bar[i] + cur_o_bar[i];
    numerator += (cur_o_bar[i] + 0.5) * o_prime[i];
  }

  double max_ip    = 0.0;
  double t_val     = 0.0;
  bool improvement = true;

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

  sqr_denom = D * 0.25;
  numerator = 0.0;
  int o_bar[MAX_D];

  // Third loop: compute final o_bar, update sqr_denom and numerator.
  for (int i = 0; i < D; i++) {
    o_bar[i] = int(t_val * o_prime[i] + eps);
    if (o_bar[i] >= (1 << EX_BITS)) o_bar[i] = (1 << EX_BITS) - 1;
    sqr_denom += o_bar[i] * o_bar[i] + o_bar[i];
    numerator += (o_bar[i] + 0.5) * o_prime[i];
  }

  *ip_norm = 1.0f / (float)numerator;
  if (!isfinite(*ip_norm)) { *ip_norm = 1.0f; }

  // Fourth loop: assign codes.
  for (int i = 0; i < D; i++) {
    code[i] = (uint8_t)o_bar[i];
  }
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
  // 1. Data Transformation:
  float* d_XP_norm = nullptr;
  int* d_bin_XP    = nullptr;
  RAFT_CUDA_TRY(cudaMallocAsync((void**)&d_XP_norm, num_points * D * sizeof(float), stream_));
  RAFT_CUDA_TRY(cudaMallocAsync((void**)&d_bin_XP, num_points * D * sizeof(int), stream_));
  raft::resource::sync_stream(handle_);
  data_transformation(
    d_data, d_centroid, d_IDs, num_points, rotator, d_rotated_c, d_XP_norm, d_bin_XP);

  // 2. (skipped) Compute total blocks for factors and short codes.

  // 3. Allocate intermediate buffers on device.
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

  // 4. Compute RaBitQ quantization codes.
  rabitq_codes(d_bin_XP, d_all_short_codes, num_points);

  // 5. Compute re-ranking factors.
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

  // 6. Compute ExRaBitQ quantization codes.
  exrabitq_codes(d_bin_XP, d_XP_norm, d_long_code, d_ex_factor, d_all_factor_x2, num_points);

  // 7. Copy short codes and factor blocks into final output d_short_data.
  uint32_t* cur_block = d_short_data;
  // copy point by point (follows point-factor data layout)
  for (size_t i = 0; i < num_points; i++) {
    size_t block_code_bytes = code_len * sizeof(uint32_t);
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

  // 8. Free intermediate buffers.
  RAFT_CUDA_TRY(cudaFreeAsync(d_all_short_codes, stream_));
  RAFT_CUDA_TRY(cudaFreeAsync(d_all_factor_x2, stream_));
  RAFT_CUDA_TRY(cudaFreeAsync(d_all_factor_ip, stream_));
  RAFT_CUDA_TRY(cudaFreeAsync(d_all_factor_sumxb, stream_));
  RAFT_CUDA_TRY(cudaFreeAsync(d_all_factor_err, stream_));
  RAFT_CUDA_TRY(cudaFreeAsync(d_XP_norm, stream_));
  RAFT_CUDA_TRY(cudaFreeAsync(d_bin_XP, stream_));

  raft::resource::sync_stream(handle_);
}

void DataQuantizerGPU::alloc_buffers(size_t num_points)
{
  const int64_t size_norm = static_cast<int64_t>(num_points) * D;
  const int64_t size_bin  = static_cast<int64_t>(num_points) * D;
  const int64_t size_xp   = static_cast<int64_t>(num_points + 1) * D;

  // Overwrite RAFT device vectors with new allocations
  d_XP_norm     = raft::make_device_vector<float, int64_t>(handle_, size_norm);
  d_bin_XP      = raft::make_device_vector<int, int64_t>(handle_, size_bin);
  d_XP          = raft::make_device_vector<float, int64_t>(handle_, size_xp);
  d_X_and_C_pad = raft::make_device_vector<float, int64_t>(handle_, size_xp);
}

}  // namespace cuvs::neighbors::ivf_rabitq::detail
