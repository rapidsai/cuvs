/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Created by Stardust on 3/24/25.
//

#include <cuvs/neighbors/ivf_rabitq/gpu_index/rotator_gpu.cuh>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/detail/qr.cuh>
#include <raft/linalg/gemm.hpp>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_rt_essentials.hpp>

namespace cuvs::neighbors::ivf_rabitq::detail {

RotatorGPU::RotatorGPU(raft::resources const& handle, uint32_t dim)
  : handle_(const_cast<raft::resources*>(&handle)),
    stream_(raft::resource::get_cuda_stream(*handle_))
{
  // keep track of cuda stream
  // Compute padded dimension
  // A padding function that rounds up to a multiple of 64.
  auto rd_up_to_multiple_of = [](uint32_t dim, uint32_t mult) -> size_t {
    return ((dim + mult - 1) / mult) * mult;
  };
  D = rd_up_to_multiple_of(dim, 64);
  // Create a random matrix (size D x D)
  RAFT_CUDA_TRY(cudaMallocAsync((void**)&d_P, sizeof(float) * D * D, stream_));
  raft::random::RngState rng(7ULL);
  raft::random::normal(handle, rng, d_P, D * D, 0.0f, 1.0f);
  // Compute the random rotation matrix in-place
  raft::linalg::detail::qrGetQ_inplace(handle, d_P, D, D, stream_);
}

RotatorGPU::~RotatorGPU()
{
  if (d_P) { RAFT_CUDA_TRY(cudaFreeAsync(d_P, stream_)); }
}

RotatorGPU& RotatorGPU::operator=(const RotatorGPU& other)
{
  if (this != &other) {
    D       = other.D;
    handle_ = other.handle_;
    stream_ = other.stream_;
    // Firstly free it in case not dimension not match
    if (d_P) { RAFT_CUDA_TRY(cudaFreeAsync(d_P, stream_)); }
    RAFT_CUDA_TRY(cudaMallocAsync((void**)&d_P, sizeof(float) * D * D, stream_));
    // Copy the rotation matrix from the other object.
    RAFT_CUDA_TRY(
      cudaMemcpyAsync(d_P, other.d_P, sizeof(float) * D * D, cudaMemcpyDeviceToDevice, stream_));
    raft::resource::sync_stream(*handle_);
  }
  return *this;
}

size_t RotatorGPU::size() const { return D; }

void RotatorGPU::load(std::ifstream& input)
{
  float* hostP = new float[D * D];
  for (size_t i = 0; i < D * D; ++i) {
    input.read(reinterpret_cast<char*>(&hostP[i]), sizeof(float));
  }
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(d_P, hostP, sizeof(float) * D * D, cudaMemcpyHostToDevice, stream_));
  raft::resource::sync_stream(*handle_);
  delete[] hostP;
}

void RotatorGPU::save(std::ofstream& output) const
{
  float* hostP = new float[D * D];
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(hostP, d_P, sizeof(float) * D * D, cudaMemcpyDeviceToHost, stream_));
  raft::resource::sync_stream(*handle_);
  for (size_t i = 0; i < D * D; ++i) {
    output.write(reinterpret_cast<char*>(&hostP[i]), sizeof(float));
  }
  delete[] hostP;
}

// Rotate the matrix A and store the result in RAND_A on the GPU.
// A and RAND_A are assumed to be stored in row-major order.
// A is of size N x D and P is of size D x D, so the result is N x D.
// This function uses cuBLAS to perform the matrix multiplication.
// Do note that d_P is generated from Eigen, and it is column major!!!!!!!!??
void RotatorGPU::rotate(const float* d_A, float* d_RAND_A, size_t N) const
{
  //    cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);
  // cuBLAS assumes column-major storage by default. Since our matrices are in row-major order,
  // we can perform the multiplication as:
  //   RAND_A^T = P^T * A^T
  // which is equivalent to RAND_A = A * P, if we interpret the data as row-major.
  // Note that in Cublas it is RAND_A^T in column major, which is what we want in row-major
  // Here, we use the RAFT wrapper for gemm.
  raft::linalg::gemm(
    *handle_,
    raft::make_device_matrix_view<float, int64_t, raft::col_major>(const_cast<float*>(d_P), D, D),
    raft::make_device_matrix_view<float, int64_t, raft::col_major>(const_cast<float*>(d_A), D, N),
    raft::make_device_matrix_view<float, int64_t, raft::col_major>(d_RAND_A, D, N));
  raft::resource::sync_stream(*handle_);
}

}  // namespace cuvs::neighbors::ivf_rabitq::detail
