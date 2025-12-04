/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Created by Stardust on 3/24/25.
//

#include "rotator_gpu.cuh"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/detail/qr.cuh>
#include <raft/linalg/gemm.hpp>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_rt_essentials.hpp>

namespace cuvs::neighbors::ivf_rabitq::detail {

RotatorGPU::RotatorGPU(raft::resources const& handle, uint32_t dim) : handle_(handle)
{
  // keep track of cuda stream
  // Compute padded dimension
  // A padding function that rounds up to a multiple of 64.
  auto rd_up_to_multiple_of = [](uint32_t dim, uint32_t mult) -> size_t {
    return ((dim + mult - 1) / mult) * mult;
  };
  D = rd_up_to_multiple_of(dim, 64);
  // Create a random matrix (size D x D)
  rotation_matrix_ = raft::make_device_matrix<float, int64_t, raft::row_major>(handle_, D, D);
  raft::random::RngState rng(7ULL);
  raft::random::normal(handle, rng, rotation_matrix_.data_handle(), D * D, 0.0f, 1.0f);
  // Compute the random rotation matrix in-place
  raft::linalg::detail::qrGetQ_inplace(handle, rotation_matrix_.data_handle(), D, D, stream_);
}

size_t RotatorGPU::size() const { return D; }

void RotatorGPU::load(std::ifstream& input)
{
  auto host_buf = raft::make_host_vector<float, int64_t>(D * D);
  for (size_t i = 0; i < D * D; ++i) {
    input.read(reinterpret_cast<char*>(&host_buf(i)), sizeof(float));
  }
  raft::copy(rotation_matrix_.data_handle(), host_buf.data_handle(), D * D, stream_);
  raft::resource::sync_stream(handle_);
}

void RotatorGPU::save(std::ofstream& output) const
{
  auto host_buf = raft::make_host_vector<float, int64_t>(D * D);
  raft::copy(host_buf.data_handle(), rotation_matrix_.data_handle(), D * D, stream_);
  raft::resource::sync_stream(handle_);
  for (size_t i = 0; i < D * D; ++i) {
    output.write(reinterpret_cast<char*>(&host_buf(i)), sizeof(float));
  }
}

// Rotate the matrix A and store the result in RAND_A on the GPU.
// A and RAND_A are assumed to be stored in row-major order.
// A is of size N x D and P is of size D x D, so the result is N x D.
// This function uses cuBLAS to perform the matrix multiplication.
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
    handle_,
    raft::make_device_matrix_view<float, int64_t, raft::col_major>(
      const_cast<float*>(rotation_matrix_.data_handle()), D, D),
    raft::make_device_matrix_view<float, int64_t, raft::col_major>(const_cast<float*>(d_A), D, N),
    raft::make_device_matrix_view<float, int64_t, raft::col_major>(d_RAND_A, D, N));
  raft::resource::sync_stream(handle_);
}

}  // namespace cuvs::neighbors::ivf_rabitq::detail
