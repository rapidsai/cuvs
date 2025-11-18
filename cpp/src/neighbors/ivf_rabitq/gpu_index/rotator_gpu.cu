/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Created by Stardust on 3/24/25.
//

#include <Eigen/Dense>
#include <cuvs/neighbors/ivf_rabitq/gpu_index/rotator_gpu.cuh>

#include <raft/core/cublas_macros.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cuda_rt_essentials.hpp>

RotatorGPU::RotatorGPU(raft::resources const& handle, uint32_t dim)
{
  // keep track of cuda stream
  // note that the cuBlas handle `m_handle` is not set to `m_stream`, or the results would end up
  // being incorrect
  m_stream = raft::resource::get_cuda_stream(handle);
  // Compute padded dimension
  // A padding function that rounds up to a multiple of 64.
  auto rd_up_to_multiple_of = [](uint32_t dim, uint32_t mult) -> size_t {
    return ((dim + mult - 1) / mult) * mult;
  };
  D = rd_up_to_multiple_of(dim, 64);
  // Create a random matrix (size D x D) using Eigen.
#ifdef DEBUG_BATCH_CONSTRUCT
  srand(1);  // fix seed
#endif

  //    FloatRowMat RAND = FloatRowMat::Random(D, D);
  FloatRowMat RAND = random_gaussian_matrix<float>(D, D);

  //    printf("1st element: %f\n", RAND(0, 0));
  // Householder QR decomposition.
  Eigen::HouseholderQR<FloatRowMat> qr(RAND);

  // Get the orthonormal Q matrix.
  FloatRowMat Q = qr.householderQ();
  // Set P to be the transpose of Q (which is the inverse of Q, since Q is orthogonal).
  FloatRowMat P = Q.transpose();

  float* hostP = new float[D * D];

  std::memcpy(hostP, P.data(), sizeof(float) * D * D);
  RAFT_CUDA_TRY(cudaMallocAsync((void**)&d_P, sizeof(float) * D * D, m_stream));
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(d_P, hostP, sizeof(float) * D * D, cudaMemcpyHostToDevice, m_stream));
  RAFT_CUDA_TRY(cudaStreamSynchronize(m_stream));
  delete[] hostP;

  RAFT_CUBLAS_TRY(cublasCreate(&m_handle));
}

RotatorGPU::~RotatorGPU()
{
  if (d_P) { RAFT_CUDA_TRY(cudaFreeAsync(d_P, m_stream)); }
  RAFT_CUBLAS_TRY(cublasDestroy(m_handle));
}

RotatorGPU& RotatorGPU::operator=(const RotatorGPU& other)
{
  if (this != &other) {
    D        = other.D;
    m_stream = other.m_stream;
    RAFT_CUBLAS_TRY(cublasCreate(&m_handle));
    // Firstly free it in case not dimension not match
    if (d_P) { RAFT_CUDA_TRY(cudaFreeAsync(d_P, m_stream)); }
    RAFT_CUDA_TRY(cudaMallocAsync((void**)&d_P, sizeof(float) * D * D, m_stream));
    // Copy the rotation matrix from the other object.
    RAFT_CUDA_TRY(
      cudaMemcpyAsync(d_P, other.d_P, sizeof(float) * D * D, cudaMemcpyDeviceToDevice, m_stream));
    RAFT_CUDA_TRY(cudaStreamSynchronize(m_stream));
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
    cudaMemcpyAsync(d_P, hostP, sizeof(float) * D * D, cudaMemcpyHostToDevice, m_stream));
  RAFT_CUDA_TRY(cudaStreamSynchronize(m_stream));
  delete[] hostP;
}

void RotatorGPU::save(std::ofstream& output) const
{
  float* hostP = new float[D * D];
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(hostP, d_P, sizeof(float) * D * D, cudaMemcpyDeviceToHost, m_stream));
  RAFT_CUDA_TRY(cudaStreamSynchronize(m_stream));
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
  // Here, we use cublasSgemm with appropriate parameters.
  const float alpha = 1.0f;
  const float beta  = 0.0f;
  RAFT_CUBLAS_TRY(cublasSgemm(
    m_handle, CUBLAS_OP_N, CUBLAS_OP_N, D, N, D, &alpha, d_P, D, d_A, D, &beta, d_RAND_A, D));
}
