/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//
// Created by Stardust on 3/24/25.
//

#include <cuvs/neighbors/ivf_rabitq/gpu_index/rotator_gpu.cuh>
#include <cuvs/neighbors/ivf_rabitq/third/Eigen/Dense>

RotatorGPU::RotatorGPU(uint32_t dim)
{
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
  CUDA_CHECK(cudaMalloc((void**)&d_P, sizeof(float) * D * D));
  CUDA_CHECK(cudaMemcpy(d_P, hostP, sizeof(float) * D * D, cudaMemcpyHostToDevice));

  cublasStatus_t status = cublasCreate(&m_handle);
}

RotatorGPU::~RotatorGPU()
{
  if (d_P) { cudaFree(d_P); }
  cublasDestroy(m_handle);
}

RotatorGPU& RotatorGPU::operator=(const RotatorGPU& other)
{
  if (this != &other) {
    D = other.D;
    // Firstly free it in case not dimension not match
    if (d_P) { cudaFree(d_P); }
    cudaMalloc((void**)&d_P, sizeof(float) * D * D);
    // Copy the rotation matrix from the other object.
    cudaMemcpy(d_P, other.d_P, sizeof(float) * D * D, cudaMemcpyDeviceToDevice);
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
  CUDA_CHECK(cudaMemcpy(d_P, hostP, sizeof(float) * D * D, cudaMemcpyHostToDevice));
  delete[] hostP;
}

void RotatorGPU::save(std::ofstream& output) const
{
  float* hostP = new float[D * D];
  CUDA_CHECK(cudaMemcpy(hostP, d_P, sizeof(float) * D * D, cudaMemcpyDeviceToHost));
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
  const float alpha     = 1.0f;
  const float beta      = 0.0f;
  cublasStatus_t status = cublasSgemm(
    m_handle, CUBLAS_OP_N, CUBLAS_OP_N, D, N, D, &alpha, d_P, D, d_A, D, &beta, d_RAND_A, D);
  if (status != CUBLAS_STATUS_SUCCESS) {
    // jamxia edit
    // std::cerr << "cuBLAS sgemm failed" << std::endl;
    std::cerr << "cuBLAS sgemm failed with status: " << status << std::endl;
    exit(EXIT_FAILURE);
  }
}
