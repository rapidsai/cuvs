/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Created by Stardust on 3/24/25.
//

#pragma once

#include <raft/core/device_mdarray.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resources.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cstdint>
#include <cuvs/neighbors/ivf_rabitq/defines.hpp>
#include <cuvs/neighbors/ivf_rabitq/utils/utils_cuda.cuh>
#include <fstream>

namespace cuvs::neighbors::ivf_rabitq::detail {

// The RotatorGPU class holds a rotation matrix (P) on the GPU. The matrix is computed
// on the CPU (using Eigen, similar to your CPU code) and then copied to device memory.
// The rotate() function uses cuBLAS to compute the product: RAND_A = A * P.
// It is assumed that A and RAND_A reside in GPU memory.
class RotatorGPU {
 public: /**
          * @brief Construct a new RotatorGPU object.
          * @param dim The original dimension; the padded dimension D is computed as
          * rd_up_to_multiple_of(dim, 64).
          *
          * The constructor generates a random rotation matrix on the CPU (using Eigen) and then
          * copies it into                    device memory in column-major order.
          */
  explicit RotatorGPU(raft::resources const& handle, uint32_t dim);

  // Disable copy assignment
  RotatorGPU& operator=(const RotatorGPU& other) = delete;

  size_t size() const;

  /**
   * @brief Load the rotation matrix from a file.
   * @param input Input stream (the file stores the matrix in row-major order).
   *
   * The function reads the D×D matrix from the file, transposes it into column-major order,
   * and copies it into device memory.
   */
  void load(std::ifstream& input);

  /**
   * @brief Save the rotation matrix to a file.
   * @param handle Resource handle
   * @param output Output stream.
   *
   * The function copies the rotation matrix from device memory, transposes it from column-major to
   * row-major, and writes it to the file.
   */
  void save(std::ofstream& output) const;

  // Rotate matrix A and store the result in RAND_A.
  // A and RAND_A are device pointers representing matrices of size N x D.
  // This function computes: RAND_A = A * P using cuBLAS.
  void rotate(const float* d_A, float* d_RAND_A, size_t N) const;

 private:
  raft::resources const& handle_;  // reusable resource handle
  rmm::cuda_stream_view stream_;   // CUDA stream obtained from handle_
  size_t D;                        // Padded dimension
  raft::device_matrix<float, uint32_t, raft::row_major> rotation_matrix_;  // Rotation matrix P
};

}  // namespace cuvs::neighbors::ivf_rabitq::detail
