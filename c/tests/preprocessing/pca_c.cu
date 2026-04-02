/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda.h>

#include <gtest/gtest.h>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_uvector.hpp>

#include <cuvs/preprocessing/pca.h>

#include <cmath>
#include <vector>

extern "C" void run_pca(int64_t n_rows,
                        int64_t n_cols,
                        int n_components,
                        float* input_data,
                        float* trans_data,
                        float* components_data,
                        float* explained_var_data,
                        float* explained_var_ratio_data,
                        float* singular_vals_data,
                        float* mu_data,
                        float* noise_vars_data,
                        float* output_data);

TEST(PcaC, FitTransformInverseTransform)
{
  raft::device_resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  int64_t n_rows      = 256;
  int64_t n_cols      = 32;
  int n_components    = 32;

  rmm::device_uvector<float> input(n_rows * n_cols, stream);
  rmm::device_uvector<float> input_copy(n_rows * n_cols, stream);
  rmm::device_uvector<float> trans(n_rows * n_components, stream);
  rmm::device_uvector<float> components(n_components * n_cols, stream);
  rmm::device_uvector<float> explained_var(n_components, stream);
  rmm::device_uvector<float> explained_var_ratio(n_components, stream);
  rmm::device_uvector<float> singular_vals(n_components, stream);
  rmm::device_uvector<float> mu(n_cols, stream);
  rmm::device_uvector<float> noise_vars(1, stream);
  rmm::device_uvector<float> output(n_rows * n_cols, stream);

  raft::random::RngState rng(1234ULL);
  raft::random::uniform(handle, rng, input.data(), n_rows * n_cols, -1.0f, 1.0f);

  RAFT_CUDA_TRY(cudaMemcpyAsync(input_copy.data(),
                                 input.data(),
                                 sizeof(float) * n_rows * n_cols,
                                 cudaMemcpyDeviceToDevice,
                                 stream));

  run_pca(n_rows,
          n_cols,
          n_components,
          input.data(),
          trans.data(),
          components.data(),
          explained_var.data(),
          explained_var_ratio.data(),
          singular_vals.data(),
          mu.data(),
          noise_vars.data(),
          output.data());

  std::vector<float> input_h(n_rows * n_cols);
  std::vector<float> output_h(n_rows * n_cols);
  RAFT_CUDA_TRY(cudaMemcpyAsync(
    input_h.data(), input_copy.data(), sizeof(float) * n_rows * n_cols, cudaMemcpyDeviceToHost, stream));
  RAFT_CUDA_TRY(cudaMemcpyAsync(
    output_h.data(), output.data(), sizeof(float) * n_rows * n_cols, cudaMemcpyDeviceToHost, stream));
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

  float max_err = 0.0f;
  for (int64_t i = 0; i < n_rows * n_cols; ++i) {
    max_err = std::max(max_err, std::abs(input_h[i] - output_h[i]));
  }
  EXPECT_LT(max_err, 1e-3f)
    << "Reconstruction with all components should be near-lossless";
}

TEST(PcaC, DimReduction)
{
  raft::device_resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  int64_t n_rows   = 512;
  int64_t n_cols   = 64;
  int n_components = 16;

  rmm::device_uvector<float> input(n_rows * n_cols, stream);
  rmm::device_uvector<float> input_copy(n_rows * n_cols, stream);
  rmm::device_uvector<float> trans(n_rows * n_components, stream);
  rmm::device_uvector<float> components(n_components * n_cols, stream);
  rmm::device_uvector<float> explained_var(n_components, stream);
  rmm::device_uvector<float> explained_var_ratio(n_components, stream);
  rmm::device_uvector<float> singular_vals(n_components, stream);
  rmm::device_uvector<float> mu(n_cols, stream);
  rmm::device_uvector<float> noise_vars(1, stream);
  rmm::device_uvector<float> output(n_rows * n_cols, stream);

  raft::random::RngState rng(5678ULL);
  raft::random::uniform(handle, rng, input.data(), n_rows * n_cols, -1.0f, 1.0f);

  RAFT_CUDA_TRY(cudaMemcpyAsync(input_copy.data(),
                                 input.data(),
                                 sizeof(float) * n_rows * n_cols,
                                 cudaMemcpyDeviceToDevice,
                                 stream));

  run_pca(n_rows,
          n_cols,
          n_components,
          input.data(),
          trans.data(),
          components.data(),
          explained_var.data(),
          explained_var_ratio.data(),
          singular_vals.data(),
          mu.data(),
          noise_vars.data(),
          output.data());

  std::vector<float> input_h(n_rows * n_cols);
  std::vector<float> output_h(n_rows * n_cols);
  RAFT_CUDA_TRY(cudaMemcpyAsync(
    input_h.data(), input_copy.data(), sizeof(float) * n_rows * n_cols, cudaMemcpyDeviceToHost, stream));
  RAFT_CUDA_TRY(cudaMemcpyAsync(
    output_h.data(), output.data(), sizeof(float) * n_rows * n_cols, cudaMemcpyDeviceToHost, stream));
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

  float max_err = 0.0f;
  for (int64_t i = 0; i < n_rows * n_cols; ++i) {
    max_err = std::max(max_err, std::abs(input_h[i] - output_h[i]));
  }
  EXPECT_GT(max_err, 1e-5f)
    << "With fewer components, reconstruction error should be non-zero";
  EXPECT_LT(max_err, 2.0f)
    << "Reconstruction error should still be bounded";
}
