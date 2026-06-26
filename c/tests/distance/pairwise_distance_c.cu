/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda.h>
#include <cuda_fp16.h>

#include <gtest/gtest.h>
#include <raft/core/error.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/handle.hpp>
#include <raft/random/rng.cuh>

#include <cuvs/core/c_api.h>
#include <cuvs/distance/distance.h>
#include <cuvs/distance/pairwise_distance.h>

#include <string>
#include <string_view>

extern "C" void run_pairwise_distance(int64_t n_rows,
                                      int64_t n_queries,
                                      int64_t n_dim,
                                      float* index_data,
                                      float* query_data,
                                      float* distances_data,
                                      cuvsDistanceType metric);

template <typename T>
void generate_random_data(T* devPtr, size_t size)
{
  raft::handle_t handle;
  raft::random::RngState r(1234ULL);
  raft::random::uniform(handle, r, devPtr, size, T(0.1), T(2.0));
};

namespace {

struct DeviceMatrixTensor {
  DLManagedTensor tensor{};
  int64_t shape[2]{};

  DeviceMatrixTensor(void* data, int64_t rows, int64_t cols, DLDataType dtype)
  {
    shape[0]                              = rows;
    shape[1]                              = cols;
    tensor.dl_tensor.data                 = data;
    tensor.dl_tensor.device               = DLDevice{kDLCUDA, 0};
    tensor.dl_tensor.ndim                 = 2;
    tensor.dl_tensor.dtype                = dtype;
    tensor.dl_tensor.shape                = shape;
    tensor.dl_tensor.strides              = nullptr;
    tensor.dl_tensor.byte_offset          = 0;
  }
};

DLDataType float_dtype(uint8_t bits) { return DLDataType{kDLFloat, bits, 1}; }

void expect_pairwise_distance_error_contains(DLDataType x_dtype,
                                             DLDataType y_dtype,
                                             DLDataType distances_dtype,
                                             std::string_view expected_error)
{
  cuvsResources_t res;
  ASSERT_EQ(cuvsResourcesCreate(&res), CUVS_SUCCESS);

  void *x_data, *y_data, *distances_data;
  RAFT_CUDA_TRY(cudaMalloc(&x_data, 2 * 3 * sizeof(double)));
  RAFT_CUDA_TRY(cudaMalloc(&y_data, 4 * 3 * sizeof(double)));
  RAFT_CUDA_TRY(cudaMalloc(&distances_data, 2 * 4 * sizeof(double)));

  DeviceMatrixTensor x_tensor{x_data, 2, 3, x_dtype};
  DeviceMatrixTensor y_tensor{y_data, 4, 3, y_dtype};
  DeviceMatrixTensor distances_tensor{distances_data, 2, 4, distances_dtype};

  auto status = cuvsPairwiseDistance(res,
                                     &x_tensor.tensor,
                                     &y_tensor.tensor,
                                     &distances_tensor.tensor,
                                     L2Expanded,
                                     2.0f);
  EXPECT_EQ(status, CUVS_ERROR);
  if (status == CUVS_ERROR) {
    const char* error_text = cuvsGetLastErrorText();
    if (error_text == nullptr) {
      ADD_FAILURE() << "Expected cuvsPairwiseDistance to set an error message";
    } else {
      EXPECT_NE(std::string{error_text}.find(expected_error), std::string::npos) << error_text;
    }
  }

  RAFT_CUDA_TRY(cudaFree(x_data));
  RAFT_CUDA_TRY(cudaFree(y_data));
  RAFT_CUDA_TRY(cudaFree(distances_data));
  ASSERT_EQ(cuvsResourcesDestroy(res), CUVS_SUCCESS);
}

}  // namespace

TEST(PairwiseDistanceC, Distance)
{
  int64_t n_rows    = 8096;
  int64_t n_queries = 128;
  int64_t n_dim     = 32;

  cuvsDistanceType metric = L2Expanded;

  float *index_data, *query_data, *distances_data;
  cudaMalloc(&index_data, sizeof(float) * n_rows * n_dim);
  cudaMalloc(&query_data, sizeof(float) * n_queries * n_dim);
  cudaMalloc(&distances_data, sizeof(float) * n_queries * n_rows);

  generate_random_data(index_data, n_rows * n_dim);
  generate_random_data(query_data, n_queries * n_dim);

  run_pairwise_distance(n_rows, n_queries, n_dim, index_data, query_data, distances_data, metric);

  // delete device memory
  cudaFree(index_data);
  cudaFree(query_data);
  cudaFree(distances_data);
}

TEST(PairwiseDistanceC, FailsWithMismatchedInputDtypes)
{
  expect_pairwise_distance_error_contains(float_dtype(32),
                                          float_dtype(64),
                                          float_dtype(32),
                                          "X and Y inputs to cuvsPairwiseDistance must have the "
                                          "same dtype");
}

TEST(PairwiseDistanceC, FailsWithMismatchedFloatOutputDtype)
{
  expect_pairwise_distance_error_contains(
    float_dtype(32),
    float_dtype(32),
    float_dtype(64),
    "distances output to cuvsPairwiseDistance must have dtype float32 for float16 inputs");
}

TEST(PairwiseDistanceC, FailsWithFloat16OutputForFloat16Inputs)
{
  expect_pairwise_distance_error_contains(
    float_dtype(16),
    float_dtype(16),
    float_dtype(16),
    "distances output to cuvsPairwiseDistance must have dtype float32 for float16 inputs");
}

TEST(PairwiseDistanceC, AllowsFloat32OutputForFloat16Inputs)
{
  cuvsResources_t res;
  ASSERT_EQ(cuvsResourcesCreate(&res), CUVS_SUCCESS);

  constexpr int64_t n_rows    = 2;
  constexpr int64_t n_queries = 3;
  constexpr int64_t n_dim     = 4;

  half *x_data, *y_data;
  float* distances_data;
  RAFT_CUDA_TRY(cudaMalloc(&x_data, sizeof(half) * n_rows * n_dim));
  RAFT_CUDA_TRY(cudaMalloc(&y_data, sizeof(half) * n_queries * n_dim));
  RAFT_CUDA_TRY(cudaMalloc(&distances_data, sizeof(float) * n_rows * n_queries));
  RAFT_CUDA_TRY(cudaMemset(x_data, 0, sizeof(half) * n_rows * n_dim));
  RAFT_CUDA_TRY(cudaMemset(y_data, 0, sizeof(half) * n_queries * n_dim));

  DeviceMatrixTensor x_tensor{x_data, n_rows, n_dim, float_dtype(16)};
  DeviceMatrixTensor y_tensor{y_data, n_queries, n_dim, float_dtype(16)};
  DeviceMatrixTensor distances_tensor{distances_data, n_rows, n_queries, float_dtype(32)};

  auto status = cuvsPairwiseDistance(
    res, &x_tensor.tensor, &y_tensor.tensor, &distances_tensor.tensor, L2Expanded, 2.0f);
  EXPECT_EQ(status, CUVS_SUCCESS) << (cuvsGetLastErrorText() ? cuvsGetLastErrorText() : "");
  if (status == CUVS_SUCCESS) { EXPECT_EQ(cuvsStreamSync(res), CUVS_SUCCESS); }

  RAFT_CUDA_TRY(cudaFree(x_data));
  RAFT_CUDA_TRY(cudaFree(y_data));
  RAFT_CUDA_TRY(cudaFree(distances_data));
  ASSERT_EQ(cuvsResourcesDestroy(res), CUVS_SUCCESS);
}
