/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda.h>

#include <gtest/gtest.h>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/handle.hpp>
#include <raft/random/rng.cuh>

#include <cuvs/distance/distance.h>

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
