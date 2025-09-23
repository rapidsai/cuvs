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

#include <cuda.h>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/handle.hpp>
#include <raft/random/rng.cuh>
#include <rmm/device_uvector.hpp>

#include "ann_utils.cuh"
#include "naive_knn.cuh"
#include <cuvs/neighbors/all_neighbors.h>

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <unordered_set>
#include <vector>

extern "C" void run_all_neighbors(int64_t n_rows,
                                  int64_t n_dim,
                                  int64_t k,
                                  float* dataset_data,
                                  int64_t* indices_data,
                                  float* distances_data,
                                  float* core_distances_data,
                                  cuvsDistanceType metric,
                                  cuvsAllNeighborsAlgo algo,
                                  float alpha);

template <typename T>
void generate_random_data(T* devPtr, size_t size)
{
  raft::handle_t handle;
  raft::random::RngState r(1234ULL);
  // Generate more diverse data with larger range to ensure different neighbors
  raft::random::uniform(handle, r, devPtr, size, T(-10.0), T(10.0));
}

struct AllNeighborsCInputs {
  int64_t n_rows;
  int64_t n_dim;
  int64_t k;
  cuvsDistanceType metric;
  cuvsAllNeighborsAlgo algo;
  float alpha;
  bool include_distances;
  bool include_core_distances;
};

class AllNeighborsCTest : public ::testing::TestWithParam<AllNeighborsCInputs> {
 public:
  AllNeighborsCTest()
    : stream_(0),
      data_(0, stream_),
      indices_(0, stream_),
      distances_(0, stream_),
      core_distances_(0, stream_)
  {
  }

 protected:
  void SetUp() override
  {
    params_ = ::testing::TestWithParam<AllNeighborsCInputs>::GetParam();
    RAFT_CUDA_TRY(cudaStreamCreate(&stream_));
  }

  void TearDown() override { RAFT_CUDA_TRY(cudaStreamDestroy(stream_)); }

  void run()
  {
    size_t data_size    = params_.n_rows * params_.n_dim;
    size_t indices_size = params_.n_rows * params_.k;

    data_.resize(data_size, stream_);
    indices_.resize(indices_size, stream_);

    if (params_.include_distances) { distances_.resize(indices_size, stream_); }

    if (params_.include_core_distances) { core_distances_.resize(params_.n_rows, stream_); }

    generate_random_data(data_.data(), data_size);

    float* distances_ptr      = params_.include_distances ? distances_.data() : nullptr;
    float* core_distances_ptr = params_.include_core_distances ? core_distances_.data() : nullptr;

    // Test the C API
    run_all_neighbors(params_.n_rows,
                      params_.n_dim,
                      params_.k,
                      data_.data(),
                      indices_.data(),
                      distances_ptr,
                      core_distances_ptr,
                      params_.metric,
                      params_.algo,
                      params_.alpha);

    RAFT_CUDA_TRY(cudaStreamSynchronize(stream_));

    // Basic sanity checks
    verify_results();
  }

 private:
  template <typename T, typename IdxT>
  void recall_eval(T* query_data,
                   T* index_data,
                   IdxT* neighbors,
                   T* distances,
                   size_t n_queries,
                   size_t n_rows,
                   size_t n_dim,
                   size_t n_neighbors,
                   cuvsDistanceType metric)
  {
    raft::handle_t handle;
    auto distances_ref = raft::make_device_matrix<T, IdxT>(handle, n_queries, n_neighbors);
    auto neighbors_ref = raft::make_device_matrix<IdxT, IdxT>(handle, n_queries, n_neighbors);
    cuvs::neighbors::naive_knn<T, T, IdxT>(
      handle,
      distances_ref.data_handle(),
      neighbors_ref.data_handle(),
      query_data,
      index_data,
      n_queries,
      n_rows,
      n_dim,
      n_neighbors,
      static_cast<cuvs::distance::DistanceType>((uint16_t)metric));

    size_t size = n_queries * n_neighbors;
    std::vector<IdxT> neighbors_h(size);
    std::vector<T> distances_h(size);
    std::vector<IdxT> neighbors_ref_h(size);
    std::vector<T> distances_ref_h(size);

    auto stream = raft::resource::get_cuda_stream(handle);
    raft::copy(neighbors_h.data(), neighbors, size, stream);
    raft::copy(distances_h.data(), distances, size, stream);
    raft::copy(neighbors_ref_h.data(), neighbors_ref.data_handle(), size, stream);
    raft::copy(distances_ref_h.data(), distances_ref.data_handle(), size, stream);

    // verify output with algorithm-specific minimum recall
    double min_recall = 0.8;  // Default for approximate algorithms

    if (params_.algo == CUVS_ALL_NEIGHBORS_ALGO_BRUTE_FORCE) {
      min_recall = 0.95;  // Brute force should be nearly perfect
    } else if (params_.algo == CUVS_ALL_NEIGHBORS_ALGO_NN_DESCENT) {
      min_recall = 0.7;  // NN Descent is approximate
    }

    // Lower recall threshold when using mutual reachability distance (core_distances)
    if (params_.include_core_distances) {
      min_recall = std::min(min_recall, 0.8);  // Mutual reachability is more approximate
    }

    ASSERT_TRUE(cuvs::neighbors::eval_neighbours(neighbors_ref_h,
                                                 neighbors_h,
                                                 distances_ref_h,
                                                 distances_h,
                                                 n_queries,
                                                 n_neighbors,
                                                 0.001,
                                                 min_recall));
  }

  void verify_results()
  {
    // Copy results to host for verification
    std::vector<int64_t> indices_host(params_.n_rows * params_.k);
    RAFT_CUDA_TRY(cudaMemcpy(indices_host.data(),
                             indices_.data(),
                             indices_host.size() * sizeof(int64_t),
                             cudaMemcpyDeviceToHost));

    // Verify that all indices are valid (within bounds) or indicate missing neighbors
    for (size_t i = 0; i < indices_host.size(); ++i) {
      // Accept either valid indices or max value (indicating missing neighbors)
      EXPECT_TRUE(indices_host[i] >= 0 && indices_host[i] < params_.n_rows ||
                  indices_host[i] == std::numeric_limits<int64_t>::max())
        << "Index " << i << " has value " << indices_host[i] << " which is out of bounds";
    }

    // Perform recall evaluation using the same approach as IVF-PQ tests
    if (params_.include_distances) {
      // We have distances, so we can do full recall evaluation
      std::vector<float> distances_host(params_.n_rows * params_.k);
      RAFT_CUDA_TRY(cudaMemcpy(distances_host.data(),
                               distances_.data(),
                               distances_host.size() * sizeof(float),
                               cudaMemcpyDeviceToHost));

      recall_eval<float, int64_t>(data_.data(),     // query_data (same as index for all_neighbors)
                                  data_.data(),     // index_data
                                  indices_.data(),  // neighbors
                                  distances_.data(),  // distances
                                  params_.n_rows,     // n_queries
                                  params_.n_rows,     // n_rows
                                  params_.n_dim,      // n_dim
                                  params_.k,          // n_neighbors
                                  params_.metric);    // metric

      // Verify distances are non-negative for valid neighbors
      for (size_t i = 0; i < distances_host.size(); ++i) {
        int64_t corresponding_index = indices_host[i];
        if (corresponding_index != std::numeric_limits<int64_t>::max()) {
          // Allow tiny negative distances due to floating-point precision errors
          EXPECT_GE(distances_host[i], -1e-6f)
            << "Distance at index " << i << " is significantly negative: " << distances_host[i];
        }
      }
    } else {
      // No distances available, do basic neighbor validation only
      for (int64_t row = 0; row < params_.n_rows; ++row) {
        std::unordered_set<int64_t> unique_neighbors;
        int64_t valid_neighbors = 0;
        for (int64_t neighbor_idx = 0; neighbor_idx < params_.k; ++neighbor_idx) {
          int64_t neighbor = indices_host[row * params_.k + neighbor_idx];
          if (neighbor != std::numeric_limits<int64_t>::max() && neighbor >= 0 &&
              neighbor < params_.n_rows) {
            unique_neighbors.insert(neighbor);
            valid_neighbors++;
          }
        }

        // For all_neighbors, we should have at least some valid neighbors
        EXPECT_GT(valid_neighbors, 0) << "Row " << row << " has no valid neighbors";

        // The number of unique neighbors should be reasonable
        EXPECT_GE(unique_neighbors.size(), 1) << "Row " << row << " has no unique neighbors";

        // We should have at most min(k, n_rows) neighbors
        EXPECT_LE(unique_neighbors.size(), static_cast<size_t>(std::min(params_.k, params_.n_rows)))
          << "Row " << row << " has more unique neighbors than expected";
      }
    }

    // If core distances are computed, verify they are non-negative
    if (params_.include_core_distances) {
      std::vector<float> core_distances_host(params_.n_rows);
      RAFT_CUDA_TRY(cudaMemcpy(core_distances_host.data(),
                               core_distances_.data(),
                               core_distances_host.size() * sizeof(float),
                               cudaMemcpyDeviceToHost));

      for (size_t i = 0; i < core_distances_host.size(); ++i) {
        EXPECT_GE(core_distances_host[i], 0.0f) << "Core distance at index " << i << " is negative";
      }
    }
  }

  cudaStream_t stream_;
  AllNeighborsCInputs params_;
  rmm::device_uvector<float> data_;
  rmm::device_uvector<int64_t> indices_;
  rmm::device_uvector<float> distances_;
  rmm::device_uvector<float> core_distances_;
};

const std::vector<AllNeighborsCInputs> inputs = {
  // Basic brute force tests
  {100, 16, 10, L2Expanded, CUVS_ALL_NEIGHBORS_ALGO_BRUTE_FORCE, 1.0f, true, false},
  {100, 16, 10, L2Expanded, CUVS_ALL_NEIGHBORS_ALGO_BRUTE_FORCE, 1.0f, false, false},
  {100, 16, 10, L2Expanded, CUVS_ALL_NEIGHBORS_ALGO_BRUTE_FORCE, 1.0f, true, true},

  // Different metrics
  {50, 8, 5, InnerProduct, CUVS_ALL_NEIGHBORS_ALGO_BRUTE_FORCE, 1.0f, true, false},
  {50, 8, 5, CosineExpanded, CUVS_ALL_NEIGHBORS_ALGO_BRUTE_FORCE, 1.0f, true, false},

  // NN Descent tests
  {200, 32, 15, L2Expanded, CUVS_ALL_NEIGHBORS_ALGO_NN_DESCENT, 1.0f, true, false},
  {200, 32, 15, L2Expanded, CUVS_ALL_NEIGHBORS_ALGO_NN_DESCENT, 1.0f, false, false},

  // Different k values
  {80, 12, 20, L2Expanded, CUVS_ALL_NEIGHBORS_ALGO_BRUTE_FORCE, 1.0f, true, false},
  {120, 24, 30, L2Expanded, CUVS_ALL_NEIGHBORS_ALGO_BRUTE_FORCE, 1.0f, true, false},

  // Different alpha values for mutual reachability
  {60, 10, 8, L2Expanded, CUVS_ALL_NEIGHBORS_ALGO_BRUTE_FORCE, 0.5f, true, true},
  {60, 10, 8, L2Expanded, CUVS_ALL_NEIGHBORS_ALGO_BRUTE_FORCE, 2.0f, true, true},
};

TEST_P(AllNeighborsCTest, AllNeighborsC) { run(); }

INSTANTIATE_TEST_CASE_P(AllNeighborsCTest, AllNeighborsCTest, ::testing::ValuesIn(inputs));
