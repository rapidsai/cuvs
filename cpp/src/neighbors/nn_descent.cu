/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "detail/nn_descent.cuh"
#include <cuvs/neighbors/nn_descent.hpp>
#include <raft/core/logger.hpp>

namespace cuvs::neighbors::nn_descent {

/**
 * @brief Test if we have enough GPU memory to run NN descent algorithm.
 * *
 * @param res
 * @param dataset shape of the dataset
 * @param idx_size the size of index type in bytes
 * @return true if enough GPU memory could be allocated
 * @return false otherwise
 */
auto has_enough_device_memory(raft::resources const& res,
                              raft::matrix_extent<int64_t> dataset,
                              size_t idx_size) -> bool
{
  using DistData_t = float;
  try {
    auto d_data = raft::make_device_matrix<__half, size_t, raft::row_major>(
      res, dataset.extent(0), dataset.extent(1));
    auto l2_norms     = raft::make_device_vector<DistData_t, size_t>(res, dataset.extent(0));
    auto graph_buffer = raft::make_device_vector<uint32_t, size_t>(
      res, dataset.extent(0) * idx_size * detail::kDegreeOnDevice);

    auto dists_buffer = raft::make_device_matrix<DistData_t, size_t, raft::row_major>(
      res, dataset.extent(0), detail::kDegreeOnDevice);

    auto d_locks = raft::make_device_vector<int, size_t>(res, dataset.extent(0));

    auto d_list_sizes_new = raft::make_device_vector<int2, size_t>(res, dataset.extent(0));
    auto d_list_sizes_old = raft::make_device_vector<int2, size_t>(res, dataset.extent(0));
    RAFT_LOG_DEBUG("Sufficient memory for NN descent");
    return true;
  } catch (std::bad_alloc& e) {
    RAFT_LOG_DEBUG("Insufficient memory for NN descent");
    return false;
  } catch (raft::logic_error& e) {
    RAFT_LOG_DEBUG("Insufficient memory for NN descent (logic error)");
    return false;
  }
}

}  // namespace cuvs::neighbors::nn_descent
