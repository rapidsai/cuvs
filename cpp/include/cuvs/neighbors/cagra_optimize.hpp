/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resources.hpp>

namespace cuvs::neighbors::cagra::helpers {

/**
 * @brief Optimize a KNN graph into a CAGRA graph.
 *
 * This function optimizes a k-NN graph to create a CAGRA graph.
 * The input/output graphs must be on host memory.
 *
 * Usage example:
 * @code{.cpp}
 *   raft::resources res;
 *   auto h_knn = raft::make_host_matrix<uint32_t, int64_t>(N, K_in);
 *   // Fill h_knn with KNN graph
 *   auto h_out = raft::make_host_matrix<uint32_t, int64_t>(N, K_out);
 *   cuvs::neighbors::cagra::helpers::optimize(res, h_knn.view(), h_out.view());
 * @endcode
 *
 * @param[in] handle RAFT resources
 * @param[in] knn_graph Input KNN graph on host [n_rows, k_in]
 * @param[out] new_graph Output CAGRA graph on host [n_rows, k_out]
 */
void optimize(raft::resources const& handle,
              raft::host_matrix_view<uint32_t, int64_t, raft::row_major> knn_graph,
              raft::host_matrix_view<uint32_t, int64_t, raft::row_major> new_graph);

}  // namespace cuvs::neighbors::cagra::helpers
