/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include "cuvs/neighbors/ivf_pq.hpp"
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resources.hpp>

/**
 * @brief Public helper API for fetching a trained index's IVF centroids into a buffer that may be
 * allocated on either host or device.
 *
 * Usage example:
 * @code{.cpp}
 *   raft::resources res;
 *   // allocate the buffer for the output centers
 *   auto cluster_centers = raft::make_device_matrix<float, uint32_t>(
 *     res, index.n_lists(), index.dim());
 *   // Extract the IVF centroids into the buffer
 *   raft::neighbors::ivf_pq::helpers::extract_centers(res, index, cluster_centers.data_handle());
 * @endcode
 *
 * @tparam IdxT
 *
 * @param[in] res raft resource
 * @param[in] index IVF-PQ index (passed by reference)
 * @param[out] cluster_centers IVF cluster centers [index.n_lists(), index.dim]
 */
template <typename IdxT>
void extract_centers(raft::resources const& res,
                     const cuvs::neighbors::ivf_pq::index<int64_t>& index,
                     raft::device_matrix_view<float, uint32_t, raft::row_major> cluster_centers)
{
  RAFT_EXPECTS(cluster_centers.extent(0) == index.n_lists(),
               "Number of rows in the output buffer for cluster centers must be equal to the "
               "number of IVF lists");
  RAFT_EXPECTS(
    cluster_centers.extent(1) == index.dim(),
    "Number of columns in the output buffer for cluster centers and index dim are different");
  auto stream = raft::resource::get_cuda_stream(res);
  RAFT_CUDA_TRY(cudaMemcpy2DAsync(cluster_centers.data_handle(),
                                  sizeof(float) * index.dim(),
                                  index.centers().data_handle(),
                                  sizeof(float) * index.dim_ext(),
                                  sizeof(float) * index.dim(),
                                  index.n_lists(),
                                  cudaMemcpyDefault,
                                  stream));
}