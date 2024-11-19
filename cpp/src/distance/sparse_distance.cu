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

#include <raft/core/resource/cuda_stream.hpp>
#include <rmm/exec_policy.hpp>

#include "sparse_distance.cuh"

namespace cuvs {
namespace distance {

void pairwise_distance(raft::resources const& handle,
                       raft::device_csr_matrix_view<const float, int, int, int> x,
                       raft::device_csr_matrix_view<const float, int, int, int> y,
                       raft::device_matrix_view<float, int, raft::row_major> dist,
                       cuvs::distance::DistanceType metric,
                       float metric_arg)
{
  auto x_structure = x.structure_view();
  auto y_structure = y.structure_view();

  RAFT_EXPECTS(x_structure.get_n_cols() == y_structure.get_n_cols(),
               "Number of columns must be equal");

  RAFT_EXPECTS(dist.extent(0) == x_structure.get_n_rows(),
               "Number of rows in output must be equal to "
               "number of rows in X");
  RAFT_EXPECTS(dist.extent(1) == y_structure.get_n_rows(),
               "Number of columns in output must be equal to "
               "number of rows in Y");

  detail::sparse::distances_config_t<int, float> input_config(handle);
  input_config.a_nrows   = x_structure.get_n_rows();
  input_config.a_ncols   = x_structure.get_n_cols();
  input_config.a_nnz     = x_structure.get_nnz();
  input_config.a_indptr  = const_cast<int*>(x_structure.get_indptr().data());
  input_config.a_indices = const_cast<int*>(x_structure.get_indices().data());
  input_config.a_data    = const_cast<float*>(x.get_elements().data());

  input_config.b_nrows   = y_structure.get_n_rows();
  input_config.b_ncols   = y_structure.get_n_cols();
  input_config.b_nnz     = y_structure.get_nnz();
  input_config.b_indptr  = const_cast<int*>(y_structure.get_indptr().data());
  input_config.b_indices = const_cast<int*>(y_structure.get_indices().data());
  input_config.b_data    = const_cast<float*>(y.get_elements().data());

  pairwiseDistance(dist.data_handle(), input_config, metric, metric_arg);
}
}  // namespace distance
}  // namespace cuvs
