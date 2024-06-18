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

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <raft/core/mdspan.hpp>

namespace cuvs::neighbors::cagra::graph_build_params {
ivf_pq_params::ivf_pq_params(raft::matrix_extent<int64_t> dataset_extents,
                             cuvs::distance::DistanceType metric)
{
  build_params = cuvs::neighbors::ivf_pq::index_params::from_dataset(dataset_extents, metric);

  search_params                         = cuvs::neighbors::ivf_pq::search_params{};
  search_params.n_probes                = std::max<uint32_t>(10, build_params.n_lists * 0.01);
  search_params.lut_dtype               = CUDA_R_16F;
  search_params.internal_distance_dtype = CUDA_R_16F;

  refinement_rate = 2;
}
}  // namespace cuvs::neighbors::cagra::graph_build_params