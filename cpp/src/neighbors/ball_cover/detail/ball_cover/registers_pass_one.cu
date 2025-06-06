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

/*
 * NOTE: this file is generated by registers_00_generate.py
 *
 * Make changes there and run in this directory:
 *
 * > python registers_00_generate.py
 *
 */

#include "../../registers.cuh"
#include <cstdint>  // int64_t
#include <cuvs/neighbors/ball_cover.hpp>

#define instantiate_cuvs_neighbors_detail_rbc_low_dim_pass_one(Mvalue_idx, Mvalue_t)             \
  template void cuvs::neighbors::ball_cover::detail::rbc_low_dim_pass_one<Mvalue_idx, Mvalue_t>( \
    raft::resources const& handle,                                                               \
    const cuvs::neighbors::ball_cover::index<Mvalue_idx, Mvalue_t>& index,                       \
    const Mvalue_t* query,                                                                       \
    const int64_t n_query_rows,                                                                  \
    const int64_t k,                                                                             \
    const Mvalue_idx* R_knn_inds,                                                                \
    const Mvalue_t* R_knn_dists,                                                                 \
    Mvalue_idx* inds,                                                                            \
    Mvalue_t* dists,                                                                             \
    float weight,                                                                                \
    int dims);

instantiate_cuvs_neighbors_detail_rbc_low_dim_pass_one(std::int64_t, float);
#undef instantiate_cuvs_neighbors_detail_rbc_low_dim_pass_one
