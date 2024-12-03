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

#include "detail/dynamic_batching.cuh"

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

namespace cuvs::neighbors::dynamic_batching {

// NB: the (template) index parameter should be the last; it may contain the spaces and so split
//       into multiple preprocessor token. Then it is consumed as __VA_ARGS__
//
#define CUVS_INST_DYNAMIC_BATCHING_INDEX(T, IdxT, Namespace, ...)                         \
  template <>                                                                             \
  template <>                                                                             \
  index<T, IdxT>::index<Namespace ::__VA_ARGS__>(                                         \
    const raft::resources& res,                                                           \
    const cuvs::neighbors::dynamic_batching::index_params& params,                        \
    const Namespace ::__VA_ARGS__& upstream_index,                                        \
    const typename Namespace ::__VA_ARGS__::search_params_type& upstream_params,          \
    const cuvs::neighbors::filtering::base_filter* sample_filter)                         \
    : runner{new detail::batch_runner<T, IdxT>(                                           \
        res, params, upstream_index, upstream_params, Namespace ::search, sample_filter)} \
  {                                                                                       \
  }

#define CUVS_INST_DYNAMIC_BATCHING_SEARCH(T, IdxT)                                 \
  void search(raft::resources const& res,                                          \
              cuvs::neighbors::dynamic_batching::search_params const& params,      \
              cuvs::neighbors::dynamic_batching::index<T, IdxT> const& index,      \
              raft::device_matrix_view<const T, int64_t, raft::row_major> queries, \
              raft::device_matrix_view<IdxT, int64_t, raft::row_major> neighbors,  \
              raft::device_matrix_view<float, int64_t, raft::row_major> distances) \
  {                                                                                \
    return index.runner->search(res, params, queries, neighbors, distances);       \
  }

CUVS_INST_DYNAMIC_BATCHING_INDEX(float, uint32_t, cuvs::neighbors::cagra, index<float, uint32_t>);
CUVS_INST_DYNAMIC_BATCHING_INDEX(half, uint32_t, cuvs::neighbors::cagra, index<half, uint32_t>);
CUVS_INST_DYNAMIC_BATCHING_INDEX(int8_t, uint32_t, cuvs::neighbors::cagra, index<int8_t, uint32_t>);
CUVS_INST_DYNAMIC_BATCHING_INDEX(uint8_t,
                                 uint32_t,
                                 cuvs::neighbors::cagra,
                                 index<uint8_t, uint32_t>);

CUVS_INST_DYNAMIC_BATCHING_INDEX(float, int64_t, cuvs::neighbors::ivf_pq, index<int64_t>);
CUVS_INST_DYNAMIC_BATCHING_INDEX(half, int64_t, cuvs::neighbors::ivf_pq, index<int64_t>);
CUVS_INST_DYNAMIC_BATCHING_INDEX(int8_t, int64_t, cuvs::neighbors::ivf_pq, index<int64_t>);
CUVS_INST_DYNAMIC_BATCHING_INDEX(uint8_t, int64_t, cuvs::neighbors::ivf_pq, index<int64_t>);

CUVS_INST_DYNAMIC_BATCHING_INDEX(float, int64_t, cuvs::neighbors::ivf_flat, index<float, int64_t>);
CUVS_INST_DYNAMIC_BATCHING_INDEX(int8_t,
                                 int64_t,
                                 cuvs::neighbors::ivf_flat,
                                 index<int8_t, int64_t>);
CUVS_INST_DYNAMIC_BATCHING_INDEX(uint8_t,
                                 int64_t,
                                 cuvs::neighbors::ivf_flat,
                                 index<uint8_t, int64_t>);

CUVS_INST_DYNAMIC_BATCHING_SEARCH(float, int64_t);
CUVS_INST_DYNAMIC_BATCHING_SEARCH(half, int64_t);
CUVS_INST_DYNAMIC_BATCHING_SEARCH(int8_t, int64_t);
CUVS_INST_DYNAMIC_BATCHING_SEARCH(uint8_t, int64_t);
CUVS_INST_DYNAMIC_BATCHING_SEARCH(float, uint32_t);  // uint32_t index type is needed for CAGRA
CUVS_INST_DYNAMIC_BATCHING_SEARCH(half, uint32_t);
CUVS_INST_DYNAMIC_BATCHING_SEARCH(int8_t, uint32_t);
CUVS_INST_DYNAMIC_BATCHING_SEARCH(uint8_t, uint32_t);

#undef CUVS_INST_DYNAMIC_BATCHING_INDEX
#undef CUVS_INST_DYNAMIC_BATCHING_SEARCH

}  // namespace cuvs::neighbors::dynamic_batching
