
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

#include <cstdint>
#include <dlpack/dlpack.h>

#include <raft/core/error.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resources.hpp>

#include <cuvs/core/c_api.h>
#include <cuvs/core/interop.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/cagra_c.h>

namespace {

template <typename T>
void* _build(cuvsResources_t res, cagraIndexParams params, DLManagedTensor* dataset_tensor)
{
  auto dataset = dataset_tensor->dl_tensor;

  auto res_ptr = reinterpret_cast<raft::resources*>(res);
  auto index   = new cuvs::neighbors::cagra::index<T, uint32_t>(*res_ptr);

  auto build_params                      = cuvs::neighbors::cagra::index_params();
  build_params.intermediate_graph_degree = params.intermediate_graph_degree;
  build_params.graph_degree              = params.graph_degree;
  build_params.build_algo =
    static_cast<cuvs::neighbors::cagra::graph_build_algo>(params.build_algo);
  build_params.nn_descent_niter = params.nn_descent_niter;

  if (cuvs::core::is_dlpack_device_compatible(dataset)) {
    using mdspan_type = raft::device_matrix_view<T const, int64_t, raft::row_major>;
    auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);
    cuvs::neighbors::cagra::build_device(*res_ptr, build_params, mds, *index);
  } else if (cuvs::core::is_dlpack_host_compatible(dataset)) {
    using mdspan_type = raft::host_matrix_view<T const, int64_t, raft::row_major>;
    auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);
    cuvs::neighbors::cagra::build_host(*res_ptr, build_params, mds, *index);
  }

  return index;
}

template <typename T>
void _search(cuvsResources_t res,
             cagraSearchParams params,
             cagraIndex index,
             DLManagedTensor* queries_tensor,
             DLManagedTensor* neighbors_tensor,
             DLManagedTensor* distances_tensor)
{
  auto res_ptr   = reinterpret_cast<raft::resources*>(res);
  auto index_ptr = reinterpret_cast<cuvs::neighbors::cagra::index<T, uint32_t>*>(index.addr);

  auto search_params              = cuvs::neighbors::cagra::search_params();
  search_params.max_queries       = params.max_queries;
  search_params.itopk_size        = params.itopk_size;
  search_params.max_iterations    = params.max_iterations;
  search_params.algo              = static_cast<cuvs::neighbors::cagra::search_algo>(params.algo);
  search_params.team_size         = params.team_size;
  search_params.search_width      = params.search_width;
  search_params.min_iterations    = params.min_iterations;
  search_params.thread_block_size = params.thread_block_size;
  search_params.hashmap_mode = static_cast<cuvs::neighbors::cagra::hash_mode>(params.hashmap_mode);
  search_params.hashmap_min_bitlen    = params.hashmap_min_bitlen;
  search_params.hashmap_max_fill_rate = params.hashmap_max_fill_rate;
  search_params.num_random_samplings  = params.num_random_samplings;
  search_params.rand_xor_mask         = params.rand_xor_mask;

  using queries_mdspan_type   = raft::device_matrix_view<T const, int64_t, raft::row_major>;
  using neighbors_mdspan_type = raft::device_matrix_view<uint32_t, int64_t, raft::row_major>;
  using distances_mdspan_type = raft::device_matrix_view<float, int64_t, raft::row_major>;
  auto queries_mds            = cuvs::core::from_dlpack<queries_mdspan_type>(queries_tensor);
  auto neighbors_mds          = cuvs::core::from_dlpack<neighbors_mdspan_type>(neighbors_tensor);
  auto distances_mds          = cuvs::core::from_dlpack<distances_mdspan_type>(distances_tensor);
  cuvs::neighbors::cagra::search(
    *res_ptr, search_params, *index_ptr, queries_mds, neighbors_mds, distances_mds);
}

}  // namespace

extern "C" cuvsError_t cagraIndexCreate(cagraIndex_t* index)
{
  try {
    *index = new cagraIndex{};
    return CUVS_SUCCESS;
  } catch (...) {
    return CUVS_ERROR;
  }
}

extern "C" cuvsError_t cagraIndexDestroy(cagraIndex_t index_c_ptr)
{
  try {
    auto index = *index_c_ptr;

    if (index.dtype.code == kDLFloat) {
      auto index_ptr =
        reinterpret_cast<cuvs::neighbors::cagra::index<float, uint32_t>*>(index.addr);
      delete index_ptr;
    } else if (index.dtype.code == kDLInt) {
      auto index_ptr =
        reinterpret_cast<cuvs::neighbors::cagra::index<int8_t, uint32_t>*>(index.addr);
      delete index_ptr;
    } else if (index.dtype.code == kDLUInt) {
      auto index_ptr =
        reinterpret_cast<cuvs::neighbors::cagra::index<uint8_t, uint32_t>*>(index.addr);
      delete index_ptr;
    }
    delete index_c_ptr;
    return CUVS_SUCCESS;
  } catch (...) {
    return CUVS_ERROR;
  }
}

extern "C" cuvsError_t cagraBuild(cuvsResources_t res,
                                  cuvsCagraIndexParams_t params,
                                  DLManagedTensor* dataset_tensor,
                                  cagraIndex_t index)
{
  try {
    auto dataset = dataset_tensor->dl_tensor;

    if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
      index->addr       = reinterpret_cast<uintptr_t>(_build<float>(res, *params, dataset_tensor));
      index->dtype.code = kDLFloat;
    } else if (dataset.dtype.code == kDLInt && dataset.dtype.bits == 8) {
      index->addr       = reinterpret_cast<uintptr_t>(_build<int8_t>(res, *params, dataset_tensor));
      index->dtype.code = kDLInt;
    } else if (dataset.dtype.code == kDLUInt && dataset.dtype.bits == 8) {
      index->addr = reinterpret_cast<uintptr_t>(_build<uint8_t>(res, *params, dataset_tensor));
      index->dtype.code = kDLUInt;
    } else {
      RAFT_FAIL("Unsupported dataset DLtensor dtype: %d and bits: %d",
                dataset.dtype.code,
                dataset.dtype.bits);
    }
    return CUVS_SUCCESS;
  } catch (...) {
    return CUVS_ERROR;
  }
}

extern "C" cuvsError_t cagraSearch(cuvsResources_t res,
                                   cuvsCagraSearchParams_t params,
                                   cagraIndex_t index_c_ptr,
                                   DLManagedTensor* queries_tensor,
                                   DLManagedTensor* neighbors_tensor,
                                   DLManagedTensor* distances_tensor)
{
  try {
    auto queries   = queries_tensor->dl_tensor;
    auto neighbors = neighbors_tensor->dl_tensor;
    auto distances = distances_tensor->dl_tensor;

    RAFT_EXPECTS(cuvs::core::is_dlpack_device_compatible(queries),
                 "queries should have device compatible memory");
    RAFT_EXPECTS(cuvs::core::is_dlpack_device_compatible(neighbors),
                 "queries should have device compatible memory");
    RAFT_EXPECTS(cuvs::core::is_dlpack_device_compatible(distances),
                 "queries should have device compatible memory");

    RAFT_EXPECTS(neighbors.dtype.code == kDLUInt && neighbors.dtype.bits == 32,
                 "neighbors should be of type uint32_t");
    RAFT_EXPECTS(distances.dtype.code == kDLFloat && neighbors.dtype.bits == 32,
                 "neighbors should be of type float32");

    auto index = *index_c_ptr;
    RAFT_EXPECTS(queries.dtype.code == index.dtype.code, "type mismatch between index and queries");

    if (queries.dtype.code == kDLFloat && queries.dtype.bits == 32) {
      _search<float>(res, *params, index, queries_tensor, neighbors_tensor, distances_tensor);
    } else if (queries.dtype.code == kDLInt && queries.dtype.bits == 8) {
      _search<int8_t>(res, *params, index, queries_tensor, neighbors_tensor, distances_tensor);
    } else if (queries.dtype.code == kDLUInt && queries.dtype.bits == 8) {
      _search<uint8_t>(res, *params, index, queries_tensor, neighbors_tensor, distances_tensor);
    } else {
      RAFT_FAIL("Unsupported queries DLtensor dtype: %d and bits: %d",
                queries.dtype.code,
                queries.dtype.bits);
    }
    return CUVS_SUCCESS;
  } catch (...) {
    return CUVS_ERROR;
  }
}

extern "C" cuvsError_t cuvsCagraIndexParamsCreate(cuvsCagraIndexParams_t* params)
{
  try {
    *params = new cagraIndexParams{.intermediate_graph_degree = 128,
                                   .graph_degree              = 64,
                                   .build_algo                = IVF_PQ,
                                   .nn_descent_niter          = 20};
    return CUVS_SUCCESS;
  } catch (...) {
    return CUVS_ERROR;
  }
}

extern "C" cuvsError_t cuvsCagraIndexParamsDestroy(cuvsCagraIndexParams_t params)
{
  try {
    delete params;
    return CUVS_SUCCESS;
  } catch (...) {
    return CUVS_ERROR;
  }
}

extern "C" cuvsError_t cuvsCagraSearchParamsCreate(cuvsCagraSearchParams_t* params)
{
  try {
    *params = new cagraSearchParams{.itopk_size            = 64,
                                    .search_width          = 1,
                                    .hashmap_max_fill_rate = 0.5,
                                    .num_random_samplings  = 1,
                                    .rand_xor_mask         = 0x128394};
    return CUVS_SUCCESS;
  } catch (...) {
    return CUVS_ERROR;
  }
}

extern "C" cuvsError_t cuvsCagraSearchParamsDestroy(cuvsCagraSearchParams_t params)
{
  try {
    delete params;
    return CUVS_SUCCESS;
  } catch (...) {
    return CUVS_ERROR;
  }
}
