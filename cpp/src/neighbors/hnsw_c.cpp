
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

#include "cuvs/distance/distance.h"
#include <cstdint>
#include <dlpack/dlpack.h>

#include <raft/core/error.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/serialize.hpp>

#include <cuvs/core/c_api.h>
#include <cuvs/core/exceptions.hpp>
#include <cuvs/core/interop.hpp>
#include <cuvs/neighbors/hnsw.h>
#include <cuvs/neighbors/hnsw.hpp>

namespace {

template <typename T>
void _from_cagra(cuvsResources_t res,
                 cuvsHnswIndexParams_t params,
                 cuvsCagraIndex_t cagra_index,
                 cuvsHnswIndex_t hnsw_index)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);
  auto index   = reinterpret_cast<cuvs::neighbors::cagra::index<T, uint32_t>*>(cagra_index->addr);
  auto cpp_params            = cuvs::neighbors::hnsw::index_params();
  cpp_params.hierarchy       = static_cast<cuvs::neighbors::hnsw::HnswHierarchy>(params->hierarchy);
  cpp_params.ef_construction = params->ef_construction;
  cpp_params.num_threads     = params->num_threads;
  std::optional<raft::host_matrix_view<const T, int64_t, raft::row_major>> dataset = std::nullopt;

  auto hnsw_index_unique_ptr =
    cuvs::neighbors::hnsw::from_cagra(*res_ptr, cpp_params, *index, dataset);
  auto hnsw_index_ptr = hnsw_index_unique_ptr.release();
  hnsw_index->addr    = reinterpret_cast<uintptr_t>(hnsw_index_ptr);
}

template <typename T>
void _extend(cuvsResources_t res,
             cuvsHnswExtendParams_t params,
             DLManagedTensor* additional_dataset,
             cuvsHnswIndex index)
{
  auto res_ptr           = reinterpret_cast<raft::resources*>(res);
  auto index_ptr         = reinterpret_cast<cuvs::neighbors::hnsw::index<T>*>(index.addr);
  auto cpp_params        = cuvs::neighbors::hnsw::extend_params();
  cpp_params.num_threads = params->num_threads;

  using additional_dataset_mdspan_type = raft::host_matrix_view<T const, int64_t, raft::row_major>;
  auto additional_dataset_mds =
    cuvs::core::from_dlpack<additional_dataset_mdspan_type>(additional_dataset);
  cuvs::neighbors::hnsw::extend(*res_ptr, cpp_params, additional_dataset_mds, *index_ptr);
}

template <typename T>
void _search(cuvsResources_t res,
             cuvsHnswSearchParams params,
             cuvsHnswIndex index,
             DLManagedTensor* queries_tensor,
             DLManagedTensor* neighbors_tensor,
             DLManagedTensor* distances_tensor)
{
  auto res_ptr   = reinterpret_cast<raft::resources*>(res);
  auto index_ptr = reinterpret_cast<cuvs::neighbors::hnsw::index<T>*>(index.addr);

  auto search_params        = cuvs::neighbors::hnsw::search_params();
  search_params.ef          = params.ef;
  search_params.num_threads = params.num_threads;

  using queries_mdspan_type   = raft::host_matrix_view<T const, int64_t, raft::row_major>;
  using neighbors_mdspan_type = raft::host_matrix_view<uint64_t, int64_t, raft::row_major>;
  using distances_mdspan_type = raft::host_matrix_view<float, int64_t, raft::row_major>;
  auto queries_mds            = cuvs::core::from_dlpack<queries_mdspan_type>(queries_tensor);
  auto neighbors_mds          = cuvs::core::from_dlpack<neighbors_mdspan_type>(neighbors_tensor);
  auto distances_mds          = cuvs::core::from_dlpack<distances_mdspan_type>(distances_tensor);
  cuvs::neighbors::hnsw::search(
    *res_ptr, search_params, *index_ptr, queries_mds, neighbors_mds, distances_mds);
}

template <typename T>
void _serialize(cuvsResources_t res, const char* filename, cuvsHnswIndex index)
{
  auto res_ptr   = reinterpret_cast<raft::resources*>(res);
  auto index_ptr = reinterpret_cast<cuvs::neighbors::hnsw::index<T>*>(index.addr);
  cuvs::neighbors::hnsw::serialize(*res_ptr, std::string(filename), *index_ptr);
}

template <typename T>
void* _deserialize(cuvsResources_t res,
                   cuvsHnswIndexParams_t params,
                   const char* filename,
                   int dim,
                   cuvsDistanceType metric)
{
  auto res_ptr                           = reinterpret_cast<raft::resources*>(res);
  cuvs::neighbors::hnsw::index<T>* index = nullptr;
  auto cpp_params                        = cuvs::neighbors::hnsw::index_params();
  cpp_params.hierarchy = static_cast<cuvs::neighbors::hnsw::HnswHierarchy>(params->hierarchy);
  cuvs::neighbors::hnsw::deserialize(
    *res_ptr, cpp_params, std::string(filename), dim, metric, &index);
  return index;
}
}  // namespace

extern "C" cuvsError_t cuvsHnswIndexParamsCreate(cuvsHnswIndexParams_t* params)
{
  return cuvs::core::translate_exceptions([=] {
    *params = new cuvsHnswIndexParams{
      .hierarchy = cuvsHnswHierarchy::NONE, .ef_construction = 200, .num_threads = 2};
  });
}

extern "C" cuvsError_t cuvsHnswIndexParamsDestroy(cuvsHnswIndexParams_t params)
{
  return cuvs::core::translate_exceptions([=] { delete params; });
}
extern "C" cuvsError_t cuvsHnswIndexCreate(cuvsHnswIndex_t* index)
{
  return cuvs::core::translate_exceptions([=] { *index = new cuvsHnswIndex{}; });
}

extern "C" cuvsError_t cuvsHnswIndexDestroy(cuvsHnswIndex_t index_c_ptr)
{
  return cuvs::core::translate_exceptions([=] {
    auto index = *index_c_ptr;

    if (index.dtype.code == kDLFloat) {
      auto index_ptr = reinterpret_cast<cuvs::neighbors::hnsw::index<float>*>(index.addr);
      delete index_ptr;
    } else if (index.dtype.code == kDLInt) {
      auto index_ptr = reinterpret_cast<cuvs::neighbors::hnsw::index<int8_t>*>(index.addr);
      delete index_ptr;
    } else if (index.dtype.code == kDLUInt) {
      auto index_ptr = reinterpret_cast<cuvs::neighbors::hnsw::index<uint8_t>*>(index.addr);
      delete index_ptr;
    }
    delete index_c_ptr;
  });
}

extern "C" cuvsError_t cuvsHnswExtendParamsCreate(cuvsHnswExtendParams_t* params)
{
  return cuvs::core::translate_exceptions(
    [=] { *params = new cuvsHnswExtendParams{.num_threads = 0}; });
}

extern "C" cuvsError_t cuvsHnswExtendParamsDestroy(cuvsHnswExtendParams_t params)
{
  return cuvs::core::translate_exceptions([=] { delete params; });
}

extern "C" cuvsError_t cuvsHnswFromCagra(cuvsResources_t res,
                                         cuvsHnswIndexParams_t params,
                                         cuvsCagraIndex_t cagra_index,
                                         cuvsHnswIndex_t hnsw_index)
{
  return cuvs::core::translate_exceptions([=] {
    auto index        = *cagra_index;
    hnsw_index->dtype = index.dtype;
    if (index.dtype.code == kDLFloat) {
      _from_cagra<float>(res, params, cagra_index, hnsw_index);
    } else if (index.dtype.code == kDLUInt) {
      _from_cagra<uint8_t>(res, params, cagra_index, hnsw_index);
    } else if (index.dtype.code == kDLInt) {
      _from_cagra<int8_t>(res, params, cagra_index, hnsw_index);
    } else {
      RAFT_FAIL("Unsupported dtype: %d", index.dtype.code);
    }
  });
}

extern "C" cuvsError_t cuvsHnswExtend(cuvsResources_t res,
                                      cuvsHnswExtendParams_t params,
                                      DLManagedTensor* additional_dataset,
                                      cuvsHnswIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    if (index->dtype.code == kDLFloat) {
      _extend<float>(res, params, additional_dataset, *index);
    } else if (index->dtype.code == kDLUInt) {
      _extend<uint8_t>(res, params, additional_dataset, *index);
    } else if (index->dtype.code == kDLInt) {
      _extend<int8_t>(res, params, additional_dataset, *index);
    } else {
      RAFT_FAIL("Unsupported dtype: %d", index->dtype.code);
    }
  });
}

extern "C" cuvsError_t cuvsHnswSearchParamsCreate(cuvsHnswSearchParams_t* params)
{
  return cuvs::core::translate_exceptions(
    [=] { *params = new cuvsHnswSearchParams{.ef = 200, .num_threads = 0}; });
}

extern "C" cuvsError_t cuvsHnswSearchParamsDestroy(cuvsHnswSearchParams_t params)
{
  return cuvs::core::translate_exceptions([=] { delete params; });
}

extern "C" cuvsError_t cuvsHnswSearch(cuvsResources_t res,
                                      cuvsHnswSearchParams_t params,
                                      cuvsHnswIndex_t index_c_ptr,
                                      DLManagedTensor* queries_tensor,
                                      DLManagedTensor* neighbors_tensor,
                                      DLManagedTensor* distances_tensor)
{
  return cuvs::core::translate_exceptions([=] {
    auto queries   = queries_tensor->dl_tensor;
    auto neighbors = neighbors_tensor->dl_tensor;
    auto distances = distances_tensor->dl_tensor;

    RAFT_EXPECTS(cuvs::core::is_dlpack_host_compatible(queries),
                 "queries should have host compatible memory");
    RAFT_EXPECTS(cuvs::core::is_dlpack_host_compatible(neighbors),
                 "neighbors should have host compatible memory");
    RAFT_EXPECTS(cuvs::core::is_dlpack_host_compatible(distances),
                 "distances should have host compatible memory");

    RAFT_EXPECTS(neighbors.dtype.code == kDLUInt && neighbors.dtype.bits == 64,
                 "neighbors should be of type uint64_t");
    RAFT_EXPECTS(distances.dtype.code == kDLFloat && distances.dtype.bits == 32,
                 "distances should be of type float32");

    auto index = *index_c_ptr;
    RAFT_EXPECTS(queries.dtype.code == index.dtype.code, "type mismatch between index and queries");

    if (index.dtype.code == kDLFloat) {
      _search<float>(res, *params, index, queries_tensor, neighbors_tensor, distances_tensor);
    } else if (index.dtype.code == kDLUInt) {
      _search<uint8_t>(res, *params, index, queries_tensor, neighbors_tensor, distances_tensor);
    } else if (index.dtype.code == kDLInt) {
      _search<int8_t>(res, *params, index, queries_tensor, neighbors_tensor, distances_tensor);
    } else {
      RAFT_FAIL("Unsupported index dtype: %d and bits: %d", queries.dtype.code, queries.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsHnswSerialize(cuvsResources_t res,
                                         const char* filename,
                                         cuvsHnswIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    if (index->dtype.code == kDLFloat) {
      _serialize<float>(res, filename, *index);
    } else if (index->dtype.code == kDLInt) {
      _serialize<int8_t>(res, filename, *index);
    } else if (index->dtype.code == kDLUInt) {
      _serialize<uint8_t>(res, filename, *index);
    } else {
      RAFT_FAIL("Unsupported index dtype: %d and bits: %d", index->dtype.code, index->dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsHnswDeserialize(cuvsResources_t res,
                                           cuvsHnswIndexParams_t params,
                                           const char* filename,
                                           int dim,
                                           cuvsDistanceType metric,
                                           cuvsHnswIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    if (index->dtype.code == kDLFloat && index->dtype.bits == 32) {
      index->addr =
        reinterpret_cast<uintptr_t>(_deserialize<float>(res, params, filename, dim, metric));
    } else if (index->dtype.code == kDLUInt && index->dtype.bits == 8) {
      index->addr =
        reinterpret_cast<uintptr_t>(_deserialize<uint8_t>(res, params, filename, dim, metric));
    } else if (index->dtype.code == kDLInt && index->dtype.bits == 8) {
      index->addr =
        reinterpret_cast<uintptr_t>(_deserialize<int8_t>(res, params, filename, dim, metric));
    } else {
      RAFT_FAIL("Unsupported dtype in file %s", filename);
    }
  });
}
