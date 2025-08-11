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

#include "cagra_c.hpp"
#include <cuvs/core/exceptions.hpp>
#include <cuvs/core/interop.hpp>
#include <cuvs/neighbors/cagra.h>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/mg_cagra.h>
#include <dlpack/dlpack.h>
#include <raft/core/error.hpp>
#include <raft/core/serialize.hpp>

#include <fstream>

extern "C" cuvsError_t cuvsMultiGpuCagraIndexParamsCreate(
  cuvsMultiGpuCagraIndexParams_t* index_params)
{
  return cuvs::core::translate_exceptions([=] {
    // Create base CAGRA parameters
    cuvsCagraIndexParams_t base_params;
    cuvsCagraIndexParamsCreate(&base_params);

    // Create MG wrapper with default values
    *index_params = new cuvsMultiGpuCagraIndexParams{
      .base_params = base_params,
      .mode        = CUVS_NEIGHBORS_MG_SHARDED  // Default to sharded mode
    };
  });
}

extern "C" cuvsError_t cuvsMultiGpuCagraIndexParamsDestroy(
  cuvsMultiGpuCagraIndexParams_t index_params)
{
  return cuvs::core::translate_exceptions([=] {
    if (index_params) {
      cuvsCagraIndexParamsDestroy(index_params->base_params);
      delete index_params;
    }
  });
}

extern "C" cuvsError_t cuvsMultiGpuCagraSearchParamsCreate(cuvsMultiGpuCagraSearchParams_t* params)
{
  return cuvs::core::translate_exceptions([=] {
    // Create base CAGRA search parameters
    cuvsCagraSearchParams_t base_params;
    cuvsCagraSearchParamsCreate(&base_params);

    // Create MG wrapper with default values
    *params = new cuvsMultiGpuCagraSearchParams{
      .base_params      = base_params,
      .search_mode      = CUVS_NEIGHBORS_MG_LOAD_BALANCER,  // Default to load balancer
      .merge_mode       = CUVS_NEIGHBORS_MG_TREE_MERGE,     // Default to tree merge
      .n_rows_per_batch = 1LL << 20                         // Default to 1M rows per batch
    };
  });
}

extern "C" cuvsError_t cuvsMultiGpuCagraSearchParamsDestroy(cuvsMultiGpuCagraSearchParams_t params)
{
  return cuvs::core::translate_exceptions([=] {
    if (params) {
      cuvsCagraSearchParamsDestroy(params->base_params);
      delete params;
    }
  });
}

extern "C" cuvsError_t cuvsMultiGpuCagraIndexCreate(cuvsMultiGpuCagraIndex_t* index)
{
  return cuvs::core::translate_exceptions([=] { *index = new cuvsMultiGpuCagraIndex{}; });
}

extern "C" cuvsError_t cuvsMultiGpuCagraIndexDestroy(cuvsMultiGpuCagraIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    if (index) {
      // Properly clean up the templated inner object based on dtype, like single GPU API
      if (index->dtype.code == kDLFloat && index->dtype.bits == 32) {
        auto mg_index_ptr =
          reinterpret_cast<cuvs::neighbors::mg_index<cuvs::neighbors::cagra::index<float, uint32_t>,
                                                     float,
                                                     uint32_t>*>(index->addr);
        delete mg_index_ptr;
      } else if (index->dtype.code == kDLFloat && index->dtype.bits == 16) {
        auto mg_index_ptr =
          reinterpret_cast<cuvs::neighbors::mg_index<cuvs::neighbors::cagra::index<half, uint32_t>,
                                                     half,
                                                     uint32_t>*>(index->addr);
        delete mg_index_ptr;
      } else if (index->dtype.code == kDLInt && index->dtype.bits == 8) {
        auto mg_index_ptr = reinterpret_cast<
          cuvs::neighbors::
            mg_index<cuvs::neighbors::cagra::index<int8_t, uint32_t>, int8_t, uint32_t>*>(
          index->addr);
        delete mg_index_ptr;
      } else if (index->dtype.code == kDLUInt && index->dtype.bits == 8) {
        auto mg_index_ptr = reinterpret_cast<
          cuvs::neighbors::
            mg_index<cuvs::neighbors::cagra::index<uint8_t, uint32_t>, uint8_t, uint32_t>*>(
          index->addr);
        delete mg_index_ptr;
      }
      delete index;
    }
  });
}

namespace cuvs::neighbors::cagra {

void convert_c_mg_index_params(
  cuvsMultiGpuCagraIndexParams params,
  int64_t n_rows,
  int64_t dim,
  cuvs::neighbors::mg_index_params<cuvs::neighbors::cagra::index_params>* out)
{
  convert_c_index_params(*params.base_params, n_rows, dim, out);
  out->mode = (params.mode == CUVS_NEIGHBORS_MG_SHARDED)
                ? cuvs::neighbors::distribution_mode::SHARDED
                : cuvs::neighbors::distribution_mode::REPLICATED;
}

void convert_c_mg_search_params(
  cuvsMultiGpuCagraSearchParams params,
  cuvs::neighbors::mg_search_params<cuvs::neighbors::cagra::search_params>* out)
{
  convert_c_search_params(*params.base_params, out);
  out->search_mode      = (params.search_mode == CUVS_NEIGHBORS_MG_LOAD_BALANCER)
                            ? cuvs::neighbors::replicated_search_mode::LOAD_BALANCER
                            : cuvs::neighbors::replicated_search_mode::ROUND_ROBIN;
  out->merge_mode       = (params.merge_mode == CUVS_NEIGHBORS_MG_TREE_MERGE)
                            ? cuvs::neighbors::sharded_merge_mode::TREE_MERGE
                            : cuvs::neighbors::sharded_merge_mode::MERGE_ON_ROOT_RANK;
  out->n_rows_per_batch = params.n_rows_per_batch;
}
}  // namespace cuvs::neighbors::cagra

namespace {

template <typename T>
void* _mg_build(cuvsResources_t res,
                cuvsMultiGpuCagraIndexParams params,
                DLManagedTensor* dataset_tensor)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);
  auto dataset = dataset_tensor->dl_tensor;

  auto mg_params = cuvs::neighbors::mg_index_params<cuvs::neighbors::cagra::index_params>();
  cuvs::neighbors::cagra::convert_c_mg_index_params(
    params, dataset.shape[0], dataset.shape[1], &mg_params);

  using mdspan_type = raft::host_matrix_view<const T, int64_t, raft::row_major>;
  auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);

  auto mg_index =
    new cuvs::neighbors::mg_index<cuvs::neighbors::cagra::index<T, uint32_t>, T, uint32_t>(
      cuvs::neighbors::cagra::build(*res_ptr, mg_params, mds));

  return mg_index;
}

template <typename T>
void _mg_search(cuvsResources_t res,
                cuvsMultiGpuCagraSearchParams params,
                cuvsMultiGpuCagraIndex index,
                DLManagedTensor* queries_tensor,
                DLManagedTensor* neighbors_tensor,
                DLManagedTensor* distances_tensor)
{
  auto res_ptr      = reinterpret_cast<raft::resources*>(res);
  auto mg_index_ptr = reinterpret_cast<
    cuvs::neighbors::mg_index<cuvs::neighbors::cagra::index<T, uint32_t>, T, uint32_t>*>(
    index.addr);

  auto mg_search_params =
    cuvs::neighbors::mg_search_params<cuvs::neighbors::cagra::search_params>();
  cuvs::neighbors::cagra::convert_c_mg_search_params(params, &mg_search_params);

  using queries_mdspan_type   = raft::host_matrix_view<const T, int64_t, raft::row_major>;
  using neighbors_mdspan_type = raft::host_matrix_view<int64_t, int64_t, raft::row_major>;
  using distances_mdspan_type = raft::host_matrix_view<float, int64_t, raft::row_major>;

  auto queries_mds   = cuvs::core::from_dlpack<queries_mdspan_type>(queries_tensor);
  auto neighbors_mds = cuvs::core::from_dlpack<neighbors_mdspan_type>(neighbors_tensor);
  auto distances_mds = cuvs::core::from_dlpack<distances_mdspan_type>(distances_tensor);

  cuvs::neighbors::cagra::search(
    *res_ptr, *mg_index_ptr, mg_search_params, queries_mds, neighbors_mds, distances_mds);
}

template <typename T>
void _mg_extend(cuvsResources_t res,
                cuvsMultiGpuCagraIndex index,
                DLManagedTensor* new_vectors_tensor,
                DLManagedTensor* new_indices_tensor)
{
  auto res_ptr      = reinterpret_cast<raft::resources*>(res);
  auto mg_index_ptr = reinterpret_cast<
    cuvs::neighbors::mg_index<cuvs::neighbors::cagra::index<T, uint32_t>, T, uint32_t>*>(
    index.addr);

  using vectors_mdspan_type = raft::host_matrix_view<const T, int64_t, raft::row_major>;
  auto new_vectors_mds      = cuvs::core::from_dlpack<vectors_mdspan_type>(new_vectors_tensor);

  std::optional<raft::host_vector_view<const uint32_t, int64_t>> new_indices_mds = std::nullopt;
  if (new_indices_tensor != nullptr) {
    using indices_mdspan_type = raft::host_vector_view<const uint32_t, int64_t>;
    new_indices_mds           = cuvs::core::from_dlpack<indices_mdspan_type>(new_indices_tensor);
  }

  cuvs::neighbors::cagra::extend(*res_ptr, *mg_index_ptr, new_vectors_mds, new_indices_mds);
}

template <typename T>
void _mg_serialize(cuvsResources_t res, cuvsMultiGpuCagraIndex index, const char* filename)
{
  auto res_ptr      = reinterpret_cast<raft::resources*>(res);
  auto mg_index_ptr = reinterpret_cast<
    cuvs::neighbors::mg_index<cuvs::neighbors::cagra::index<T, uint32_t>, T, uint32_t>*>(
    index.addr);

  cuvs::neighbors::cagra::serialize(*res_ptr, *mg_index_ptr, std::string(filename));
}

template <typename T>
void* _mg_deserialize(cuvsResources_t res, const char* filename)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);
  auto mg_index =
    new cuvs::neighbors::mg_index<cuvs::neighbors::cagra::index<T, uint32_t>, T, uint32_t>(
      cuvs::neighbors::cagra::deserialize<T, uint32_t>(*res_ptr, std::string(filename)));

  return mg_index;
}

template <typename T>
void* _mg_distribute(cuvsResources_t res, const char* filename)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);
  auto mg_index =
    new cuvs::neighbors::mg_index<cuvs::neighbors::cagra::index<T, uint32_t>, T, uint32_t>(
      cuvs::neighbors::cagra::distribute<T, uint32_t>(*res_ptr, std::string(filename)));

  return mg_index;
}

}  // anonymous namespace

extern "C" cuvsError_t cuvsMultiGpuCagraBuild(cuvsResources_t res,
                                              cuvsMultiGpuCagraIndexParams_t params,
                                              DLManagedTensor* dataset_tensor,
                                              cuvsMultiGpuCagraIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    auto dataset      = dataset_tensor->dl_tensor;
    index->dtype.code = dataset.dtype.code;
    index->dtype.bits = dataset.dtype.bits;

    if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
      index->addr = reinterpret_cast<uintptr_t>(_mg_build<float>(res, *params, dataset_tensor));
    } else if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 16) {
      index->addr = reinterpret_cast<uintptr_t>(_mg_build<half>(res, *params, dataset_tensor));
    } else if (dataset.dtype.code == kDLInt && dataset.dtype.bits == 8) {
      index->addr = reinterpret_cast<uintptr_t>(_mg_build<int8_t>(res, *params, dataset_tensor));
    } else if (dataset.dtype.code == kDLUInt && dataset.dtype.bits == 8) {
      index->addr = reinterpret_cast<uintptr_t>(_mg_build<uint8_t>(res, *params, dataset_tensor));
    } else {
      RAFT_FAIL("Unsupported dataset DLtensor dtype: %d and bits: %d",
                dataset.dtype.code,
                dataset.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsMultiGpuCagraSearch(cuvsResources_t res,
                                               cuvsMultiGpuCagraSearchParams_t params,
                                               cuvsMultiGpuCagraIndex_t index,
                                               DLManagedTensor* queries_tensor,
                                               DLManagedTensor* neighbors_tensor,
                                               DLManagedTensor* distances_tensor)
{
  return cuvs::core::translate_exceptions([=] {
    auto queries = queries_tensor->dl_tensor;

    if (queries.dtype.code == kDLFloat && queries.dtype.bits == 32) {
      _mg_search<float>(res, *params, *index, queries_tensor, neighbors_tensor, distances_tensor);
    } else if (queries.dtype.code == kDLFloat && queries.dtype.bits == 16) {
      _mg_search<half>(res, *params, *index, queries_tensor, neighbors_tensor, distances_tensor);
    } else if (queries.dtype.code == kDLInt && queries.dtype.bits == 8) {
      _mg_search<int8_t>(res, *params, *index, queries_tensor, neighbors_tensor, distances_tensor);
    } else if (queries.dtype.code == kDLUInt && queries.dtype.bits == 8) {
      _mg_search<uint8_t>(res, *params, *index, queries_tensor, neighbors_tensor, distances_tensor);
    } else {
      RAFT_FAIL("Unsupported queries DLtensor dtype: %d and bits: %d",
                queries.dtype.code,
                queries.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsMultiGpuCagraExtend(cuvsResources_t res,
                                               cuvsMultiGpuCagraIndex_t index,
                                               DLManagedTensor* new_vectors_tensor,
                                               DLManagedTensor* new_indices_tensor)
{
  return cuvs::core::translate_exceptions([=] {
    auto vectors = new_vectors_tensor->dl_tensor;

    if (vectors.dtype.code == kDLFloat && vectors.dtype.bits == 32) {
      _mg_extend<float>(res, *index, new_vectors_tensor, new_indices_tensor);
    } else if (vectors.dtype.code == kDLFloat && vectors.dtype.bits == 16) {
      _mg_extend<half>(res, *index, new_vectors_tensor, new_indices_tensor);
    } else if (vectors.dtype.code == kDLInt && vectors.dtype.bits == 8) {
      _mg_extend<int8_t>(res, *index, new_vectors_tensor, new_indices_tensor);
    } else if (vectors.dtype.code == kDLUInt && vectors.dtype.bits == 8) {
      _mg_extend<uint8_t>(res, *index, new_vectors_tensor, new_indices_tensor);
    } else {
      RAFT_FAIL("Unsupported new_vectors DLtensor dtype: %d and bits: %d",
                vectors.dtype.code,
                vectors.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsMultiGpuCagraSerialize(cuvsResources_t res,
                                                  cuvsMultiGpuCagraIndex_t index,
                                                  const char* filename)
{
  return cuvs::core::translate_exceptions([=] {
    if (index->dtype.code == kDLFloat && index->dtype.bits == 32) {
      _mg_serialize<float>(res, *index, filename);
    } else if (index->dtype.code == kDLFloat && index->dtype.bits == 16) {
      _mg_serialize<half>(res, *index, filename);
    } else if (index->dtype.code == kDLInt && index->dtype.bits == 8) {
      _mg_serialize<int8_t>(res, *index, filename);
    } else if (index->dtype.code == kDLUInt && index->dtype.bits == 8) {
      _mg_serialize<uint8_t>(res, *index, filename);
    } else {
      RAFT_FAIL("Unsupported index dtype: %d and bits: %d", index->dtype.code, index->dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsMultiGpuCagraDeserialize(cuvsResources_t res,
                                                    const char* filename,
                                                    cuvsMultiGpuCagraIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    std::ifstream is(filename, std::ios::in | std::ios::binary);
    if (!is) { RAFT_FAIL("Cannot open file %s", filename); }
    char dtype_string[4];
    is.read(dtype_string, 4);
    auto dtype = raft::detail::numpy_serializer::parse_descr(std::string(dtype_string, 4));
    is.close();

    index->dtype.bits = dtype.itemsize * 8;
    if (dtype.kind == 'f' && dtype.itemsize == 4) {
      index->dtype.code = kDLFloat;
      index->addr       = reinterpret_cast<uintptr_t>(_mg_deserialize<float>(res, filename));
    } else if (dtype.kind == 'f' && dtype.itemsize == 2) {
      index->dtype.code = kDLFloat;
      index->addr       = reinterpret_cast<uintptr_t>(_mg_deserialize<half>(res, filename));
    } else if (dtype.kind == 'i' && dtype.itemsize == 1) {
      index->dtype.code = kDLInt;
      index->addr       = reinterpret_cast<uintptr_t>(_mg_deserialize<int8_t>(res, filename));
    } else if (dtype.kind == 'u' && dtype.itemsize == 1) {
      index->dtype.code = kDLUInt;
      index->addr       = reinterpret_cast<uintptr_t>(_mg_deserialize<uint8_t>(res, filename));
    } else {
      RAFT_FAIL("Unsupported index dtype");
    }
  });
}

extern "C" cuvsError_t cuvsMultiGpuCagraDistribute(cuvsResources_t res,
                                                   const char* filename,
                                                   cuvsMultiGpuCagraIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    std::ifstream is(filename, std::ios::in | std::ios::binary);
    if (!is) { RAFT_FAIL("Cannot open file %s", filename); }
    char dtype_string[4];
    is.read(dtype_string, 4);
    auto dtype = raft::detail::numpy_serializer::parse_descr(std::string(dtype_string, 4));
    is.close();

    index->dtype.bits = dtype.itemsize * 8;
    if (dtype.kind == 'f' && dtype.itemsize == 4) {
      index->dtype.code = kDLFloat;
      index->addr       = reinterpret_cast<uintptr_t>(_mg_distribute<float>(res, filename));
    } else if (dtype.kind == 'f' && dtype.itemsize == 2) {
      index->dtype.code = kDLFloat;
      index->addr       = reinterpret_cast<uintptr_t>(_mg_distribute<half>(res, filename));
    } else if (dtype.kind == 'i' && dtype.itemsize == 1) {
      index->dtype.code = kDLInt;
      index->addr       = reinterpret_cast<uintptr_t>(_mg_distribute<int8_t>(res, filename));
    } else if (dtype.kind == 'u' && dtype.itemsize == 1) {
      index->dtype.code = kDLUInt;
      index->addr       = reinterpret_cast<uintptr_t>(_mg_distribute<uint8_t>(res, filename));
    } else {
      RAFT_FAIL("Unsupported index dtype");
    }
  });
}
