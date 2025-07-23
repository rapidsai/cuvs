/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include <cuvs/core/exceptions.hpp>
#include <cuvs/core/interop.hpp>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_flat.h>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <cuvs/neighbors/mg_ivf_flat.h>
#include <dlpack/dlpack.h>
#include <raft/core/error.hpp>

extern "C" cuvsError_t cuvsMgIvfFlatIndexParamsCreate(cuvsMgIvfFlatIndexParams_t* index_params)
{
  return cuvs::core::translate_exceptions([=] {
    // Create base IVF-Flat parameters
    cuvsIvfFlatIndexParams_t base_params;
    cuvsIvfFlatIndexParamsCreate(&base_params);

    // Create MG wrapper with default values
    *index_params = new cuvsMgIvfFlatIndexParams{
      .base_params = base_params,
      .mode        = CUVS_MG_SHARDED  // Default to sharded mode
    };
  });
}

extern "C" cuvsError_t cuvsMgIvfFlatIndexParamsDestroy(cuvsMgIvfFlatIndexParams_t index_params)
{
  return cuvs::core::translate_exceptions([=] {
    if (index_params) {
      // Base parameters are destroyed automatically
      delete index_params;
    }
  });
}

extern "C" cuvsError_t cuvsMgIvfFlatSearchParamsCreate(cuvsMgIvfFlatSearchParams_t* params)
{
  return cuvs::core::translate_exceptions([=] {
    // Create base IVF-Flat search parameters
    cuvsIvfFlatSearchParams_t base_params;
    cuvsIvfFlatSearchParamsCreate(&base_params);

    // Create MG wrapper with default values
    *params = new cuvsMgIvfFlatSearchParams{
      .base_params      = base_params,
      .search_mode      = CUVS_MG_LOAD_BALANCER,  // Default to load balancer
      .merge_mode       = CUVS_MG_TREE_MERGE,     // Default to tree merge
      .n_rows_per_batch = 1LL << 20               // Default to 1M rows per batch
    };
  });
}

extern "C" cuvsError_t cuvsMgIvfFlatSearchParamsDestroy(cuvsMgIvfFlatSearchParams_t params)
{
  return cuvs::core::translate_exceptions([=] {
    if (params) {
      // Base parameters are destroyed automatically
      delete params;
    }
  });
}

extern "C" cuvsError_t cuvsMgIvfFlatIndexCreate(cuvsMgIvfFlatIndex_t* index)
{
  return cuvs::core::translate_exceptions([=] { *index = new cuvsMgIvfFlatIndex{}; });
}

extern "C" cuvsError_t cuvsMgIvfFlatIndexDestroy(cuvsMgIvfFlatIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    if (index) {
      // Properly clean up the templated inner object based on dtype, like single GPU API
      if (index->dtype.code == kDLFloat && index->dtype.bits == 32) {
        auto mg_index_ptr = reinterpret_cast<
          cuvs::neighbors::
            mg_index<cuvs::neighbors::ivf_flat::index<float, int64_t>, float, int64_t>*>(
          index->addr);
        delete mg_index_ptr;
      } else if (index->dtype.code == kDLInt && index->dtype.bits == 8) {
        auto mg_index_ptr = reinterpret_cast<
          cuvs::neighbors::
            mg_index<cuvs::neighbors::ivf_flat::index<int8_t, int64_t>, int8_t, int64_t>*>(
          index->addr);
        delete mg_index_ptr;
      } else if (index->dtype.code == kDLUInt && index->dtype.bits == 8) {
        auto mg_index_ptr = reinterpret_cast<
          cuvs::neighbors::
            mg_index<cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>, uint8_t, int64_t>*>(
          index->addr);
        delete mg_index_ptr;
      }
      delete index;
    }
  });
}

namespace cuvs::neighbors::ivf_flat {

// Forward declarations for functions defined in ivf_flat_c.cpp
void convert_c_index_params(cuvsIvfFlatIndexParams params,
                            cuvs::neighbors::ivf_flat::index_params* out);
void convert_c_search_params(cuvsIvfFlatSearchParams params,
                             cuvs::neighbors::ivf_flat::search_params* out);

void convert_c_mg_index_params(
  cuvsMgIvfFlatIndexParams params,
  cuvs::neighbors::mg_index_params<cuvs::neighbors::ivf_flat::index_params>* out)
{
  convert_c_index_params(*params.base_params, out);
  out->mode = (params.mode == CUVS_MG_SHARDED) ? cuvs::neighbors::distribution_mode::SHARDED
                                               : cuvs::neighbors::distribution_mode::REPLICATED;
}

void convert_c_mg_search_params(
  cuvsMgIvfFlatSearchParams params,
  cuvs::neighbors::mg_search_params<cuvs::neighbors::ivf_flat::search_params>* out)
{
  convert_c_search_params(*params.base_params, out);
  out->search_mode      = (params.search_mode == CUVS_MG_LOAD_BALANCER)
                            ? cuvs::neighbors::replicated_search_mode::LOAD_BALANCER
                            : cuvs::neighbors::replicated_search_mode::ROUND_ROBIN;
  out->merge_mode       = (params.merge_mode == CUVS_MG_TREE_MERGE)
                            ? cuvs::neighbors::sharded_merge_mode::TREE_MERGE
                            : cuvs::neighbors::sharded_merge_mode::MERGE_ON_ROOT_RANK;
  out->n_rows_per_batch = params.n_rows_per_batch;
}
}  // namespace cuvs::neighbors::ivf_flat

namespace {

template <typename T>
void* _mg_build(cuvsResources_t res,
                cuvsMgIvfFlatIndexParams params,
                DLManagedTensor* dataset_tensor)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);

  auto mg_params = cuvs::neighbors::mg_index_params<cuvs::neighbors::ivf_flat::index_params>();
  cuvs::neighbors::ivf_flat::convert_c_mg_index_params(params, &mg_params);

  using mdspan_type = raft::host_matrix_view<const T, int64_t, raft::row_major>;
  auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);

  auto mg_index =
    new cuvs::neighbors::mg_index<cuvs::neighbors::ivf_flat::index<T, int64_t>, T, int64_t>(
      cuvs::neighbors::ivf_flat::build(*res_ptr, mg_params, mds));

  return mg_index;
}

template <typename T>
void _mg_search(cuvsResources_t res,
                cuvsMgIvfFlatSearchParams params,
                cuvsMgIvfFlatIndex index,
                DLManagedTensor* queries_tensor,
                DLManagedTensor* neighbors_tensor,
                DLManagedTensor* distances_tensor)
{
  auto res_ptr      = reinterpret_cast<raft::resources*>(res);
  auto mg_index_ptr = reinterpret_cast<
    cuvs::neighbors::mg_index<cuvs::neighbors::ivf_flat::index<T, int64_t>, T, int64_t>*>(
    index.addr);

  auto mg_search_params =
    cuvs::neighbors::mg_search_params<cuvs::neighbors::ivf_flat::search_params>();
  cuvs::neighbors::ivf_flat::convert_c_mg_search_params(params, &mg_search_params);

  using queries_mdspan_type   = raft::host_matrix_view<const T, int64_t, raft::row_major>;
  using neighbors_mdspan_type = raft::host_matrix_view<int64_t, int64_t, raft::row_major>;
  using distances_mdspan_type = raft::host_matrix_view<float, int64_t, raft::row_major>;

  auto queries_mds   = cuvs::core::from_dlpack<queries_mdspan_type>(queries_tensor);
  auto neighbors_mds = cuvs::core::from_dlpack<neighbors_mdspan_type>(neighbors_tensor);
  auto distances_mds = cuvs::core::from_dlpack<distances_mdspan_type>(distances_tensor);

  cuvs::neighbors::ivf_flat::search(
    *res_ptr, *mg_index_ptr, mg_search_params, queries_mds, neighbors_mds, distances_mds);
}

template <typename T>
void _mg_extend(cuvsResources_t res,
                cuvsMgIvfFlatIndex index,
                DLManagedTensor* new_vectors_tensor,
                DLManagedTensor* new_indices_tensor)
{
  auto res_ptr      = reinterpret_cast<raft::resources*>(res);
  auto mg_index_ptr = reinterpret_cast<
    cuvs::neighbors::mg_index<cuvs::neighbors::ivf_flat::index<T, int64_t>, T, int64_t>*>(
    index.addr);

  using vectors_mdspan_type = raft::host_matrix_view<const T, int64_t, raft::row_major>;
  auto new_vectors_mds      = cuvs::core::from_dlpack<vectors_mdspan_type>(new_vectors_tensor);

  std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices_mds = std::nullopt;
  if (new_indices_tensor != nullptr) {
    using indices_mdspan_type = raft::host_vector_view<const int64_t, int64_t>;
    new_indices_mds           = cuvs::core::from_dlpack<indices_mdspan_type>(new_indices_tensor);
  }

  cuvs::neighbors::ivf_flat::extend(*res_ptr, *mg_index_ptr, new_vectors_mds, new_indices_mds);
}

template <typename T>
void _mg_serialize(cuvsResources_t res, cuvsMgIvfFlatIndex index, const char* filename)
{
  auto res_ptr      = reinterpret_cast<raft::resources*>(res);
  auto mg_index_ptr = reinterpret_cast<
    cuvs::neighbors::mg_index<cuvs::neighbors::ivf_flat::index<T, int64_t>, T, int64_t>*>(
    index.addr);

  cuvs::neighbors::ivf_flat::serialize(*res_ptr, *mg_index_ptr, std::string(filename));
}

template <typename T>
void* _mg_deserialize(cuvsResources_t res, const char* filename)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);
  auto mg_index =
    new cuvs::neighbors::mg_index<cuvs::neighbors::ivf_flat::index<T, int64_t>, T, int64_t>(
      cuvs::neighbors::ivf_flat::deserialize<T, int64_t>(*res_ptr, std::string(filename)));

  return mg_index;
}

template <typename T>
void* _mg_distribute(cuvsResources_t res, const char* filename)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);
  auto mg_index =
    new cuvs::neighbors::mg_index<cuvs::neighbors::ivf_flat::index<T, int64_t>, T, int64_t>(
      cuvs::neighbors::ivf_flat::distribute<T, int64_t>(*res_ptr, std::string(filename)));

  return mg_index;
}

}  // anonymous namespace

extern "C" cuvsError_t cuvsMgIvfFlatBuild(cuvsResources_t res,
                                          cuvsMgIvfFlatIndexParams_t params,
                                          DLManagedTensor* dataset_tensor,
                                          cuvsMgIvfFlatIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    auto dataset      = dataset_tensor->dl_tensor;
    index->dtype.code = dataset.dtype.code;
    index->dtype.bits = dataset.dtype.bits;

    if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
      index->addr = reinterpret_cast<uintptr_t>(_mg_build<float>(res, *params, dataset_tensor));
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

extern "C" cuvsError_t cuvsMgIvfFlatSearch(cuvsResources_t res,
                                           cuvsMgIvfFlatSearchParams_t params,
                                           cuvsMgIvfFlatIndex_t index,
                                           DLManagedTensor* queries_tensor,
                                           DLManagedTensor* neighbors_tensor,
                                           DLManagedTensor* distances_tensor)
{
  return cuvs::core::translate_exceptions([=] {
    auto queries = queries_tensor->dl_tensor;

    if (queries.dtype.code == kDLFloat && queries.dtype.bits == 32) {
      _mg_search<float>(res, *params, *index, queries_tensor, neighbors_tensor, distances_tensor);
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

extern "C" cuvsError_t cuvsMgIvfFlatExtend(cuvsResources_t res,
                                           cuvsMgIvfFlatIndex_t index,
                                           DLManagedTensor* new_vectors_tensor,
                                           DLManagedTensor* new_indices_tensor)
{
  return cuvs::core::translate_exceptions([=] {
    auto vectors = new_vectors_tensor->dl_tensor;

    if (vectors.dtype.code == kDLFloat && vectors.dtype.bits == 32) {
      _mg_extend<float>(res, *index, new_vectors_tensor, new_indices_tensor);
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

extern "C" cuvsError_t cuvsMgIvfFlatSerialize(cuvsResources_t res,
                                              cuvsMgIvfFlatIndex_t index,
                                              const char* filename)
{
  return cuvs::core::translate_exceptions([=] {
    if (index->dtype.code == kDLFloat && index->dtype.bits == 32) {
      _mg_serialize<float>(res, *index, filename);
    } else if (index->dtype.code == kDLInt && index->dtype.bits == 8) {
      _mg_serialize<int8_t>(res, *index, filename);
    } else if (index->dtype.code == kDLUInt && index->dtype.bits == 8) {
      _mg_serialize<uint8_t>(res, *index, filename);
    } else {
      RAFT_FAIL("Unsupported index dtype: %d and bits: %d", index->dtype.code, index->dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsMgIvfFlatDeserialize(cuvsResources_t res,
                                                const char* filename,
                                                cuvsMgIvfFlatIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    // We need to read dtype from file since we don't know it yet
    // For now, we'll default to float32
    index->dtype.code = kDLFloat;
    index->dtype.bits = 32;
    index->addr       = reinterpret_cast<uintptr_t>(_mg_deserialize<float>(res, filename));
  });
}

extern "C" cuvsError_t cuvsMgIvfFlatDistribute(cuvsResources_t res,
                                               const char* filename,
                                               cuvsMgIvfFlatIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    // We need to read dtype from file since we don't know it yet
    // For now, we'll default to float32
    index->dtype.code = kDLFloat;
    index->dtype.bits = 32;
    index->addr       = reinterpret_cast<uintptr_t>(_mg_distribute<float>(res, filename));
  });
}
