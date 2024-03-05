
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
#include <cuvs/neighbors/ivf_flat.h>
#include <cuvs/neighbors/ivf_flat.hpp>

namespace {

template <typename T, typename IdxT>
void* _build(cuvsResources_t res, ivfFlatIndexParams params, DLManagedTensor* dataset_tensor)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);

  auto build_params              = cuvs::neighbors::ivf_flat::index_params();
  build_params.metric            = static_cast<cuvs::distance::DistanceType>((int)params.metric),
  build_params.metric_arg        = params.metric_arg;
  build_params.add_data_on_build = params.add_data_on_build;
  build_params.n_lists           = params.n_lists;
  build_params.kmeans_n_iters    = params.kmeans_n_iters;
  build_params.kmeans_trainset_fraction       = params.kmeans_trainset_fraction;
  build_params.adaptive_centers               = params.adaptive_centers;
  build_params.conservative_memory_allocation = params.conservative_memory_allocation;

  auto dataset = dataset_tensor->dl_tensor;
  auto dim     = dataset.shape[0];

  auto index = new cuvs::neighbors::ivf_flat::index<T, IdxT>(*res_ptr, build_params, dim);

  using mdspan_type = raft::device_matrix_view<T const, IdxT, raft::row_major>;
  auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);

  cuvs::neighbors::ivf_flat::build(*res_ptr, build_params, mds, *index);

  return index;
}

template <typename T, typename IdxT>
void _search(cuvsResources_t res,
             ivfFlatSearchParams params,
             ivfFlatIndex index,
             DLManagedTensor* queries_tensor,
             DLManagedTensor* neighbors_tensor,
             DLManagedTensor* distances_tensor)
{
  auto res_ptr   = reinterpret_cast<raft::resources*>(res);
  auto index_ptr = reinterpret_cast<cuvs::neighbors::ivf_flat::index<T, IdxT>*>(index.addr);

  auto search_params     = cuvs::neighbors::ivf_flat::search_params();
  search_params.n_probes = params.n_probes;

  using queries_mdspan_type   = raft::device_matrix_view<T const, IdxT, raft::row_major>;
  using neighbors_mdspan_type = raft::device_matrix_view<IdxT, IdxT, raft::row_major>;
  using distances_mdspan_type = raft::device_matrix_view<float, IdxT, raft::row_major>;
  auto queries_mds            = cuvs::core::from_dlpack<queries_mdspan_type>(queries_tensor);
  auto neighbors_mds          = cuvs::core::from_dlpack<neighbors_mdspan_type>(neighbors_tensor);
  auto distances_mds          = cuvs::core::from_dlpack<distances_mdspan_type>(distances_tensor);

  cuvs::neighbors::ivf_flat::search(
    *res_ptr, search_params, *index_ptr, queries_mds, neighbors_mds, distances_mds);
}

}  // namespace

extern "C" cuvsError_t ivfFlatIndexCreate(cuvsIvfFlatIndex_t* index)
{
  try {
    *index = new ivfFlatIndex{};
    return CUVS_SUCCESS;
  } catch (...) {
    return CUVS_ERROR;
  }
}

extern "C" cuvsError_t ivfFlatIndexDestroy(cuvsIvfFlatIndex_t index_c_ptr)
{
  try {
    auto index = *index_c_ptr;

    if (index.dtype.code == kDLFloat) {
      auto index_ptr =
        reinterpret_cast<cuvs::neighbors::ivf_flat::index<float, int64_t>*>(index.addr);
      delete index_ptr;
    } else if (index.dtype.code == kDLInt) {
      auto index_ptr =
        reinterpret_cast<cuvs::neighbors::ivf_flat::index<int8_t, int64_t>*>(index.addr);
      delete index_ptr;
    } else if (index.dtype.code == kDLUInt) {
      auto index_ptr =
        reinterpret_cast<cuvs::neighbors::ivf_flat::index<uint8_t, int64_t>*>(index.addr);
      delete index_ptr;
    }
    delete index_c_ptr;
    return CUVS_SUCCESS;
  } catch (...) {
    return CUVS_ERROR;
  }
}

extern "C" cuvsError_t ivfFlatBuild(cuvsResources_t res,
                                    cuvsIvfFlatIndexParams_t params,
                                    DLManagedTensor* dataset_tensor,
                                    cuvsIvfFlatIndex_t index)
{
  try {
    auto dataset = dataset_tensor->dl_tensor;

    if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
      index->addr =
        reinterpret_cast<uintptr_t>(_build<float, int64_t>(res, *params, dataset_tensor));
      index->dtype.code = kDLFloat;
    } else if (dataset.dtype.code == kDLInt && dataset.dtype.bits == 8) {
      index->addr =
        reinterpret_cast<uintptr_t>(_build<int8_t, int64_t>(res, *params, dataset_tensor));
      index->dtype.code = kDLInt;
    } else if (dataset.dtype.code == kDLUInt && dataset.dtype.bits == 8) {
      index->addr =
        reinterpret_cast<uintptr_t>(_build<uint8_t, int64_t>(res, *params, dataset_tensor));
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

extern "C" cuvsError_t ivfFlatSearch(cuvsResources_t res,
                                     cuvsIvfFlatSearchParams_t params,
                                     cuvsIvfFlatIndex_t index_c_ptr,
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
                 "neighbors should have device compatible memory");
    RAFT_EXPECTS(cuvs::core::is_dlpack_device_compatible(distances),
                 "distances should have device compatible memory");

    RAFT_EXPECTS(neighbors.dtype.code == kDLInt && neighbors.dtype.bits == 64,
                 "neighbors should be of type int64_t");
    RAFT_EXPECTS(distances.dtype.code == kDLFloat && distances.dtype.bits == 32,
                 "distances should be of type float32");

    auto index = *index_c_ptr;
    RAFT_EXPECTS(queries.dtype.code == index.dtype.code, "type mismatch between index and queries");

    if (queries.dtype.code == kDLFloat && queries.dtype.bits == 32) {
      _search<float, int64_t>(
        res, *params, index, queries_tensor, neighbors_tensor, distances_tensor);
    } else if (queries.dtype.code == kDLInt && queries.dtype.bits == 8) {
      _search<int8_t, int64_t>(
        res, *params, index, queries_tensor, neighbors_tensor, distances_tensor);
    } else if (queries.dtype.code == kDLUInt && queries.dtype.bits == 8) {
      _search<uint8_t, int64_t>(
        res, *params, index, queries_tensor, neighbors_tensor, distances_tensor);
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

extern "C" cuvsError_t cuvsIvfFlatIndexParamsCreate(cuvsIvfFlatIndexParams_t* params)
{
  try {
    *params = new ivfFlatIndexParams{.metric                         = L2Expanded,
                                     .metric_arg                     = 2.0f,
                                     .add_data_on_build              = true,
                                     .n_lists                        = 1024,
                                     .kmeans_n_iters                 = 20,
                                     .kmeans_trainset_fraction       = 0.5,
                                     .adaptive_centers               = false,
                                     .conservative_memory_allocation = false};
    return CUVS_SUCCESS;
  } catch (...) {
    return CUVS_ERROR;
  }
}

extern "C" cuvsError_t cuvsIvfFlatIndexParamsDestroy(cuvsIvfFlatIndexParams_t params)
{
  try {
    delete params;
    return CUVS_SUCCESS;
  } catch (...) {
    return CUVS_ERROR;
  }
}

extern "C" cuvsError_t cuvsIvfFlatSearchParamsCreate(cuvsIvfFlatSearchParams_t* params)
{
  try {
    *params = new ivfFlatSearchParams{.n_probes = 20};
    return CUVS_SUCCESS;
  } catch (...) {
    return CUVS_ERROR;
  }
}

extern "C" cuvsError_t cuvsIvfFlatSearchParamsDestroy(cuvsIvfFlatSearchParams_t params)
{
  try {
    delete params;
    return CUVS_SUCCESS;
  } catch (...) {
    return CUVS_ERROR;
  }
}
