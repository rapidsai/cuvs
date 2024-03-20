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
#include <cuvs/neighbors/ivf_pq.h>
#include <cuvs/neighbors/ivf_pq.hpp>

namespace {

template <typename IdxT>
void* _build(cuvsResources_t res, ivfPqIndexParams params, DLManagedTensor* dataset_tensor)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);

  auto build_params              = cuvs::neighbors::ivf_pq::index_params();
  build_params.metric            = static_cast<cuvs::distance::DistanceType>((int)params.metric),
  build_params.metric_arg        = params.metric_arg;
  build_params.add_data_on_build = params.add_data_on_build;
  build_params.n_lists           = params.n_lists;
  build_params.kmeans_n_iters    = params.kmeans_n_iters;
  build_params.kmeans_trainset_fraction = params.kmeans_trainset_fraction;
  build_params.pq_bits                  = params.pq_bits;
  build_params.pq_dim                   = params.pq_dim;
  build_params.codebook_kind =
    static_cast<cuvs::neighbors::ivf_pq::codebook_gen>((int)params.codebook_kind);
  build_params.force_random_rotation          = params.force_random_rotation;
  build_params.conservative_memory_allocation = params.conservative_memory_allocation;

  auto dataset = dataset_tensor->dl_tensor;
  auto dim     = dataset.shape[0];

  auto index = new cuvs::neighbors::ivf_pq::index<IdxT>(*res_ptr, build_params, dim);

  using mdspan_type = raft::device_matrix_view<float const, IdxT, raft::row_major>;
  auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);

  cuvs::neighbors::ivf_pq::build(*res_ptr, build_params, mds, index);

  return index;
}

template <typename IdxT>
void _search(cuvsResources_t res,
             ivfPqSearchParams params,
             ivfPqIndex index,
             DLManagedTensor* queries_tensor,
             DLManagedTensor* neighbors_tensor,
             DLManagedTensor* distances_tensor)
{
  auto res_ptr   = reinterpret_cast<raft::resources*>(res);
  auto index_ptr = reinterpret_cast<cuvs::neighbors::ivf_pq::index<IdxT>*>(index.addr);

  auto search_params                     = cuvs::neighbors::ivf_pq::search_params();
  search_params.n_probes                 = params.n_probes;
  search_params.lut_dtype                = params.lut_dtype;
  search_params.internal_distance_dtype  = params.internal_distance_dtype;
  search_params.preferred_shmem_carveout = params.preferred_shmem_carveout;

  using queries_mdspan_type   = raft::device_matrix_view<float const, IdxT, raft::row_major>;
  using neighbors_mdspan_type = raft::device_matrix_view<IdxT, IdxT, raft::row_major>;
  using distances_mdspan_type = raft::device_matrix_view<float, IdxT, raft::row_major>;
  auto queries_mds            = cuvs::core::from_dlpack<queries_mdspan_type>(queries_tensor);
  auto neighbors_mds          = cuvs::core::from_dlpack<neighbors_mdspan_type>(neighbors_tensor);
  auto distances_mds          = cuvs::core::from_dlpack<distances_mdspan_type>(distances_tensor);

  cuvs::neighbors::ivf_pq::search(
    *res_ptr, search_params, *index_ptr, queries_mds, neighbors_mds, distances_mds);
}

}  // namespace

extern "C" cuvsError_t ivfPqIndexCreate(cuvsIvfPqIndex_t* index)
{
  try {
    *index = new ivfPqIndex{};
    return CUVS_SUCCESS;
  } catch (...) {
    return CUVS_ERROR;
  }
}

extern "C" cuvsError_t ivfPqIndexDestroy(cuvsIvfPqIndex_t index_c_ptr)
{
  try {
    auto index = *index_c_ptr;

    auto index_ptr = reinterpret_cast<cuvs::neighbors::ivf_pq::index<int64_t>*>(index.addr);
    delete index_ptr;
    delete index_c_ptr;
    return CUVS_SUCCESS;
  } catch (...) {
    return CUVS_ERROR;
  }
}

extern "C" cuvsError_t ivfPqBuild(cuvsResources_t res,
                                  cuvsIvfPqIndexParams_t params,
                                  DLManagedTensor* dataset_tensor,
                                  cuvsIvfPqIndex_t index)
{
  try {
    auto dataset = dataset_tensor->dl_tensor;

    if ((dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) ||
        (dataset.dtype.code == kDLInt && dataset.dtype.bits == 8) ||
        (dataset.dtype.code == kDLUInt && dataset.dtype.bits == 8)) {
      index->addr = reinterpret_cast<uintptr_t>(_build<int64_t>(res, *params, dataset_tensor));
      index->dtype.code = dataset.dtype.code;
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

extern "C" cuvsError_t ivfPqSearch(cuvsResources_t res,
                                   cuvsIvfPqSearchParams_t params,
                                   cuvsIvfPqIndex_t index_c_ptr,
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

    if ((queries.dtype.code == kDLFloat && queries.dtype.bits == 32) ||
        (queries.dtype.code == kDLInt && queries.dtype.bits == 8) ||
        (queries.dtype.code == kDLUInt && queries.dtype.bits == 8)) {
      _search<int64_t>(res, *params, index, queries_tensor, neighbors_tensor, distances_tensor);
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

extern "C" cuvsError_t cuvsIvfPqIndexParamsCreate(cuvsIvfPqIndexParams_t* params)
{
  try {
    *params = new ivfPqIndexParams{.metric                         = L2Expanded,
                                   .metric_arg                     = 2.0f,
                                   .add_data_on_build              = true,
                                   .n_lists                        = 1024,
                                   .kmeans_n_iters                 = 20,
                                   .kmeans_trainset_fraction       = 0.5,
                                   .pq_bits                        = 8,
                                   .pq_dim                         = 0,
                                   .codebook_kind                  = codebook_gen::PER_SUBSPACE,
                                   .force_random_rotation          = false,
                                   .conservative_memory_allocation = false};
    return CUVS_SUCCESS;
  } catch (...) {
    return CUVS_ERROR;
  }
}

extern "C" cuvsError_t cuvsIvfPqIndexParamsDestroy(cuvsIvfPqIndexParams_t params)
{
  try {
    delete params;
    return CUVS_SUCCESS;
  } catch (...) {
    return CUVS_ERROR;
  }
}

extern "C" cuvsError_t cuvsIvfPqSearchParamsCreate(cuvsIvfPqSearchParams_t* params)
{
  try {
    *params = new ivfPqSearchParams{.n_probes                 = 20,
                                    .lut_dtype                = CUDA_R_32F,
                                    .internal_distance_dtype  = CUDA_R_32F,
                                    .preferred_shmem_carveout = 1.0};
    return CUVS_SUCCESS;
  } catch (...) {
    return CUVS_ERROR;
  }
}

extern "C" cuvsError_t cuvsIvfPqSearchParamsDestroy(cuvsIvfPqSearchParams_t params)
{
  try {
    delete params;
    return CUVS_SUCCESS;
  } catch (...) {
    return CUVS_ERROR;
  }
}
