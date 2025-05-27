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
#include <raft/core/serialize.hpp>

#include <cuvs/core/c_api.h>
#include <cuvs/core/exceptions.hpp>
#include <cuvs/core/interop.hpp>
#include <cuvs/neighbors/ivf_pq.h>
#include <cuvs/neighbors/ivf_pq.hpp>

namespace cuvs::neighbors::ivf_pq {
void convert_c_index_params(cuvsIvfPqIndexParams params, cuvs::neighbors::ivf_pq::index_params* out)
{
  out->metric                   = static_cast<cuvs::distance::DistanceType>((int)params.metric),
  out->metric_arg               = params.metric_arg;
  out->add_data_on_build        = params.add_data_on_build;
  out->n_lists                  = params.n_lists;
  out->kmeans_n_iters           = params.kmeans_n_iters;
  out->kmeans_trainset_fraction = params.kmeans_trainset_fraction;
  out->pq_bits                  = params.pq_bits;
  out->pq_dim                   = params.pq_dim;
  out->codebook_kind =
    static_cast<cuvs::neighbors::ivf_pq::codebook_gen>((int)params.codebook_kind);
  out->force_random_rotation          = params.force_random_rotation;
  out->conservative_memory_allocation = params.conservative_memory_allocation;
  out->max_train_points_per_pq_code   = params.max_train_points_per_pq_code;
}
void convert_c_search_params(cuvsIvfPqSearchParams params,
                             cuvs::neighbors::ivf_pq::search_params* out)
{
  out->n_probes                 = params.n_probes;
  out->lut_dtype                = params.lut_dtype;
  out->internal_distance_dtype  = params.internal_distance_dtype;
  out->preferred_shmem_carveout = params.preferred_shmem_carveout;
  out->coarse_search_dtype      = params.coarse_search_dtype;
  out->max_internal_batch_size  = params.max_internal_batch_size;
}

}  // namespace cuvs::neighbors::ivf_pq

namespace {

template <typename T, typename IdxT>
void* _build(cuvsResources_t res, cuvsIvfPqIndexParams params, DLManagedTensor* dataset_tensor)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);

  auto build_params = cuvs::neighbors::ivf_pq::index_params();
  convert_c_index_params(params, &build_params);

  auto dataset = dataset_tensor->dl_tensor;
  auto dim     = dataset.shape[1];

  auto index = new cuvs::neighbors::ivf_pq::index<IdxT>(*res_ptr, build_params, dim);

  if (cuvs::core::is_dlpack_device_compatible(dataset)) {
    using mdspan_type = raft::device_matrix_view<const T, IdxT, raft::row_major>;
    auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);
    cuvs::neighbors::ivf_pq::build(*res_ptr, build_params, mds, index);
  } else {
    using mdspan_type = raft::host_matrix_view<T const, int64_t, raft::row_major>;
    auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);
    cuvs::neighbors::ivf_pq::build(*res_ptr, build_params, mds, index);
  }

  return index;
}

template <typename T, typename IdxT>
void _search(cuvsResources_t res,
             cuvsIvfPqSearchParams params,
             cuvsIvfPqIndex index,
             DLManagedTensor* queries_tensor,
             DLManagedTensor* neighbors_tensor,
             DLManagedTensor* distances_tensor)
{
  auto res_ptr   = reinterpret_cast<raft::resources*>(res);
  auto index_ptr = reinterpret_cast<cuvs::neighbors::ivf_pq::index<IdxT>*>(index.addr);

  auto search_params = cuvs::neighbors::ivf_pq::search_params();
  cuvs::neighbors::ivf_pq::convert_c_search_params(params, &search_params);

  using queries_mdspan_type   = raft::device_matrix_view<const T, IdxT, raft::row_major>;
  using neighbors_mdspan_type = raft::device_matrix_view<IdxT, IdxT, raft::row_major>;
  using distances_mdspan_type = raft::device_matrix_view<float, IdxT, raft::row_major>;
  auto queries_mds            = cuvs::core::from_dlpack<queries_mdspan_type>(queries_tensor);
  auto neighbors_mds          = cuvs::core::from_dlpack<neighbors_mdspan_type>(neighbors_tensor);
  auto distances_mds          = cuvs::core::from_dlpack<distances_mdspan_type>(distances_tensor);

  cuvs::neighbors::ivf_pq::search(
    *res_ptr, search_params, *index_ptr, queries_mds, neighbors_mds, distances_mds);
}

template <typename IdxT>
void _serialize(cuvsResources_t res, const char* filename, cuvsIvfPqIndex index)
{
  auto res_ptr   = reinterpret_cast<raft::resources*>(res);
  auto index_ptr = reinterpret_cast<cuvs::neighbors::ivf_pq::index<IdxT>*>(index.addr);
  cuvs::neighbors::ivf_pq::serialize(*res_ptr, std::string(filename), *index_ptr);
}

template <typename IdxT>
void* _deserialize(cuvsResources_t res, const char* filename)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);
  auto index   = new cuvs::neighbors::ivf_pq::index<IdxT>(*res_ptr);
  cuvs::neighbors::ivf_pq::deserialize(*res_ptr, std::string(filename), index);
  return index;
}

template <typename T, typename IdxT>
void _extend(cuvsResources_t res,
             DLManagedTensor* new_vectors,
             DLManagedTensor* new_indices,
             cuvsIvfPqIndex index)
{
  auto res_ptr   = reinterpret_cast<raft::resources*>(res);
  auto index_ptr = reinterpret_cast<cuvs::neighbors::ivf_pq::index<IdxT>*>(index.addr);

  bool on_device = cuvs::core::is_dlpack_device_compatible(new_vectors->dl_tensor);
  if (on_device != cuvs::core::is_dlpack_device_compatible(new_indices->dl_tensor)) {
    RAFT_FAIL("extend inputs must both either be on device memory or host memory");
  }

  if (on_device) {
    using vectors_mdspan_type = raft::device_matrix_view<const T, IdxT, raft::row_major>;
    using indices_mdspan_type = raft::device_vector_view<IdxT, IdxT>;
    auto vectors_mds          = cuvs::core::from_dlpack<vectors_mdspan_type>(new_vectors);
    auto indices_mds          = cuvs::core::from_dlpack<indices_mdspan_type>(new_indices);
    cuvs::neighbors::ivf_pq::extend(*res_ptr, vectors_mds, indices_mds, index_ptr);
  } else {
    using vectors_mdspan_type = raft::host_matrix_view<const T, IdxT, raft::row_major>;
    using indices_mdspan_type = raft::host_vector_view<IdxT, IdxT>;
    auto vectors_mds          = cuvs::core::from_dlpack<vectors_mdspan_type>(new_vectors);
    auto indices_mds          = cuvs::core::from_dlpack<indices_mdspan_type>(new_indices);
    cuvs::neighbors::ivf_pq::extend(*res_ptr, vectors_mds, indices_mds, index_ptr);
  }
}

template <typename output_mdspan_type, typename IdxT>
void _get_centers(cuvsResources_t res, cuvsIvfPqIndex index, DLManagedTensor* centers)
{
  auto res_ptr   = reinterpret_cast<raft::resources*>(res);
  auto index_ptr = reinterpret_cast<cuvs::neighbors::ivf_pq::index<IdxT>*>(index.addr);
  auto dst       = cuvs::core::from_dlpack<output_mdspan_type>(centers);

  cuvs::neighbors::ivf_pq::helpers::extract_centers(*res_ptr, *index_ptr, dst);
}
}  // namespace

extern "C" cuvsError_t cuvsIvfPqIndexCreate(cuvsIvfPqIndex_t* index)
{
  return cuvs::core::translate_exceptions([=] { *index = new cuvsIvfPqIndex{}; });
}

extern "C" cuvsError_t cuvsIvfPqIndexDestroy(cuvsIvfPqIndex_t index_c_ptr)
{
  return cuvs::core::translate_exceptions([=] {
    auto index = *index_c_ptr;

    auto index_ptr = reinterpret_cast<cuvs::neighbors::ivf_pq::index<int64_t>*>(index.addr);
    delete index_ptr;
    delete index_c_ptr;
  });
}

extern "C" cuvsError_t cuvsIvfPqBuild(cuvsResources_t res,
                                      cuvsIvfPqIndexParams_t params,
                                      DLManagedTensor* dataset_tensor,
                                      cuvsIvfPqIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    auto dataset      = dataset_tensor->dl_tensor;
    index->dtype.code = dataset.dtype.code;
    index->dtype.bits = dataset.dtype.bits;

    if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
      index->addr =
        reinterpret_cast<uintptr_t>(_build<float, int64_t>(res, *params, dataset_tensor));
    } else if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 16) {
      index->addr =
        reinterpret_cast<uintptr_t>(_build<half, int64_t>(res, *params, dataset_tensor));
    } else if (dataset.dtype.code == kDLInt && dataset.dtype.bits == 8) {
      index->addr =
        reinterpret_cast<uintptr_t>(_build<int8_t, int64_t>(res, *params, dataset_tensor));
    } else if (dataset.dtype.code == kDLUInt && dataset.dtype.bits == 8) {
      index->addr =
        reinterpret_cast<uintptr_t>(_build<uint8_t, int64_t>(res, *params, dataset_tensor));
    } else {
      RAFT_FAIL("Unsupported dataset DLtensor dtype: %d and bits: %d",
                dataset.dtype.code,
                dataset.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsIvfPqSearch(cuvsResources_t res,
                                       cuvsIvfPqSearchParams_t params,
                                       cuvsIvfPqIndex_t index_c_ptr,
                                       DLManagedTensor* queries_tensor,
                                       DLManagedTensor* neighbors_tensor,
                                       DLManagedTensor* distances_tensor)
{
  return cuvs::core::translate_exceptions([=] {
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
    if (queries.dtype.code == kDLFloat && queries.dtype.bits == 32) {
      _search<float, int64_t>(
        res, *params, index, queries_tensor, neighbors_tensor, distances_tensor);
    } else if (queries.dtype.code == kDLFloat && queries.dtype.bits == 16) {
      _search<half, int64_t>(
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
  });
}

extern "C" cuvsError_t cuvsIvfPqIndexParamsCreate(cuvsIvfPqIndexParams_t* params)
{
  return cuvs::core::translate_exceptions([=] {
    *params = new cuvsIvfPqIndexParams{.metric                         = L2Expanded,
                                       .metric_arg                     = 2.0f,
                                       .add_data_on_build              = true,
                                       .n_lists                        = 1024,
                                       .kmeans_n_iters                 = 20,
                                       .kmeans_trainset_fraction       = 0.5,
                                       .pq_bits                        = 8,
                                       .pq_dim                         = 0,
                                       .codebook_kind                  = codebook_gen::PER_SUBSPACE,
                                       .force_random_rotation          = false,
                                       .conservative_memory_allocation = false,
                                       .max_train_points_per_pq_code   = 256};
  });
}

extern "C" cuvsError_t cuvsIvfPqIndexParamsDestroy(cuvsIvfPqIndexParams_t params)
{
  return cuvs::core::translate_exceptions([=] { delete params; });
}

extern "C" cuvsError_t cuvsIvfPqSearchParamsCreate(cuvsIvfPqSearchParams_t* params)
{
  return cuvs::core::translate_exceptions([=] {
    *params = new cuvsIvfPqSearchParams{.n_probes                 = 20,
                                        .lut_dtype                = CUDA_R_32F,
                                        .internal_distance_dtype  = CUDA_R_32F,
                                        .coarse_search_dtype      = CUDA_R_32F,
                                        .max_internal_batch_size  = 4096,
                                        .preferred_shmem_carveout = 1.0};
  });
}

extern "C" cuvsError_t cuvsIvfPqSearchParamsDestroy(cuvsIvfPqSearchParams_t params)
{
  return cuvs::core::translate_exceptions([=] { delete params; });
}

extern "C" cuvsError_t cuvsIvfPqDeserialize(cuvsResources_t res,
                                            const char* filename,
                                            cuvsIvfPqIndex_t index)
{
  return cuvs::core::translate_exceptions(
    [=] { index->addr = reinterpret_cast<uintptr_t>(_deserialize<int64_t>(res, filename)); });
}

extern "C" cuvsError_t cuvsIvfPqSerialize(cuvsResources_t res,
                                          const char* filename,
                                          cuvsIvfPqIndex_t index)
{
  return cuvs::core::translate_exceptions([=] { _serialize<int64_t>(res, filename, *index); });
}

extern "C" cuvsError_t cuvsIvfPqExtend(cuvsResources_t res,
                                       DLManagedTensor* new_vectors,
                                       DLManagedTensor* new_indices,
                                       cuvsIvfPqIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    auto vectors = new_vectors->dl_tensor;

    if (vectors.dtype.code == kDLFloat && vectors.dtype.bits == 32) {
      _extend<float, int64_t>(res, new_vectors, new_indices, *index);
    } else if (vectors.dtype.code == kDLFloat && vectors.dtype.bits == 16) {
      _extend<half, int64_t>(res, new_vectors, new_indices, *index);
    } else if (vectors.dtype.code == kDLInt && vectors.dtype.bits == 8) {
      _extend<int8_t, int64_t>(res, new_vectors, new_indices, *index);
    } else if (vectors.dtype.code == kDLUInt && vectors.dtype.bits == 8) {
      _extend<uint8_t, int64_t>(res, new_vectors, new_indices, *index);
    } else {
      RAFT_FAIL("Unsupported index dtype: %d and bits: %d", vectors.dtype.code, vectors.dtype.bits);
    }
  });
}

extern "C" uint32_t cuvsIvfPqIndexGetNLists(cuvsIvfPqIndex_t index)
{
  auto index_ptr = reinterpret_cast<cuvs::neighbors::ivf_pq::index<int64_t>*>(index->addr);
  return index_ptr->n_lists();
}

extern "C" uint32_t cuvsIvfPqIndexGetDim(cuvsIvfPqIndex_t index)
{
  auto index_ptr = reinterpret_cast<cuvs::neighbors::ivf_pq::index<int64_t>*>(index->addr);
  return index_ptr->dim();
}

extern "C" cuvsError_t cuvsIvfPqIndexGetCenters(cuvsResources_t res,
                                                cuvsIvfPqIndex_t index,
                                                DLManagedTensor* centers)
{
  return cuvs::core::translate_exceptions([=] {
    if (cuvs::core::is_dlpack_device_compatible(centers->dl_tensor)) {
      using output_mdspan_type = raft::device_matrix_view<float, int64_t, raft::row_major>;
      _get_centers<output_mdspan_type, int64_t>(res, *index, centers);
    } else {
      using output_mdspan_type = raft::host_matrix_view<float, int64_t, raft::row_major>;
      _get_centers<output_mdspan_type, int64_t>(res, *index, centers);
    }
  });
}
