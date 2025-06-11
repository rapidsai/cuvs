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

#include <cstdint>
#include <dlpack/dlpack.h>

#include <raft/core/error.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/serialize.hpp>

#include <cuvs/core/c_api.h>
#include <cuvs/core/exceptions.hpp>
#include <cuvs/core/interop.hpp>
#include <cuvs/neighbors/cagra.h>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/common.h>

#include "cagra_c.hpp"
#include <fstream>

namespace {

static void _set_graph_build_params(
  std::variant<std::monostate,
               cuvs::neighbors::cagra::graph_build_params::ivf_pq_params,
               cuvs::neighbors::cagra::graph_build_params::nn_descent_params,
               cuvs::neighbors::cagra::graph_build_params::iterative_search_params>& out_params,
  cuvsCagraIndexParams& params,
  cuvsCagraGraphBuildAlgo algo,
  int64_t n_rows,
  int64_t dim)

{
  auto metric = static_cast<cuvs::distance::DistanceType>((int)params.metric);
  switch (algo) {
    case cuvsCagraGraphBuildAlgo::AUTO_SELECT: break;
    case cuvsCagraGraphBuildAlgo::IVF_PQ: {
      auto pq_params = cuvs::neighbors::cagra::graph_build_params::ivf_pq_params(
        raft::matrix_extent<int64_t>(n_rows, dim), metric);
      if (params.graph_build_params) {
        auto ivf_params = static_cast<cuvsIvfPqParams*>(params.graph_build_params);
        if (ivf_params->ivf_pq_build_params) {
          auto bp                                         = ivf_params->ivf_pq_build_params;
          pq_params.build_params.add_data_on_build        = bp->add_data_on_build;
          pq_params.build_params.n_lists                  = bp->n_lists;
          pq_params.build_params.kmeans_n_iters           = bp->kmeans_n_iters;
          pq_params.build_params.kmeans_trainset_fraction = bp->kmeans_trainset_fraction;
          pq_params.build_params.pq_bits                  = bp->pq_bits;
          pq_params.build_params.pq_dim                   = bp->pq_dim;
          pq_params.build_params.codebook_kind =
            static_cast<cuvs::neighbors::ivf_pq::codebook_gen>(bp->codebook_kind);
          pq_params.build_params.force_random_rotation = bp->force_random_rotation;
          pq_params.build_params.conservative_memory_allocation =
            bp->conservative_memory_allocation;
          pq_params.build_params.max_train_points_per_pq_code = bp->max_train_points_per_pq_code;
        }
        if (ivf_params->ivf_pq_search_params) {
          auto sp                                          = ivf_params->ivf_pq_search_params;
          pq_params.search_params.n_probes                 = sp->n_probes;
          pq_params.search_params.lut_dtype                = sp->lut_dtype;
          pq_params.search_params.internal_distance_dtype  = sp->internal_distance_dtype;
          pq_params.search_params.preferred_shmem_carveout = sp->preferred_shmem_carveout;
        }
        if (ivf_params->refinement_rate > 1.0f) {
          pq_params.refinement_rate = ivf_params->refinement_rate;
        }
      }
      out_params = pq_params;
      break;
    }
    case cuvsCagraGraphBuildAlgo::NN_DESCENT: {
      auto nn_params =
        cuvs::neighbors::nn_descent::index_params(params.intermediate_graph_degree, metric);
      nn_params.max_iterations = params.nn_descent_niter;
      out_params               = nn_params;
      break;
    }
    case cuvsCagraGraphBuildAlgo::ITERATIVE_CAGRA_SEARCH: {
      cuvs::neighbors::cagra::graph_build_params::iterative_search_params p;
      out_params = p;
      break;
    }
  }
}

template <typename T>
void* _build(cuvsResources_t res, cuvsCagraIndexParams params, DLManagedTensor* dataset_tensor)
{
  auto dataset = dataset_tensor->dl_tensor;

  auto res_ptr = reinterpret_cast<raft::resources*>(res);
  auto index   = new cuvs::neighbors::cagra::index<T, uint32_t>(*res_ptr);

  auto index_params = cuvs::neighbors::cagra::index_params();
  convert_c_index_params(params, dataset.shape[0], dataset.shape[1], &index_params);

  if (cuvs::core::is_dlpack_device_compatible(dataset)) {
    using mdspan_type = raft::device_matrix_view<T const, int64_t, raft::row_major>;
    auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);
    *index            = cuvs::neighbors::cagra::build(*res_ptr, index_params, mds);
  } else if (cuvs::core::is_dlpack_host_compatible(dataset)) {
    using mdspan_type = raft::host_matrix_view<T const, int64_t, raft::row_major>;
    auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);
    *index            = cuvs::neighbors::cagra::build(*res_ptr, index_params, mds);
  }
  return index;
}

template <typename T>
void _extend(cuvsResources_t res,
             cuvsCagraExtendParams params,
             cuvsCagraIndex index,
             DLManagedTensor* additional_dataset_tensor,
             DLManagedTensor* return_tensor)
{
  auto dataset          = additional_dataset_tensor->dl_tensor;
  auto return_dl_tensor = return_tensor->dl_tensor;
  auto index_ptr        = reinterpret_cast<cuvs::neighbors::cagra::index<T, uint32_t>*>(index.addr);
  auto res_ptr          = reinterpret_cast<raft::resources*>(res);

  // TODO: use C struct here (see issue #487)
  auto extend_params           = cuvs::neighbors::cagra::extend_params();
  extend_params.max_chunk_size = params.max_chunk_size;

  if (cuvs::core::is_dlpack_device_compatible(dataset) &&
      cuvs::core::is_dlpack_device_compatible(return_dl_tensor)) {
    using mdspan_type        = raft::device_matrix_view<T const, int64_t, raft::row_major>;
    using mdspan_return_type = raft::device_matrix_view<T, int64_t, raft::row_major>;
    auto mds                 = cuvs::core::from_dlpack<mdspan_type>(additional_dataset_tensor);
    auto return_mds          = cuvs::core::from_dlpack<mdspan_return_type>(return_tensor);
    cuvs::neighbors::cagra::extend(*res_ptr, extend_params, mds, *index_ptr, return_mds);
  } else if (cuvs::core::is_dlpack_host_compatible(dataset) &&
             cuvs::core::is_dlpack_host_compatible(return_dl_tensor)) {
    using mdspan_type        = raft::host_matrix_view<T const, int64_t, raft::row_major>;
    using mdspan_return_type = raft::device_matrix_view<T, int64_t, raft::row_major>;
    auto mds                 = cuvs::core::from_dlpack<mdspan_type>(additional_dataset_tensor);
    auto return_mds          = cuvs::core::from_dlpack<mdspan_return_type>(return_tensor);
    cuvs::neighbors::cagra::extend(*res_ptr, extend_params, mds, *index_ptr, return_mds);
  } else {
    RAFT_FAIL("Unsupported dataset DLtensor dtype: %d and bits: %d",
              dataset.dtype.code,
              dataset.dtype.bits);
  }
}

template <typename T, typename IdxT>
void _search(cuvsResources_t res,
             cuvsCagraSearchParams params,
             cuvsCagraIndex index,
             DLManagedTensor* queries_tensor,
             DLManagedTensor* neighbors_tensor,
             DLManagedTensor* distances_tensor,
             cuvsFilter filter)
{
  auto res_ptr   = reinterpret_cast<raft::resources*>(res);
  auto index_ptr = reinterpret_cast<cuvs::neighbors::cagra::index<T, uint32_t>*>(index.addr);

  auto search_params = cuvs::neighbors::cagra::search_params();
  convert_c_search_params(params, &search_params);

  using queries_mdspan_type   = raft::device_matrix_view<T const, int64_t, raft::row_major>;
  using neighbors_mdspan_type = raft::device_matrix_view<IdxT, int64_t, raft::row_major>;
  using distances_mdspan_type = raft::device_matrix_view<float, int64_t, raft::row_major>;
  auto queries_mds            = cuvs::core::from_dlpack<queries_mdspan_type>(queries_tensor);
  auto neighbors_mds          = cuvs::core::from_dlpack<neighbors_mdspan_type>(neighbors_tensor);
  auto distances_mds          = cuvs::core::from_dlpack<distances_mdspan_type>(distances_tensor);
  if (filter.type == NO_FILTER) {
    cuvs::neighbors::cagra::search(
      *res_ptr, search_params, *index_ptr, queries_mds, neighbors_mds, distances_mds);
  } else if (filter.type == BITSET) {
    using filter_mdspan_type    = raft::device_vector_view<std::uint32_t, int64_t, raft::row_major>;
    auto removed_indices_tensor = reinterpret_cast<DLManagedTensor*>(filter.addr);
    auto removed_indices = cuvs::core::from_dlpack<filter_mdspan_type>(removed_indices_tensor);
    cuvs::core::bitset_view<std::uint32_t, int64_t> removed_indices_bitset(
      removed_indices, index_ptr->dataset().extent(0));
    auto bitset_filter_obj = cuvs::neighbors::filtering::bitset_filter(removed_indices_bitset);
    cuvs::neighbors::cagra::search(*res_ptr,
                                   search_params,
                                   *index_ptr,
                                   queries_mds,
                                   neighbors_mds,
                                   distances_mds,
                                   bitset_filter_obj);
  } else {
    RAFT_FAIL("Unsupported filter type: BITMAP");
  }
}

template <typename T>
void _search(cuvsResources_t res,
             cuvsCagraSearchParams params,
             cuvsCagraIndex index,
             DLManagedTensor* queries_tensor,
             DLManagedTensor* neighbors_tensor,
             DLManagedTensor* distances_tensor,
             cuvsFilter filter)
{
  if (neighbors_tensor->dl_tensor.dtype.code == kDLUInt &&
      neighbors_tensor->dl_tensor.dtype.bits == 32) {
    _search<T, uint32_t>(
      res, params, index, queries_tensor, neighbors_tensor, distances_tensor, filter);
  } else if (neighbors_tensor->dl_tensor.dtype.code == kDLInt &&
             neighbors_tensor->dl_tensor.dtype.bits == 64) {
    _search<T, int64_t>(
      res, params, index, queries_tensor, neighbors_tensor, distances_tensor, filter);
  } else {
    RAFT_FAIL("neighbors should be of type uint32_t or int64_t");
  }
}

template <typename T>
void _serialize(cuvsResources_t res,
                const char* filename,
                cuvsCagraIndex_t index,
                bool include_dataset)
{
  auto res_ptr   = reinterpret_cast<raft::resources*>(res);
  auto index_ptr = reinterpret_cast<cuvs::neighbors::cagra::index<T, uint32_t>*>(index->addr);
  cuvs::neighbors::cagra::serialize(*res_ptr, std::string(filename), *index_ptr, include_dataset);
}

template <typename T>
void _serialize_to_hnswlib(cuvsResources_t res, const char* filename, cuvsCagraIndex_t index)
{
  auto res_ptr   = reinterpret_cast<raft::resources*>(res);
  auto index_ptr = reinterpret_cast<cuvs::neighbors::cagra::index<T, uint32_t>*>(index->addr);
  cuvs::neighbors::cagra::serialize_to_hnswlib(*res_ptr, std::string(filename), *index_ptr);
}

template <typename T>
void* _deserialize(cuvsResources_t res, const char* filename)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);
  auto index   = new cuvs::neighbors::cagra::index<T, uint32_t>(*res_ptr);
  cuvs::neighbors::cagra::deserialize(*res_ptr, std::string(filename), index);
  return index;
}

template <typename T>
void* _merge(cuvsResources_t res,
             cuvsCagraMergeParams params,
             cuvsCagraIndex_t* indices,
             size_t num_indices)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);
  cuvs::neighbors::cagra::merge_params merge_params_cpp;
  auto& out_idx_params = *params.output_index_params;

  merge_params_cpp.output_index_params.metric =
    static_cast<cuvs::distance::DistanceType>((int)out_idx_params.metric);
  merge_params_cpp.output_index_params.intermediate_graph_degree =
    out_idx_params.intermediate_graph_degree;
  merge_params_cpp.output_index_params.graph_degree = out_idx_params.graph_degree;

  int64_t total_size = 0;
  int64_t dim        = 0;
  if (out_idx_params.build_algo == cuvsCagraGraphBuildAlgo::IVF_PQ) {
    auto first_idx_ptr =
      reinterpret_cast<cuvs::neighbors::cagra::index<T, uint32_t>*>(indices[0]->addr);
    dim = first_idx_ptr->dim();
    for (size_t i = 0; i < num_indices; ++i) {
      auto idx_ptr =
        reinterpret_cast<cuvs::neighbors::cagra::index<T, uint32_t>*>(indices[i]->addr);
      total_size += idx_ptr->size();
    }
  }

  _set_graph_build_params(merge_params_cpp.output_index_params.graph_build_params,
                          out_idx_params,
                          out_idx_params.build_algo,
                          total_size,
                          dim);

  std::vector<cuvs::neighbors::cagra::index<T, uint32_t>*> index_ptrs;
  index_ptrs.reserve(num_indices);
  for (size_t i = 0; i < num_indices; ++i) {
    auto idx_ptr = reinterpret_cast<cuvs::neighbors::cagra::index<T, uint32_t>*>(indices[i]->addr);
    index_ptrs.push_back(idx_ptr);
  }

  auto merged_index = new cuvs::neighbors::cagra::index<T, uint32_t>(
    cuvs::neighbors::cagra::merge(*res_ptr, merge_params_cpp, index_ptrs));

  return merged_index;
}
}  // namespace

namespace cuvs::neighbors::cagra {
void convert_c_index_params(cuvsCagraIndexParams params,
                            int64_t n_rows,
                            int64_t dim,
                            cuvs::neighbors::cagra::index_params* out)
{
  out->metric                    = static_cast<cuvs::distance::DistanceType>((int)params.metric);
  out->intermediate_graph_degree = params.intermediate_graph_degree;
  out->graph_degree              = params.graph_degree;
  _set_graph_build_params(out->graph_build_params, params, params.build_algo, n_rows, dim);

  if (auto* cparams = params.compression; cparams != nullptr) {
    auto compression_params                        = cuvs::neighbors::vpq_params();
    compression_params.pq_bits                     = cparams->pq_bits;
    compression_params.pq_dim                      = cparams->pq_dim;
    compression_params.vq_n_centers                = cparams->vq_n_centers;
    compression_params.kmeans_n_iters              = cparams->kmeans_n_iters;
    compression_params.vq_kmeans_trainset_fraction = cparams->vq_kmeans_trainset_fraction;
    compression_params.pq_kmeans_trainset_fraction = cparams->pq_kmeans_trainset_fraction;
    out->compression.emplace(compression_params);
  }
}
void convert_c_search_params(cuvsCagraSearchParams params,
                             cuvs::neighbors::cagra::search_params* out)
{
  out->max_queries           = params.max_queries;
  out->itopk_size            = params.itopk_size;
  out->max_iterations        = params.max_iterations;
  out->algo                  = static_cast<cuvs::neighbors::cagra::search_algo>(params.algo);
  out->team_size             = params.team_size;
  out->search_width          = params.search_width;
  out->min_iterations        = params.min_iterations;
  out->thread_block_size     = params.thread_block_size;
  out->hashmap_mode          = static_cast<cuvs::neighbors::cagra::hash_mode>(params.hashmap_mode);
  out->hashmap_min_bitlen    = params.hashmap_min_bitlen;
  out->hashmap_max_fill_rate = params.hashmap_max_fill_rate;
  out->num_random_samplings  = params.num_random_samplings;
  out->rand_xor_mask         = params.rand_xor_mask;
  out->persistent            = params.persistent;
  out->persistent_lifetime   = params.persistent_lifetime;
  out->persistent_device_usage = params.persistent_device_usage;
}
}  // namespace cuvs::neighbors::cagra
extern "C" cuvsError_t cuvsCagraIndexCreate(cuvsCagraIndex_t* index)
{
  return cuvs::core::translate_exceptions([=] { *index = new cuvsCagraIndex{}; });
}

extern "C" cuvsError_t cuvsCagraIndexDestroy(cuvsCagraIndex_t index_c_ptr)
{
  return cuvs::core::translate_exceptions([=] {
    auto index = *index_c_ptr;

    if (index.dtype.code == kDLFloat && index.dtype.bits == 32) {
      auto index_ptr =
        reinterpret_cast<cuvs::neighbors::cagra::index<float, uint32_t>*>(index.addr);
      delete index_ptr;
    } else if (index.dtype.code == kDLFloat && index.dtype.bits == 16) {
      auto index_ptr = reinterpret_cast<cuvs::neighbors::cagra::index<half, uint32_t>*>(index.addr);
      delete index_ptr;
    } else if (index.dtype.code == kDLInt && index.dtype.bits == 8) {
      auto index_ptr =
        reinterpret_cast<cuvs::neighbors::cagra::index<int8_t, uint32_t>*>(index.addr);
      delete index_ptr;
    } else if (index.dtype.code == kDLUInt && index.dtype.bits == 8) {
      auto index_ptr =
        reinterpret_cast<cuvs::neighbors::cagra::index<uint8_t, uint32_t>*>(index.addr);
      delete index_ptr;
    }
    delete index_c_ptr;
  });
}

extern "C" cuvsError_t cuvsCagraIndexGetDims(cuvsCagraIndex_t index, int* dim)
{
  return cuvs::core::translate_exceptions([=] {
    auto index_ptr = reinterpret_cast<cuvs::neighbors::cagra::index<float, uint32_t>*>(index->addr);
    *dim           = index_ptr->dim();
  });
}

extern "C" cuvsError_t cuvsCagraBuild(cuvsResources_t res,
                                      cuvsCagraIndexParams_t params,
                                      DLManagedTensor* dataset_tensor,
                                      cuvsCagraIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    auto dataset = dataset_tensor->dl_tensor;
    index->dtype = dataset.dtype;
    if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
      index->addr = reinterpret_cast<uintptr_t>(_build<float>(res, *params, dataset_tensor));
    } else if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 16) {
      index->addr = reinterpret_cast<uintptr_t>(_build<half>(res, *params, dataset_tensor));
    } else if (dataset.dtype.code == kDLInt && dataset.dtype.bits == 8) {
      index->addr = reinterpret_cast<uintptr_t>(_build<int8_t>(res, *params, dataset_tensor));
    } else if (dataset.dtype.code == kDLUInt && dataset.dtype.bits == 8) {
      index->addr = reinterpret_cast<uintptr_t>(_build<uint8_t>(res, *params, dataset_tensor));
    } else {
      RAFT_FAIL("Unsupported dataset DLtensor dtype: %d and bits: %d",
                dataset.dtype.code,
                dataset.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsCagraExtend(cuvsResources_t res,
                                       cuvsCagraExtendParams_t params,
                                       DLManagedTensor* additional_dataset_tensor,
                                       cuvsCagraIndex_t index_c_ptr,
                                       DLManagedTensor* return_dataset_tensor)
{
  return cuvs::core::translate_exceptions([=] {
    auto dataset = additional_dataset_tensor->dl_tensor;
    auto index   = *index_c_ptr;

    if ((dataset.dtype.code == kDLFloat) && (dataset.dtype.bits == 32)) {
      _extend<float>(res, *params, index, additional_dataset_tensor, return_dataset_tensor);
    } else if (dataset.dtype.code == kDLInt && dataset.dtype.bits == 8) {
      _extend<int8_t>(res, *params, index, additional_dataset_tensor, return_dataset_tensor);
    } else if (dataset.dtype.code == kDLUInt && dataset.dtype.bits == 8) {
      _extend<uint8_t>(res, *params, index, additional_dataset_tensor, return_dataset_tensor);
    } else {
      RAFT_FAIL("Unsupported dataset DLtensor dtype: %d and bits: %d",
                dataset.dtype.code,
                dataset.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsCagraSearch(cuvsResources_t res,
                                       cuvsCagraSearchParams_t params,
                                       cuvsCagraIndex_t index_c_ptr,
                                       DLManagedTensor* queries_tensor,
                                       DLManagedTensor* neighbors_tensor,
                                       DLManagedTensor* distances_tensor,
                                       cuvsFilter filter)
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

    // NB: the dtype of neighbors is checked later in _search function
    RAFT_EXPECTS(distances.dtype.code == kDLFloat && distances.dtype.bits == 32,
                 "distances should be of type float32");

    auto index = *index_c_ptr;
    RAFT_EXPECTS(queries.dtype.code == index.dtype.code, "type mismatch between index and queries");

    if (queries.dtype.code == kDLFloat && queries.dtype.bits == 32) {
      _search<float>(
        res, *params, index, queries_tensor, neighbors_tensor, distances_tensor, filter);
    } else if (queries.dtype.code == kDLFloat && queries.dtype.bits == 16) {
      _search<half>(
        res, *params, index, queries_tensor, neighbors_tensor, distances_tensor, filter);
    } else if (queries.dtype.code == kDLInt && queries.dtype.bits == 8) {
      _search<int8_t>(
        res, *params, index, queries_tensor, neighbors_tensor, distances_tensor, filter);
    } else if (queries.dtype.code == kDLUInt && queries.dtype.bits == 8) {
      _search<uint8_t>(
        res, *params, index, queries_tensor, neighbors_tensor, distances_tensor, filter);
    } else {
      RAFT_FAIL("Unsupported queries DLtensor dtype: %d and bits: %d",
                queries.dtype.code,
                queries.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsCagraMerge(cuvsResources_t res,
                                      cuvsCagraMergeParams_t params,
                                      cuvsCagraIndex_t* indices,
                                      size_t num_indices,
                                      cuvsCagraIndex_t output_index)
{
  return cuvs::core::translate_exceptions([=] {
    // Basic checks on inputs
    RAFT_EXPECTS(indices != nullptr && num_indices > 0, "indices array cannot be null or empty");
    // Use first index dtype as reference
    auto dtype = (*indices[0]).dtype;
    for (size_t i = 1; i < num_indices; ++i) {
      RAFT_EXPECTS((*indices[i]).dtype.code == dtype.code && (*indices[i]).dtype.bits == dtype.bits,
                   "All input indices must have the same data type");
      RAFT_EXPECTS((*indices[i]).addr != 0, "All input indices must be built (non-empty)");
    }
    RAFT_EXPECTS(output_index != nullptr, "Output index pointer must not be null");
    output_index->dtype = dtype;  // output index type matches inputs
    // Dispatch based on data type
    if (dtype.code == kDLFloat && dtype.bits == 32) {
      output_index->addr =
        reinterpret_cast<uintptr_t>(_merge<float>(res, *params, indices, num_indices));
    } else if (dtype.code == kDLFloat && dtype.bits == 16) {
      output_index->addr =
        reinterpret_cast<uintptr_t>(_merge<half>(res, *params, indices, num_indices));
    } else if (dtype.code == kDLInt && dtype.bits == 8) {
      output_index->addr =
        reinterpret_cast<uintptr_t>(_merge<int8_t>(res, *params, indices, num_indices));
    } else if (dtype.code == kDLUInt && dtype.bits == 8) {
      output_index->addr =
        reinterpret_cast<uintptr_t>(_merge<uint8_t>(res, *params, indices, num_indices));
    } else {
      RAFT_FAIL("Unsupported index data type: code=%d, bits=%d", dtype.code, dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsCagraIndexParamsCreate(cuvsCagraIndexParams_t* params)
{
  return cuvs::core::translate_exceptions([=] {
    *params                       = new cuvsCagraIndexParams{.metric                    = L2Expanded,
                                                             .intermediate_graph_degree = 128,
                                                             .graph_degree              = 64,
                                                             .build_algo                = IVF_PQ,
                                                             .nn_descent_niter          = 20};
    (*params)->graph_build_params = new cuvsIvfPqParams{nullptr, nullptr, 1};
  });
}

extern "C" cuvsError_t cuvsCagraIndexParamsDestroy(cuvsCagraIndexParams_t params)
{
  return cuvs::core::translate_exceptions([=] {
    delete params->graph_build_params;
    delete params;
  });
}

extern "C" cuvsError_t cuvsCagraCompressionParamsCreate(cuvsCagraCompressionParams_t* params)
{
  return cuvs::core::translate_exceptions([=] {
    auto ps = cuvs::neighbors::vpq_params();
    *params =
      new cuvsCagraCompressionParams{.pq_bits                     = ps.pq_bits,
                                     .pq_dim                      = ps.pq_dim,
                                     .vq_n_centers                = ps.vq_n_centers,
                                     .kmeans_n_iters              = ps.kmeans_n_iters,
                                     .vq_kmeans_trainset_fraction = ps.vq_kmeans_trainset_fraction,
                                     .pq_kmeans_trainset_fraction = ps.pq_kmeans_trainset_fraction};
  });
}

extern "C" cuvsError_t cuvsCagraCompressionParamsDestroy(cuvsCagraCompressionParams_t params)
{
  return cuvs::core::translate_exceptions([=] { delete params; });
}

extern "C" cuvsError_t cuvsCagraExtendParamsCreate(cuvsCagraExtendParams_t* params)
{
  return cuvs::core::translate_exceptions(
    [=] { *params = new cuvsCagraExtendParams{.max_chunk_size = 0}; });
}

extern "C" cuvsError_t cuvsCagraExtendParamsDestroy(cuvsCagraExtendParams_t params)
{
  return cuvs::core::translate_exceptions([=] { delete params; });
}

extern "C" cuvsError_t cuvsCagraSearchParamsCreate(cuvsCagraSearchParams_t* params)
{
  return cuvs::core::translate_exceptions([=] {
    *params = new cuvsCagraSearchParams{
      .itopk_size              = 64,
      .search_width            = 1,
      .hashmap_max_fill_rate   = 0.5,
      .num_random_samplings    = 1,
      .rand_xor_mask           = 0x128394,
      .persistent              = false,
      .persistent_lifetime     = 2,
      .persistent_device_usage = 1.0,
    };
  });
}

extern "C" cuvsError_t cuvsCagraSearchParamsDestroy(cuvsCagraSearchParams_t params)
{
  return cuvs::core::translate_exceptions([=] { delete params; });
}

extern "C" cuvsError_t cuvsCagraMergeParamsCreate(cuvsCagraMergeParams_t* params)
{
  return cuvs::core::translate_exceptions([=] {
    cuvsCagraIndexParams_t idx_params;
    cuvsCagraIndexParamsCreate(&idx_params);
    *params = new cuvsCagraMergeParams{.output_index_params = idx_params,
                                       .strategy            = MERGE_STRATEGY_PHYSICAL};
  });
}

extern "C" cuvsError_t cuvsCagraMergeParamsDestroy(cuvsCagraMergeParams_t params)
{
  return cuvs::core::translate_exceptions([=] {
    cuvsCagraIndexParamsDestroy(params->output_index_params);
    delete params;
  });
}

extern "C" cuvsError_t cuvsCagraDeserialize(cuvsResources_t res,
                                            const char* filename,
                                            cuvsCagraIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    // read the numpy dtype from the beginning of the file
    std::ifstream is(filename, std::ios::in | std::ios::binary);
    if (!is) { RAFT_FAIL("Cannot open file %s", filename); }
    char dtype_string[4];
    is.read(dtype_string, 4);
    auto dtype = raft::detail::numpy_serializer::parse_descr(std::string(dtype_string, 4));

    index->dtype.bits = dtype.itemsize * 8;
    if (dtype.kind == 'f' && dtype.itemsize == 4) {
      index->addr       = reinterpret_cast<uintptr_t>(_deserialize<float>(res, filename));
      index->dtype.code = kDLFloat;
    } else if (dtype.kind == 'f' && dtype.itemsize == 2) {
      index->addr       = reinterpret_cast<uintptr_t>(_deserialize<half>(res, filename));
      index->dtype.code = kDLFloat;
    } else if (dtype.kind == 'i' && dtype.itemsize == 1) {
      index->addr       = reinterpret_cast<uintptr_t>(_deserialize<int8_t>(res, filename));
      index->dtype.code = kDLInt;
    } else if (dtype.kind == 'u' && dtype.itemsize == 1) {
      index->addr       = reinterpret_cast<uintptr_t>(_deserialize<uint8_t>(res, filename));
      index->dtype.code = kDLUInt;
    } else {
      RAFT_FAIL("Unsupported dtype in file %s", filename);
    }
  });
}

extern "C" cuvsError_t cuvsCagraSerialize(cuvsResources_t res,
                                          const char* filename,
                                          cuvsCagraIndex_t index,
                                          bool include_dataset)
{
  return cuvs::core::translate_exceptions([=] {
    if (index->dtype.code == kDLFloat && index->dtype.bits == 32) {
      _serialize<float>(res, filename, index, include_dataset);
    } else if (index->dtype.code == kDLFloat && index->dtype.bits == 16) {
      _serialize<half>(res, filename, index, include_dataset);
    } else if (index->dtype.code == kDLInt && index->dtype.bits == 8) {
      _serialize<int8_t>(res, filename, index, include_dataset);
    } else if (index->dtype.code == kDLUInt && index->dtype.bits == 8) {
      _serialize<uint8_t>(res, filename, index, include_dataset);
    } else {
      RAFT_FAIL("Unsupported index dtype: %d and bits: %d", index->dtype.code, index->dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsCagraSerializeToHnswlib(cuvsResources_t res,
                                                   const char* filename,
                                                   cuvsCagraIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    if (index->dtype.code == kDLFloat && index->dtype.bits == 32) {
      _serialize_to_hnswlib<float>(res, filename, index);
    } else if (index->dtype.code == kDLFloat && index->dtype.bits == 16) {
      _serialize_to_hnswlib<half>(res, filename, index);
    } else if (index->dtype.code == kDLInt && index->dtype.bits == 8) {
      _serialize_to_hnswlib<int8_t>(res, filename, index);
    } else if (index->dtype.code == kDLUInt && index->dtype.bits == 8) {
      _serialize_to_hnswlib<uint8_t>(res, filename, index);
    } else {
      RAFT_FAIL("Unsupported index dtype: %d and bits: %d", index->dtype.code, index->dtype.bits);
    }
  });
}
