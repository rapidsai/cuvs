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

#include <cstdint>
#include <dlpack/dlpack.h>

#include <raft/core/error.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/serialize.hpp>

#include <cuvs/core/c_api.h>
#include <cuvs/core/exceptions.hpp>
#include <cuvs/core/interop.hpp>
#include <cuvs/neighbors/tiered_index.h>
#include <cuvs/neighbors/tiered_index.hpp>

#include <fstream>

#include "cagra_c.hpp"
#include "ivf_flat_c.hpp"
#include "ivf_pq_c.hpp"

namespace {
using namespace cuvs::neighbors;

template <typename T>
void* _build(cuvsResources_t res, cuvsTieredIndexParams params, DLManagedTensor* dataset_tensor)
{
  auto res_ptr      = reinterpret_cast<raft::resources*>(res);
  using mdspan_type = raft::device_matrix_view<const T, int64_t, raft::row_major>;
  auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);

  switch (params.algo) {
    case CUVS_TIERED_INDEX_ALGO_CAGRA: {
      auto build_params = tiered_index::index_params<cagra::index_params>();
      if (params.cagra_params != NULL) {
        auto dataset = dataset_tensor->dl_tensor;
        cagra::convert_c_index_params(
          *params.cagra_params, dataset.shape[0], dataset.shape[1], &build_params);
      }
      build_params.min_ann_rows               = params.min_ann_rows;
      build_params.create_ann_index_on_extend = params.create_ann_index_on_extend;
      build_params.metric                     = params.metric;
      return new tiered_index::index<cagra::index<T, uint32_t>>(
        tiered_index::build(*res_ptr, build_params, mds));
    }
    case CUVS_TIERED_INDEX_ALGO_IVF_FLAT: {
      auto build_params = tiered_index::index_params<ivf_flat::index_params>();
      if (params.ivf_flat_params != NULL) {
        ivf_flat::convert_c_index_params(*params.ivf_flat_params, &build_params);
      }
      build_params.metric                     = params.metric;
      build_params.min_ann_rows               = params.min_ann_rows;
      build_params.create_ann_index_on_extend = params.create_ann_index_on_extend;
      return new tiered_index::index<ivf_flat::index<T, int64_t>>(
        tiered_index::build(*res_ptr, build_params, mds));
    }
    case CUVS_TIERED_INDEX_ALGO_IVF_PQ: {
      auto build_params   = tiered_index::index_params<ivf_pq::index_params>();
      build_params.metric = params.metric;
      if (params.ivf_pq_params != NULL) {
        ivf_pq::convert_c_index_params(*params.ivf_pq_params, &build_params);
      }
      build_params.metric                     = params.metric;
      build_params.min_ann_rows               = params.min_ann_rows;
      build_params.create_ann_index_on_extend = params.create_ann_index_on_extend;
      return new tiered_index::index<ivf_pq::typed_index<T, int64_t>>(
        tiered_index::build(*res_ptr, build_params, mds));
    }
    default: RAFT_FAIL("unsupported tiered index algorithm");
  }
}

template <typename UpstreamT>
void _search(cuvsResources_t res,
             void* params,
             cuvsTieredIndex index,
             DLManagedTensor* queries_tensor,
             DLManagedTensor* neighbors_tensor,
             DLManagedTensor* distances_tensor,
             cuvsFilter filter)
{
  auto res_ptr   = reinterpret_cast<raft::resources*>(res);
  auto index_ptr = reinterpret_cast<tiered_index::index<UpstreamT>*>(index.addr);

  auto search_params = typename UpstreamT::search_params_type();
  if (params != NULL) {
    if constexpr (std::is_same_v<typename UpstreamT::search_params_type, cagra::search_params>) {
      auto c_params = reinterpret_cast<cuvsCagraSearchParams*>(params);
      cagra::convert_c_search_params(*c_params, &search_params);
    }
    if constexpr (std::is_same_v<typename UpstreamT::search_params_type, ivf_flat::search_params>) {
      auto c_params = reinterpret_cast<cuvsIvfFlatSearchParams*>(params);
      ivf_flat::convert_c_search_params(*c_params, &search_params);
    }
    if constexpr (std::is_same_v<typename UpstreamT::search_params_type, ivf_pq::search_params>) {
      auto c_params = reinterpret_cast<cuvsIvfPqSearchParams*>(params);
      ivf_pq::convert_c_search_params(*c_params, &search_params);
    }
  }

  using T = typename UpstreamT::value_type;

  using queries_mdspan_type   = raft::device_matrix_view<T const, int64_t, raft::row_major>;
  using neighbors_mdspan_type = raft::device_matrix_view<int64_t, int64_t, raft::row_major>;
  using distances_mdspan_type = raft::device_matrix_view<T, int64_t, raft::row_major>;
  auto queries_mds            = cuvs::core::from_dlpack<queries_mdspan_type>(queries_tensor);
  auto neighbors_mds          = cuvs::core::from_dlpack<neighbors_mdspan_type>(neighbors_tensor);
  auto distances_mds          = cuvs::core::from_dlpack<distances_mdspan_type>(distances_tensor);

  if (filter.type == NO_FILTER) {
    tiered_index::search(
      *res_ptr, search_params, *index_ptr, queries_mds, neighbors_mds, distances_mds);
  } else if (filter.type == BITSET) {
    using filter_mdspan_type    = raft::device_vector_view<std::uint32_t, int64_t, raft::row_major>;
    auto removed_indices_tensor = reinterpret_cast<DLManagedTensor*>(filter.addr);
    auto removed_indices = cuvs::core::from_dlpack<filter_mdspan_type>(removed_indices_tensor);
    cuvs::core::bitset_view<std::uint32_t, int64_t> removed_indices_bitset(removed_indices,
                                                                           index_ptr->size());
    auto bitset_filter_obj = cuvs::neighbors::filtering::bitset_filter(removed_indices_bitset);

    tiered_index::search(*res_ptr,
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

template <typename UpstreamT>
void _extend(cuvsResources_t res, DLManagedTensor* new_vectors, cuvsTieredIndex index)
{
  auto res_ptr              = reinterpret_cast<raft::resources*>(res);
  auto index_ptr            = reinterpret_cast<tiered_index::index<UpstreamT>*>(index.addr);
  using T                   = typename UpstreamT::value_type;
  using vectors_mdspan_type = raft::device_matrix_view<T const, int64_t, raft::row_major>;

  auto vectors_mds = cuvs::core::from_dlpack<vectors_mdspan_type>(new_vectors);

  tiered_index::extend(*res_ptr, vectors_mds, index_ptr);
}

template <typename UpstreamT>
void _compact(cuvsResources_t res, cuvsTieredIndex index)
{
  auto res_ptr   = reinterpret_cast<raft::resources*>(res);
  auto index_ptr = reinterpret_cast<tiered_index::index<UpstreamT>*>(index.addr);

  tiered_index::compact(*res_ptr, index_ptr);
}
}  // namespace

extern "C" cuvsError_t cuvsTieredIndexCreate(cuvsTieredIndex_t* index)
{
  return cuvs::core::translate_exceptions([=] { *index = new cuvsTieredIndex{}; });
}

extern "C" cuvsError_t cuvsTieredIndexDestroy(cuvsTieredIndex_t index_c_ptr)
{
  return cuvs::core::translate_exceptions([=] {
    auto index = *index_c_ptr;
    if (index.dtype.code == kDLFloat && index.dtype.bits == 32) {
      switch (index.algo) {
        case CUVS_TIERED_INDEX_ALGO_CAGRA: {
          auto index_ptr =
            reinterpret_cast<tiered_index::index<cagra::index<float, uint32_t>>*>(index.addr);
          delete index_ptr;
          break;
        }
        case CUVS_TIERED_INDEX_ALGO_IVF_FLAT: {
          auto index_ptr =
            reinterpret_cast<tiered_index::index<ivf_flat::index<float, int64_t>>*>(index.addr);
          delete index_ptr;
          break;
        }
        case CUVS_TIERED_INDEX_ALGO_IVF_PQ: {
          auto index_ptr =
            reinterpret_cast<tiered_index::index<ivf_pq::typed_index<float, int64_t>>*>(index.addr);
          delete index_ptr;
          break;
        }
        default: RAFT_FAIL("unsupported tiered index algorithm");
      }
    }
    delete index_c_ptr;
  });
}

extern "C" cuvsError_t cuvsTieredIndexBuild(cuvsResources_t res,
                                            cuvsTieredIndexParams_t params,
                                            DLManagedTensor* dataset_tensor,
                                            cuvsTieredIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    auto dataset = dataset_tensor->dl_tensor;
    index->dtype = dataset.dtype;
    index->algo  = params->algo;
    if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
      index->addr = reinterpret_cast<uintptr_t>(_build<float>(res, *params, dataset_tensor));
    } else {
      RAFT_FAIL("Unsupported dataset DLtensor dtype: %d and bits: %d",
                dataset.dtype.code,
                dataset.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsTieredIndexSearch(cuvsResources_t res,
                                             void* search_params,
                                             cuvsTieredIndex_t index_c_ptr,
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

    RAFT_EXPECTS(neighbors.dtype.code == kDLInt && neighbors.dtype.bits == 64,
                 "neighbors should be of type int64_t");
    RAFT_EXPECTS(distances.dtype.code == kDLFloat && distances.dtype.bits == 32,
                 "distances should be of type float32");

    auto index = *index_c_ptr;
    RAFT_EXPECTS(queries.dtype.code == index.dtype.code, "type mismatch between index and queries");

    switch (index.algo) {
      case CUVS_TIERED_INDEX_ALGO_CAGRA: {
        _search<cagra::index<float, uint32_t>>(
          res, search_params, index, queries_tensor, neighbors_tensor, distances_tensor, filter);
        break;
      }
      case CUVS_TIERED_INDEX_ALGO_IVF_FLAT: {
        _search<ivf_flat::index<float, int64_t>>(
          res, search_params, index, queries_tensor, neighbors_tensor, distances_tensor, filter);
        break;
      }
      case CUVS_TIERED_INDEX_ALGO_IVF_PQ: {
        _search<ivf_pq::typed_index<float, int64_t>>(
          res, search_params, index, queries_tensor, neighbors_tensor, distances_tensor, filter);
        break;
      }
      default: RAFT_FAIL("unsupported tiered index algorithm");
    }
  });
}

extern "C" cuvsError_t cuvsTieredIndexParamsCreate(cuvsTieredIndexParams_t* params)
{
  return cuvs::core::translate_exceptions([=] {
    cuvs::neighbors::tiered_index::index_params<cagra::index_params> cpp_params;
    *params = new cuvsTieredIndexParams{
      .metric                     = cpp_params.metric,
      .algo                       = CUVS_TIERED_INDEX_ALGO_CAGRA,
      .min_ann_rows               = cpp_params.min_ann_rows,
      .create_ann_index_on_extend = cpp_params.create_ann_index_on_extend};
  });
}

extern "C" cuvsError_t cuvsTieredIndexParamsDestroy(cuvsTieredIndexParams_t params)
{
  return cuvs::core::translate_exceptions([=] { delete params; });
}

extern "C" cuvsError_t cuvsTieredIndexExtend(cuvsResources_t res,
                                             DLManagedTensor* new_vectors,
                                             cuvsTieredIndex_t index_c_ptr)
{
  return cuvs::core::translate_exceptions([=] {
    auto index = *index_c_ptr;
    switch (index.algo) {
      case CUVS_TIERED_INDEX_ALGO_CAGRA: {
        _extend<cagra::index<float, uint32_t>>(res, new_vectors, index);
        break;
      }
      case CUVS_TIERED_INDEX_ALGO_IVF_FLAT: {
        _extend<ivf_flat::index<float, int64_t>>(res, new_vectors, index);
        break;
      }
      case CUVS_TIERED_INDEX_ALGO_IVF_PQ: {
        _extend<ivf_pq::typed_index<float, int64_t>>(res, new_vectors, index);
        break;
      }
      default: RAFT_FAIL("unsupported tiered index algorithm");
    }
  });
}

extern "C" cuvsError_t cuvsTieredIndexCompact(cuvsResources_t res, cuvsTieredIndex_t index_c_ptr)
{
  return cuvs::core::translate_exceptions([=] {
    auto index = *index_c_ptr;
    switch (index.algo) {
      case CUVS_TIERED_INDEX_ALGO_CAGRA: {
        _compact<cagra::index<float, uint32_t>>(res, index);
        break;
      }
      case CUVS_TIERED_INDEX_ALGO_IVF_FLAT: {
        _compact<ivf_flat::index<float, int64_t>>(res, index);
        break;
      }
      case CUVS_TIERED_INDEX_ALGO_IVF_PQ: {
        _compact<ivf_pq::typed_index<float, int64_t>>(res, index);
        break;
      }
      default: RAFT_FAIL("unsupported tiered index algorithm");
    }
  });
}
