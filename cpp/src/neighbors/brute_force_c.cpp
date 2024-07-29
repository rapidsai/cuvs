
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
#include <cuvs/core/exceptions.hpp>
#include <cuvs/core/interop.hpp>
#include <cuvs/neighbors/brute_force.h>
#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/common.h>

namespace {

template <typename T>
void* _build(cuvsResources_t res,
             DLManagedTensor* dataset_tensor,
             cuvsDistanceType metric,
             T metric_arg)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);

  using mdspan_type = raft::device_matrix_view<T const, int64_t, raft::row_major>;
  auto mds          = cuvs::core::from_dlpack<mdspan_type>(dataset_tensor);

  auto index_on_stack = cuvs::neighbors::brute_force::build(
    *res_ptr, mds, static_cast<cuvs::distance::DistanceType>((int)metric), metric_arg);
  auto index_on_heap = new cuvs::neighbors::brute_force::index<T>(std::move(index_on_stack));

  return index_on_heap;
}

template <typename T>
void _search(cuvsResources_t res,
             cuvsBruteForceIndex index,
             DLManagedTensor* queries_tensor,
             DLManagedTensor* neighbors_tensor,
             DLManagedTensor* distances_tensor,
             cuvsFilter prefilter)
{
  auto res_ptr   = reinterpret_cast<raft::resources*>(res);
  auto index_ptr = reinterpret_cast<cuvs::neighbors::brute_force::index<T>*>(index.addr);

  using queries_mdspan_type   = raft::device_matrix_view<T const, int64_t, raft::row_major>;
  using neighbors_mdspan_type = raft::device_matrix_view<int64_t, int64_t, raft::row_major>;
  using distances_mdspan_type = raft::device_matrix_view<float, int64_t, raft::row_major>;
  using prefilter_mds_type    = raft::device_vector_view<const uint32_t, int64_t>;
  using prefilter_opt_type    = cuvs::core::bitmap_view<const uint32_t, int64_t>;

  auto queries_mds   = cuvs::core::from_dlpack<queries_mdspan_type>(queries_tensor);
  auto neighbors_mds = cuvs::core::from_dlpack<neighbors_mdspan_type>(neighbors_tensor);
  auto distances_mds = cuvs::core::from_dlpack<distances_mdspan_type>(distances_tensor);

  std::optional<cuvs::core::bitmap_view<const uint32_t, int64_t>> filter_opt;

  if (prefilter.type == NO_FILTER) {
    filter_opt = std::nullopt;
  } else {
    auto prefilter_ptr  = reinterpret_cast<DLManagedTensor*>(prefilter.addr);
    auto prefilter_mds  = cuvs::core::from_dlpack<prefilter_mds_type>(prefilter_ptr);
    auto prefilter_view = prefilter_opt_type((const uint32_t*)prefilter_mds.data_handle(),
                                             queries_mds.extent(0),
                                             index_ptr->dataset().extent(0));

    filter_opt = std::make_optional<prefilter_opt_type>(prefilter_view);
  }

  cuvs::neighbors::brute_force::search(
    *res_ptr, *index_ptr, queries_mds, neighbors_mds, distances_mds, filter_opt);
}

}  // namespace

extern "C" cuvsError_t cuvsBruteForceIndexCreate(cuvsBruteForceIndex_t* index)
{
  return cuvs::core::translate_exceptions([=] { *index = new cuvsBruteForceIndex{}; });
}

extern "C" cuvsError_t cuvsBruteForceIndexDestroy(cuvsBruteForceIndex_t index_c_ptr)
{
  return cuvs::core::translate_exceptions([=] {
    auto index = *index_c_ptr;

    if (index.dtype.code == kDLFloat) {
      auto index_ptr = reinterpret_cast<cuvs::neighbors::brute_force::index<float>*>(index.addr);
      delete index_ptr;
    } else if (index.dtype.code == kDLInt) {
      auto index_ptr = reinterpret_cast<cuvs::neighbors::brute_force::index<int8_t>*>(index.addr);
      delete index_ptr;
    } else if (index.dtype.code == kDLUInt) {
      auto index_ptr = reinterpret_cast<cuvs::neighbors::brute_force::index<uint8_t>*>(index.addr);
      delete index_ptr;
    }
    delete index_c_ptr;
  });
}

extern "C" cuvsError_t cuvsBruteForceBuild(cuvsResources_t res,
                                           DLManagedTensor* dataset_tensor,
                                           cuvsDistanceType metric,
                                           float metric_arg,
                                           cuvsBruteForceIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    auto dataset = dataset_tensor->dl_tensor;

    if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
      index->addr =
        reinterpret_cast<uintptr_t>(_build<float>(res, dataset_tensor, metric, metric_arg));
      index->dtype.code = kDLFloat;
    } else {
      RAFT_FAIL("Unsupported dataset DLtensor dtype: %d and bits: %d",
                dataset.dtype.code,
                dataset.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsBruteForceSearch(cuvsResources_t res,
                                            cuvsBruteForceIndex_t index_c_ptr,
                                            DLManagedTensor* queries_tensor,
                                            DLManagedTensor* neighbors_tensor,
                                            DLManagedTensor* distances_tensor,
                                            cuvsFilter prefilter)
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

    if (queries.dtype.code == kDLFloat && queries.dtype.bits == 32) {
      _search<float>(res, index, queries_tensor, neighbors_tensor, distances_tensor, prefilter);
    } else {
      RAFT_FAIL("Unsupported queries DLtensor dtype: %d and bits: %d",
                queries.dtype.code,
                queries.dtype.bits);
    }
  });
}
