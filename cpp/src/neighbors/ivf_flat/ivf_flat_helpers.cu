/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include "../ivf_common.cuh"
#include "ivf_flat_helpers.cuh"
#include <cuvs/neighbors/ivf_flat.hpp>

namespace cuvs::neighbors::ivf_flat::helpers {

namespace codepacker {

void pack(raft::resources const& res,
          raft::device_matrix_view<const float, uint32_t, raft::row_major> codes,
          uint32_t veclen,
          uint32_t offset,
          raft::device_mdspan<float,
                              typename list_spec<uint32_t, float, int64_t>::list_extents,
                              raft::row_major> list_data)
{
  detail::pack<float, int64_t>(res, codes, veclen, offset, list_data);
}

void pack(raft::resources const& res,
          raft::device_matrix_view<const int8_t, uint32_t, raft::row_major> codes,
          uint32_t veclen,
          uint32_t offset,
          raft::device_mdspan<int8_t,
                              typename list_spec<uint32_t, int8_t, int64_t>::list_extents,
                              raft::row_major> list_data)
{
  detail::pack<int8_t, int64_t>(res, codes, veclen, offset, list_data);
}

void pack(raft::resources const& res,
          raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major> codes,
          uint32_t veclen,
          uint32_t offset,
          raft::device_mdspan<uint8_t,
                              typename list_spec<uint32_t, uint8_t, int64_t>::list_extents,
                              raft::row_major> list_data)
{
  detail::pack<uint8_t, int64_t>(res, codes, veclen, offset, list_data);
}

void unpack(raft::resources const& res,
            raft::device_mdspan<const float,
                                typename list_spec<uint32_t, float, int64_t>::list_extents,
                                raft::row_major> list_data,
            uint32_t veclen,
            uint32_t offset,
            raft::device_matrix_view<float, uint32_t, raft::row_major> codes)
{
  detail::unpack<float, int64_t>(res, list_data, veclen, offset, codes);
}

void unpack(raft::resources const& res,
            raft::device_mdspan<const int8_t,
                                typename list_spec<uint32_t, int8_t, int64_t>::list_extents,
                                raft::row_major> list_data,
            uint32_t veclen,
            uint32_t offset,
            raft::device_matrix_view<int8_t, uint32_t, raft::row_major> codes)
{
  detail::unpack<int8_t, int64_t>(res, list_data, veclen, offset, codes);
}

void unpack(raft::resources const& res,
            raft::device_mdspan<const uint8_t,
                                typename list_spec<uint32_t, uint8_t, int64_t>::list_extents,
                                raft::row_major> list_data,
            uint32_t veclen,
            uint32_t offset,
            raft::device_matrix_view<uint8_t, uint32_t, raft::row_major> codes)
{
  detail::unpack<uint8_t, int64_t>(res, list_data, veclen, offset, codes);
}

void pack_1(const float* flat_code, float* block, uint32_t dim, uint32_t veclen, uint32_t offset)
{
  detail::pack_1<float>(flat_code, block, dim, veclen, offset);
}

void pack_1(const int8_t* flat_code, int8_t* block, uint32_t dim, uint32_t veclen, uint32_t offset)
{
  detail::pack_1<int8_t>(flat_code, block, dim, veclen, offset);
}

void pack_1(
  const uint8_t* flat_code, uint8_t* block, uint32_t dim, uint32_t veclen, uint32_t offset)
{
  detail::pack_1<uint8_t>(flat_code, block, dim, veclen, offset);
}

void unpack_1(const float* block, float* flat_code, uint32_t dim, uint32_t veclen, uint32_t offset)
{
  detail::unpack_1<float>(block, flat_code, dim, veclen, offset);
}

void unpack_1(
  const int8_t* block, int8_t* flat_code, uint32_t dim, uint32_t veclen, uint32_t offset)
{
  detail::unpack_1<int8_t>(block, flat_code, dim, veclen, offset);
}

void unpack_1(
  const uint8_t* block, uint8_t* flat_code, uint32_t dim, uint32_t veclen, uint32_t offset)
{
  detail::unpack_1<uint8_t>(block, flat_code, dim, veclen, offset);
}

}  // namespace codepacker

namespace detail {

template <typename T, typename IdxT>
void reset_index(const raft::resources& res, index<T, IdxT>* idx)
{
  auto stream = raft::resource::get_cuda_stream(res);

  cuvs::spatial::knn::detail::utils::memzero(
    idx->accum_sorted_sizes().data_handle(), idx->accum_sorted_sizes().size(), stream);
  cuvs::spatial::knn::detail::utils::memzero(
    idx->list_sizes().data_handle(), idx->list_sizes().size(), stream);
  cuvs::spatial::knn::detail::utils::memzero(
    idx->data_ptrs().data_handle(), idx->data_ptrs().size(), stream);
  cuvs::spatial::knn::detail::utils::memzero(
    idx->inds_ptrs().data_handle(), idx->inds_ptrs().size(), stream);
}

}  // namespace detail

void reset_index(const raft::resources& res, index<float, int64_t>* index)
{
  detail::reset_index<float, int64_t>(res, index);
}

/**
 * @brief Public helper API to reset the data and indices ptrs, and the list sizes. Useful for
 * externally modifying the index without going through the build stage. The data and indices of the
 * IVF lists will be lost.
 *
 * Usage example:
 * @code{.cpp}
 *   raft::resources res;
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_flat::index_params index_params;
 *   // initialize an empty index
 *   ivf_flat::index<int8_t, int64_t> index(res, index_params, D);
 *   // reset the index's state and list sizes
 *   ivf_flat::helpers::reset_index(res, &index);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[inout] index pointer to IVF-Flat index
 */
void reset_index(const raft::resources& res, index<int8_t, int64_t>* index)
{
  detail::reset_index<int8_t, int64_t>(res, index);
}

/**
 * @brief Public helper API to reset the data and indices ptrs, and the list sizes. Useful for
 * externally modifying the index without going through the build stage. The data and indices of the
 * IVF lists will be lost.
 *
 * Usage example:
 * @code{.cpp}
 *   raft::resources res;
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   ivf_flat::index_params index_params;
 *   // initialize an empty index
 *   ivf_flat::index<uint8_t, int64_t> index(res, index_params, D);
 *   // reset the index's state and list sizes
 *   ivf_flat::helpers::reset_index(res, &index);
 * @endcode
 *
 * @param[in] res raft resource
 * @param[inout] index pointer to IVF-Flat index
 */
void reset_index(const raft::resources& res, index<uint8_t, int64_t>* index)
{
  detail::reset_index<uint8_t, int64_t>(res, index);
}

void recompute_internal_state(const raft::resources& res, index<float, int64_t>* index)
{
  ivf::detail::recompute_internal_state(res, *index);
}

void recompute_internal_state(const raft::resources& res, index<int8_t, int64_t>* index)
{
  ivf::detail::recompute_internal_state(res, *index);
}

void recompute_internal_state(const raft::resources& res, index<uint8_t, int64_t>* index)
{
  ivf::detail::recompute_internal_state(res, *index);
}

}  // namespace cuvs::neighbors::ivf_flat::helpers
