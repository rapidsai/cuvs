/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include "../../core/nvtx.hpp"
#include "ivf_pq_build.cuh"
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <raft/core/mdspan_types.hpp>

namespace cuvs::neighbors::ivf_pq::helpers {

namespace codepacker {

void unpack(
  raft::resources const& res,
  raft::device_mdspan<const uint8_t, list_spec<uint32_t, uint32_t>::list_extents, raft::row_major>
    list_data,
  uint32_t pq_bits,
  uint32_t offset,
  raft::device_matrix_view<uint8_t, uint32_t, raft::row_major> codes)
{
  detail::unpack_list_data(codes, list_data, offset, pq_bits, raft::resource::get_cuda_stream(res));
}

void unpack_contiguous(
  raft::resources const& res,
  raft::device_mdspan<const uint8_t, list_spec<uint32_t, uint32_t>::list_extents, raft::row_major>
    list_data,
  uint32_t pq_bits,
  uint32_t offset,
  uint32_t n_rows,
  uint32_t pq_dim,
  uint8_t* codes)
{
  detail::unpack_contiguous_list_data(
    codes, list_data, n_rows, pq_dim, offset, pq_bits, raft::resource::get_cuda_stream(res));
}
void pack(raft::resources const& res,
          raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major> codes,
          uint32_t pq_bits,
          uint32_t offset,
          raft::device_mdspan<uint8_t, list_spec<uint32_t, uint32_t>::list_extents, raft::row_major>
            list_data)
{
  detail::pack_list_data(list_data, codes, offset, pq_bits, raft::resource::get_cuda_stream(res));
}

void pack_contiguous(
  raft::resources const& res,
  const uint8_t* codes,
  uint32_t n_rows,
  uint32_t pq_dim,
  uint32_t pq_bits,
  uint32_t offset,
  raft::device_mdspan<uint8_t, list_spec<uint32_t, uint32_t>::list_extents, raft::row_major>
    list_data)
{
  detail::pack_contiguous_list_data(
    list_data, codes, n_rows, pq_dim, offset, pq_bits, raft::resource::get_cuda_stream(res));
}

void pack_list_data(raft::resources const& res,
                    index<int64_t>* index,
                    raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major> codes,
                    uint32_t label,
                    uint32_t offset)
{
  detail::pack_list_data(res, index, codes, label, offset);
}

void pack_contiguous_list_data(raft::resources const& res,
                               index<int64_t>* index,
                               uint8_t* codes,
                               uint32_t n_rows,
                               uint32_t label,
                               uint32_t offset)
{
  detail::pack_contiguous_list_data(res, index, codes, n_rows, label, offset);
}

void unpack_list_data(raft::resources const& res,
                      const index<int64_t>& index,
                      raft::device_matrix_view<uint8_t, uint32_t, raft::row_major> out_codes,
                      uint32_t label,
                      uint32_t offset)
{
  detail::unpack_list_data(res, index, out_codes, label, offset);
}

void unpack_list_data(raft::resources const& res,
                      const index<int64_t>& index,
                      raft::device_vector_view<const uint32_t> in_cluster_indices,
                      raft::device_matrix_view<uint8_t, uint32_t, raft::row_major> out_codes,
                      uint32_t label)
{
  detail::unpack_list_data<int64_t>(res, index, out_codes, label, in_cluster_indices.data_handle());
}

void unpack_contiguous_list_data(raft::resources const& res,
                                 const index<int64_t>& index,
                                 uint8_t* out_codes,
                                 uint32_t n_rows,
                                 uint32_t label,
                                 uint32_t offset)
{
  detail::unpack_contiguous_list_data(res, index, out_codes, n_rows, label, offset);
}

void reconstruct_list_data(raft::resources const& res,
                           const index<int64_t>& index,
                           raft::device_matrix_view<float, uint32_t, raft::row_major> out_vectors,
                           uint32_t label,
                           uint32_t offset)
{
  detail::reconstruct_list_data<float, int64_t>(res, index, out_vectors, label, offset);
}

void reconstruct_list_data(raft::resources const& res,
                           const index<int64_t>& index,
                           raft::device_matrix_view<int8_t, uint32_t, raft::row_major> out_vectors,
                           uint32_t label,
                           uint32_t offset)
{
  detail::reconstruct_list_data<int8_t, int64_t>(res, index, out_vectors, label, offset);
}
void reconstruct_list_data(raft::resources const& res,
                           const index<int64_t>& index,
                           raft::device_matrix_view<uint8_t, uint32_t, raft::row_major> out_vectors,
                           uint32_t label,
                           uint32_t offset)
{
  detail::reconstruct_list_data<uint8_t, int64_t>(res, index, out_vectors, label, offset);
}

void reconstruct_list_data(raft::resources const& res,
                           const index<int64_t>& index,
                           raft::device_vector_view<const uint32_t> in_cluster_indices,
                           raft::device_matrix_view<float, uint32_t, raft::row_major> out_vectors,
                           uint32_t label)
{
  detail::reconstruct_list_data<float, int64_t>(
    res, index, out_vectors, label, in_cluster_indices.data_handle());
}
void reconstruct_list_data(raft::resources const& res,
                           const index<int64_t>& index,
                           raft::device_vector_view<const uint32_t> in_cluster_indices,
                           raft::device_matrix_view<int8_t, uint32_t, raft::row_major> out_vectors,
                           uint32_t label)
{
  detail::reconstruct_list_data<int8_t, int64_t>(
    res, index, out_vectors, label, in_cluster_indices.data_handle());
}
void reconstruct_list_data(raft::resources const& res,
                           const index<int64_t>& index,
                           raft::device_vector_view<const uint32_t> in_cluster_indices,
                           raft::device_matrix_view<uint8_t, uint32_t, raft::row_major> out_vectors,
                           uint32_t label)
{
  detail::reconstruct_list_data<uint8_t, int64_t>(
    res, index, out_vectors, label, in_cluster_indices.data_handle());
}

void extend_list_with_codes(
  raft::resources const& res,
  index<int64_t>* index,
  raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major> new_codes,
  raft::device_vector_view<const int64_t, uint32_t, raft::row_major> new_indices,
  uint32_t label)
{
  detail::extend_list_with_codes<int64_t>(res, index, new_codes, new_indices, label);
}

void extend_list(raft::resources const& res,
                 index<int64_t>* index,
                 raft::device_matrix_view<const float, uint32_t, raft::row_major> new_vectors,
                 raft::device_vector_view<const int64_t, uint32_t, raft::row_major> new_indices,
                 uint32_t label)
{
  detail::extend_list<float, int64_t>(res, index, new_vectors, new_indices, label);
}
void extend_list(raft::resources const& res,
                 index<int64_t>* index,
                 raft::device_matrix_view<const int8_t, uint32_t, raft::row_major> new_vectors,
                 raft::device_vector_view<const int64_t, uint32_t, raft::row_major> new_indices,
                 uint32_t label)
{
  detail::extend_list<int8_t, int64_t>(res, index, new_vectors, new_indices, label);
}
void extend_list(raft::resources const& res,
                 index<int64_t>* index,
                 raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major> new_vectors,
                 raft::device_vector_view<const int64_t, uint32_t, raft::row_major> new_indices,
                 uint32_t label)
{
  detail::extend_list<uint8_t, int64_t>(res, index, new_vectors, new_indices, label);
}

};  // namespace codepacker

void erase_list(raft::resources const& res, index<int64_t>* index, uint32_t label)
{
  detail::erase_list<int64_t>(res, index, label);
}

void reset_index(const raft::resources& res, index<int64_t>* index)
{
  auto stream = raft::resource::get_cuda_stream(res);

  cuvs::spatial::knn::detail::utils::memzero(
    index->accum_sorted_sizes().data_handle(), index->accum_sorted_sizes().size(), stream);
  cuvs::spatial::knn::detail::utils::memzero(
    index->list_sizes().data_handle(), index->list_sizes().size(), stream);
  cuvs::spatial::knn::detail::utils::memzero(
    index->data_ptrs().data_handle(), index->data_ptrs().size(), stream);
  cuvs::spatial::knn::detail::utils::memzero(
    index->inds_ptrs().data_handle(), index->inds_ptrs().size(), stream);
}

void make_rotation_matrix(raft::resources const& handle,
                          bool force_random_rotation,
                          uint32_t n_rows,
                          uint32_t n_cols,
                          float* rotation_matrix,
                          raft::random::RngState rng = raft::random::RngState(7ULL))
{
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "ivf_pq::make_rotation_matrix(%u * %u)", n_rows, n_cols);
  auto stream  = raft::resource::get_cuda_stream(handle);
  bool inplace = n_rows == n_cols;
  uint32_t n   = std::max(n_rows, n_cols);
  if (force_random_rotation || !inplace) {
    rmm::device_uvector<float> buf(inplace ? 0 : n * n, stream);
    float* mat = inplace ? rotation_matrix : buf.data();
    raft::random::normal(handle, rng, mat, n * n, 0.0f, 1.0f);
    raft::linalg::detail::qrGetQ_inplace(handle, mat, n, n, stream);
    if (!inplace) {
      RAFT_CUDA_TRY(cudaMemcpy2DAsync(rotation_matrix,
                                      sizeof(float) * n_cols,
                                      mat,
                                      sizeof(float) * n,
                                      sizeof(float) * n_cols,
                                      n_rows,
                                      cudaMemcpyDefault,
                                      stream));
    }
  } else {
    uint32_t stride = n + 1;
    auto rotation_matrix_view =
      raft::make_device_vector_view<float, uint32_t>(rotation_matrix, n * n);
    raft::linalg::map_offset(handle, rotation_matrix_view, [stride] __device__(uint32_t i) {
      return static_cast<float>(i % stride == 0u);
    });
  }
}

void make_rotation_matrix(raft::resources const& res,
                          index<int64_t>* index,
                          bool force_random_rotation)
{
  make_rotation_matrix(res,
                       force_random_rotation,
                       index->rot_dim(),
                       index->dim(),
                       index->rotation_matrix().data_handle());
}

void set_centers(raft::resources const& handle,
                 index<int64_t>* index,
                 raft::device_matrix_view<const float, uint32_t, raft::row_major> cluster_centers)
{
  RAFT_EXPECTS(cluster_centers.extent(0) == index->n_lists(),
               "Number of rows in the new centers must be equal to the number of IVF lists");
  RAFT_EXPECTS(cluster_centers.extent(1) == index->dim(),
               "Number of columns in the new cluster centers and index dim are different");
  RAFT_EXPECTS(index->size() == 0, "Index must be empty");
  detail::set_centers(handle, index, cluster_centers.data_handle());
}

void extract_centers(raft::resources const& res,
                     const cuvs::neighbors::ivf_pq::index<int64_t>& index,
                     raft::device_matrix_view<float, uint32_t, raft::row_major> cluster_centers)
{
  RAFT_EXPECTS(cluster_centers.extent(0) == index.n_lists(),
               "Number of rows in the output buffer for cluster centers must be equal to the "
               "number of IVF lists");
  RAFT_EXPECTS(
    cluster_centers.extent(1) == index.dim(),
    "Number of columns in the output buffer for cluster centers and index dim are different");
  auto stream = raft::resource::get_cuda_stream(res);
  RAFT_CUDA_TRY(cudaMemcpy2DAsync(cluster_centers.data_handle(),
                                  sizeof(float) * index.dim(),
                                  index.centers().data_handle(),
                                  sizeof(float) * index.dim_ext(),
                                  sizeof(float) * index.dim(),
                                  index.n_lists(),
                                  cudaMemcpyDefault,
                                  stream));
}

void recompute_internal_state(const raft::resources& res, index<int64_t>* index)
{
  ivf::detail::recompute_internal_state(res, *index);
}

}  // namespace cuvs::neighbors::ivf_pq::helpers
