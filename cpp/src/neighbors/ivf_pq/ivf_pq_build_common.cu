/*
 * raft::copyright (c) 2022-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a raft::copy of the License at
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
#include <cuvs/distance/distance_types.hpp>
#include <cuvs/neighbors/ivf_list.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>

namespace cuvs::neighbors::ivf_pq::helpers {

namespace codepacker {

void unpack_list_data(raft::resources const& res,
                      const index<int64_t>& index,
                      raft::device_matrix_view<uint8_t, uint32_t, raft::row_major> out_codes,
                      uint32_t label,
                      std::variant<uint32_t, const uint32_t*> offset_or_indices)
{
  detail::unpack_list_data(res, index, out_codes, label, offset_or_indices);
}

void pack_list_data(raft::resources const& res,
                    index<int64_t>* index,
                    raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major> new_codes,
                    uint32_t label,
                    std::variant<uint32_t, const uint32_t*> offset_or_indices)
{
  detail::pack_list_data(res, index, new_codes, label, offset_or_indices);
}

};  // namespace codepacker

void make_rotation_matrix(raft::resources const& handle,
                          bool force_random_rotation,
                          uint32_t n_rows,
                          uint32_t n_cols,
                          float* rotation_matrix,
                          raft::random::RngState rng)
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

void set_centers(raft::resources const& handle, index<int64_t>* index, const float* cluster_centers)
{
  detail::set_centers(handle, index, cluster_centers);
}

void reconstruct_list_data(raft::resources const& res,
                           const index<int64_t>& index,
                           raft::device_matrix_view<float, uint32_t, raft::row_major> out_vectors,
                           uint32_t label,
                           std::variant<uint32_t, const uint32_t*> offset_or_indices)
{
  detail::reconstruct_list_data<float, int64_t>(res, index, out_vectors, label, offset_or_indices);
}

void reconstruct_list_data(raft::resources const& res,
                           const index<int64_t>& index,
                           raft::device_matrix_view<int8_t, uint32_t, raft::row_major> out_vectors,
                           uint32_t label,
                           std::variant<uint32_t, const uint32_t*> offset_or_indices)
{
  detail::reconstruct_list_data<int8_t, int64_t>(res, index, out_vectors, label, offset_or_indices);
}
void reconstruct_list_data(raft::resources const& res,
                           const index<int64_t>& index,
                           raft::device_matrix_view<uint8_t, uint32_t, raft::row_major> out_vectors,
                           uint32_t label,
                           std::variant<uint32_t, const uint32_t*> offset_or_indices)
{
  detail::reconstruct_list_data<uint8_t, int64_t>(
    res, index, out_vectors, label, offset_or_indices);
}

void pack_contiguous_list_data(raft::resources const& res,
                               index<int64_t>* index,
                               const uint8_t* new_codes,
                               uint32_t n_rows,
                               uint32_t label,
                               std::variant<uint32_t, const uint32_t*> offset_or_indices)
{
  detail::pack_contiguous_list_data<int64_t>(
    res, index, new_codes, n_rows, label, offset_or_indices);
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

void erase_list(raft::resources const& res, index<int64_t>* index, uint32_t label)
{
  detail::erase_list<int64_t>(res, index, label);
}

}  // namespace cuvs::neighbors::ivf_pq::helpers
