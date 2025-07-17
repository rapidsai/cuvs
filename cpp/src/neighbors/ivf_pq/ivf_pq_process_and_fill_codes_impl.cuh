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

#pragma once
#include "ivf_pq_codepacking.cuh"
#include "ivf_pq_process_and_fill_codes.cuh"
#include <cstdint>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/util/cuda_dev_essentials.cuh>
#include <raft/util/pow2_utils.cuh>
#include <variant>

namespace cuvs::neighbors::ivf_pq::detail {

/**
 *
 * A producer for the `write_list` and `write_vector` that encodes level-1 input vector residuals
 * into lvl-2 PQ codes.
 * Computing a PQ code means finding the closest cluster in a pq_dim-subspace.
 *
 * @tparam SubWarpSize
 *   how many threads work on a single vector;
 *   bounded by either WarpSize or pq_book_size.
 *
 * @param pq_centers
 *   - codebook_gen::PER_SUBSPACE: [pq_dim , pq_len, pq_book_size]
 *   - codebook_gen::PER_CLUSTER:  [n_lists, pq_len, pq_book_size]
 * @param new_vector a single input of length rot_dim, reinterpreted as [pq_dim, pq_len].
 *   the input must be already transformed to floats, rotated, and the level 1 cluster
 *   center must be already substructed (i.e. this is the residual of a single input vector).
 * @param codebook_kind
 * @param j index along pq_dim "dimension"
 * @param cluster_ix is used for PER_CLUSTER codebooks.
 */
/**
 */
template <uint32_t SubWarpSize, typename IdxT>
struct encode_vectors {
  codebook_gen codebook_kind;
  uint32_t cluster_ix;
  raft::device_mdspan<const float, raft::extent_3d<uint32_t>, raft::row_major> pq_centers;
  raft::device_mdspan<const float, raft::extent_3d<IdxT>, raft::row_major> in_vectors;

  __device__ inline encode_vectors(
    raft::device_mdspan<const float, raft::extent_3d<uint32_t>, raft::row_major> pq_centers,
    raft::device_matrix_view<const float, IdxT, raft::row_major> in_vectors,
    codebook_gen codebook_kind,
    uint32_t cluster_ix)
    : codebook_kind{codebook_kind},
      cluster_ix{cluster_ix},
      pq_centers{pq_centers},
      in_vectors{reinterpret_vectors(in_vectors, pq_centers)}
  {
  }

  /**
   * Decode j-th component of the i-th vector by its code and write it into a chunk of the output
   * vectors (pq_len elements).
   */
  __device__ inline auto operator()(IdxT i, uint32_t j) -> uint8_t
  {
    uint32_t lane_id = raft::Pow2<SubWarpSize>::mod(raft::laneId());
    uint32_t partition_ix;
    switch (codebook_kind) {
      case codebook_gen::PER_CLUSTER: {
        partition_ix = cluster_ix;
      } break;
      case codebook_gen::PER_SUBSPACE: {
        partition_ix = j;
      } break;
      default: __builtin_unreachable();
    }

    const uint32_t pq_book_size = pq_centers.extent(2);
    const uint32_t pq_len       = pq_centers.extent(1);
    float min_dist              = std::numeric_limits<float>::infinity();
    uint8_t code                = 0;
    // calculate the distance for each PQ cluster, find the minimum for each thread
    for (uint32_t l = lane_id; l < pq_book_size; l += SubWarpSize) {
      // NB: the L2 quantifiers on residuals are always trained on L2 metric.
      float d = 0.0f;
      for (uint32_t k = 0; k < pq_len; k++) {
        auto t = in_vectors(i, j, k) - pq_centers(partition_ix, k, l);
        d += t * t;
      }
      if (d < min_dist) {
        min_dist = d;
        code     = uint8_t(l);
      }
    }
    // reduce among threads
#pragma unroll
    for (uint32_t stride = SubWarpSize >> 1; stride > 0; stride >>= 1) {
      const auto other_dist = raft::shfl_xor(min_dist, stride, SubWarpSize);
      const auto other_code = raft::shfl_xor(code, stride, SubWarpSize);
      if (other_dist < min_dist) {
        min_dist = other_dist;
        code     = other_code;
      }
    }
    return code;
  }
};

template <uint32_t BlockSize, uint32_t PqBits>
__launch_bounds__(BlockSize) RAFT_KERNEL encode_list_data_kernel(
  raft::device_mdspan<uint8_t, list_spec<uint32_t, uint32_t>::list_extents, row_major> list_data,
  raft::device_matrix_view<const float, uint32_t, row_major> new_vectors,
  raft::device_mdspan<const float, extent_3d<uint32_t>, row_major> pq_centers,
  codebook_gen codebook_kind,
  uint32_t cluster_ix,
  std::variant<uint32_t, const uint32_t*> offset_or_indices)
{
  constexpr uint32_t kSubWarpSize = std::min<uint32_t>(WarpSize, 1u << PqBits);
  const uint32_t pq_dim           = new_vectors.extent(1) / pq_centers.extent(1);
  auto encode_action =
    encode_vectors<kSubWarpSize, uint32_t>{pq_centers, new_vectors, codebook_kind, cluster_ix};
  write_list<PqBits, kSubWarpSize>(
    list_data, offset_or_indices, new_vectors.extent(0), pq_dim, encode_action);
}

template <typename IdxT>
void launch_encode_list_data_kernel(raft::resources const& handle,
                                    index<IdxT>* index,
                                    raft::device_matrix_view<float> new_vectors_residual,
                                    uint32_t label,
                                    std::variant<uint32_t, const uint32_t*> offset_or_indices,
                                    IdxT n_rows)
{
  constexpr uint32_t kBlockSize  = 256;
  const uint32_t threads_per_vec = std::min<uint32_t>(raft::WarpSize, index->pq_book_size());
  dim3 blocks(raft::div_rounding_up_safe<uint32_t>(n_rows, kBlockSize / threads_per_vec), 1, 1);
  dim3 threads(kBlockSize, 1, 1);
  auto kernel = [](uint32_t pq_bits) {
    switch (pq_bits) {
      case 4: return encode_list_data_kernel<kBlockSize, 4>;
      case 5: return encode_list_data_kernel<kBlockSize, 5>;
      case 6: return encode_list_data_kernel<kBlockSize, 6>;
      case 7: return encode_list_data_kernel<kBlockSize, 7>;
      case 8: return encode_list_data_kernel<kBlockSize, 8>;
      default: RAFT_FAIL("Invalid pq_bits (%u), the value must be within [4, 8]", pq_bits);
    }
  }(index->pq_bits());
  kernel<<<blocks, threads, 0, raft::resource::get_cuda_stream(handle)>>>(
    index->lists()[label]->data.view(),
    new_vectors_residual,
    index->pq_centers(),
    index->codebook_kind(),
    label,
    offset_or_indices);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <uint32_t BlockSize, uint32_t PqBits, typename IdxT>
__launch_bounds__(BlockSize) static __global__ void process_and_fill_codes_kernel(
  raft::device_matrix_view<const float, IdxT, raft::row_major> new_vectors,
  std::variant<IdxT, const IdxT*> src_offset_or_indices,
  const uint32_t* new_labels,
  raft::device_vector_view<uint32_t, uint32_t, raft::row_major> list_sizes,
  raft::device_vector_view<IdxT*, uint32_t, raft::row_major> inds_ptrs,
  raft::device_vector_view<uint8_t*, uint32_t, raft::row_major> data_ptrs,
  raft::device_mdspan<const float, raft::extent_3d<uint32_t>, raft::row_major> pq_centers,
  codebook_gen codebook_kind)
{
  constexpr uint32_t kSubWarpSize = std::min<uint32_t>(raft::WarpSize, 1u << PqBits);
  using subwarp_align             = raft::Pow2<kSubWarpSize>;
  const uint32_t lane_id          = subwarp_align::mod(threadIdx.x);
  const IdxT row_ix = subwarp_align::div(IdxT{threadIdx.x} + IdxT{BlockSize} * IdxT{blockIdx.x});
  if (row_ix >= new_vectors.extent(0)) { return; }

  const uint32_t cluster_ix = new_labels[row_ix];
  uint32_t out_ix;
  if (lane_id == 0) { out_ix = atomicAdd(&list_sizes(cluster_ix), 1); }
  out_ix = raft::shfl(out_ix, 0, kSubWarpSize);

  // write the label  (one record per subwarp)
  auto pq_indices = inds_ptrs(cluster_ix);
  if (lane_id == 0) {
    if (std::holds_alternative<IdxT>(src_offset_or_indices)) {
      pq_indices[out_ix] = std::get<IdxT>(src_offset_or_indices) + row_ix;
    } else {
      pq_indices[out_ix] = std::get<const IdxT*>(src_offset_or_indices)[row_ix];
    }
  }

  // write the codes (one record per subwarp):
  const uint32_t pq_dim = new_vectors.extent(1) / pq_centers.extent(1);
  auto pq_extents = list_spec<uint32_t, IdxT>{PqBits, pq_dim, true}.make_list_extents(out_ix + 1);
  auto pq_dataset = raft::make_mdspan<uint8_t, uint32_t, raft::row_major, false, true>(
    data_ptrs[cluster_ix], pq_extents);
  write_vector<PqBits, kSubWarpSize>(
    pq_dataset,
    out_ix,
    row_ix,
    pq_dim,
    encode_vectors<kSubWarpSize, IdxT>{pq_centers, new_vectors, codebook_kind, cluster_ix});
}

template <typename IdxT>
void launch_process_and_fill_codes_kernel(raft::resources const& handle,
                                          index<IdxT>& index,
                                          raft::device_matrix_view<float> new_vectors_residual,
                                          std::variant<IdxT, const IdxT*> src_offset_or_indices,
                                          const uint32_t* new_labels,
                                          IdxT n_rows)
{
  constexpr uint32_t kBlockSize  = 256;
  const uint32_t threads_per_vec = std::min<uint32_t>(raft::WarpSize, index.pq_book_size());
  dim3 blocks(raft::div_rounding_up_safe<IdxT>(n_rows, kBlockSize / threads_per_vec), 1, 1);
  dim3 threads(kBlockSize, 1, 1);
  auto kernel = [](uint32_t pq_bits) {
    switch (pq_bits) {
      case 4: return process_and_fill_codes_kernel<kBlockSize, 4, IdxT>;
      case 5: return process_and_fill_codes_kernel<kBlockSize, 5, IdxT>;
      case 6: return process_and_fill_codes_kernel<kBlockSize, 6, IdxT>;
      case 7: return process_and_fill_codes_kernel<kBlockSize, 7, IdxT>;
      case 8: return process_and_fill_codes_kernel<kBlockSize, 8, IdxT>;
      default: RAFT_FAIL("Invalid pq_bits (%u), the value must be within [4, 8]", pq_bits);
    }
  }(index.pq_bits());

  kernel<<<blocks, threads, 0, raft::resource::get_cuda_stream(handle)>>>(new_vectors_residual,
                                                                          src_offset_or_indices,
                                                                          new_labels,
                                                                          index.list_sizes(),
                                                                          index.inds_ptrs(),
                                                                          index.data_ptrs(),
                                                                          index.pq_centers(),
                                                                          index.codebook_kind());

  RAFT_CUDA_TRY(cudaPeekAtLastError());
}
}  // namespace cuvs::neighbors::ivf_pq::detail
