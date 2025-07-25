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

#include <cstdint>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <raft/core/device_mdspan.hpp>
#include <variant>

namespace cuvs::neighbors::ivf_pq::detail {
/**
 * A helper function: given the dataset in the rotated space
 *  [n_rows, rot_dim] = [n_rows, pq_dim * pq_len],
 * reinterpret the last dimension as two: [n_rows, pq_dim, pq_len]
 *
 * @tparam T
 * @tparam IdxT
 *
 * @param vectors input data [n_rows, rot_dim]
 * @param pq_centers codebook (used to infer the structure - pq_len)
 * @return reinterpreted vectors [n_rows, pq_dim, pq_len]
 */
template <typename T, typename IdxT>
static __device__ auto reinterpret_vectors(
  raft::device_matrix_view<T, IdxT, raft::row_major> vectors,
  raft::device_mdspan<const float, raft::extent_3d<uint32_t>, raft::row_major> pq_centers)
  -> raft::device_mdspan<T, raft::extent_3d<IdxT>, raft::row_major>
{
  const uint32_t pq_len = pq_centers.extent(1);
  const uint32_t pq_dim = vectors.extent(1) / pq_len;
  using layout_t        = typename decltype(vectors)::layout_type;
  using accessor_t      = typename decltype(vectors)::accessor_type;
  return raft::mdspan<T, raft::extent_3d<IdxT>, layout_t, accessor_t>(
    vectors.data_handle(), raft::extent_3d<IdxT>{vectors.extent(0), pq_dim, pq_len});
}

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

template <typename IdxT>
void launch_process_and_fill_codes_kernel(raft::resources const& handle,
                                          index<IdxT>& index,
                                          raft::device_matrix_view<float> new_vectors_residual,
                                          std::variant<IdxT, const IdxT*> src_offset_or_indices,
                                          const uint32_t* new_labels,
                                          IdxT n_rows);

}  // namespace cuvs::neighbors::ivf_pq::detail
