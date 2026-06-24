/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "fused_1nn_tile.hpp"

#include "fused_1nn_planner.hpp"

#include <cuvs/core/export.hpp>
#include <raft/util/cuda_utils.cuh>

namespace cuvs {
namespace distance {
namespace detail {

namespace {

template <typename IdxT, typename OutT>
__global__ void pack_fused_1nn_kvp(OutT* out, const int64_t* idx, const float* dist, IdxT len)
{
  IdxT i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    out[i].key   = static_cast<IdxT>(idx[i]);
    out[i].value = static_cast<decltype(out[i].value)>(dist[i]);
  }
}

template <typename DataTag, typename DataT, typename IdxT, typename OutT>
bool launch_fused_1nn_tile(const DataT* x,
                           const DataT* y,
                           OutT* out,
                           IdxT m,
                           IdxT n,
                           IdxT k,
                           cudaStream_t stream)
{
  Fused1nnTilePlanner<DataTag> planner;
  planner.add_entrypoint();
  planner.add_tileir_fallback();
  auto launcher = planner.try_get_launcher();
  if (!launcher) { return false; }

  int64_t* d_idx = nullptr;
  float* d_dist  = nullptr;
  RAFT_CUDA_TRY(cudaMallocAsync(&d_idx, m * sizeof(int64_t), stream));
  RAFT_CUDA_TRY(cudaMallocAsync(&d_dist, m * sizeof(float), stream));

  int64_t shape_x[2]     = {m, k};
  int64_t stride_x[2]    = {k, 1};
  int64_t shape_y[2]     = {n, k};
  int64_t stride_y[2]    = {k, 1};
  int64_t shape_idx[1]   = {m};
  int64_t stride_idx[1]  = {1};
  int64_t shape_dist[1]  = {m};
  int64_t stride_dist[1] = {1};

  int64_t M = m, N = n, K = k;
  constexpr int64_t tm = 128, tn = 256, tk = 64;

  void* x_ptr    = const_cast<DataT*>(x);
  void* y_ptr    = const_cast<DataT*>(y);
  void* idx_ptr  = d_idx;
  void* dist_ptr = d_dist;

  dim3 grid((m + tm - 1) / tm, 1, 1);
  dim3 block(1, 1, 1);

  using fused_1nn_cutile_kernel_t = void(void*,
                                         int64_t*,
                                         int64_t*,
                                         void*,
                                         int64_t*,
                                         int64_t*,
                                         void*,
                                         int64_t*,
                                         int64_t*,
                                         void*,
                                         int64_t*,
                                         int64_t*,
                                         int64_t,
                                         int64_t,
                                         int64_t,
                                         int64_t,
                                         int64_t,
                                         int64_t);
  launcher->template dispatch<fused_1nn_cutile_kernel_t>(
    stream,
    grid,
    block,
    0,
    x_ptr,
    shape_x,
    stride_x,
    y_ptr,
    shape_y,
    stride_y,
    idx_ptr,
    shape_idx,
    stride_idx,
    dist_ptr,
    shape_dist,
    stride_dist,
    M,
    N,
    K,
    tm,
    tn,
    tk);

  pack_fused_1nn_kvp<IdxT, OutT><<<(m + 255) / 256, 256, 0, stream>>>(out, d_idx, d_dist, m);
  RAFT_CUDA_TRY(cudaGetLastError());
  RAFT_CUDA_TRY(cudaFreeAsync(d_idx, stream));
  RAFT_CUDA_TRY(cudaFreeAsync(d_dist, stream));
  return true;
}

}  // namespace

template <typename DataT,
          typename OutT,
          typename IdxT,
          std::enable_if_t<is_fused_1nn_kvp_output_v<OutT, IdxT, DataT>, int>>
bool try_fused_1nn_tile(OutT* min,
                        const DataT* x,
                        const DataT* y,
                        IdxT m,
                        IdxT n,
                        IdxT k,
                        cuvs::distance::DistanceType metric,
                        cudaStream_t stream)
{
  if (metric != cuvs::distance::DistanceType::InnerProduct) { return false; }

  if constexpr (std::is_same_v<DataT, half>) {
    return launch_fused_1nn_tile<cuvs::neighbors::detail::tag_h, DataT, IdxT, OutT>(
      x, y, min, m, n, k, stream);
  } else if constexpr (std::is_same_v<DataT, float>) {
    return launch_fused_1nn_tile<cuvs::neighbors::detail::tag_f, DataT, IdxT, OutT>(
      x, y, min, m, n, k, stream);
  } else {
    return false;
  }
}

using kvp_i_f   = raft::KeyValuePair<int, float>;
using kvp_i64_f = raft::KeyValuePair<int64_t, float>;
using kvp_i_h   = raft::KeyValuePair<int, half>;
using kvp_i64_h = raft::KeyValuePair<int64_t, half>;

#define CUVS_INST_TRY_FUSED_1NN_TILE(DataT, OutT, IdxT)                                          \
  template CUVS_EXPORT bool try_fused_1nn_tile<DataT, OutT, IdxT>(OutT*,                        \
                                                                  const DataT*,                  \
                                                                  const DataT*,                  \
                                                                  IdxT,                          \
                                                                  IdxT,                          \
                                                                  IdxT,                          \
                                                                  cuvs::distance::DistanceType, \
                                                                  cudaStream_t)

// int and int32_t are the same on LP64; one instantiation covers both.
CUVS_INST_TRY_FUSED_1NN_TILE(float, kvp_i_f, int);
CUVS_INST_TRY_FUSED_1NN_TILE(float, kvp_i64_f, int64_t);
CUVS_INST_TRY_FUSED_1NN_TILE(half, kvp_i_f, int);
CUVS_INST_TRY_FUSED_1NN_TILE(half, kvp_i64_f, int64_t);
CUVS_INST_TRY_FUSED_1NN_TILE(half, kvp_i_h, int);
CUVS_INST_TRY_FUSED_1NN_TILE(half, kvp_i64_h, int64_t);

#undef CUVS_INST_TRY_FUSED_1NN_TILE

}  // namespace detail
}  // namespace distance
}  // namespace cuvs
