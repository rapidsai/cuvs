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
__global__ void pack_fused_1nn_kvp(
  OutT* out, const int64_t* idx, const float* dist, IdxT len, bool apply_sqrt)
{
  IdxT i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    out[i].key  = static_cast<IdxT>(idx[i]);
    float value = dist[i];
    if (apply_sqrt) { value = sqrtf(value); }
    out[i].value = static_cast<decltype(out[i].value)>(value);
  }
}

template <cuvs::distance::DistanceType Metric, typename DataT, typename IdxT, typename OutT>
bool launch_fused_1nn_tile(const DataT* x,
                           const DataT* y,
                           const DataT* xn,
                           const DataT* yn,
                           OutT* out,
                           IdxT m,
                           IdxT n,
                           IdxT k,
                           bool is_sqrt,
                           cudaStream_t stream)
{
  if constexpr (!std::is_same_v<DataT, float> && !std::is_same_v<DataT, half>) { return false; }

  Fused1nnTilePlanner<DataT, Metric> planner;
  planner.add_entrypoint();
  planner.add_tileir_fallback();
  const CutileTileConfig tile_cfg = planner.tile_config();
  auto launcher                   = planner.try_get_launcher();
  if (!launcher) { return false; }

  const bool apply_sqrt = fused_1nn_apply_sqrt_at_pack<Metric>(is_sqrt);

  int64_t* d_idx = nullptr;
  float* d_dist  = nullptr;
  RAFT_CUDA_TRY(cudaMallocAsync(&d_idx, m * sizeof(int64_t), stream));
  RAFT_CUDA_TRY(cudaMallocAsync(&d_dist, m * sizeof(float), stream));

  int64_t shape_x[2]  = {m, k};
  int64_t stride_x[2] = {k, 1};
  int64_t shape_y[2]  = {n, k};
  int64_t stride_y[2] = {k, 1};
  int64_t shape_xn    = m;
  int64_t stride_xn   = 1;
  int64_t shape_yn    = n;
  int64_t stride_yn   = 1;
  int64_t shape_idx   = m;
  int64_t stride_idx  = 1;
  int64_t shape_dist  = m;
  int64_t stride_dist = 1;

  int64_t M = m, N = n, K = k;

  void* x_ptr    = const_cast<DataT*>(x);
  void* y_ptr    = const_cast<DataT*>(y);
  void* xn_ptr   = const_cast<DataT*>(xn);
  void* yn_ptr   = const_cast<DataT*>(yn);
  void* idx_ptr  = d_idx;
  void* dist_ptr = d_dist;

  const int64_t tile_m = tile_cfg.tile_m;
  dim3 grid((m + tile_m - 1) / tile_m, 1, 1);
  dim3 block(1, 1, 1);

  // cutile_python_v1: 2D array (ptr, shape0, shape1, stride0, stride1);
  // 1D array (ptr, shape, stride); tile sizes are embedded constants.
  using fused_1nn_cutile_kernel_t = void(void*,
                                         int64_t,
                                         int64_t,
                                         int64_t,
                                         int64_t,
                                         void*,
                                         int64_t,
                                         int64_t,
                                         int64_t,
                                         int64_t,
                                         void*,
                                         int64_t,
                                         int64_t,
                                         void*,
                                         int64_t,
                                         int64_t,
                                         void*,
                                         int64_t,
                                         int64_t,
                                         void*,
                                         int64_t,
                                         int64_t,
                                         int64_t,
                                         int64_t,
                                         int64_t);
  std::cout << "Launching cuTile kernel" << std::endl;
  launcher->template dispatch<fused_1nn_cutile_kernel_t>(stream,
                                                         grid,
                                                         block,
                                                         0,
                                                         x_ptr,
                                                         shape_x[0],
                                                         shape_x[1],
                                                         stride_x[0],
                                                         stride_x[1],
                                                         y_ptr,
                                                         shape_y[0],
                                                         shape_y[1],
                                                         stride_y[0],
                                                         stride_y[1],
                                                         xn_ptr,
                                                         shape_xn,
                                                         stride_xn,
                                                         yn_ptr,
                                                         shape_yn,
                                                         stride_yn,
                                                         idx_ptr,
                                                         shape_idx,
                                                         stride_idx,
                                                         dist_ptr,
                                                         shape_dist,
                                                         stride_dist,
                                                         M,
                                                         N,
                                                         K);

  pack_fused_1nn_kvp<IdxT, OutT>
    <<<(m + 255) / 256, 256, 0, stream>>>(out, d_idx, d_dist, m, apply_sqrt);
  RAFT_CUDA_TRY(cudaGetLastError());
  RAFT_CUDA_TRY(cudaFreeAsync(d_idx, stream));
  RAFT_CUDA_TRY(cudaFreeAsync(d_dist, stream));
  return true;
}

template <typename DataT, typename OutT, typename IdxT>
bool try_fused_1nn_tile_dispatch(OutT* min,
                                 const DataT* x,
                                 const DataT* y,
                                 const DataT* xn,
                                 const DataT* yn,
                                 IdxT m,
                                 IdxT n,
                                 IdxT k,
                                 cuvs::distance::DistanceType metric,
                                 bool is_sqrt,
                                 cudaStream_t stream)
{
  switch (metric) {
    case cuvs::distance::DistanceType::InnerProduct:
      return launch_fused_1nn_tile<cuvs::distance::DistanceType::InnerProduct, DataT, IdxT, OutT>(
        x, y, xn, yn, min, m, n, k, is_sqrt, stream);
    case cuvs::distance::DistanceType::L2Expanded:
      return launch_fused_1nn_tile<cuvs::distance::DistanceType::L2Expanded, DataT, IdxT, OutT>(
        x, y, xn, yn, min, m, n, k, is_sqrt, stream);
    case cuvs::distance::DistanceType::L2SqrtExpanded:
      return launch_fused_1nn_tile<cuvs::distance::DistanceType::L2SqrtExpanded, DataT, IdxT, OutT>(
        x, y, xn, yn, min, m, n, k, is_sqrt, stream);
    case cuvs::distance::DistanceType::CosineExpanded:
      return launch_fused_1nn_tile<cuvs::distance::DistanceType::CosineExpanded, DataT, IdxT, OutT>(
        x, y, xn, yn, min, m, n, k, is_sqrt, stream);
    default: return false;
  }
}

}  // namespace

template <typename DataT, typename OutT, typename IdxT>
  requires Fused1nnKvpOutput<OutT, IdxT, DataT>
bool try_fused_1nn_tile(OutT* min,
                        const DataT* x,
                        const DataT* y,
                        const DataT* xn,
                        const DataT* yn,
                        IdxT m,
                        IdxT n,
                        IdxT k,
                        cuvs::distance::DistanceType metric,
                        bool is_sqrt,
                        cudaStream_t stream)
{
  if (!cuvs::detail::jit_lto::cutile_launch_available_on_current_device()) { return false; }
  return try_fused_1nn_tile_dispatch<DataT, OutT, IdxT>(
    min, x, y, xn, yn, m, n, k, metric, is_sqrt, stream);
}

using kvp_i_f   = raft::KeyValuePair<int, float>;
using kvp_i64_f = raft::KeyValuePair<int64_t, float>;
using kvp_i_h   = raft::KeyValuePair<int, half>;
using kvp_i64_h = raft::KeyValuePair<int64_t, half>;

#define CUVS_INST_TRY_FUSED_1NN_TILE(DataT, OutT, IdxT)                                         \
  template CUVS_EXPORT bool try_fused_1nn_tile<DataT, OutT, IdxT>(OutT*,                        \
                                                                  const DataT*,                 \
                                                                  const DataT*,                 \
                                                                  const DataT*,                 \
                                                                  const DataT*,                 \
                                                                  IdxT,                         \
                                                                  IdxT,                         \
                                                                  IdxT,                         \
                                                                  cuvs::distance::DistanceType, \
                                                                  bool,                         \
                                                                  cudaStream_t)

// int and int64_t are the same on LP64; one instantiation covers both.
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
