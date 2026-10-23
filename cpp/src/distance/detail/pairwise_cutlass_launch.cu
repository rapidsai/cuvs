/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../util/cutlass_utils.hpp"
#include "distance_ops/l2_exp.cuh"
#include "pairwise_cutlass_launch.hpp"
#include "pairwise_distance_epilogue_elementwise.h"
#include "pairwise_distance_gemm.h"
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <raft/core/operators.hpp>
#include <rmm/device_uvector.hpp>

namespace cuvs::distance::detail {
template <typename T, int Alignment, bool RowMajor>
using pairwise_cutlass_l2 =
  cutlass::gemm::device::GemmUniversalAdapter<typename gemm::kernel::PairwiseDistanceGemm<
    T,
    Alignment,
    T,
    Alignment,
    T,
    T,
    epilogue::thread::PairwiseDistanceEpilogueElementwise<T,
                                                          T,
                                                          T,
                                                          T,
                                                          T,
                                                          1,
                                                          ops::l2_exp_cutlass_op<T, T>,
                                                          raft::identity_op>,
    3,
    RowMajor>::GemmKernel>;

template <typename T, int Alignment, bool RowMajor>
void launch_pairwise_cutlass_l2(int m,
                                int n,
                                int k,
                                const T* a,
                                const T* b,
                                const T* xn,
                                T* yn,
                                T* output,
                                int64_t lda,
                                int64_t ldb,
                                int64_t ldd,
                                bool sqrt,
                                cudaStream_t stream)
{
  using Gemm = pairwise_cutlass_l2<T, Alignment, RowMajor>;
  using Epilogue =
    epilogue::thread::PairwiseDistanceEpilogueElementwise<T,
                                                          T,
                                                          T,
                                                          T,
                                                          T,
                                                          1,
                                                          ops::l2_exp_cutlass_op<T, T>,
                                                          raft::identity_op>;
  typename Epilogue::Params epilogue(ops::l2_exp_cutlass_op<T, T>(sqrt), raft::identity_op{});
  typename Gemm::Arguments arguments{cutlass::gemm::GemmUniversalMode::kGemm,
                                     cutlass::gemm::GemmCoord(m, n, k),
                                     1,
                                     epilogue,
                                     a,
                                     b,
                                     xn,
                                     nullptr,
                                     yn,
                                     output,
                                     int64_t{0},
                                     int64_t{0},
                                     int64_t{0},
                                     int64_t{0},
                                     int64_t{0},
                                     int64_t{0},
                                     lda,
                                     ldb,
                                     1,
                                     0,
                                     0,
                                     ldd};
  rmm::device_uvector<uint8_t> workspace(Gemm::get_workspace_size(arguments), stream);
  Gemm operation;
  CUVS_CUTLASS_TRY(operation.can_implement(arguments));
  CUVS_CUTLASS_TRY(operation.initialize(arguments, workspace.data(), stream));
  CUVS_CUTLASS_TRY(operation(stream));
}

#define INSTANTIATE(T, Alignment, RowMajor)                                  \
  template void launch_pairwise_cutlass_l2<T, Alignment, RowMajor>(int,      \
                                                                   int,      \
                                                                   int,      \
                                                                   const T*, \
                                                                   const T*, \
                                                                   const T*, \
                                                                   T*,       \
                                                                   T*,       \
                                                                   int64_t,  \
                                                                   int64_t,  \
                                                                   int64_t,  \
                                                                   bool,     \
                                                                   cudaStream_t);
INSTANTIATE(float, 1, true)
INSTANTIATE(float, 1, false)
INSTANTIATE(float, 2, true)
INSTANTIATE(float, 2, false)
INSTANTIATE(float, 4, true)
INSTANTIATE(float, 4, false)
INSTANTIATE(double, 1, true)
INSTANTIATE(double, 1, false)
INSTANTIATE(double, 2, true)
INSTANTIATE(double, 2, false)

#undef INSTANTIATE
}  // namespace cuvs::distance::detail
