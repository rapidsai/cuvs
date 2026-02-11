/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Wtautological-compare"

// We define CUTLASS_NAMESPACE in case
// RAFT cmake is not used
#ifndef CUTLASS_NAMESPACE
#define cutlass cuvs_cutlass
#endif

#include "../../../util/cutlass_utils.hpp"  // CUVS_CUTLASS_TRY
#include "epilogue_elementwise.cuh"         // FusedDistanceNNEpilogueElementwise
#include "gemm.h"                           // FusedDistanceNNGemm
#include <raft/util/cudart_utils.hpp>       // getMultiProcessorCount

#include <rmm/device_uvector.hpp>

#include <cuda/semaphore>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/layout/tensor.h>
#include <cutlass/matrix_coord.h>
#include <cutlass/tensor_view.h>

namespace cuvs {
namespace distance {
namespace detail {

template <typename IdxT>
RAFT_KERNEL init_bin_mutex_kernel(cuda::binary_semaphore<cuda::thread_scope_device>* mut, IdxT m)
{
  auto tid = IdxT(blockIdx.x) * blockDim.x + threadIdx.x;

  if (tid < m) { mut[tid].release(); }
}

template <typename DataT,
          typename AccT,
          typename OutT,
          typename IdxT,
          int VecLen,
          typename CGReduceOpT,
          typename DistanceFn,
          typename ReduceOpT,
          typename KVPReduceOpT>
void cutlass_fused_distance_nn(const DataT* x,
                               const DataT* y,
                               const DataT* xn,
                               const DataT* yn,
                               IdxT m,
                               IdxT n,
                               IdxT k,
                               IdxT lda,
                               IdxT ldb,
                               IdxT ldd,
                               OutT* dOutput,
                               int* mutexes,
                               CGReduceOpT cg_reduce_op,
                               DistanceFn dist_op,
                               ReduceOpT redOp,
                               KVPReduceOpT pairRedOp,
                               cudaStream_t stream)
{
  using epilogue_output_op_t = cuvs::epilogue::thread::FusedDistanceNNEpilogueElementwise<
    DataT,  // ElementC_
    AccT,   // ElementAccumulator_
    DataT,  // ElementCompute_
    AccT,   // ElementZ_
    OutT,   // ElementT_
    // 128 / cutlass::sizeof_bits<DataT>::value,
    1,  // Elements per access 1
    DistanceFn,
    CGReduceOpT,
    ReduceOpT,
    KVPReduceOpT>;
  constexpr int kBatchCount = 1;

  rmm::device_uvector<cuda::binary_semaphore<cuda::thread_scope_device>> bin_mutex(m, stream);

  int nblks = (m / 256) + 1;

  init_bin_mutex_kernel<<<nblks, 256, 0, stream>>>(bin_mutex.data(), m);

  typename epilogue_output_op_t::Params epilog_op_param(
    dist_op, cg_reduce_op, redOp, pairRedOp, mutexes, bin_mutex.data());

  // Number of pipelines you want to use
  constexpr int kNumStages = 3;
  // Alignment
  constexpr int kAlignment = VecLen;

  // default initialize problem size with row major inputs
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k);

  constexpr bool kIsRowMajor = true;

  using fused_distance_nn_kernel_t =
    typename cuvs::gemm::kernel::FusedDistanceNNGemm<DataT,
                                                     kAlignment,
                                                     DataT,
                                                     kAlignment,
                                                     AccT,
                                                     AccT,
                                                     epilogue_output_op_t,
                                                     kNumStages,  // Number of pipeline stages
                                                     kIsRowMajor>::GemmKernel;

  using fused_distance_nn_t = cutlass::gemm::device::GemmGrouped<fused_distance_nn_kernel_t>;

  int num_blocks_per_sm    = fused_distance_nn_t::maximum_active_blocks();
  int num_sms              = raft::getMultiProcessorCount();
  int full_wave            = num_blocks_per_sm * num_sms;
  constexpr int kMmaShapeM = fused_distance_nn_kernel_t::Mma::Shape::kM;
  constexpr int kMmaShapeN = fused_distance_nn_kernel_t::Mma::Shape::kN;
  int column_tiles         = (problem_size.n() - 1 + kMmaShapeN) / kMmaShapeN;
  int row_tiles            = (problem_size.m() - 1 + kMmaShapeM) / kMmaShapeM;
  int total_tiles          = column_tiles * row_tiles;
  int thread_blocks =
    row_tiles < full_wave ? (total_tiles < full_wave ? total_tiles : full_wave) : row_tiles;

  typename fused_distance_nn_t::Arguments arguments{
    problem_size,
    kBatchCount,  // num of problems.
    thread_blocks,
    epilog_op_param,
    x,
    y,
    xn,                         // C matrix eq vector param, which here is A norm
    const_cast<DataT*>(yn),     // this is broadcast vec, which is required to be non-const param
    dOutput,                    // Output distance matrix
    static_cast<int64_t>(lda),  // stride A
    static_cast<int64_t>(ldb),  // stride B
    static_cast<int64_t>(1),    // stride A norm
    static_cast<int64_t>(ldd)   // stride Output matrix
  };

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = fused_distance_nn_t::get_workspace_size(arguments);
  // Allocate workspace memory
  rmm::device_uvector<uint8_t> workspace(workspace_size, stream);
  // Instantiate CUTLASS kernel depending on templates
  fused_distance_nn_t fused_distance_nn_op;
  // Check the problem size is supported or not
  CUVS_CUTLASS_TRY(fused_distance_nn_op.can_implement(arguments));
  // Initialize CUTLASS kernel with arguments and workspace pointer
  CUVS_CUTLASS_TRY(fused_distance_nn_op.initialize(arguments, workspace.data(), stream));
  // Launch initialized CUTLASS kernel
  CUVS_CUTLASS_TRY(fused_distance_nn_op.run(stream));
}

};  // namespace detail
};  // namespace distance
};  // namespace cuvs

#pragma GCC diagnostic pop
