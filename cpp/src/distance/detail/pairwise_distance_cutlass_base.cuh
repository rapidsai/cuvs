/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
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

#include "pairwise_distance_epilogue_elementwise.h"
#include "pairwise_distance_gemm.h"

#include "../../util/cutlass_utils.hpp"
#include "distance_ops/cutlass.cuh"

#include <rmm/device_uvector.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/layout/tensor.h>
#include <cutlass/matrix_coord.h>
#include <cutlass/tensor_view.h>

#include <type_traits>

namespace cuvs {
namespace distance {
namespace detail {

template <typename DataT,
          typename AccT,
          typename OutT,
          typename IdxT,
          int VecLen,
          typename FinalLambda,
          typename OpT,
          bool isRowMajor>
auto cutlassDistanceKernel(const DataT* x,  // NOLINT(readability-identifier-naming)
                           const DataT* y,
                           const OutT* xn,
                           const OutT* yn,
                           IdxT m,
                           IdxT n,
                           IdxT k,
                           IdxT lda,
                           IdxT ldb,
                           IdxT ldd,
                           OutT* dOutput,
                           FinalLambda fin_op,
                           OpT distance_op,
                           cudaStream_t stream) -> std::enable_if_t<ops::has_cutlass_op<OpT>::value>
{
  static_assert(!(std::is_same_v<OutT, bool>), "OutType bool is not supported use uint8_t instead");

  auto dist_op           = distance_op.get_cutlass_op();
  using DistanceFn       = decltype(dist_op);  // NOLINT(readability-identifier-naming)
  using EpilogueOutputOp =                     // NOLINT(readability-identifier-naming)
    epilogue::thread::PairwiseDistanceEpilogueElementwise<OutT,  // ElementC_
                                                          AccT,  // ElementAccumulator_
                                                          AccT,  // ElementCompute_
                                                          OutT,  // ElementZ_
                                                          OutT,  // ElementT_
                                                          1,     // Elements per access 1
                                                          DistanceFn,
                                                          FinalLambda>;
  constexpr int batch_count = 1;  // NOLINT(readability-identifier-naming)

  constexpr auto mode =
    cutlass::gemm::GemmUniversalMode::kGemm;  // NOLINT(readability-identifier-naming)

  typename EpilogueOutputOp::Params epilog_op_param(dist_op, fin_op);

  // Number of pipelines you want to use
  constexpr int NumStages = 3;  // NOLINT(readability-identifier-naming)
  // Alignment
  constexpr int Alignment = VecLen;  // NOLINT(readability-identifier-naming)

  using cutlassDistKernel =  // NOLINT(readability-identifier-naming)
    typename gemm::kernel::PairwiseDistanceGemm<DataT,
                                                Alignment,
                                                DataT,
                                                Alignment,
                                                AccT,
                                                AccT,
                                                EpilogueOutputOp,
                                                NumStages,  // Number of pipeline stages
                                                isRowMajor>::GemmKernel;

  using cutlassDist = cutlass::gemm::device::GemmUniversalAdapter<cutlassDistKernel>;

  constexpr uint32_t gridYZMax =
    ((1 << (sizeof(uint16_t) * 8)) - 1);  // NOLINT(readability-identifier-naming)
  constexpr uint32_t max_batch_size =
    gridYZMax * cutlassDistKernel::ThreadblockShape::kN;  // NOLINT(readability-identifier-naming)
  IdxT numNbatches =
    (n - 1 + max_batch_size) / max_batch_size;  // NOLINT(readability-identifier-naming)

  for (IdxT i = 0; i < numNbatches; i++) {
    const DataT *a, *b;
    IdxT gemm_lda, gemm_ldb;
    size_t offsetN = i * max_batch_size;  // NOLINT(readability-identifier-naming)

    if constexpr (isRowMajor) {
      gemm_lda = ldb;
      gemm_ldb = lda;
      a        = y + offsetN * gemm_lda;
      b        = x;
    } else {
      gemm_lda = lda;
      gemm_ldb = ldb;
      a        = x;
      b        = y + offsetN;
    }
    IdxT chunkN = (i + 1) * max_batch_size;  // NOLINT(readability-identifier-naming)
    IdxT currentN =
      (chunkN < n) ? max_batch_size : (n - offsetN);  // NOLINT(readability-identifier-naming)

    // default initialize problem size with row major inputs
    auto problem_size = isRowMajor ? cutlass::gemm::GemmCoord(currentN, m, k)
                                   : cutlass::gemm::GemmCoord(m, currentN, k);

    typename cutlassDist::Arguments arguments{
      mode,
      problem_size,
      batch_count,
      epilog_op_param,
      a,
      b,
      xn,       // C matrix eq vector param, which here is A norm
      nullptr,  // tensor_Z,
      const_cast<OutT*>(yn) +
        offsetN,                // this is broadcast vec, which is required to be non-const param
      dOutput + offsetN,        // Output distance matrix
      static_cast<int64_t>(0),  // batch stride A
      static_cast<int64_t>(0),  // batch stride B
      static_cast<int64_t>(0),  // batch stride Norm A
      static_cast<int64_t>(0),
      static_cast<int64_t>(0),  // batch stride Norm B
      static_cast<int64_t>(0),  // batch stride Output
      gemm_lda,                 // stride A
      gemm_ldb,                 // stride B
      1,                        // stride A norm
      0,                        // this is no-op for Z
      0,                        // This must be zero
      ldd                       // stride Output matrix
    };

    // Using the arguments, query for extra workspace required for matrix multiplication computation
    size_t workspace_size = cutlassDist::get_workspace_size(arguments);
    // Allocate workspace memory
    rmm::device_uvector<uint8_t> workspace(workspace_size, stream);
    // Instantiate CUTLASS kernel depending on templates
    cutlassDist cutlassDist_op;  // NOLINT(readability-identifier-naming)
    // Check the problem size is supported or not
    CUVS_CUTLASS_TRY(cutlassDist_op.can_implement(arguments));

    // Initialize CUTLASS kernel with arguments and workspace pointer
    CUVS_CUTLASS_TRY(cutlassDist_op.initialize(arguments, workspace.data(), stream));

    // Launch initialized CUTLASS kernel
    CUVS_CUTLASS_TRY(cutlassDist_op(stream));
  }
}

};  // namespace detail
};  // namespace distance
};  // namespace cuvs

#pragma GCC diagnostic pop
