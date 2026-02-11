/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "./pairwise_distance_epilogue.h"

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/threadblock/default_epilogue_direct_store.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/gemm/kernel/default_gemm_universal.h>
#include <cutlass/gemm/kernel/gemm_with_fused_epilogue.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/layout/tensor.h>

#include <cuda_fp16.h>

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cuvs::gemm::kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  /// Element type for A matrix operand
  typename ElementA_,  // NOLINT(readability-identifier-naming)
  /// Layout type for A matrix operand
  int kAlignmentA,
  /// Element type for B matrix operand
  typename ElementB_,  // NOLINT(readability-identifier-naming)
  /// Layout type for B matrix operand
  int kAlignmentB,
  /// Element type for C and D matrix operands
  typename ElementC_,  // NOLINT(readability-identifier-naming)
  /// Element type for internal accumulation
  typename ElementAccumulator,
  /// Element type for final output
  // typename ElementOutT,
  /// Epilogue output operator      - must satisfy concept of 'EpilogueWithBroadcastOp'
  typename EpilogueOutputOp,
  /// Number of stages used in the pipelined mainloop
  int Stages,
  /// data layout row/column major of inputs
  bool isRowMajor>
struct PairwiseDistanceGemm {  // NOLINT(readability-identifier-naming)
  // This struct is specialized for fp32/3xTF32

  /// Threadblock-level tile size (concept: GemmShape)
  using ThreadblockShape =                   // NOLINT(readability-identifier-naming)
    cutlass::gemm::GemmShape<128, 128, 16>;  // <- threadblock tile M = 128, N = 128, K = 16
  /// Warp-level tile size (concept: GemmShape)
  // This code section describes tile size a warp will compute
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;  // <- warp tile M = 64, N = 64, K = 16 //
                                                           // NOLINT(readability-identifier-naming)
  /// Warp-level tile size (concept: GemmShape)
  // This code section describes the size of MMA op
  using InstructionShape =               // NOLINT(readability-identifier-naming)
    cutlass::gemm::GemmShape<16, 8, 4>;  // <- MMA Op tile M = 16, N = 8, K = 4

  /// Operation performed by GEMM
  using Operator = cutlass::arch::OpMultiplyAddFastF32;  // NOLINT(readability-identifier-naming)

  // This code section describes whether you want to use tensor cores or regular SIMT cores on GPU
  // SM
  using OperatorClass = cutlass::arch::OpClassTensorOp;  // NOLINT(readability-identifier-naming)

  // This code section describes CUDA SM architecture number
  using ArchTag = cutlass::arch::Sm80;  // NOLINT(readability-identifier-naming)

  // This code section describes how threadblocks are scheduled on GPU
  /// Threadblock-level swizzling operator
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  /// data layout for final output matrix.
  // we keep this same layout even for column major inputs
  using LayoutOutput = cutlass::layout::RowMajor;  // NOLINT(readability-identifier-naming)

  using NormXLayout =
    std::conditional_t<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>;

  using LayoutA_ =
    std::conditional_t<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>;

  using LayoutB_ =
    std::conditional_t<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;

  using GemmBase =
    typename cutlass::gemm::kernel::DefaultGemmUniversal<ElementA_,
                                                         LayoutA_,
                                                         cutlass::ComplexTransform::kNone,
                                                         kAlignmentA,
                                                         ElementB_,
                                                         LayoutB_,
                                                         cutlass::ComplexTransform::kNone,
                                                         kAlignmentB,
                                                         ElementC_,
                                                         LayoutOutput,
                                                         ElementAccumulator,
                                                         OperatorClass,
                                                         ArchTag,
                                                         ThreadblockShape,
                                                         WarpShape,
                                                         InstructionShape,
                                                         EpilogueOutputOp,
                                                         ThreadblockSwizzle,
                                                         Stages,
                                                         Operator>::GemmKernel;

  // Replace epilogue
  using Epilogue = typename cuvs::epilogue::threadblock::PairwiseDistanceEpilogue<
    typename GemmBase::Epilogue::Shape,
    typename GemmBase::Epilogue::WarpMmaOperator,
    GemmBase::Epilogue::kPartitionsK,
    ElementAccumulator,
    typename EpilogueOutputOp::ElementT,
    ElementAccumulator,
    EpilogueOutputOp,
    NormXLayout,
    GemmBase::Epilogue::kElementsPerAccess>::Epilogue;

  // Compose the GEMM kernel
  using GemmKernel = cutlass::gemm::kernel::
    GemmWithFusedEpilogue<typename GemmBase::Mma, Epilogue, ThreadblockSwizzle>;
};

template <
  /// Layout type for A matrix operand
  int kAlignmentA,
  /// Layout type for B matrix operand
  int kAlignmentB,
  /// Element type for C and D matrix operands
  typename ElementC_,  // NOLINT(readability-identifier-naming)
  /// Element type for internal accumulation
  typename ElementAccumulator,
  /// Epilogue output operator      - must satisfy concept of 'EpilogueWithBroadcastOp'
  typename EpilogueOutputOp,
  /// Number of stages used in the pipelined mainloop
  int Stages,
  /// data layout row/column major of inputs
  bool isRowMajor>
struct PairwiseDistanceGemm<double,
                            kAlignmentA,
                            double,
                            kAlignmentB,
                            ElementC_,
                            ElementAccumulator,
                            EpilogueOutputOp,
                            Stages,
                            isRowMajor> {
  // using Transform = cutlass::ComplexTransform::kNone;
  // Threadblock-level tile size (concept: GemmShape)
  using ThreadblockShape =                 // NOLINT(readability-identifier-naming)
    cutlass::gemm::GemmShape<64, 64, 16>;  // <- threadblock tile M = 64, N = 64, K = 16
  /// Warp-level tile size (concept: GemmShape)
  // This code section describes tile size a warp will compute
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 16>;  // <- warp tile M = 32, N = 32, K = 16 //
                                                           // NOLINT(readability-identifier-naming)
  /// Warp-level tile size (concept: GemmShape)
  // This code section describes the size of MMA op
  using InstructionShape =
    cutlass::gemm::GemmShape<8, 8, 4>;  // NOLINT(readability-identifier-naming)

  // Operation performed by GEMM
  using Operator = cutlass::arch::OpMultiplyAdd;  // NOLINT(readability-identifier-naming)
  // This code section describes whether you want to use tensor cores or regular SIMT cores on GPU
  // SM
  using OperatorClass = cutlass::arch::OpClassTensorOp;  // NOLINT(readability-identifier-naming)

  // This code section describes CUDA SM architecture number
  using ArchTag = cutlass::arch::Sm80;  // NOLINT(readability-identifier-naming)

  // This code section describes how threadblocks are scheduled on GPU
  /// Threadblock-level swizzling operator
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  /// data layout for final output matrix.
  // we keep this same layout even for column major inputs
  using LayoutOutput = cutlass::layout::RowMajor;  // NOLINT(readability-identifier-naming)

  using NormXLayout =
    std::conditional_t<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>;

  using LayoutA_ =
    std::conditional_t<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>;

  using LayoutB_ =
    std::conditional_t<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;

  using GemmBase =
    typename cutlass::gemm::kernel::DefaultGemmUniversal<double,
                                                         LayoutA_,
                                                         cutlass::ComplexTransform::kNone,
                                                         1,
                                                         double,
                                                         LayoutB_,
                                                         cutlass::ComplexTransform::kNone,
                                                         1,
                                                         ElementC_,
                                                         LayoutOutput,
                                                         ElementAccumulator,
                                                         OperatorClass,
                                                         ArchTag,
                                                         ThreadblockShape,
                                                         WarpShape,
                                                         InstructionShape,
                                                         EpilogueOutputOp,
                                                         ThreadblockSwizzle,
                                                         Stages,
                                                         Operator>::GemmKernel;

  // Replace epilogue
  using Epilogue = typename cuvs::epilogue::threadblock::PairwiseDistanceEpilogue<
    typename GemmBase::Epilogue::Shape,
    typename GemmBase::Epilogue::WarpMmaOperator,
    GemmBase::Epilogue::kPartitionsK,
    ElementC_,
    typename EpilogueOutputOp::ElementT,
    ElementC_,
    EpilogueOutputOp,
    NormXLayout,
    GemmBase::Epilogue::kElementsPerAccess>::Epilogue;

  // Compose the GEMM kernel
  using GemmKernel = cutlass::gemm::kernel::
    GemmWithFusedEpilogue<typename GemmBase::Mma, Epilogue, ThreadblockSwizzle>;
};

template <
  /// Layout type for A matrix operand
  int kAlignmentA,
  /// Layout type for B matrix operand
  int kAlignmentB,
  /// Element type for C and D matrix operands
  typename ElementC_,  // NOLINT(readability-identifier-naming)
  /// Element type for internal accumulation
  typename ElementAccumulator,
  /// Epilogue output operator      - must satisfy concept of 'EpilogueWithBroadcastOp'
  typename EpilogueOutputOp,
  /// Number of stages used in the pipelined mainloop
  int Stages,
  /// data layout row/column major of inputs
  bool isRowMajor>
struct PairwiseDistanceGemm<half,
                            kAlignmentA,
                            half,
                            kAlignmentB,
                            ElementC_,
                            ElementAccumulator,
                            EpilogueOutputOp,
                            Stages,
                            isRowMajor> {
  // using Transform = cutlass::ComplexTransform::kNone;
  // Threadblock-level tile size (concept: GemmShape)
  using ThreadblockShape =                   // NOLINT(readability-identifier-naming)
    cutlass::gemm::GemmShape<128, 128, 32>;  // <- threadblock tile M = 64, N = 64, K = 16
  /// Warp-level tile size (concept: GemmShape)
  // This code section describes tile size a warp will compute
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;  // <- warp tile M = 32, N = 32, K = 16 //
                                                           // NOLINT(readability-identifier-naming)
  /// Warp-level tile size (concept: GemmShape)
  // This code section describes the size of MMA op
  using InstructionShape =
    cutlass::gemm::GemmShape<16, 8, 16>;  // NOLINT(readability-identifier-naming)

  // Operation performed by GEMM
  using Operator = cutlass::arch::OpMultiplyAdd;  // NOLINT(readability-identifier-naming)
  // This code section describes whether you want to use tensor cores or regular SIMT cores on GPU
  // SM
  using OperatorClass = cutlass::arch::OpClassTensorOp;  // NOLINT(readability-identifier-naming)

  // This code section describes CUDA SM architecture number
  using ArchTag = cutlass::arch::Sm80;  // NOLINT(readability-identifier-naming)

  // This code section describes how threadblocks are scheduled on GPU
  /// Threadblock-level swizzling operator
  using ThreadblockSwizzle = cutlass::gemm::threadblock::
    GemmBatchedIdentityThreadblockSwizzle;  // NOLINT(readability-identifier-naming)

  /// data layout for final output matrix.
  // we keep this same layout even for column major inputs
  using LayoutOutput = cutlass::layout::RowMajor;  // NOLINT(readability-identifier-naming)

  using NormXLayout =
    std::conditional_t<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>;

  using LayoutA_ =
    std::conditional_t<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>;

  using LayoutB_ =
    std::conditional_t<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;

  using GemmBase = typename cutlass::gemm::device::GemmUniversal<cutlass::half_t,
                                                                 LayoutA_,
                                                                 cutlass::half_t,
                                                                 LayoutB_,
                                                                 ElementC_,
                                                                 LayoutOutput,
                                                                 ElementAccumulator,
                                                                 OperatorClass,
                                                                 ArchTag,
                                                                 ThreadblockShape,
                                                                 WarpShape,
                                                                 InstructionShape,
                                                                 EpilogueOutputOp,
                                                                 ThreadblockSwizzle,
                                                                 3,
                                                                 2,
                                                                 2>::GemmKernel;

  // Replace epilogue
  using Epilogue = typename cuvs::epilogue::threadblock::PairwiseDistanceEpilogue<
    typename GemmBase::Epilogue::Shape,
    typename GemmBase::Epilogue::WarpMmaOperator,
    GemmBase::Epilogue::kPartitionsK,
    ElementC_,
    typename EpilogueOutputOp::ElementT,
    ElementC_,
    EpilogueOutputOp,
    NormXLayout,
    GemmBase::Epilogue::kElementsPerAccess>::Epilogue;

  // Compose the GEMM kernel
  using GemmKernel = cutlass::gemm::kernel::
    GemmWithFusedEpilogue<typename GemmBase::Mma, Epilogue, ThreadblockSwizzle>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cuvs::gemm::kernel
