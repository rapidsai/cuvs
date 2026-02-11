/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/math.hpp>

#include <cuvs/distance/distance.hpp>

#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>

namespace cuvs::neighbors::detail {

template <typename ValueIdx,
          typename value_t,
          typename OutputT,
          typename ExpansionF>  // NOLINT(readability-identifier-naming)
RAFT_KERNEL epilogue_on_csr_kernel(OutputT* __restrict__ compressed_C,
                                   const ValueIdx* __restrict__ rows,
                                   const ValueIdx* __restrict__ cols,
                                   const OutputT* __restrict__ Q_sq_norms,
                                   const value_t* __restrict__ R_sq_norms,
                                   ValueIdx nnz,
                                   ExpansionF expansion_func)
{
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid >= nnz) return;
  const ValueIdx i = rows[tid];
  const ValueIdx j = cols[tid];

  compressed_C[tid] = expansion_func(compressed_C[tid], Q_sq_norms[i], R_sq_norms[j]);
}

template <typename ValueIdx,
          typename value_t,
          typename OutputT,
          int tpb = 256>  // NOLINT(readability-identifier-naming)
void epilogue_on_csr(raft::resources const& handle,
                     OutputT* compressed_C,
                     const ValueIdx nnz,
                     const ValueIdx* rows,
                     const ValueIdx* cols,
                     const OutputT* Q_sq_norms,
                     const value_t* R_sq_norms,
                     cuvs::distance::DistanceType metric)
{
  if (nnz == 0) return;
  auto stream = raft::resource::get_cuda_stream(handle);

  int blocks = raft::ceildiv<size_t>((size_t)nnz, tpb);
  if (metric == cuvs::distance::DistanceType::L2Expanded) {
    epilogue_on_csr_kernel<<<blocks, tpb, 0, stream>>>(
      compressed_C,
      rows,
      cols,
      Q_sq_norms,
      R_sq_norms,
      nnz,
      [] __device__ __host__(OutputT dot, OutputT q_norm, value_t r_norm) -> OutputT {
        if constexpr (std::is_same_v<value_t, half>) {
          return OutputT(-2.0) * dot + q_norm + __half2float(r_norm);
        } else {
          return OutputT(-2.0) * dot + q_norm + r_norm;
        }
      });
  } else if (metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
    epilogue_on_csr_kernel<<<blocks, tpb, 0, stream>>>(
      compressed_C,
      rows,
      cols,
      Q_sq_norms,
      R_sq_norms,
      nnz,
      [] __device__ __host__(OutputT dot, OutputT q_norm, value_t r_norm) -> OutputT {
        if constexpr (std::is_same_v<value_t, half>) {
          return raft::sqrt(OutputT(-2.0) * dot + q_norm + __half2float(r_norm));
        } else {
          return raft::sqrt(OutputT(-2.0) * dot + q_norm + r_norm);
        }
      });
  } else if (metric == cuvs::distance::DistanceType::CosineExpanded) {
    epilogue_on_csr_kernel<<<blocks, tpb, 0, stream>>>(
      compressed_C,
      rows,
      cols,
      Q_sq_norms,
      R_sq_norms,
      nnz,
      [] __device__ __host__(OutputT dot, OutputT q_norm, value_t r_norm) -> OutputT {
        if constexpr (std::is_same_v<value_t, half>) {
          return OutputT(1.0) - dot / (q_norm * __half2float(r_norm));
        } else {
          return OutputT(1.0) - dot / (q_norm * r_norm);
        }
      });
  }
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}
}  // namespace cuvs::neighbors::detail
