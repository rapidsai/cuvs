/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../distance_ops/l2_exp.cuh"     // ops::l2_exp_distance_op
#include "../pairwise_distance_base.cuh"  // PairwiseDistances
#include "cutlass_base.cuh"
#include "simt_kernel.cuh"
#include <raft/core/kvp.hpp>             // raft::KeyValuePair
#include <raft/core/operators.hpp>       // raft::identity_op
#include <raft/linalg/contractions.cuh>  // policy
#include <raft/util/arch.cuh>            // raft::util::arch::SM_*
#include <raft/util/cuda_utils.cuh>      // raft::ceildiv, raft::shfl
#include <raft/util/device_atomics.cuh>

#include <cstddef>  // size_t
#include <limits>   // std::numeric_limits

namespace cuvs::distance::detail {

template <typename label_t, typename DataT>
struct kvp_min_reduce_impl {
  using kvp = raft::KeyValuePair<label_t, DataT>;
  DI auto operator()(label_t rit, const kvp& a, const kvp& b) -> kvp
  {
    return b.value < a.value ? b : a;
  }
  DI auto operator()(const kvp& a, const kvp& b) -> kvp { return b.value < a.value ? b : a; }

};  // kvp_min_reduce

template <typename label_t, typename DataT>
struct min_and_distance_reduce_op_impl {
  using kvp = typename raft::KeyValuePair<label_t, DataT>;

  DI void operator()(label_t rid, kvp* out, const kvp& other) const
  {
    if (other.value < out->value) {
      out->key   = other.key;
      out->value = other.value;
    }
  }
  DI void operator()(label_t rid, volatile kvp* out, const kvp& other) const
  {
    if (other.value < out->value) {
      out->key   = other.key;
      out->value = other.value;
    }
  }

  DI void operator()(label_t rid, DataT* out, const kvp& other) const
  {
    if (other.value < *out) { *out = other.value; }
  }

  DI void operator()(label_t rid, volatile DataT* out, const kvp& other) const
  {
    if (other.value < *out) { *out = other.value; }
  }

  DI void operator()(label_t rid, DataT* out, const DataT& other) const
  {
    if (other < *out) { *out = other; }
  }

  DI void operator()(label_t rid, volatile DataT* out, const DataT& other) const
  {
    if (other < *out) { *out = other; }
  }

  DI void init(DataT* out, DataT maxVal) const { *out = maxVal; }
  DI void init(kvp* out, DataT maxVal) const
  {
    out->value = maxVal;
    out->key   = 0xfffffff0;
  }

  DI void init_key(DataT& out, label_t idx) const { return; }
  DI void init_key(kvp& out, label_t idx) const { out.key = idx; }

  DI auto get_value(kvp& out) const -> DataT { return out.value; }
  DI auto get_value(DataT& out) const -> DataT { return out; }
};

template <typename label_t, typename DataT>
struct min_reduce_op_impl {
  using kvp = typename raft::KeyValuePair<label_t, DataT>;
  DI void operator()(label_t rid, DataT* out, const kvp& other)
  {
    if (other.value < *out) { *out = other.value; }
  }

  DI void init(DataT* out, DataT maxVal) { *out = maxVal; }
};

template <typename DataT, typename OutT, typename IdxT, typename ReduceOpT>
RAFT_KERNEL init_kernel(OutT* min, IdxT m, DataT maxVal, ReduceOpT redOp)
{
  auto tid = IdxT(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid < m) { redOp.init(min + tid, maxVal); }
}

template <typename DataT, typename OutT, typename IdxT, typename ReduceOpT>
void initialize(OutT* min, IdxT m, DataT maxVal, ReduceOpT redOp, cudaStream_t stream)
{
  auto blks = raft::ceildiv(m, 256);
  init_kernel<DataT, OutT, IdxT><<<blks, 256, 0, stream>>>(min, m, maxVal, redOp);
}

// cg::reduce functor for FusedDistanceNN used in its cutlass version
// to output the min distance value & key(loc id).
// This is used in fused_distance_nn/predicated_tile_iterator_reduced_vec.h
// store_with_byte_offset() passed to cg::reduce() & select_reduce.
template <typename AccType, typename Index, typename OutType>
struct kvp_cg_min_reduce_op {
  using kvp = typename raft::KeyValuePair<Index, AccType>;

  __host__ __device__ kvp_cg_min_reduce_op() noexcept = default;

  using acc_type_t = AccType;
  using index_t    = Index;
  // functor signature.
  __host__ __device__ auto operator()(kvp a, kvp b) const -> kvp
  {
    return a.value < b.value ? a : b;
  }

  __host__ __device__ auto operator()(AccType a, AccType b) const -> AccType { return min(a, b); }

  __host__ __device__ auto is_amin(AccType a, AccType b) const -> bool
  {
    return a < b ? true : false;
  }
};

}  // namespace cuvs::distance::detail
