/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/neighbors/epsilon_neighborhood.hpp>
#include <raft/linalg/contractions.cuh>
#include <raft/util/device_utils.cuh>

namespace cuvs::neighbors::epsilon_neighborhood::detail {

template <typename DataT,
          typename IdxT,
          typename Policy,
          typename BaseClass = raft::linalg::Contractions_NT<DataT, IdxT, Policy>>
struct eps_unexp_l2_sq_neighborhood_impl : public BaseClass {
 private:
  using p = Policy;

  bool* adj_;
  DataT eps_;
  IdxT* vd_;

  char* smem_;  // for final reductions

  DataT acc_[p::AccRowsPerTh][p::AccColsPerTh];

 public:
  DI eps_unexp_l2_sq_neighborhood_impl(bool* _adj,
                                       IdxT* _vd,
                                       const DataT* _x,
                                       const DataT* _y,
                                       IdxT _m,
                                       IdxT _n,
                                       IdxT _k,
                                       DataT _eps,
                                       char* _smem)
    : BaseClass(_x, _y, _m, _n, _k, _smem), adj_(_adj), eps_(_eps), vd_(_vd), smem_(_smem)
  {
  }

  DI void run()
  {
    prolog();
    loop();
    epilog();
  }

 private:
  DI void prolog()
  {
    this->ldgXY(IdxT(blockIdx.x) * p::Mblk, IdxT(blockIdx.y) * p::Nblk, 0);
#pragma unroll
    for (int i = 0; i < p::AccRowsPerTh; ++i) {
#pragma unroll
      for (int j = 0; j < p::AccColsPerTh; ++j) {
        acc_[i][j] = BaseClass::Zero();
      }
    }
    this->stsXY();
    __syncthreads();
    this->switch_write_buffer();
  }

  DI void loop()
  {
    for (int kidx = p::Kblk; kidx < this->k; kidx += p::Kblk) {
      this->ldgXY(IdxT(blockIdx.x) * p::Mblk, IdxT(blockIdx.y) * p::Nblk, kidx);
      accumulate();  // on the previous k-block
      this->stsXY();
      __syncthreads();
      this->switch_write_buffer();
      this->switch_read_buffer();
    }
    accumulate();  // last iteration
  }

  DI void epilog()
  {
    IdxT startx = blockIdx.x * p::Mblk + this->accrowid;
    IdxT starty = blockIdx.y * p::Nblk + this->acccolid;
    auto lid    = raft::laneId();
    IdxT sums[p::AccRowsPerTh];
#pragma unroll
    for (int i = 0; i < p::AccRowsPerTh; ++i) {
      sums[i]  = 0;
      auto xid = startx + i * p::AccThRows;
#pragma unroll
      for (int j = 0; j < p::AccColsPerTh; ++j) {
        auto yid      = starty + j * p::AccThCols;
        auto is_neigh = acc_[i][j] <= eps_;
        ///@todo: fix uncoalesced writes using shared mem
        if (xid < this->m && yid < this->n) {
          adj_[xid * this->n + yid] = is_neigh;
          sums[i] += is_neigh;
        }
      }
    }
    // perform reduction of adjacency values to compute vertex degrees
    if (vd_ != nullptr) { update_vertex_degree(sums); }
  }

  DI void accumulate()
  {
#pragma unroll
    for (int ki = 0; ki < p::Kblk; ki += p::Veclen) {
      this->ldsXY(ki);
#pragma unroll
      for (int i = 0; i < p::AccRowsPerTh; ++i) {
#pragma unroll
        for (int j = 0; j < p::AccColsPerTh; ++j) {
#pragma unroll
          for (int v = 0; v < p::Veclen; ++v) {
            auto diff = this->regx[i][v] - this->regy[j][v];
            acc_[i][j] += diff * diff;
          }
        }
      }
    }
  }

  DI void update_vertex_degree(IdxT (&sums)[p::AccRowsPerTh])
  {
    __syncthreads();  // so that we can safely reuse smem_
    int gid        = this->accrowid;
    int lid        = this->acccolid;
    auto cidx      = IdxT(blockIdx.x) * p::Mblk + gid;
    IdxT total_sum = 0;
    // update the individual vertex degrees
#pragma unroll
    for (int i = 0; i < p::AccRowsPerTh; ++i) {
      // p::AccThCols neighboring threads need to reduce
      // -> we have p::Nblk/p::AccThCols individual reductions
      auto cid = cidx + i * p::AccThRows;
      sums[i]  = raft::logicalWarpReduce<p::AccThCols>(sums[i], raft::add_op());
      if (lid == 0 && cid < this->m) {
        atomic_update(cid, sums[i]);
        total_sum += sums[i];
      }
      __syncthreads();  // for safe smem_ reuse
    }
    // update the total edge count
    total_sum = raft::blockReduce<IdxT>(total_sum, smem_);
    if (threadIdx.x == 0) { atomic_update(this->m, total_sum); }
  }

  DI void atomic_update(IdxT addrId, IdxT val)
  {
    if (sizeof(IdxT) == 4) {
      raft::myAtomicAdd<unsigned>(reinterpret_cast<unsigned*>((vd_ + addrId)), val);
    } else if (sizeof(IdxT) == 8) {
      raft::myAtomicAdd<unsigned long long>(reinterpret_cast<unsigned long long*>((vd_ + addrId)),
                                            val);
    }
  }
};  // struct eps_unexp_l2_sq_neighborhood_impl

template <typename DataT, typename IdxT, typename Policy>
__launch_bounds__(Policy::Nthreads, 2) RAFT_KERNEL  // NOLINT(readability-identifier-naming)
  epsUnexpL2SqNeighKernel(                          // NOLINT(readability-identifier-naming)
    bool* adj_,
    IdxT* vd_,
    const DataT* x,
    const DataT* y,
    IdxT m,
    IdxT n,
    IdxT k,
    DataT eps_)
{
  extern __shared__ char smem[];  // NOLINT(modernize-avoid-c-arrays)
  eps_unexp_l2_sq_neighborhood_impl<DataT, IdxT, Policy> obj(adj_, vd_, x, y, m, n, k, eps_, smem);
  obj.run();
}

template <typename DataT, typename IdxT, int VecLen>
void eps_unexp_l2_sq_neigh_impl(bool* adj_,
                                IdxT* vd_,
                                const DataT* x,
                                const DataT* y,
                                IdxT m,
                                IdxT n,
                                IdxT k,
                                DataT eps_,
                                cudaStream_t stream)
{
  using policy = typename raft::linalg::Policy4x4<DataT, VecLen>::Policy;
  dim3 grid(raft::ceildiv<int>(m, policy::Mblk), raft::ceildiv<int>(n, policy::Nblk));
  dim3 blk(policy::Nthreads);
  epsUnexpL2SqNeighKernel<DataT, IdxT, policy>
    <<<grid, blk, policy::SmemSize, stream>>>(adj_, vd_, x, y, m, n, k, eps_);
  RAFT_CUDA_TRY(cudaGetLastError());
}

/**
 * @brief Computes epsilon neighborhood for the L2-Squared distance metric
 *
 * @tparam DataT   IO and math type
 * @tparam IdxT    Index type
 *
 * @param[out] adj_    adjacency matrix [row-major] [on device] [dim = m x n]
 * @param[out] vd_     vertex degree array [on device] [len = m + 1]
 *                    `vd_ + m` stores the total number of edges in the adjacency
 *                    matrix. Pass a nullptr if you don't need this info.
 * @param[in]  x      first matrix [row-major] [on device] [dim = m x k]
 * @param[in]  y      second matrix [row-major] [on device] [dim = n x k]
 * @param[in]  eps_    defines epsilon neighborhood radius (should be passed as
 *                    squared as we compute L2-squared distance in this method)
 * @param[in]  fop    device lambda to do any other custom functions
 * @param[in]  stream cuda stream
 */
template <typename DataT, typename IdxT>
void eps_unexp_l2_sq_neighborhood(bool* adj_,
                                  IdxT* vd_,
                                  const DataT* x,
                                  const DataT* y,
                                  IdxT m,
                                  IdxT n,
                                  IdxT k,
                                  DataT eps_,
                                  cudaStream_t stream)
{
  if (vd_ != nullptr) { RAFT_CUDA_TRY(cudaMemsetAsync(vd_, 0, (m + 1) * sizeof(IdxT), stream)); }

  size_t bytes = sizeof(DataT) * k;
  if (16 % sizeof(DataT) == 0 && bytes % 16 == 0) {
    eps_unexp_l2_sq_neigh_impl<DataT, IdxT, 16 / sizeof(DataT)>(
      adj_, vd_, x, y, m, n, k, eps_, stream);
  } else if (8 % sizeof(DataT) == 0 && bytes % 8 == 0) {
    eps_unexp_l2_sq_neigh_impl<DataT, IdxT, 8 / sizeof(DataT)>(
      adj_, vd_, x, y, m, n, k, eps_, stream);
  } else {
    eps_unexp_l2_sq_neigh_impl<DataT, IdxT, 1>(adj_, vd_, x, y, m, n, k, eps_, stream);
  }
}
}  // namespace cuvs::neighbors::epsilon_neighborhood::detail
