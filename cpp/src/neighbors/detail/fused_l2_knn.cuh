/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "faiss_select/Select.cuh"
#include <raft/core/operators.hpp>
#include <raft/linalg/norm.cuh>
#include <raft/util/cuda_utils.cuh>

#include <cub/cub.cuh>

#include <limits>

#include "../../distance/detail/distance.cuh"
#include "../../distance/detail/distance_ops/l2_exp.cuh"
#include "../../distance/detail/distance_ops/l2_unexp.cuh"
#include "../../distance/detail/pairwise_distance_base.cuh"
#include "../../distance/distance.cuh"

namespace cuvs::neighbors::detail {

template <typename policy, typename PairT, typename MyWarpSelect, typename IdxT>
DI void load_all_warp_q_shmem(MyWarpSelect** heap_arr,
                              PairT* sh_dump_kv,
                              const IdxT m,
                              const unsigned int numOfNN)
{
  const int lid = raft::laneId();
#pragma unroll
  for (int i = 0; i < policy::AccRowsPerTh; ++i) {
    const auto row_id = (threadIdx.x / policy::AccThCols) + i * policy::AccThRows;
    if (row_id < m) {
#pragma unroll
      for (int j = 0; j < MyWarpSelect::kNumWarpQRegisters; ++j) {
        const int idx = j * warpSize + lid;
        if (idx < numOfNN) {
          PairT kv_pair          = sh_dump_kv[row_id * numOfNN + idx];
          heap_arr[i]->warp_v[j] = kv_pair.key;
          heap_arr[i]->warp_k[j] = kv_pair.value;
        }
      }
    }
  }
}

template <typename policy, typename PairT, typename MyWarpSelect>
DI void load_warp_q_shmem(MyWarpSelect* heap_arr,
                          PairT* sh_dump_kv,
                          const int row_id,
                          const unsigned int numOfNN)
{
  const int lid = raft::laneId();
#pragma unroll
  for (int j = 0; j < MyWarpSelect::kNumWarpQRegisters; ++j) {
    const int idx = j * warpSize + lid;
    if (idx < numOfNN) {
      PairT kv_pair       = sh_dump_kv[row_id * numOfNN + idx];
      heap_arr->warp_v[j] = kv_pair.key;
      heap_arr->warp_k[j] = kv_pair.value;
    }
  }
}

template <typename policy, typename PairT, typename MyWarpSelect, typename IdxT>
DI void store_warp_q_shmem(MyWarpSelect* heap_arr,
                           PairT* sh_dump_kv,
                           const IdxT row_id,
                           const unsigned int numOfNN)
{
  const int lid = raft::laneId();

#pragma unroll
  for (int j = 0; j < MyWarpSelect::kNumWarpQRegisters; ++j) {
    const int idx = j * warpSize + lid;
    if (idx < numOfNN) {
      PairT other_kv                     = PairT(heap_arr->warp_v[j], heap_arr->warp_k[j]);
      sh_dump_kv[row_id * numOfNN + idx] = other_kv;
    }
  }
}

template <typename policy, typename PairT, typename MyWarpSelect, typename IdxT, typename OutT>
DI void store_warp_q_gmem(MyWarpSelect** heap_arr,
                          volatile OutT* out_dists,
                          volatile IdxT* out_inds,
                          const IdxT m,
                          const unsigned int numOfNN,
                          const IdxT starty)
{
  const int lid = raft::laneId();
#pragma unroll
  for (int i = 0; i < policy::AccRowsPerTh; ++i) {
    const auto gmem_row_id = starty + i * policy::AccThRows;
    if (gmem_row_id < m) {
#pragma unroll
      for (int j = 0; j < MyWarpSelect::kNumWarpQRegisters; ++j) {
        const auto idx = j * warpSize + lid;
        if (idx < numOfNN) {
          out_dists[std::size_t(gmem_row_id) * numOfNN + idx] = heap_arr[i]->warp_k[j];
          out_inds[std::size_t(gmem_row_id) * numOfNN + idx] =
            static_cast<IdxT>(heap_arr[i]->warp_v[j]);
        }
      }
    }
  }
}

template <typename policy, typename PairT, typename MyWarpSelect, typename IdxT, typename OutT>
DI void load_prev_top_ks_gmem_warp_q(MyWarpSelect** heap_arr,
                                     volatile OutT* out_dists,
                                     volatile IdxT* out_inds,
                                     const IdxT m,
                                     const unsigned int numOfNN,
                                     const IdxT starty)
{
  const int lid = raft::laneId();
#pragma unroll
  for (int i = 0; i < policy::AccRowsPerTh; ++i) {
    const auto gmem_row_id = starty + i * policy::AccThRows;
    if (gmem_row_id < m) {
#pragma unroll
      for (int j = 0; j < MyWarpSelect::kNumWarpQRegisters; ++j) {
        const auto idx = j * warpSize + lid;
        if (idx < numOfNN) {
          heap_arr[i]->warp_k[j] = out_dists[std::size_t(gmem_row_id) * numOfNN + idx];
          heap_arr[i]->warp_v[j] =
            static_cast<uint32_t>(out_inds[std::size_t(gmem_row_id) * numOfNN + idx]);
        }
      }
      static constexpr auto kLaneWarpKTop = MyWarpSelect::kNumWarpQRegisters - 1;
      heap_arr[i]->warp_k_top = raft::shfl(heap_arr[i]->warp_k[kLaneWarpKTop], heap_arr[i]->k_lane);
    }
  }
}

template <typename PairT, int NumWarpQRegs, typename MyWarpSelect>
DI void update_sorted_warp_q(
  MyWarpSelect& heap_arr, PairT* all_warp_top_ks, int row_id, int final_num_vals, int startId = 0)
{
  constexpr uint32_t kMask = 0xffffffffu;
  const int lid            = raft::laneId();
  // calculate src_lane such that tid 0 -> 31, 1 -> 0,... 31 -> 30.
  // warp around 0 to 31 required for NN > 32
  const auto src_lane = (warpSize + (lid - 1)) & (warpSize - 1);

  for (int k = startId; k < final_num_vals; k++) {
    PairT kv_pair = all_warp_top_ks[row_id * (256) + k];
#pragma unroll
    for (int i = 0; i < NumWarpQRegs; i++) {
      unsigned active_lanes = __ballot_sync(kMask, kv_pair.value < heap_arr->warp_k[i]);
      if (active_lanes) {
        PairT temp_kv;
        temp_kv.value                = raft::shfl(heap_arr->warp_k[i], src_lane);
        temp_kv.key                  = raft::shfl(heap_arr->warp_v[i], src_lane);
        const auto first_active_lane = __ffs(active_lanes) - 1;
        if (first_active_lane == lid) {
          heap_arr->warp_k[i] = kv_pair.value;
          heap_arr->warp_v[i] = kv_pair.key;
        } else if (lid > first_active_lane) {
          heap_arr->warp_k[i] = temp_kv.value;
          heap_arr->warp_v[i] = temp_kv.key;
        }
        if (i == 0 && NumWarpQRegs > 1) {
          heap_arr->warp_k[1] = __shfl_up_sync(kMask, heap_arr->warp_k[1], 1);
          heap_arr->warp_v[1] = __shfl_up_sync(kMask, heap_arr->warp_v[1], 1);
          if (lid == 0) {
            heap_arr->warp_k[1] = temp_kv.value;
            heap_arr->warp_v[1] = temp_kv.key;
          }
          break;
        }
      }
    }
  }
}

template <typename DataT,
          typename OutT,
          typename IdxT,
          typename policy,
          typename OpT,
          typename FinalLambda,
          int NumWarpQ,
          int NumThreadQ,
          bool usePrevTopKs = false,
          bool isRowMajor   = true>
__launch_bounds__(policy::Nthreads, 2) RAFT_KERNEL  // NOLINT(readability-identifier-naming)
  fused_l2_knn_kernel(const DataT* x,
                      const DataT* y,
                      const OutT* _xn,
                      const OutT* _yn,
                      const IdxT m,
                      const IdxT n,
                      const IdxT k,
                      const IdxT lda,
                      const IdxT ldb,
                      const IdxT ldd,
                      OpT distance_op,
                      FinalLambda fin_op,
                      unsigned int numOfNN,
                      volatile int* mutexes,
                      volatile OutT* out_dists,
                      volatile IdxT* out_inds)
{
  using AccT = typename OpT::acc_t;
  extern __shared__ char smem[];

  using pair_t             = cub::KeyValuePair<uint32_t, AccT>;
  constexpr auto kIdentity = std::numeric_limits<AccT>::max();
  constexpr auto kKeyMax   = std::numeric_limits<uint32_t>::max();
  constexpr auto kDir      = false;
  using cuvs::neighbors::detail::faiss_select::comparator;
  using cuvs::neighbors::detail::faiss_select::warp_select;
  using MyWarpSelect =
    warp_select<AccT, uint32_t, kDir, comparator<AccT>, NumWarpQ, NumThreadQ, 32>;

  auto row_epilog_lambda = [m, n, &distance_op, numOfNN, out_dists, out_inds, mutexes] __device__(
                             IdxT gridStrideY) -> void {
    if (gridDim.x == 1) { return; }

    // Use ::template to disambiguate (See:
    // https://en.cppreference.com/w/cpp/language/dependent_name)
    int smem_offset    = OpT::template shared_mem_size<policy>();
    pair_t* sh_dump_kv = (pair_t*)(&smem[smem_offset]);

    const int lid     = threadIdx.x % warpSize;
    const IdxT starty = gridStrideY + (threadIdx.x / policy::AccThCols);

    //  0 -> consumer done consuming the buffer.
    // -1 -> consumer started consuming the buffer
    // -2 -> producer done filling the buffer
    //  1 -> prod acquired to fill the buffer
    if (blockIdx.x == 0) {
      auto cta_processed = 0;
      MyWarpSelect heap_arr1(kIdentity, kKeyMax, numOfNN);
      MyWarpSelect heap_arr2(kIdentity, kKeyMax, numOfNN);
      MyWarpSelect* heap_arr[] = {&heap_arr1, &heap_arr2};
      __syncwarp();

      load_all_warp_q_shmem<policy, pair_t>(heap_arr, &sh_dump_kv[0], m, numOfNN);

      while (cta_processed < gridDim.x - 1) {
        if (threadIdx.x == 0) {
          while (atomicCAS((int*)&mutexes[gridStrideY / policy::Mblk], -2, -1) != -2) {
            ;
          }
        }
        __threadfence();
        __syncthreads();

#pragma unroll
        for (int i = 0; i < policy::AccRowsPerTh; ++i) {
          const auto row_id = starty + i * policy::AccThRows;
          if (row_id < m) {
#pragma unroll
            for (int j = 0; j < MyWarpSelect::kNumWarpQRegisters; ++j) {
              pair_t other_kv;
              other_kv.value = kIdentity;
              other_kv.key   = kKeyMax;
              const auto idx = j * warpSize + lid;
              if (idx < numOfNN) {
                other_kv.value = out_dists[row_id * numOfNN + idx];
                other_kv.key   = static_cast<uint32_t>(out_inds[row_id * numOfNN + idx]);
                const auto sh_mem_row_id =
                  (threadIdx.x / policy::AccThCols) + i * policy::AccThRows;
                sh_dump_kv[sh_mem_row_id * numOfNN + idx] = other_kv;
              }
            }
          }
        }
        __threadfence();
        __syncthreads();

        if (threadIdx.x == 0) { atomicExch((int*)&mutexes[gridStrideY / policy::Mblk], 0); }
        __threadfence();

        // Perform merging of other_kv with topk's across warp.
#pragma unroll
        for (int i = 0; i < policy::AccRowsPerTh; ++i) {
          const auto row_id = starty + i * policy::AccThRows;
          if (row_id < m) {
#pragma unroll
            for (int j = 0; j < MyWarpSelect::kNumWarpQRegisters; ++j) {
              pair_t other_kv;
              other_kv.value = kIdentity;
              other_kv.key   = kKeyMax;
              const auto idx = j * warpSize + lid;
              if (idx < numOfNN) {
                const auto sh_mem_row_id =
                  (threadIdx.x / policy::AccThCols) + i * policy::AccThRows;
                other_kv = sh_dump_kv[sh_mem_row_id * numOfNN + idx];
              }
              heap_arr[i]->add(other_kv.value, other_kv.key);
            }
          }
        }
        cta_processed++;
      }
#pragma unroll
      for (int i = 0; i < policy::AccRowsPerTh; ++i) {
        const auto row_id = starty + i * policy::AccThRows;
        if (row_id < m) {
          bool need_sort = (heap_arr[i]->num_vals > 0);
          need_sort      = __any_sync(0xffffffff, need_sort);
          if (need_sort) { heap_arr[i]->reduce(); }
        }
      }
      store_warp_q_gmem<policy, pair_t>(heap_arr, out_dists, out_inds, m, numOfNN, starty);
    } else {
      if (threadIdx.x == 0) {
        while (atomicCAS((int*)&mutexes[gridStrideY / policy::Mblk], 0, 1) != 0) {
          ;
        }
      }
      __threadfence();
      __syncthreads();

#pragma unroll
      for (int i = 0; i < policy::AccRowsPerTh; ++i) {
        const auto row_id = starty + i * policy::AccThRows;
        if (row_id < m) {
          for (int idx = lid; idx < numOfNN; idx += warpSize) {
            const auto sh_mem_row_id = (threadIdx.x / policy::AccThCols) + i * policy::AccThRows;
            pair_t kv_pair           = sh_dump_kv[sh_mem_row_id * numOfNN + idx];
            out_dists[row_id * numOfNN + idx] = kv_pair.value;
            out_inds[row_id * numOfNN + idx]  = (IdxT)kv_pair.key;
          }
        }
      }
      __threadfence();
      __syncthreads();

      if (threadIdx.x == 0) { atomicExch((int*)&mutexes[gridStrideY / policy::Mblk], -2); }
      __threadfence();
    }
  };

  // epilogue operation lambda for final value calculation
  auto epilog_lambda =
    [&distance_op, numOfNN, m, n, ldd, out_dists, out_inds, kKeyMax, kIdentity] __device__(
      AccT acc[policy::AccRowsPerTh][policy::AccColsPerTh],
      OutT * regxn,
      OutT * regyn,
      IdxT gridStrideX,
      IdxT gridStrideY) -> void {
    // Use ::template to disambiguate (See:
    // https://en.cppreference.com/w/cpp/language/dependent_name)
    int smem_offset    = OpT::template shared_mem_size<policy>();
    pair_t* sh_dump_kv = (pair_t*)(&smem[smem_offset]);

    constexpr uint32_t kMask = 0xffffffffu;
    const IdxT starty        = gridStrideY + (threadIdx.x / policy::AccThCols);
    const IdxT startx        = gridStrideX + (threadIdx.x % policy::AccThCols);
    const int lid            = raft::laneId();

    MyWarpSelect heap_arr1(kIdentity, kKeyMax, numOfNN);
    MyWarpSelect heap_arr2(kIdentity, kKeyMax, numOfNN);
    MyWarpSelect* heap_arr[] = {&heap_arr1, &heap_arr2};
    if (usePrevTopKs) {
      if (gridStrideX == blockIdx.x * policy::Nblk) {
        load_prev_top_ks_gmem_warp_q<policy, pair_t>(
          heap_arr, out_dists, out_inds, m, numOfNN, starty);
      }
    }

    if (gridStrideX > blockIdx.x * policy::Nblk) {
#pragma unroll
      for (int i = 0; i < policy::AccRowsPerTh; ++i) {
        const auto row_id       = (threadIdx.x / policy::AccThCols) + i * policy::AccThRows;
        pair_t temp_kv          = sh_dump_kv[(row_id * numOfNN) + numOfNN - 1];
        heap_arr[i]->warp_k_top = temp_kv.value;
      }

      // total vals can atmost be 256, (32*8)
      int num_vals_warp_top_k[policy::AccRowsPerTh];
      int any_warp_top_ks = 0;
#pragma unroll
      for (int i = 0; i < policy::AccRowsPerTh; ++i) {
        const auto row_id      = starty + i * policy::AccThRows;
        num_vals_warp_top_k[i] = 0;
        if (row_id < m) {
#pragma unroll
          for (int j = 0; j < policy::AccColsPerTh; ++j) {
            const auto col_id = startx + j * policy::AccThCols;
            if (col_id < ldd) {
              if (acc[i][j] < heap_arr[i]->warp_k_top) { num_vals_warp_top_k[i]++; }
            }
          }
          any_warp_top_ks += num_vals_warp_top_k[i];
        }
      }
      any_warp_top_ks = __syncthreads_or(any_warp_top_ks > 0);
      if (any_warp_top_ks) {
        pair_t* all_warp_top_ks = (pair_t*)(&smem[0]);
        uint32_t need_scan_sort[policy::AccRowsPerTh];

#pragma unroll
        for (int i = 0; i < policy::AccRowsPerTh; ++i) {
          const auto gmem_row_id = starty + i * policy::AccThRows;
          need_scan_sort[i]      = 0;
          if (gmem_row_id < m) {
            int my_vals       = num_vals_warp_top_k[i];
            need_scan_sort[i] = __ballot_sync(kMask, my_vals > 0);
            if (need_scan_sort[i]) {
#pragma unroll
              for (unsigned int k = 1; k <= 16; k *= 2) {
                const unsigned int n = __shfl_up_sync(kMask, num_vals_warp_top_k[i], k);
                if (lid >= k) { num_vals_warp_top_k[i] += n; }
              }
            }
            // As each thread will know its total vals to write.
            // we only store its starting location.
            num_vals_warp_top_k[i] -= my_vals;
          }

          if (need_scan_sort[i]) {
            const auto row_id = (threadIdx.x / policy::AccThCols) + i * policy::AccThRows;
            if (gmem_row_id < m) {
              if (need_scan_sort[i] & (static_cast<uint32_t>(1) << lid)) {
#pragma unroll
                for (int j = 0; j < policy::AccColsPerTh; ++j) {
                  const auto col_id = startx + j * policy::AccThCols;
                  if (col_id < ldd) {
                    if (acc[i][j] < heap_arr[i]->warp_k_top) {
                      pair_t other_kv = {col_id, acc[i][j]};
                      all_warp_top_ks[row_id * (256) + num_vals_warp_top_k[i]] = other_kv;
                      num_vals_warp_top_k[i]++;
                    }
                  }
                }
              }
              __syncwarp();
              const int final_num_vals = raft::shfl(num_vals_warp_top_k[i], 31);
              load_warp_q_shmem<policy, pair_t>(heap_arr[i], &sh_dump_kv[0], row_id, numOfNN);
              update_sorted_warp_q<pair_t, MyWarpSelect::kNumWarpQRegisters>(
                heap_arr[i], &all_warp_top_ks[0], row_id, final_num_vals);
            }
          }
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < policy::AccRowsPerTh; ++i) {
          if (need_scan_sort[i]) {
            const auto row_id      = (threadIdx.x / policy::AccThCols) + i * policy::AccThRows;
            const auto gmem_row_id = starty + i * policy::AccThRows;
            if (gmem_row_id < m) {
              store_warp_q_shmem<policy, pair_t>(heap_arr[i], sh_dump_kv, row_id, numOfNN);
            }
          }
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < policy::AccRowsPerTh; ++i) {
        const auto gmem_row_id   = starty + i * policy::AccThRows;
        const auto sh_mem_row_id = (threadIdx.x / policy::AccThCols) + i * policy::AccThRows;
        if (gmem_row_id < m) {
#pragma unroll
          for (int j = 0; j < policy::AccColsPerTh; ++j) {
            const auto col_id = startx + j * policy::AccThCols;
            pair_t other_kv   = {kKeyMax, kIdentity};
            if (col_id < ldd) {
              other_kv.value = acc[i][j];
              other_kv.key   = col_id;
            }
            heap_arr[i]->add(other_kv.value, other_kv.key);
          }

          bool need_sort = (heap_arr[i]->num_vals > 0);
          need_sort      = __any_sync(kMask, need_sort);
          if (need_sort) { heap_arr[i]->reduce(); }
          store_warp_q_shmem<policy, pair_t>(heap_arr[i], sh_dump_kv, sh_mem_row_id, numOfNN);
        }
      }
    }

    if (((gridStrideX + policy::Nblk * gridDim.x) >= n) && gridDim.x == 1) {
      // This is last iteration of grid stride X
      load_all_warp_q_shmem<policy, pair_t>(heap_arr, &sh_dump_kv[0], m, numOfNN);
      store_warp_q_gmem<policy, pair_t>(heap_arr, out_dists, out_inds, m, numOfNN, starty);
    }
  };

  constexpr bool kWriteOut = false;
  cuvs::distance::detail::pairwise_distances<DataT,
                                             OutT,
                                             IdxT,
                                             policy,
                                             OpT,
                                             decltype(epilog_lambda),
                                             FinalLambda,
                                             decltype(row_epilog_lambda),
                                             isRowMajor,
                                             kWriteOut>
    obj(x,
        y,
        m,
        n,
        k,
        lda,
        ldb,
        ldd,
        _xn,
        _yn,
        nullptr,  // output ptr, can be null as kWriteOut == false.
        smem,
        distance_op,
        epilog_lambda,
        fin_op,
        row_epilog_lambda);
  obj.run();
}

template <typename DataT,
          typename AccT,
          typename OutT,
          typename IdxT,
          int VecLen,
          bool usePrevTopKs,
          bool isRowMajor>
auto fused_l2_unexp_knn_impl(const DataT* x,
                             const DataT* y,
                             IdxT m,
                             IdxT n,
                             IdxT k,
                             IdxT lda,
                             IdxT ldb,
                             IdxT ldd,
                             bool sqrt,
                             OutT* out_dists,
                             IdxT* out_inds,
                             IdxT numOfNN,
                             cudaStream_t stream,
                             void* workspace,
                             size_t& worksize) -> void
{
  using row_policy = typename raft::linalg::Policy2x8<AccT, 1>::Policy;
  using col_policy = typename raft::linalg::Policy4x4<AccT, VecLen>::ColPolicy;

  using KPolicy = std::conditional_t<true, row_policy, col_policy>;

  ASSERT(isRowMajor, "Only Row major inputs are allowed");

  dim3 blk{KPolicy::Nthreads};
  // Accumulation operation lambda
  using pair_t = cub::KeyValuePair<uint32_t, AccT>;

  cuvs::distance::detail::ops::l2_unexp_distance_op<DataT, AccT, IdxT> distance_op{sqrt};
  raft::identity_op fin_op{};

  if constexpr (isRowMajor) {
    constexpr auto kFusedL2UnexpKnn32RowMajor =
      fused_l2_knn_kernel<DataT,
                          OutT,     // NOLINT(readability-identifier-naming)
                          IdxT,     // NOLINT(readability-identifier-naming)
                          KPolicy,  // NOLINT(readability-identifier-naming)
                          decltype(distance_op),
                          decltype(fin_op),
                          32,
                          2,
                          usePrevTopKs,
                          isRowMajor>;
    constexpr auto kFusedL2UnexpKnn64RowMajor =
      fused_l2_knn_kernel<DataT,
                          OutT,     // NOLINT(readability-identifier-naming)
                          IdxT,     // NOLINT(readability-identifier-naming)
                          KPolicy,  // NOLINT(readability-identifier-naming)
                          decltype(distance_op),
                          decltype(fin_op),
                          64,
                          3,
                          usePrevTopKs,
                          isRowMajor>;

    auto fused_l2_unexp_knn_row_major = kFusedL2UnexpKnn32RowMajor;
    if (numOfNN <= 32) {
      fused_l2_unexp_knn_row_major = kFusedL2UnexpKnn32RowMajor;
    } else if (numOfNN <= 64) {
      fused_l2_unexp_knn_row_major = kFusedL2UnexpKnn64RowMajor;
    } else {
      ASSERT(numOfNN <= 64, "fused_l2_knn_kernel: num of nearest neighbors must be <= 64");
    }

    const auto shared_mem_size =
      distance_op.template shared_mem_size<KPolicy>() + KPolicy::Mblk * numOfNN * sizeof(pair_t);

    dim3 grid = cuvs::distance::detail::launch_config_generator<KPolicy>(
      m, n, shared_mem_size, fused_l2_unexp_knn_row_major);

    if (grid.x > 1) {
      const auto num_mutexes = raft::ceildiv<int>(m, KPolicy::Mblk);
      if (workspace == nullptr || worksize < (sizeof(int32_t) * num_mutexes)) {
        worksize = sizeof(int32_t) * num_mutexes;
        return;
      } else {
        RAFT_CUDA_TRY(cudaMemsetAsync(workspace, 0, sizeof(int32_t) * num_mutexes, stream));
      }
    }

    fused_l2_unexp_knn_row_major<<<grid, blk, shared_mem_size, stream>>>(x,
                                                                         y,
                                                                         nullptr,
                                                                         nullptr,
                                                                         m,
                                                                         n,
                                                                         k,
                                                                         lda,
                                                                         ldb,
                                                                         ldd,
                                                                         distance_op,
                                                                         fin_op,
                                                                         (uint32_t)numOfNN,
                                                                         (int*)workspace,
                                                                         out_dists,
                                                                         out_inds);
  } else {
  }

  RAFT_CUDA_TRY(cudaGetLastError());
}

template <typename DataT,
          typename AccT,
          typename OutT,
          typename IdxT,
          bool usePrevTopKs,
          bool isRowMajor>
void fused_l2_unexp_knn(IdxT m,
                        IdxT n,
                        IdxT k,
                        IdxT lda,
                        IdxT ldb,
                        IdxT ldd,
                        const DataT* x,
                        const DataT* y,
                        bool sqrt,
                        OutT* out_dists,
                        IdxT* out_inds,
                        IdxT numOfNN,
                        cudaStream_t stream,
                        void* workspace,
                        size_t& worksize)
{
  size_t bytes_a = sizeof(DataT) * lda;
  size_t bytes_b = sizeof(DataT) * ldb;
  if (16 % sizeof(DataT) == 0 && bytes_a % 16 == 0 && bytes_b % 16 == 0) {
    fused_l2_unexp_knn_impl<DataT, AccT, OutT, IdxT, 16 / sizeof(DataT), usePrevTopKs, isRowMajor>(
      x,
      y,
      m,
      n,
      k,
      lda,
      ldb,
      ldd,
      sqrt,
      out_dists,
      out_inds,
      numOfNN,
      stream,
      workspace,
      worksize);
  } else if (8 % sizeof(DataT) == 0 && bytes_a % 8 == 0 && bytes_b % 8 == 0) {
    fused_l2_unexp_knn_impl<DataT, AccT, OutT, IdxT, 8 / sizeof(DataT), usePrevTopKs, isRowMajor>(
      x,
      y,
      m,
      n,
      k,
      lda,
      ldb,
      ldd,
      sqrt,
      out_dists,
      out_inds,
      numOfNN,
      stream,
      workspace,
      worksize);
  } else {
    fused_l2_unexp_knn_impl<DataT, AccT, OutT, IdxT, 1, usePrevTopKs, isRowMajor>(x,
                                                                                  y,
                                                                                  m,
                                                                                  n,
                                                                                  k,
                                                                                  lda,
                                                                                  ldb,
                                                                                  ldd,
                                                                                  sqrt,
                                                                                  out_dists,
                                                                                  out_inds,
                                                                                  numOfNN,
                                                                                  stream,
                                                                                  workspace,
                                                                                  worksize);
  }
}

template <typename DataT,
          typename AccT,
          typename OutT,
          typename IdxT,
          int VecLen,
          bool usePrevTopKs,
          bool isRowMajor>
auto fused_l2_exp_knn_impl(const DataT* x,
                           const DataT* y,
                           const AccT* xn,
                           const AccT* yn,
                           IdxT m,
                           IdxT n,
                           IdxT k,
                           IdxT lda,
                           IdxT ldb,
                           IdxT ldd,
                           bool sqrt,
                           OutT* out_dists,
                           IdxT* out_inds,
                           IdxT numOfNN,
                           cudaStream_t stream,
                           void* workspace,
                           size_t& worksize) -> void
{
  using row_policy = typename raft::linalg::Policy2x8<AccT, 1>::Policy;
  using col_policy = typename raft::linalg::Policy4x4<AccT, VecLen>::ColPolicy;

  using KPolicy = std::conditional_t<true, row_policy, col_policy>;

  ASSERT(isRowMajor, "Only Row major inputs are allowed");

  ASSERT(!(((x != y) && (worksize < (m + n) * sizeof(AccT))) || (worksize < m * sizeof(AccT))),
         "workspace size error");
  ASSERT(workspace != nullptr, "workspace is null");

  dim3 blk{KPolicy::Nthreads};

  using pair_t = cub::KeyValuePair<uint32_t, AccT>;

  cuvs::distance::detail::ops::l2_exp_distance_op<DataT, AccT, IdxT> distance_op{sqrt};
  raft::identity_op fin_op{};

  if constexpr (isRowMajor) {
    constexpr auto kFusedL2ExpKnn32RowMajor =
      fused_l2_knn_kernel<DataT,
                          OutT,     // NOLINT(readability-identifier-naming)
                          IdxT,     // NOLINT(readability-identifier-naming)
                          KPolicy,  // NOLINT(readability-identifier-naming)
                          decltype(distance_op),
                          decltype(fin_op),
                          32,
                          2,
                          usePrevTopKs,
                          isRowMajor>;
    constexpr auto kFusedL2ExpKnn64RowMajor =
      fused_l2_knn_kernel<DataT,
                          OutT,     // NOLINT(readability-identifier-naming)
                          IdxT,     // NOLINT(readability-identifier-naming)
                          KPolicy,  // NOLINT(readability-identifier-naming)
                          decltype(distance_op),
                          decltype(fin_op),
                          64,
                          3,
                          usePrevTopKs,
                          isRowMajor>;  // NOLINT(readability-identifier-naming)

    auto fused_l2_exp_knn_row_major = kFusedL2ExpKnn32RowMajor;
    if (numOfNN <= 32) {
      fused_l2_exp_knn_row_major = kFusedL2ExpKnn32RowMajor;
    } else if (numOfNN <= 64) {
      fused_l2_exp_knn_row_major = kFusedL2ExpKnn64RowMajor;
    } else {
      ASSERT(numOfNN <= 64, "fused_l2_knn_kernel: num of nearest neighbors must be <= 64");
    }

    const auto shared_mem_size =
      distance_op.template shared_mem_size<KPolicy>() + (KPolicy::Mblk * numOfNN * sizeof(pair_t));
    dim3 grid = cuvs::distance::detail::launch_config_generator<KPolicy>(
      m, n, shared_mem_size, fused_l2_exp_knn_row_major);
    int32_t* mutexes = nullptr;
    if (grid.x > 1) {
      const auto num_mutexes   = raft::ceildiv<int>(m, KPolicy::Mblk);
      const auto norms_size    = (x != y) ? (m + n) * sizeof(AccT) : n * sizeof(AccT);
      const auto required_size = sizeof(int32_t) * num_mutexes + norms_size;
      if (worksize < required_size) {
        worksize = required_size;
        return;
      } else {
        mutexes = reinterpret_cast<int32_t*>(static_cast<char*>(workspace) + norms_size);
        RAFT_CUDA_TRY(cudaMemsetAsync(mutexes, 0, sizeof(int32_t) * num_mutexes, stream));
      }
    }

    // calculate norms if they haven't been passed in
    if (!xn) {
      AccT* xn_buf = reinterpret_cast<AccT*>(workspace);
      workspace    = xn_buf + m;
      if (isRowMajor) {
        raft::linalg::rowNorm<raft::linalg::L2Norm, true>(
          xn_buf, x, k, m, stream, raft::identity_op{});
      } else {
        raft::linalg::rowNorm<raft::linalg::L2Norm, false>(
          xn_buf, x, k, m, stream, raft::identity_op{});
      }
      xn = xn_buf;
    }
    if (!yn) {
      if (x == y) {
        yn = xn;
      } else {
        AccT* yn_buf = reinterpret_cast<AccT*>(workspace);
        if (isRowMajor) {
          raft::linalg::rowNorm<raft::linalg::L2Norm, true>(
            yn_buf, y, k, n, stream, raft::identity_op{});
        } else {
          raft::linalg::rowNorm<raft::linalg::L2Norm, false>(
            yn_buf, y, k, n, stream, raft::identity_op{});
        }
        yn = yn_buf;
      }
    }

    fused_l2_exp_knn_row_major<<<grid, blk, shared_mem_size, stream>>>(x,
                                                                       y,
                                                                       xn,
                                                                       yn,
                                                                       m,
                                                                       n,
                                                                       k,
                                                                       lda,
                                                                       ldb,
                                                                       ldd,
                                                                       distance_op,
                                                                       fin_op,
                                                                       (uint32_t)numOfNN,
                                                                       mutexes,
                                                                       out_dists,
                                                                       out_inds);
  } else {
  }

  RAFT_CUDA_TRY(cudaGetLastError());
}

template <typename DataT,
          typename AccT,
          typename OutT,
          typename IdxT,
          bool usePrevTopKs,
          bool isRowMajor>
void fused_l2_exp_knn(IdxT m,
                      IdxT n,
                      IdxT k,
                      IdxT lda,
                      IdxT ldb,
                      IdxT ldd,
                      const DataT* x,
                      const DataT* y,
                      const AccT* xn,
                      const AccT* yn,
                      bool sqrt,
                      OutT* out_dists,
                      IdxT* out_inds,
                      IdxT numOfNN,
                      cudaStream_t stream,
                      void* workspace,
                      size_t& worksize)
{
  size_t bytes_a = sizeof(DataT) * lda;
  size_t bytes_b = sizeof(DataT) * ldb;
  if (16 % sizeof(DataT) == 0 && bytes_a % 16 == 0 && bytes_b % 16 == 0) {
    fused_l2_exp_knn_impl<DataT, AccT, OutT, IdxT, 16 / sizeof(DataT), usePrevTopKs, isRowMajor>(
      x,
      y,
      xn,
      yn,
      m,
      n,
      k,
      lda,
      ldb,
      ldd,
      sqrt,
      out_dists,
      out_inds,
      numOfNN,
      stream,
      workspace,
      worksize);
  } else if (8 % sizeof(DataT) == 0 && bytes_a % 8 == 0 && bytes_b % 8 == 0) {
    fused_l2_exp_knn_impl<DataT, AccT, OutT, IdxT, 8 / sizeof(DataT), usePrevTopKs, isRowMajor>(
      x,
      y,
      xn,
      yn,
      m,
      n,
      k,
      lda,
      ldb,
      ldd,
      sqrt,
      out_dists,
      out_inds,
      numOfNN,
      stream,
      workspace,
      worksize);
  } else {
    fused_l2_exp_knn_impl<DataT, AccT, OutT, IdxT, 1, usePrevTopKs, isRowMajor>(x,
                                                                                y,
                                                                                xn,
                                                                                yn,
                                                                                m,
                                                                                n,
                                                                                k,
                                                                                lda,
                                                                                ldb,
                                                                                ldd,
                                                                                sqrt,
                                                                                out_dists,
                                                                                out_inds,
                                                                                numOfNN,
                                                                                stream,
                                                                                workspace,
                                                                                worksize);
  }
}

/**
 * Compute the k-nearest neighbors using L2 expanded/unexpanded distance.

 * @tparam value_idx
 * @tparam value_t
 * @param[out] out_inds output indices array on device (size n_query_rows * k)
 * @param[out] out_dists output dists array on device (size n_query_rows * k)
 * @param[in] index input index array on device (size n_index_rows * D)
 * @param[in] query input query array on device (size n_query_rows * D)
 * @param[in] n_index_rows number of rows in index array
 * @param[in] n_query_rows number of rows in query array
 * @param[in] k number of closest neighbors to return
 * @param[in] rowMajorIndex are the index arrays in row-major layout?
 * @param[in] rowMajorQuery are the query array in row-major layout?
 * @param[in] stream stream to order kernel launch
 */
template <typename ValueIdx,
          typename ValueT,
          bool usePrevTopKs   = false,
          typename distance_t = float>
void fused_l2_knn(size_t D,
                  ValueIdx* out_inds,
                  distance_t* out_dists,
                  const ValueT* index,
                  const ValueT* query,
                  size_t n_index_rows,
                  size_t n_query_rows,
                  int k,
                  bool rowMajorIndex,
                  bool rowMajorQuery,
                  cudaStream_t stream,
                  cuvs::distance::DistanceType metric,
                  const distance_t* index_norms = nullptr,
                  const distance_t* query_norms = nullptr)
{
  // Validate the input data
  ASSERT(k > 0, "l2Knn: k must be > 0");
  ASSERT(D > 0, "l2Knn: D must be > 0");
  ASSERT(n_index_rows > 0, "l2Knn: n_index_rows must be > 0");
  ASSERT(index, "l2Knn: index must be provided (passed null)");
  ASSERT(n_query_rows > 0, "l2Knn: n_query_rows must be > 0");
  ASSERT(query, "l2Knn: query must be provided (passed null)");
  ASSERT(out_dists, "l2Knn: out_dists must be provided (passed null)");
  ASSERT(out_inds, "l2Knn: out_inds must be provided (passed null)");
  // Currently we only support same layout for x & y inputs.
  ASSERT(rowMajorIndex == rowMajorQuery,
         "l2Knn: rowMajorIndex and rowMajorQuery should have same layout");
  // TODO(snanditale): Add support for column major layout
  ASSERT(rowMajorIndex == true, "l2Knn: only rowMajor inputs are supported for now.");

  // Even for L2 Sqrt distance case we use non-sqrt version as FAISS bfKNN only support
  // non-sqrt metric & some tests in RAFT/cuML (like Linkage) fails if we use L2 sqrt.
  constexpr bool kSqrt = false;

  size_t worksize = 0, tempWorksize = 0;
  rmm::device_uvector<char> workspace(worksize, stream);
  ValueIdx lda = D, ldb = D, ldd = n_index_rows;
  // <cuvs::distance::DistanceType::L2Expanded, float, float, float, ValueIdx>
  switch (metric) {
    case cuvs::distance::DistanceType::L2SqrtExpanded:
    case cuvs::distance::DistanceType::L2Expanded:
      tempWorksize =
        cuvs::distance::get_workspace_size<cuvs::distance::DistanceType::L2Expanded,
                                           ValueT,
                                           distance_t,
                                           distance_t,
                                           ValueIdx>(query, index, n_query_rows, n_index_rows, D);
      worksize = tempWorksize;
      workspace.resize(worksize, stream);
      fused_l2_exp_knn<ValueT, distance_t, distance_t, ValueIdx, usePrevTopKs, true>(
        n_query_rows,
        n_index_rows,
        D,
        lda,
        ldb,
        ldd,
        query,
        index,
        query_norms,
        index_norms,
        kSqrt,
        out_dists,
        out_inds,
        k,
        stream,
        workspace.data(),
        worksize);
      if (worksize > tempWorksize) {
        workspace.resize(worksize, stream);
        fused_l2_exp_knn<ValueT, distance_t, distance_t, ValueIdx, usePrevTopKs, true>(
          n_query_rows,
          n_index_rows,
          D,
          lda,
          ldb,
          ldd,
          query,
          index,
          query_norms,
          index_norms,
          kSqrt,
          out_dists,
          out_inds,
          k,
          stream,
          workspace.data(),
          worksize);
      }
      break;
    case cuvs::distance::DistanceType::L2Unexpanded:
    case cuvs::distance::DistanceType::L2SqrtUnexpanded:
      fused_l2_unexp_knn<ValueT, distance_t, distance_t, ValueIdx, usePrevTopKs, true>(
        n_query_rows,
        n_index_rows,
        D,
        lda,
        ldb,
        ldd,
        query,
        index,
        kSqrt,
        out_dists,
        out_inds,
        k,
        stream,
        workspace.data(),
        worksize);
      if (worksize) {
        workspace.resize(worksize, stream);
        fused_l2_unexp_knn<ValueT, distance_t, distance_t, ValueIdx, usePrevTopKs, true>(
          n_query_rows,
          n_index_rows,
          D,
          lda,
          ldb,
          ldd,
          query,
          index,
          kSqrt,
          out_dists,
          out_inds,
          k,
          stream,
          workspace.data(),
          worksize);
      }
      break;
    default: printf("only L2 distance metric is supported\n"); break;
  };
}

}  // namespace cuvs::neighbors::detail
