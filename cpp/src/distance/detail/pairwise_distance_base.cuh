/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <raft/linalg/contractions.cuh>       // raft::linalg::Contractions_NT
#include <raft/util/cuda_dev_essentials.cuh>  // ceildiv
#include <raft/util/cuda_rt_essentials.hpp>   // RAFT_CUDA_TRY

#include <cstddef>  // size_t

namespace cuvs {
namespace distance {
namespace detail {

/**
 * @brief Device class for L1, L2 and cosine distance metrics.
 * @tparam DataT          input data-type (for A and B matrices)
 * @tparam AccT           accumulation data-type
 * @tparam OutT           output data-type (for C and D matrices)
 * @tparam IdxT           index data-type
 * @tparam policy         struct which tunes the Contraction kernel
 * @tparam OpT            A distance operation, e.g., cosine_distance_op.
 * @tparam EpilogueLambda applies an elementwise function to compute final
    values. Its signature is:
    template <typename AccT, typename DataT> void epilogue_lambda
    (AccT acc[][], DataT* regxn, DataT* regyn);
 * @tparam FinalLambda the final lambda called on final distance value
 * @param[in] x input matrix
 * @param[in] y input matrix
 * @param[in] m number of rows of A and C/D
 * @param[in] n number of columns of B and C/D
 * @param[in] k number of cols of A and rows of B
 * @param[in] lda leading dimension of A
 * @param[in] ldb leading dimension of B
 * @param[in] ldd leading dimension of C/D
 * @param[in] xn row norms of input matrix A. Required for expanded L2, cosine
 * @param[in] yn row norms of input matrix B. Required for expanded L2, cosine
 * @param[output] pD output matrix
 * @param[in] smem shared mem buffer for intermediate storage of A, B, xn & yn.
 * @param distance_op the distance operation, e.g. cosine_distance_op
 * @param epilog_op the epilog operation lambda
 * @param fin_op the final gemm epilogue lambda
 * @param rowEpilog_op epilog lambda that executes when a full row has been processed
 */

template <typename DataT,
          typename OutT,
          typename IdxT,
          typename policy,
          typename OpT,
          typename EpilogueLambda,
          typename FinalLambda,
          typename RowEpilogueLambda,
          bool isRowMajor    = true,
          bool write_out     = true,
          typename BaseClass = raft::linalg::Contractions_NT<DataT, IdxT, policy, isRowMajor>>
struct pairwise_distances : public BaseClass {
  // Get accumulation type from distance_op
  using acc_t = typename OpT::acc_t;

 private:
  using p = policy;
  const OutT* xn_;
  const OutT* yn_;
  const DataT* const y_base_;
  OutT* d_output_;
  char* smem_;
  OpT distance_op_;
  EpilogueLambda epilog_op_;
  FinalLambda fin_op_;
  RowEpilogueLambda row_epilog_op_;

  const IdxT grid_stride_m_;
  const IdxT grid_stride_n_;
  const IdxT grid_offset_m_;
  const IdxT grid_offset_n_;

  acc_t acc_[p::AccRowsPerTh][p::AccColsPerTh];

 public:
  // Constructor
  DI pairwise_distances(const DataT* _x,
                        const DataT* _y,
                        IdxT _m,
                        IdxT _n,
                        IdxT _k,
                        IdxT _lda,
                        IdxT _ldb,
                        IdxT _ldd,
                        const OutT* _xn,
                        const OutT* _yn,
                        OutT* _dOutput,
                        char* _smem,
                        OpT _distance_op,
                        EpilogueLambda _epilog_op,
                        FinalLambda _fin_op,
                        RowEpilogueLambda _rowEpilog_op)
    : BaseClass(_x, _y, _m, _n, _k, _lda, _ldb, _ldd, _smem),
      xn_(_xn),
      yn_(_yn),
      y_base_(_y),
      d_output_(_dOutput),
      smem_(_smem),
      distance_op_(_distance_op),
      epilog_op_(_epilog_op),
      fin_op_(_fin_op),
      row_epilog_op_(_rowEpilog_op),
      grid_stride_m_(p::Mblk * gridDim.y),
      grid_stride_n_(p::Nblk * gridDim.x),
      grid_offset_m_(p::Mblk * blockIdx.y),
      grid_offset_n_(p::Nblk * blockIdx.x)
  {
  }

  DI void run()
  {
    for (auto tile_idx_m = grid_offset_m_; tile_idx_m < this->m; tile_idx_m += grid_stride_m_) {
      this->ldgXY(tile_idx_m, grid_offset_n_, 0);
      for (auto tile_idx_n = grid_offset_n_; tile_idx_n < this->n; tile_idx_n += grid_stride_n_) {
        // Prolog:
        reset_accumulator();
        this->stsXY();
        __syncthreads();
        this->switch_write_buffer();

        // Main loop:
        for (int kidx = p::Kblk; kidx < this->k; kidx += p::Kblk) {
          this->ldgXY(tile_idx_m, tile_idx_n, kidx);
          // Process all data in shared memory (previous k-block) and
          // accumulate in registers.
          accumulate();
          this->stsXY();
          __syncthreads();
          this->switch_write_buffer();
          this->switch_read_buffer();
        }
        accumulate();  // last iteration
        // The pre-condition for the loop over tile_idx_n is that write_buffer
        // and read_buffer point to the same buffer. This flips read_buffer back
        // so that it satisfies the pre-condition of this loop.
        this->switch_read_buffer();

        // Epilog:
        if (distance_op_.kUseNorms) {
          OutT regxn[p::AccRowsPerTh], regyn[p::AccColsPerTh];
          load_norms(tile_idx_m, tile_idx_n, regxn, regyn);
          // Overlap ldg with epilog computation
          ldg_next_grid_stride(tile_idx_m, tile_idx_n);
          // Calculate distance_op epilog.
          // Use .template to disambiguate (See:
          // https://en.cppreference.com/w/cpp/language/dependent_name)
          distance_op_.template epilog<policy>(acc_, regxn, regyn, tile_idx_n, tile_idx_m);
          // And any possible additional epilogs
          epilog_op_(acc_, regxn, regyn, tile_idx_n, tile_idx_m);
        } else {
          // Overlap ldg with epilog computation
          ldg_next_grid_stride(tile_idx_m, tile_idx_n);
          // Calculate distance_op epilog.
          // Use .template to disambiguate (See:
          // https://en.cppreference.com/w/cpp/language/dependent_name)
          distance_op_.template epilog<policy>(acc_, nullptr, nullptr, tile_idx_n, tile_idx_m);
          // And any possible additional epilogs
          epilog_op_(acc_, nullptr, nullptr, tile_idx_n, tile_idx_m);
        }
        if (write_out) { store_output(tile_idx_m, tile_idx_n); }
      }
      row_epilog_op_(tile_idx_m);
    }
  }

 private:
  DI void ldg_next_grid_stride(IdxT tile_idx_m, IdxT tile_idx_n)
  {
    // Fetch next grid stride ldg if within range
    const auto next_tile_tile_idx_n = tile_idx_n + grid_stride_n_;
    const auto next_tile_tile_idx_m = tile_idx_m + grid_stride_m_;
    if ((next_tile_tile_idx_n) < this->n) {
      this->ldgXY(tile_idx_m, next_tile_tile_idx_n, 0);
    } else if ((next_tile_tile_idx_m) < this->m) {
      this->ldgXY(next_tile_tile_idx_m, grid_offset_n_, 0);
    }
  }

  DI void reset_accumulator()
  {
    // Reset accumulator registers to zero.
#pragma unroll
    for (int i = 0; i < p::AccRowsPerTh; ++i) {
#pragma unroll
      for (int j = 0; j < p::AccColsPerTh; ++j) {
        acc_[i][j] = BaseClass::Zero();
      }
    }
  }

  DI void accumulate_reg_tile(DataT (&reg_x)[p::AccRowsPerTh][p::Veclen],
                              DataT (&reg_y)[p::AccColsPerTh][p::Veclen])
  {
#pragma unroll
    for (int v = 0; v < p::Veclen; ++v) {
#pragma unroll
      for (int i = 0; i < p::AccRowsPerTh; ++i) {
#pragma unroll
        for (int j = 0; j < p::AccColsPerTh; ++j) {
          distance_op_.core(acc_[i][j], reg_x[i][v], reg_y[j][v]);
        }
      }
    }
  }

  DI void accumulate()
  {
    // We have a separate ldsXY and accumulate_reg_tile outside the loop body,
    // so that these separated calls can be interspersed with preceding and
    // following instructions, thereby hiding latency.
    this->ldsXY(0);

    // If expensive inner loop, do not unroll loop.
    constexpr int kNumIterations = p::Kblk / p::Veclen - 1;
    constexpr int kUnrollCount   = decltype(distance_op_)::kExpensiveInnerLoop ? 1 : kNumIterations;
#pragma unroll kUnrollCount
    for (int ki = p::Veclen; ki < p::Kblk; ki += p::Veclen) {
      accumulate_reg_tile(this->regx, this->regy);
      this->ldsXY(ki);
    }

    // Accumulate last loaded tile.
    accumulate_reg_tile(this->regx, this->regy);
  }

  DI void load_norms(IdxT tile_idx_m,
                     IdxT tile_idx_n,
                     OutT (&regxn)[p::AccRowsPerTh],
                     OutT (&regyn)[p::AccColsPerTh])
  {
    OutT* sx_norm = reinterpret_cast<OutT*>((&smem_[p::SmemSize]));
    OutT* sy_norm = (&sx_norm[p::Mblk]);

    // Load x & y norms required by this threadblock in shmem buffer
    if (tile_idx_n == blockIdx.x * p::Nblk) {
      for (int i = threadIdx.x; i < p::Mblk; i += p::Nthreads) {
        auto idx   = tile_idx_m + i;
        sx_norm[i] = idx < this->m ? xn_[idx] : OutT(0);
      }
    }

    for (int i = threadIdx.x; i < p::Nblk; i += p::Nthreads) {
      auto idx   = tile_idx_n + i;
      sy_norm[i] = idx < this->n ? yn_[idx] : OutT(0);
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < p::AccRowsPerTh; ++i) {
      regxn[i] = sx_norm[i * p::AccThRows + (threadIdx.x / p::AccThCols)];
    }
#pragma unroll
    for (int i = 0; i < p::AccColsPerTh; ++i) {
      regyn[i] = sy_norm[i * p::AccThCols + (threadIdx.x % p::AccThCols)];
    }
  }

  DI void store_output(IdxT tile_idx_m, IdxT tile_idx_n)
  {
    IdxT starty = tile_idx_m + this->accrowid;
    IdxT startx = tile_idx_n + this->acccolid;

#pragma unroll
    for (int i = 0; i < p::AccRowsPerTh; ++i) {
      auto row_id = starty + i * p::AccThRows;
#pragma unroll
      for (int j = 0; j < p::AccColsPerTh; ++j) {
        auto col_id = startx + j * p::AccThCols;
        if (row_id < this->m && col_id < this->n) {
          // Promote to 64 bit index for final write, as output array can be > 2^31
          d_output_[std::size_t(row_id) * this->n + col_id] = fin_op_(acc_[i][j], acc_t(0));
        }
      }
    }
  }
};  // struct PairwiseDistances

template <typename P, typename IdxT, typename T>
auto launch_config_generator(IdxT m, IdxT n, std::size_t sMemSize, T func) -> dim3
{
  int dev_id;
  RAFT_CUDA_TRY(cudaGetDevice(&dev_id));
  int num_s_ms;
  RAFT_CUDA_TRY(cudaDeviceGetAttribute(&num_s_ms, cudaDevAttrMultiProcessorCount, dev_id));

  int num_blocks_per_sm = 0;
  dim3 grid;

  RAFT_CUDA_TRY(
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, func, P::Nthreads, sMemSize));
  std::size_t min_grid_size = num_s_ms * num_blocks_per_sm;
  std::size_t y_chunks      = raft::ceildiv<int>(m, P::Mblk);
  std::size_t x_chunks      = raft::ceildiv<int>(n, P::Nblk);
  grid.y                    = y_chunks > min_grid_size ? min_grid_size : y_chunks;
  grid.x                    = (min_grid_size - grid.y) <= 0 ? 1 : x_chunks;
  if (grid.x != 1) {
    std::size_t i = 1;
    while (grid.y * i < min_grid_size) {
      i++;
    }
    grid.x = i >= x_chunks ? x_chunks : i;
  }

  return grid;
}

};  // namespace detail
};  // namespace distance
};  // namespace cuvs
