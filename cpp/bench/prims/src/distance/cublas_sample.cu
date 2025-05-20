/**
 * @file cublas_sample.cu
 * @brief Sample code demonstrating cuBLAS GEMM (General Matrix Multiplication)
 */

#include <cublas_v2.h>
#include <benchmark/benchmark.h>
#include "common.cuh"

template <typename DataT, typename IdxT>
struct Reducer {
    typedef raft::KeyValuePair<IdxT, DataT> KVType;

    __device__ KVType operator()(const KVType& a, const KVType& b) {
        if ((a.value < b.value) || (a.value == b.value && a.key < b.key)) {
          return a;
        } else {
          return b;
        }
    }

    __device__ DataT operator()(const DataT& a, const DataT& b) {
      return a < b? a : b;
    }
};

template <typename OutT, typename DataT, typename IdxT, int TPB>
__global__ void neo_reduce_min_kernel(OutT* out, const DataT* z, const DataT* x_norm, const DataT* y_norm, IdxT m, IdxT n) {
  IdxT tid = threadIdx.x + blockIdx.x * blockDim.x;
  IdxT row = blockIdx.x;

  DataT x_norm_row = x_norm[row];

  raft::KeyValuePair<IdxT, DataT> thread_min;

  thread_min.value = max_val<DataT>();
  thread_min.key= max_val<IdxT>();

  for (IdxT col = threadIdx.x; col < n; col+=TPB) {
      auto dist = x_norm_row + y_norm[col] - 2*z[row*n + col];
      if (dist < thread_min.value) {
        thread_min.value = dist;
        thread_min.key = col;
      }
  }
  typedef cub::BlockReduce<OutT, TPB> BlockReduceT;
  __shared__ typename BlockReduceT::TempStorage temp_storage;
  auto block_result = BlockReduceT(temp_storage).Reduce(thread_min, Reducer<DataT, IdxT>{});

  if (threadIdx.x == 0) {
    out[row] = block_result;
  }
}

template <typename OutT, typename DataT, typename IdxT, int TPB, int M_TILE, int N_TILE>
__global__ void reduce_min_kernel1(OutT* interim_out, const DataT* z, const DataT* x_norm, const DataT* y_norm, IdxT m, IdxT n, IdxT num_n_tiles) {

  // This block reduces rows from M_TILE * blockIdx.x to M_TILE * (blockIdx.x + 1)
  // and cols from N_TILE * blockIdx.y to N_TILE * (blockIdx.y + 1)
  // produces output M_TILE * 1 rows
  // in total m * (n / N_TILE) output will be produced,
  // which needs to reduced to m * 1 output in stage 2 kernel

  typedef cub::BlockReduce<OutT, TPB> BlockReduceT;
  raft::KeyValuePair<IdxT, DataT> thread_min;


  constexpr int num_elems = 16/sizeof(DataT);
  //constexpr int num_elems = 1;

  union access {
    float4 f4;
    DataT dt[num_elems];
  };

  __shared__ DataT y_norm_sh[N_TILE];

  IdxT first_row = M_TILE * blockIdx.x;
  IdxT last_row = M_TILE * (blockIdx.x + 1) - 1;

  IdxT first_col = N_TILE * blockIdx.y;
  IdxT last_col = N_TILE * (blockIdx.y + 1) - 1;

  for (IdxT col = first_col + threadIdx.x; col <= last_col; col+=blockDim.x) {
    y_norm_sh[col - first_col] = y_norm[col];
  }

  __syncthreads();

  const float4* z_as_f4 = reinterpret_cast<const float4*>(z);
  for (IdxT row = first_row; row <= last_row; row++) {
    thread_min.value = max_val<DataT>();
    thread_min.key= max_val<IdxT>();
    if (row >= m) break;
    for (IdxT col = first_col + num_elems*threadIdx.x; col <= last_col; col+=num_elems*blockDim.x) {
      access z_r;
      z_r.f4 = z_as_f4[(row * n + col) / num_elems];
      for (int c = 0; c < num_elems; c++) {
        if (c + col >= n) break;
        auto dist = x_norm[row] + y_norm_sh[c + col - first_col] - 2 * z_r.dt[c];
        //auto dist = x_norm[row] + y_norm[c + col] - 2 * z[row * n + c + col];
        /*if (row == 1 &&  ((c+col) == 2612 || (c+col) == 7754)) {
          printf("%f %f\n", z[row * n + c + col], z_r.dt[c]);
          printf("%d -> %f\n", c+col, dist);
        }*/
        if (dist < thread_min.value) {
          thread_min.value = dist;
          thread_min.key = c+col;
        }
      }
    }

    __syncthreads();
    __shared__ typename BlockReduceT::TempStorage temp_storage;
    auto block_result = BlockReduceT(temp_storage).Reduce(thread_min, Reducer<DataT, IdxT>{});
    if (threadIdx.x == 0) {
      if constexpr (std::is_floating_point<OutT>::value) {
        interim_out[row*num_n_tiles + blockIdx.y] = block_result.value;
      } else {
        interim_out[row*num_n_tiles + blockIdx.y] = block_result;
      }
    }
    __syncthreads();
  }
}

template <typename OutT, typename DataT, typename IdxT, int TPB>
__global__ void reduce_min_kernel2(OutT* out, const OutT* in, IdxT m, IdxT num_n_tiles) {

  if (blockIdx.x >= m) return;
  OutT thread_min;

  IdxT row = blockIdx.x;
  Reducer<DataT, IdxT> reducer;

  thread_min.value = max_val<DataT>();
  thread_min.key= max_val<IdxT>();

  for (IdxT blk_col = threadIdx.x; blk_col < num_n_tiles; blk_col += blockDim.x) {
    auto dist = in[row * num_n_tiles + blk_col];
    thread_min = reducer(thread_min, dist);
  }

  typedef cub::BlockReduce<OutT, TPB> BlockReduceT;
  __shared__ typename BlockReduceT::TempStorage temp_storage;
  auto block_result = BlockReduceT(temp_storage).Reduce(thread_min, Reducer<DataT, IdxT>{});
  if (threadIdx.x == 0) {
    //if (row == 1) printf("## %d, %f, %d\n", row, block_result.value, block_result.key);
    out[row] = block_result;
  }
}


template <typename OutT, typename DataT, typename IdxT>
void reduce_min(OutT* out, const DataT* z, const DataT* x_norm, const DataT* y_norm, IdxT m, IdxT n, OutT* interim_out, cudaStream_t stream) {
  const int TPB = 128;
  const IdxT M_TILE = 128;
  const IdxT N_TILE = 2*TPB;

  IdxT num_m_tiles = (m + M_TILE - 1) / M_TILE;
  IdxT num_n_tiles = (n + N_TILE - 1) / N_TILE;
  dim3 blocks(num_m_tiles, num_n_tiles);
  //printf("Launching %d %d tiles\n", num_m_tiles, num_n_tiles);
  reduce_min_kernel1<OutT, DataT, IdxT, TPB, M_TILE, N_TILE><<<blocks, TPB, 0, stream>>>
    (interim_out, z, x_norm, y_norm, m, n, num_n_tiles);

  reduce_min_kernel2<OutT, DataT, IdxT, TPB><<<m, TPB, 0, stream>>>
    (out, interim_out, m, num_n_tiles);
}

template <typename OutT, typename DataT, typename IdxT>
void neo_reduce_min(OutT* out, const DataT* z, const DataT* x_norm, const DataT* y_norm, IdxT m, IdxT n, cudaStream_t stream) {
  const int TPB = 128;

  int blocks = m;
  neo_reduce_min_kernel<OutT, DataT, IdxT, TPB><<<blocks, TPB, 0, stream>>>(out, z, x_norm, y_norm, m, n);
  CHECK_CUDA(cudaDeviceSynchronize());
}
template <typename OutT, typename DataT, typename IdxT, bool GEMM_ONLY>
void cublas_l2nn(OutT* out, const DataT* x, const DataT* y, IdxT M, IdxT N, IdxT K,
                 DataT* x_norm, DataT* y_norm,
                 DataT* z, size_t ws_size, OutT* interim_out, cublasHandle_t& handle, cudaStream_t stream) {
  // Set up scaling factors
  const DataT alpha = 1.0f;
  const DataT beta = 0.0f;

  // Enable TF32 mode
  CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));

  if constexpr (std::is_same_v<DataT, float>) {
    CHECK_CUBLAS(cublasSgemm(
      handle,
      CUBLAS_OP_T, CUBLAS_OP_N,
      N, M, K,                   // Dimensions (swapped due to row/col-major difference)
      &alpha,                    // alpha
      y, K,                      // B and its leading dimension
      x, K,                      // A and its leading dimension
      &beta,                     // beta
      z, N                       // C and its leading dimension
    ));
  } else if constexpr (std::is_same_v<DataT, double>) {
    CHECK_CUBLAS(cublasDgemm(
      handle,
      CUBLAS_OP_T, CUBLAS_OP_N,
      N, M, K,                   // Dimensions (swapped due to row/col-major difference)
      &alpha,                    // alpha
      y, K,                      // B and its leading dimension
      x, K,                      // A and its leading dimension
      &beta,                     // beta
      z, N                       // C and its leading dimension
    ));
  }

  if constexpr(!GEMM_ONLY) {
    //reduce_min<OutT, DataT, IdxT>(out, z, x_norm, y_norm, M, N, interim_out, stream);
    neo_reduce_min<OutT, DataT, IdxT>(out, z, x_norm, y_norm, M, N, stream);
  }
}

