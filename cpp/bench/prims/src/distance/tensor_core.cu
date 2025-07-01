
#include "common.cuh"
// MMA stuff
__device__ inline void mma_16x8x8(const half2 *a, const half2 *b, const half2 *c, half2 *d)
{
  const uint *ua = reinterpret_cast<const uint *>(a);
  const uint *ub = reinterpret_cast<const uint *>(b);
  const uint *uc = reinterpret_cast<const uint *>(c);
  uint *ud = reinterpret_cast<uint *>(d);
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%5,%6};\n"
                : "=r"(ud[0]), "=r"(ud[1])
                : "r"(ua[0]), "r"(ua[1]), "r"(ub[0]), "r"(uc[0]), "r"(uc[1]));
}

template <typename IdxT>
__global__ void ref_kernel(half* out, const half* a_array, const half* b_array, IdxT M, IdxT N, IdxT K)
{

  for (int x = 0; x < M; x++) {
    for (int y = 0; y < N; y++) {
      out[x * N + y] = 0.0;
      for (int z = 0; z < K; z++) {
        out[x * N + y] += (a_array[x*K + z] * b_array[y*K + z]);
      }
    }
  }
}

template <typename IdxT>
__global__ void kernel(half* out, const half* a_mat, const half* b_mat, IdxT M, IdxT N, IdxT K)
{
  // input for index calculation
  /*int gt_id = threadIdx.x + blockIdx.x * blockDim.x;
  int lt_id = threadIdx.x;
  int b_id = blockIdx.x;
  int gwarp_id = gt_id / 32;
  int lwarp_id = lt_id / 32;*/
  int lane_id = threadIdx.x % 32;

  constexpr int t_m = 16;
  constexpr int t_n = 8;
  constexpr int t_k = 8;


  // x is split as x1 * t_m + x2
  // y is split as y1 * t_n + y2
  // z is split as z1 * t_k + z2

  half2 a[2];
  half2 b[1];
  half2 c[2];

  // Assuming blockDim.x is divisible by 32
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  int warp_id = tidx / 32;
  int n_warps = (blockDim.x / 32) * gridDim.x;

  for (int x1 = blockIdx.y; x1 < M / t_m; x1+=gridDim.y) {
    for (int y1 = warp_id; y1 < N / t_n; y1+=n_warps) {

      c[0] = make_half2(0.0f, 0.0f);
      c[1] = make_half2(0.0f, 0.0f);

      int x2 = lane_id / 4;
      int z2 = 2*(lane_id % 4);
      int y2 = x2;

      int a_offset = (x1*t_m + x2) * K + (z2);
      int b_offset = (y1*t_n + y2) * K + (z2);

      for (int z1 = 0; z1 < K / t_k; z1++) {
        // load a tile
        a[0] = make_half2(a_mat[a_offset + (z1 * t_k)], a_mat[a_offset + (z1 * t_k) + 1]);
        a[1] = make_half2(a_mat[a_offset + (z1 * t_k) + 8*K], a_mat[a_offset + (z1 * t_k) + 8*K + 1]);
        // load b tile
        b[0] = make_half2(b_mat[b_offset + (z1 * t_k)], b_mat[b_offset + (z1 * t_k) + 1]);

        // compute c = a * b + c
        mma_16x8x8 (a, b, c, c);
      }
      x2 = lane_id / 4;
      y2 = 2*(lane_id % 4);
      int out_offset = (x1 * t_m + x2) * N + (y1 * t_n + y2);
      out[out_offset]             = c[0].x;
      out[out_offset + 1]         = c[0].y;
      out[out_offset + 8 * N]     = c[1].x;
      out[out_offset + 8 * N + 1] = c[1].y;
    }
  }


  /*half2 a[2] = {make_half2(a_mat[lane_id*2], a_mat[lane_id*2+1]),
               make_half2(a_mat[64+lane_id*2], a_mat[64+lane_id*2+1])};

  half2 b[1] = {make_half2 (b_mat[lane_id*2], b_mat[lane_id*2+1])};

  half2 c[2] = {make_half2(0.0f, 0.0f), make_half2(0.0f, 0.0f)};
  half2 d[2];

  mma_16x8x8 (a, b, c, d);

  OutAccessor<DataT, AccT, OutT, IdxT>::set_value(out[lane_id*2], d[0].x);
  OutAccessor<DataT, AccT, OutT, IdxT>::set_value(out[lane_id*2+1], d[0].y);
  OutAccessor<DataT, AccT, OutT, IdxT>::set_value(out[64 + lane_id*2], d[1].x);
  OutAccessor<DataT, AccT, OutT, IdxT>::set_value(out[64 + lane_id*2+1], d[1].y);*/

}
template <typename DataT>
__global__ void comapre_kernel(const DataT* ref, const DataT* computed, int size, double threshold = 1e-2) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (; idx < size; idx += stride) {
    double ref_val = double(ref[idx]);
    double comp_val = double(computed[idx]);
    double diff = fabs(ref_val - comp_val);

    if (diff > threshold) {
      printf("Mismatch at index %d: ref=%f, computed=%f, diff=%f\n",
             idx, ref_val, comp_val, diff);
      return;
    }
  }
}

template <typename DataT, typename AccT, typename OutT, typename IdxT>
__global__ void print_data(const OutT* x, IdxT m, IdxT n) {
  for (IdxT i = 0; i < m; i++) {
    printf("[");
    for (IdxT j = 0; j < n; j++) {
      AccT val = OutAccessor<DataT, AccT, OutT, IdxT>::get_value(x[i*n+j]);
      printf("%f, ", double(float(val)));
    }
    printf("],\n");
  }
}

template <typename DataT, typename AccT, typename OutT, typename IdxT>
void tensor_l2nn(OutT* out, const DataT* x, const DataT* y, IdxT M, IdxT N, IdxT K,
                 AccT* x_norm, AccT* y_norm, cudaStream_t stream) {

  //DataT* z;
  //cudaMalloc(&z, sizeof(DataT)*M*N);

  dim3 blocks(N/64, M/16, 1);
  kernel<IdxT><<<blocks, 128, 0, stream>>>(out, x, y, M, N, K);
  //CHECK_CUDA(cudaDeviceSynchronize());
  //print_data<half, half, half, IdxT><<<1, 1>>>(z, M, N);
  //CHECK_CUDA(cudaDeviceSynchronize());

  //DataT* z_ref;
  //cudaMalloc(&z_ref, sizeof(DataT)*M*N);
  //ref_kernel<IdxT><<<1, 1>>>(z_ref, x, y, M, N, K);
  //CHECK_CUDA(cudaDeviceSynchronize());
  //printf("---\n");
  //print_data<half, half, half, IdxT><<<1, 1>>>(z_ref, M, N);
  //CHECK_CUDA(cudaDeviceSynchronize());

  //comapre_kernel<<<1, 1>>>(z_ref, z, M*N, 1e-1);
  //CHECK_CUDA(cudaFree(z));
  //CHECK_CUDA(cudaFree(z_ref));
}

