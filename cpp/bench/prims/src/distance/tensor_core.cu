
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

template <typename DataT, typename AccT, typename OutT, typename IdxT>
__global__ void kernel(OutT* out, const half* a_array, const half* b_array, IdxT M, IdxT N, IdxT K)
{
  int lane_id = threadIdx.x % 32;

  half2 a[2] = {make_half2(a_array[lane_id*2], a_array[lane_id*2+1]),
               make_half2(a_array[64+lane_id*2], a_array[64+lane_id*2+1])};
  //int q_id = lane_id / 4;

  half2 b[1] = {make_half2 (b_array[lane_id*2], b_array[lane_id*2+1])};

  /*if (q_id == (lane_id%4)*2) b[0].x = half(1.0);
  else b[0].x = half(0.0);

  if (q_id == (lane_id%4)*2+1) b[0].y = half(1.0);
  else b[0].y = half(0.0);*/
  half2 c[2] = {make_half2(0.0f, 0.0f), make_half2(0.0f, 0.0f)};
  half2 d[2];

  mma_16x8x8 (a, b, c, d);

  OutAccessor<DataT, AccT, OutT, IdxT>::set_value(out[lane_id*2], d[0].x);
  OutAccessor<DataT, AccT, OutT, IdxT>::set_value(out[lane_id*2+1], d[0].y);
  OutAccessor<DataT, AccT, OutT, IdxT>::set_value(out[64 + lane_id*2], d[1].x);
  OutAccessor<DataT, AccT, OutT, IdxT>::set_value(out[64 + lane_id*2+1], d[1].y);

}

template <typename DataT, typename AccT, typename OutT, typename IdxT>
__global__ void print_data(const OutT* x, IdxT m, IdxT n) {
  for (IdxT i = 0; i < m; i++) {
    printf("[");
    for (IdxT j = 0; j < n; j++) {
      AccT val = OutAccessor<DataT, AccT, OutT, IdxT>::get_value(x[i*n+j]);
      printf("%f, ", val);
    }
    printf("],\n");
  }
}

template <typename DataT, typename AccT, typename OutT, typename IdxT>
void tensor_l2nn(OutT* out, const DataT* x, const DataT* y, IdxT M, IdxT N, IdxT K,
                 AccT* x_norm, AccT* y_norm) {

  DataT* z;
  cudaMalloc(&z, sizeof(DataT)*M*N);
  print_data<DataT, AccT, DataT, IdxT><<<1, 1>>>(x, M, K);
  cudaDeviceSynchronize();
  printf("-------------------------------\n");
  print_data<DataT, AccT, DataT, IdxT><<<1, 1>>>(y, K, N);
  cudaDeviceSynchronize();
  printf("-------------------------------\n");
  kernel<DataT, AccT, DataT, IdxT><<<1, 32>>>(z, x, y, M, N, K);
  cudaDeviceSynchronize();
  print_data<DataT, AccT, DataT, IdxT><<<1, 1>>>(z, M, N);
  cudaDeviceSynchronize();
}

