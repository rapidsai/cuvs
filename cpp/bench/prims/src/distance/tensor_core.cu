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

__global__ void kernel(half2 *out)
{
  float fx = 2.0f * (threadIdx.x % 8);
  half2 a[2] = {make_half2 (fx, fx + 1.0f),
                make_half2 (fx + 64.0f, fx + 65.0f)};

  half2 b = make_half2 (fx, fx + 1.0f);
  half2 c[2] = {make_half2(0.0f, 0.0f), make_half2(0.0f, 0.0f)};
  half2 d[2];

  for (int i = 0; i < 100; i++) {
    mma_16x8x8 (a, &b, c, d);
  }
  //printf("%d %f %f\n", threadIdx.x, double(d[0].x), double(d[0].x));
  
  out[threadIdx.x % 128] = d[0];
  out[(threadIdx.x + 32)%128] = d[1];

}


