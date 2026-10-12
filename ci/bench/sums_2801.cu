/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Microbenchmarks for the k-means centroid update step:
//  - raft::linalg::reduce_rows_by_key + reduce_cols_by_key (atomics) vs.
//    sums_by_label::weighted_sums_by_label (fixed order), per call;
//  - cub::DeviceReduce sum with run_to_run vs. gpu_to_gpu determinism.
#include "cluster/detail/kmeans_centroid_sums.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/reduce_cols_by_key.cuh>
#include <raft/linalg/reduce_rows_by_key.cuh>
#include <raft/random/rng.cuh>

#include <cub/device/device_reduce.cuh>
#include <cuda/execution.determinism.h>
#include <cuda/execution.require.h>
#include <cuda/memory_resource>
#include <cuda/std/execution>
#include <cuda/std/functional>
#include <cuda/stream_ref>

#include <thrust/execution_policy.h>
#include <thrust/transform.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <tuple>
#include <vector>

template <typename F>
float time_ms(cudaStream_t stream, int reps, F f)
{
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  f();
  cudaStreamSynchronize(stream);
  cudaEventRecord(e0, stream);
  for (int r = 0; r < reps; ++r) {
    f();
  }
  cudaEventRecord(e1, stream);
  cudaEventSynchronize(e1);
  float ms = 0;
  cudaEventElapsedTime(&ms, e0, e1);
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  return ms / reps;
}

// Per-cluster weighted sums and weight sums: base (raft reduce_rows_by_key + reduce_cols_by_key,
// floating-point atomics) vs. #2801 (sums_by_label::weighted_sums_by_label, fixed order). Times
// both, checks run-to-run reproducibility (two calls each) and, when `check` is set, compares the
// cluster means (sum / weight) with a long double host reference.
//   pattern 0: X ~ U(-1, 1);  pattern 1: X ~ 100 + U(-1, 1) (large common offset)
//   labels: uniform random, or skewed (half of the rows in cluster 0)
void bench_sums(int64_t n, int64_t d, int64_t k, int reps, int pattern, bool skew, bool check)
{
  using T      = float;
  using IndexT = int64_t;
  raft::resources handle;
  auto stream = raft::resource::get_cuda_stream(handle).get();
  rmm::device_uvector<T> X(n * d, stream), W(n, stream), sums(k * d, stream), wsum(k, stream);
  rmm::device_uvector<int> labels(n, stream);
  raft::random::RngState rng(42);
  raft::random::uniform(handle, rng, X.data(), n * d, T(-1), T(1));
  if (pattern == 1) {
    thrust::transform(thrust::cuda::par.on(stream),
                      X.begin(),
                      X.end(),
                      X.begin(),
                      [] __device__(T x) { return x + T(100); });
  }
  raft::random::uniform(handle, rng, W.data(), n, T(0.5), T(1.5));
  raft::random::uniformInt(handle, rng, labels.data(), n, 0, int(k));
  if (skew) {
    thrust::transform(thrust::cuda::par.on(stream),
                      labels.begin(),
                      labels.begin() + n / 2,
                      labels.begin(),
                      [] __device__(int) { return 0; });
  }
  rmm::device_uvector<char> ws(0, stream), keys_char(n, stream);

  auto run_atomic = [&] {
    raft::linalg::reduce_rows_by_key(
      X.data(), d, labels.data(), W.data(), keys_char.data(), n, d, k, sums.data(), stream, true);
    raft::linalg::reduce_cols_by_key(
      W.data(), labels.data(), wsum.data(), IndexT(1), n, k, stream, true);
  };
  auto run_fixed = [&] {
    cuvs::cluster::kmeans::detail::sums_by_label::weighted_sums_by_label<T, IndexT>(
      handle, X.data(), W.data(), labels.data(), n, d, k, sums.data(), wsum.data(), ws, true);
  };
  auto means = [&] {
    std::vector<T> hs(k * d), hw(k);
    raft::copy(hs.data(), sums.data(), k * d, stream);
    raft::copy(hw.data(), wsum.data(), k, stream);
    cudaStreamSynchronize(stream);
    std::vector<double> m(k * d);
    for (int64_t c = 0; c < k; c++) {
      for (int64_t j = 0; j < d; j++) {
        m[c * d + j] = hw[c] > 0 ? double(hs[c * d + j]) / double(hw[c]) : 0.0;
      }
    }
    return std::make_pair(hs, m);
  };

  float t_atomic = time_ms(stream, reps, run_atomic);
  auto [a1, ma] = means();
  run_atomic();
  auto [a2, ma2] = means();
  float t_fixed = time_ms(stream, reps, run_fixed);
  auto [f1, mf] = means();
  run_fixed();
  auto [f2, mf2] = means();

  std::string acc;
  if (check) {
    auto hX = std::vector<T>(n * d);
    auto hW = std::vector<T>(n);
    auto hL = std::vector<int>(n);
    raft::copy(hX.data(), X.data(), n * d, stream);
    raft::copy(hW.data(), W.data(), n, stream);
    raft::copy(hL.data(), labels.data(), n, stream);
    cudaStreamSynchronize(stream);
    std::vector<long double> rs(k * d, 0), rw(k, 0);
    for (int64_t i = 0; i < n; i++) {
      rw[hL[i]] += hW[i];
      for (int64_t j = 0; j < d; j++) {
        rs[hL[i] * d + j] += (long double)hW[i] * hX[i * d + j];
      }
    }
    auto err = [&](const std::vector<double>& m) {
      double max_abs = 0, max_rel = 0;
      for (int64_t c = 0; c < k; c++) {
        if (rw[c] == 0) { continue; }
        for (int64_t j = 0; j < d; j++) {
          double ref = double(rs[c * d + j] / rw[c]);
          double e   = std::abs(m[c * d + j] - ref);
          max_abs    = std::max(max_abs, e);
          max_rel    = std::max(max_rel, e / std::max(std::abs(ref), 1e-30));
        }
      }
      return std::make_pair(max_abs, max_rel);
    };
    auto [ea, ra] = err(ma);
    auto [ef, rf] = err(mf);
    char buf[256];
    snprintf(buf,
             sizeof(buf),
             " mean_maxabs_vs_f64: atomic=%.3g fixed=%.3g mean_maxrel_vs_f64: atomic=%.3g "
             "fixed=%.3g",
             ea,
             ef,
             ra,
             rf);
    acc = buf;
  }
  printf(
    "MICRO centroid_sums n=%ld d=%ld k=%ld pattern=%s labels=%s base_atomic_ms=%.3f "
    "pr2801_fixed_order_ms=%.3f ratio=%.2f reproducible: atomic=%d fixed=%d%s\n",
    n,
    d,
    k,
    pattern == 1 ? "offset100" : "uniform",
    skew ? "skewed" : "uniform",
    t_atomic,
    t_fixed,
    t_fixed / t_atomic,
    int(std::memcmp(a1.data(), a2.data(), a1.size() * sizeof(T)) == 0),
    int(std::memcmp(f1.data(), f2.data(), f1.size() * sizeof(T)) == 0),
    acc.c_str());
  fflush(stdout);
}

template <typename Determinism>
void reduce(raft::resources const& handle, const float* in, int64_t n, float* out, Determinism det)
{
  auto env = cuda::std::execution::env{
    cuda::stream_ref{raft::resource::get_cuda_stream(handle)},
    cuda::execution::require(det),
    cuda::std::execution::prop{cuda::mr::get_memory_resource,
                               raft::resource::get_workspace_resource_ref(handle)}};
  RAFT_CUDA_TRY(cub::DeviceReduce::Reduce(in, out, n, cuda::std::plus<>{}, 0.0f, env));
}

void bench_reduce(int64_t n, int reps)
{
  raft::resources handle;
  auto stream = raft::resource::get_cuda_stream(handle).get();
  rmm::device_uvector<float> x(n, stream), out(1, stream);
  raft::random::RngState rng(42);
  raft::random::uniform(handle, rng, x.data(), n, -1.0f, 1.0f);
  float t_rr = time_ms(stream, reps, [&] {
    reduce(handle, x.data(), n, out.data(), cuda::execution::determinism::run_to_run);
  });
  float t_g2g = time_ms(stream, reps, [&] {
    reduce(handle, x.data(), n, out.data(), cuda::execution::determinism::gpu_to_gpu);
  });
  printf("BENCH reduce_sum n=%ld run_to_run_ms=%.4f gpu_to_gpu_ms=%.4f ratio=%.2f\n",
         n,
         t_rr,
         t_g2g,
         t_g2g / t_rr);
  fflush(stdout);
}

int main()
{
  // Timing (and reproducibility) on k-means fit shapes.
  for (auto [n, d, k] : std::vector<std::tuple<int64_t, int64_t, int64_t>>{{100000, 32, 10},
                                                                           {100000, 32, 1000},
                                                                           {100000, 128, 10},
                                                                           {100000, 128, 100},
                                                                           {100000, 128, 1000},
                                                                           {100000, 1024, 10},
                                                                           {100000, 1024, 100},
                                                                           {1000000, 32, 100},
                                                                           {1000000, 32, 1000},
                                                                           {1000000, 128, 10},
                                                                           {1000000, 128, 100},
                                                                           {1000000, 128, 1000},
                                                                           {1000000, 128, 10000},
                                                                           {1000000, 1024, 1000},
                                                                           {10000000, 32, 1000},
                                                                           {10000000, 128, 1000},
                                                                           {10000000, 2, 8}}) {
    bench_sums(n, d, k, 20, 0, false, false);
  }
  bench_sums(1000000, 128, 100, 20, 0, true, false);
  bench_sums(10000000, 32, 1000, 20, 0, true, false);
  // Accuracy of the cluster means vs. a long double reference.
  for (int pattern : {0, 1}) {
    for (bool skew : {false, true}) {
      bench_sums(1000000, 32, 10, 5, pattern, skew, true);
      bench_sums(1000000, 32, 1000, 5, pattern, skew, true);
      bench_sums(4000000, 16, 100, 5, pattern, skew, true);
    }
  }
  for (int64_t n : {1000, 100000, 1000000, 10000000, 100000000}) {
    bench_reduce(n, 50);
  }
  return 0;
}
