/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// End-to-end timing of k-means fit, balanced k-means and IVF / CAGRA builds through the public API.
// The same binary is run against several libcuvs builds (selected with LD_LIBRARY_PATH); one
// invocation runs one workload for one libcuvs build, `warmup` untimed + `reps` timed repetitions.
//
// Output: one line per timed repetition,
//   RESULT <suite> <config key=value...> | variant=<v> ms=<t> <metrics key=value...>
#ifndef RAFT_SYSTEM_LITTLE_ENDIAN
#define RAFT_SYSTEM_LITTLE_ENDIAN 1  // as set by the cuvs CMake targets
#endif

#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <cuvs/neighbors/ivf_rabitq.hpp>
#include <cuvs/neighbors/ivf_sq.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/make_blobs.cuh>
#include <raft/random/rng.cuh>

#include <thrust/execution_policy.h>
#include <thrust/transform.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;

static std::string g_variant = "?";

struct Args {
  std::map<std::string, std::string> kv;
  std::string get(const std::string& k, const std::string& d) const
  {
    auto it = kv.find(k);
    return it == kv.end() ? d : it->second;
  }
  int64_t geti(const std::string& k, int64_t d) const
  {
    auto it = kv.find(k);
    return it == kv.end() ? d : std::stoll(it->second);
  }
  double getf(const std::string& k, double d) const
  {
    auto it = kv.find(k);
    return it == kv.end() ? d : std::stod(it->second);
  }
};

std::vector<int64_t> parse_list(const std::string& s)
{
  std::vector<int64_t> out;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    if (!tok.empty()) { out.push_back(std::stoll(tok)); }
  }
  return out;
}

std::vector<std::string> split(const std::string& s, char c)
{
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, c)) {
    out.push_back(tok);
  }
  return out;
}

template <typename T>
std::vector<T> to_host(raft::resources const& res, const T* p, size_t n)
{
  std::vector<T> h(n);
  raft::copy(h.data(), p, n, raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);
  return h;
}

template <typename T>
uint64_t fnv(const std::vector<T>& v, uint64_t h = 1469598103934665603ULL)
{
  auto p = reinterpret_cast<const unsigned char*>(v.data());
  for (size_t i = 0; i < v.size() * sizeof(T); i++) {
    h = (h ^ p[i]) * 1099511628211ULL;
  }
  return h;
}

bool fits(size_t bytes, const std::string& what)
{
  size_t free_b = 0, total_b = 0;
  cudaMemGetInfo(&free_b, &total_b);
  if (bytes > free_b * 0.85) {
    printf("SKIP %s needs ~%.1f GB, free %.1f GB\n", what.c_str(), bytes / 1e9, free_b / 1e9);
    fflush(stdout);
    return false;
  }
  return true;
}

template <typename F>
double time_ms(raft::resources const& res, F&& f)
{
  raft::resource::sync_stream(res);
  cudaDeviceSynchronize();
  auto t0 = clk::now();
  f();
  raft::resource::sync_stream(res);
  cudaDeviceSynchronize();
  return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

// Prints the libcuvs.so this process loaded and exits if it is not $BENCH_EXPECT_LIB.
void check_loaded_libcuvs()
{
  std::ifstream maps("/proc/self/maps");
  std::string line, path;
  while (std::getline(maps, line)) {
    if (line.find("libcuvs.so") != std::string::npos) {
      path = line.substr(line.find('/'));
      break;
    }
  }
  printf("LIB variant=%s %s\n", g_variant.c_str(), path.c_str());
  const char* expect = std::getenv("BENCH_EXPECT_LIB");
  if (expect != nullptr && path != expect) {
    printf("ERROR variant=%s loaded %s, expected %s\n", g_variant.c_str(), path.c_str(), expect);
    fflush(stdout);
    std::exit(3);
  }
}

// Dataset: Gaussian blobs, deterministic for a given (n, d, seed).
template <typename T>
void make_data(raft::resources const& res,
               T* out,
               int64_t n,
               int64_t d,
               int64_t n_blobs,
               T stddev,
               uint64_t seed)
{
  auto labels = raft::make_device_vector<int64_t, int64_t>(res, n);
  raft::random::make_blobs<T, int64_t>(out,
                                       labels.data_handle(),
                                       n,
                                       d,
                                       n_blobs,
                                       raft::resource::get_cuda_stream(res).get(),
                                       true,
                                       nullptr,
                                       nullptr,
                                       stddev,
                                       true,
                                       T(-10),
                                       T(10),
                                       seed);
}

// ------------------------------------------------------------------------------------------------
// k-means fit (regular). Array init with the first k rows, so every build starts from the same
// centroids; tol is tiny so that the iteration count is (nearly always) max_iter.
// ------------------------------------------------------------------------------------------------
template <typename T>
void suite_kmeans(const Args& a)
{
  const int64_t n     = a.geti("n", 1000000);
  const int64_t d     = a.geti("d", 128);
  const auto ks       = parse_list(a.get("k", "100"));
  const bool weights  = a.geti("w", 0) != 0;
  const int64_t batch = a.geti("batch", 0);  // > 0: host data, batched fit
  const std::string init = a.get("init", "array");
  const int reps      = a.geti("reps", 2);
  const int warmup    = a.geti("warmup", 1);
  const bool ref64    = a.geti("ref64", 0) != 0;  // also fit in double from the same init
  const char* dtype   = sizeof(T) == 4 ? "f32" : "f64";

  raft::resources res;
  auto stream = raft::resource::get_cuda_stream(res);
  size_t need = (batch > 0 ? 2 * batch : n) * d * sizeof(T) + n * 24 + (size_t(3) << 30);
  if (ref64 && sizeof(T) == 4) { need += n * d * 8; }
  if (!fits(need, "kmeans n=" + std::to_string(n) + " d=" + std::to_string(d))) { return; }

  auto X = raft::make_device_matrix<T, int64_t>(res, n, d);
  // "uniform" (default): no cluster structure, so the cost keeps decreasing and every fit runs
  // max_iter iterations. "blobs": 1000 Gaussian blobs.
  const std::string data = a.get("data", "uniform");
  if (data == "blobs") {
    make_data<T>(res, X.data_handle(), n, d, 1000, T(2), 1234ULL);
  } else {
    raft::random::RngState rng(1234);
    raft::random::uniform(res, rng, X.data_handle(), n * d, T(-1), T(1));
  }
  const uint64_t xhash =
    fnv(to_host(res, X.data_handle(), std::min<int64_t>(n * d, int64_t(1) << 22)));
  auto W = raft::make_device_vector<T, int64_t>(res, n);
  if (weights) {
    raft::random::RngState rng(99);
    raft::random::uniform(res, rng, W.data_handle(), n, T(0.5), T(1.5));
  }
  std::vector<T> hX, hW;
  if (batch > 0) {
    hX = to_host(res, X.data_handle(), n * d);
    if (weights) { hW = to_host(res, W.data_handle(), n); }
  }
  std::optional<raft::device_matrix<double, int64_t>> X64;
  std::optional<raft::device_vector<double, int64_t>> W64;
  if (ref64 && sizeof(T) == 4) {
    X64.emplace(raft::make_device_matrix<double, int64_t>(res, n, d));
    thrust::transform(thrust::cuda::par.on(stream.get()),
                      X.data_handle(),
                      X.data_handle() + n * d,
                      X64->data_handle(),
                      [] __device__(T x) { return double(x); });
    if (weights) {
      W64.emplace(raft::make_device_vector<double, int64_t>(res, n));
      thrust::transform(thrust::cuda::par.on(stream.get()),
                        W.data_handle(),
                        W.data_handle() + n,
                        W64->data_handle(),
                        [] __device__(T x) { return double(x); });
    }
  }

  for (auto k : ks) {
    const double flops_iter = 2.0 * n * k * d;
    if (flops_iter > a.getf("max_flops_iter", 3e13)) {
      printf("SKIP kmeans n=%ld d=%ld k=%ld (too slow)\n", n, d, k);
      continue;
    }
    int max_iter = a.geti("iters", flops_iter > 3e12 ? 3 : 10);
    cuvs::cluster::kmeans::params p;
    p.n_clusters            = k;
    p.max_iter              = max_iter;
    p.tol                   = 1e-30;
    p.init                  = init == "kpp" ? cuvs::cluster::kmeans::params::KMeansPlusPlus
                                            : cuvs::cluster::kmeans::params::Array;
    p.rng_state.seed        = 42;
    p.device_buffer_samples = batch;
    auto C                  = raft::make_device_matrix<T, int64_t>(res, k, d);
    for (int r = 0; r < warmup + reps; r++) {
      raft::copy(C.data_handle(), X.data_handle(), k * d, stream);
      T inertia      = 0;
      int64_t n_iter = 0;
      double ms      = time_ms(res, [&] {
        if (batch > 0) {
          std::optional<raft::host_vector_view<const T, int64_t>> sw;
          if (weights) { sw = raft::make_host_vector_view<const T, int64_t>(hW.data(), n); }
          cuvs::cluster::kmeans::fit(res,
                                     p,
                                     raft::make_host_matrix_view<const T, int64_t>(hX.data(), n, d),
                                     sw,
                                     C.view(),
                                     raft::make_host_scalar_view(&inertia),
                                     raft::make_host_scalar_view(&n_iter));
        } else {
          std::optional<raft::device_vector_view<const T, int64_t>> sw;
          if (weights) { sw = raft::make_const_mdspan(W.view()); }
          cuvs::cluster::kmeans::fit(res,
                                     p,
                                     raft::make_const_mdspan(X.view()),
                                     sw,
                                     C.view(),
                                     raft::make_host_scalar_view(&inertia),
                                     raft::make_host_scalar_view(&n_iter));
        }
      });
      if (r < warmup) { continue; }
      auto hC = to_host(res, C.data_handle(), C.size());
      std::string ref_str;
      if (X64.has_value() && r == warmup) {
        // Reference: the same fit (same init, same iteration count) in double precision.
        auto C64 = raft::make_device_matrix<double, int64_t>(res, k, d);
        thrust::transform(thrust::cuda::par.on(stream.get()),
                          X.data_handle(),
                          X.data_handle() + k * d,
                          C64.data_handle(),
                          [] __device__(T x) { return double(x); });
        double inertia64 = 0;
        int64_t n_iter64 = 0;
        auto p64         = p;
        p64.max_iter     = n_iter;
        std::optional<raft::device_vector_view<const double, int64_t>> sw64;
        if (weights) { sw64 = raft::make_const_mdspan(W64->view()); }
        cuvs::cluster::kmeans::fit(res,
                                   p64,
                                   raft::make_const_mdspan(X64->view()),
                                   sw64,
                                   C64.view(),
                                   raft::make_host_scalar_view(&inertia64),
                                   raft::make_host_scalar_view(&n_iter64));
        auto hC64      = to_host(res, C64.data_handle(), C64.size());
        double max_abs = 0, scale = 0;
        for (size_t i = 0; i < hC.size(); i++) {
          max_abs = std::max(max_abs, std::abs(double(hC[i]) - hC64[i]));
          scale   = std::max(scale, std::abs(hC64[i]));
        }
        char buf[256];
        snprintf(buf,
                 sizeof(buf),
                 " inertia64=%.9g iters64=%ld center_maxabs_vs64=%.3g center_maxrel_vs64=%.3g",
                 inertia64,
                 n_iter64,
                 max_abs,
                 max_abs / scale);
        ref_str = buf;
      }
      printf(
        "RESULT kmeans_fit dtype=%s data=%s n=%ld d=%ld k=%ld w=%d batch=%ld init=%s max_iter=%d | "
        "variant=%s ms=%.3f iters=%ld per_iter_ms=%.4f inertia=%.9g hash=%016lx xhash=%016lx%s\n",
        dtype,
        data.c_str(),
        n,
        d,
        k,
        int(weights),
        batch,
        init.c_str(),
        max_iter,
        g_variant.c_str(),
        ms,
        n_iter,
        ms / std::max<int64_t>(n_iter, 1),
        double(inertia),
        fnv(hC),
        xhash,
        ref_str.c_str());
      fflush(stdout);
    }
  }
}

// ------------------------------------------------------------------------------------------------
// Balanced k-means fit and predict.
// ------------------------------------------------------------------------------------------------
template <typename T>
double host_inertia(const std::vector<T>& X,
                    const std::vector<float>& C,
                    const std::vector<uint32_t>& labels,
                    int64_t n,
                    int64_t d)
{
  double s = 0;
  for (int64_t i = 0; i < n; i++) {
    const float* c = C.data() + size_t(labels[i]) * d;
    const T* x     = X.data() + size_t(i) * d;
    double r       = 0;
    for (int64_t j = 0; j < d; j++) {
      double diff = double(x[j]) - double(c[j]);
      r += diff * diff;
    }
    s += r;
  }
  return s;
}

template <typename T>
void suite_balanced(const Args& a)
{
  const int64_t n       = a.geti("n", 1000000);
  const int64_t d       = a.geti("d", 128);
  const auto ks         = parse_list(a.get("k", "1024"));
  const int calls       = a.geti("calls", 1);
  const bool random     = a.get("donor", "size") == "random";
  const int n_iters     = a.geti("iters", 20);
  const int reps        = a.geti("reps", 2);
  const int warmup      = a.geti("warmup", 1);
  const bool do_predict = a.geti("predict", 0) != 0;
  const char* dtype     = std::is_same_v<T, float> ? "f32" : "i8";

  raft::resources res;
  auto stream = raft::resource::get_cuda_stream(res);
  if (!fits(n * d * 5 + n * 16 + (size_t(2) << 30), "balanced n=" + std::to_string(n))) { return; }
  auto Xf = raft::make_device_matrix<float, int64_t>(res, n, d);
  make_data<float>(res, Xf.data_handle(), n, d, 1000, 2.0f, 4321ULL);
  auto X = raft::make_device_matrix<T, int64_t>(res, n, d);
  if constexpr (std::is_same_v<T, float>) {
    raft::copy(X.data_handle(), Xf.data_handle(), n * d, stream);
  } else {
    thrust::transform(thrust::cuda::par.on(stream.get()),
                      Xf.data_handle(),
                      Xf.data_handle() + n * d,
                      X.data_handle(),
                      [] __device__(float x) {
                        return T(fmaxf(-127.f, fminf(127.f, rintf(x * 6.f))));
                      });
  }
  // distinct=1: every call gets its own dataset (like the per-subspace fits of IVF-PQ), instead
  // of refitting the same data `calls` times.
  const bool distinct = a.geti("distinct", 0) != 0 && calls > 1;
  std::optional<raft::device_matrix<T, int64_t>> XS;
  if (distinct) {
    XS.emplace(raft::make_device_matrix<T, int64_t>(res, calls * n, d));
    for (int i = 0; i < calls; i++) {
      make_data<float>(res, Xf.data_handle(), n, d, 1000, 2.0f, 4321ULL + i);
      thrust::transform(thrust::cuda::par.on(stream.get()),
                        Xf.data_handle(),
                        Xf.data_handle() + n * d,
                        XS->data_handle() + i * n * d,
                        [] __device__(float x) {
                          return std::is_same_v<T, float>
                                   ? T(x)
                                   : T(fmaxf(-127.f, fminf(127.f, rintf(x * 6.f))));
                        });
    }
  }
  std::vector<T> hX;
  if (!distinct && n * d <= int64_t(1) << 28) { hX = to_host(res, X.data_handle(), n * d); }

  cuvs::cluster::kmeans::balanced_params params;
  params.n_iters = n_iters;
  if (random) {
    params.donor_selection         = cuvs::cluster::kmeans::balanced_donor_selection::Random;
    params.balance_lower_tolerance = 0.25f;
    params.balance_upper_tolerance = 4.0f;
    params.centroid_offset         = 0.05f;
  }
  for (auto k : ks) {
    auto C      = raft::make_device_matrix<float, int64_t>(res, k, d);
    auto labels = raft::make_device_vector<uint32_t, int64_t>(res, n);
    for (int r = 0; r < warmup + reps; r++) {
      double ms = time_ms(res, [&] {
        for (int i = 0; i < calls; i++) {
          auto Xi = distinct ? raft::make_device_matrix_view<const T, int64_t>(
                                 XS->data_handle() + i * n * d, n, d)
                             : raft::make_const_mdspan(X.view());
          cuvs::cluster::kmeans::fit(res, params, Xi, C.view());
        }
      });
      if (r < warmup) { continue; }
      auto hC = to_host(res, C.data_handle(), C.size());
      cuvs::cluster::kmeans::predict(res,
                                     params,
                                     raft::make_const_mdspan(X.view()),
                                     raft::make_const_mdspan(C.view()),
                                     labels.view());
      auto hL       = to_host(res, labels.data_handle(), n);
      double inert  = hX.empty() ? -1 : host_inertia(hX, hC, hL, n, d);
      int64_t min_s = n, max_s = 0;
      {
        std::vector<int64_t> sizes(k, 0);
        for (auto l : hL) {
          sizes[l]++;
        }
        for (auto s : sizes) {
          min_s = std::min(min_s, s);
          max_s = std::max(max_s, s);
        }
      }
      printf(
        "RESULT balanced_fit dtype=%s n=%ld d=%ld k=%ld donor=%s n_iters=%d calls=%d%s | "
        "variant=%s ms=%.3f inertia64=%.9g min_size=%ld max_size=%ld hash=%016lx\n",
        dtype,
        n,
        d,
        k,
        random ? "random" : "size_sorted",
        n_iters,
        calls,
        distinct ? " distinct=1" : "",
        g_variant.c_str(),
        ms,
        inert,
        min_s,
        max_s,
        fnv(hC));
      fflush(stdout);
    }
    if (do_predict) {
      for (int r = 0; r < warmup + reps; r++) {
        double ms = time_ms(res, [&] {
          cuvs::cluster::kmeans::predict(res,
                                         params,
                                         raft::make_const_mdspan(X.view()),
                                         raft::make_const_mdspan(C.view()),
                                         labels.view());
        });
        if (r < warmup) { continue; }
        printf("RESULT balanced_predict dtype=%s n=%ld d=%ld k=%ld | variant=%s ms=%.3f\n",
               dtype,
               n,
               d,
               k,
               g_variant.c_str(),
               ms);
        fflush(stdout);
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------
// Index builds (+ recall@10 of each build, + one search timing).
// ------------------------------------------------------------------------------------------------
double recall(const std::vector<int64_t>& found, const std::vector<int64_t>& gt, int64_t nq, int k)
{
  int64_t hits = 0;
  for (int64_t q = 0; q < nq; q++) {
    for (int i = 0; i < k; i++) {
      for (int j = 0; j < k; j++) {
        if (found[q * k + i] == gt[q * k + j]) {
          hits++;
          break;
        }
      }
    }
  }
  return double(hits) / double(nq * k);
}

void suite_ivf(const Args& a)
{
  const int64_t n     = a.geti("n", 1000000);
  const int64_t d     = a.geti("d", 128);
  const int64_t nq    = a.geti("nq", 10000);
  const int topk      = 10;
  const int reps      = a.geti("reps", 2);
  const int warmup    = a.geti("warmup", 1);
  const auto specs    = split(a.get("index", "flat:1024"), ',');
  const int n_probes  = a.geti("probes", 50);
  const int search_reps = a.geti("search_reps", 5);

  raft::resources res;
  auto stream = raft::resource::get_cuda_stream(res);
  if (!fits((n + nq) * d * 4 * 3 + (size_t(3) << 30), "ivf n=" + std::to_string(n))) { return; }
  auto all = raft::make_device_matrix<float, int64_t>(res, n + nq, d);
  make_data<float>(res, all.data_handle(), n + nq, d, 10000, 2.0f, 777ULL);
  auto X = raft::make_device_matrix_view<const float, int64_t>(all.data_handle(), n, d);
  auto Q = raft::make_device_matrix_view<const float, int64_t>(all.data_handle() + n * d, nq, d);

  // Ground truth.
  auto nbrs  = raft::make_device_matrix<int64_t, int64_t>(res, nq, topk);
  auto dists = raft::make_device_matrix<float, int64_t>(res, nq, topk);
  std::vector<int64_t> gt;
  {
    auto bf = cuvs::neighbors::brute_force::build(res, {}, X);
    cuvs::neighbors::brute_force::search(res, {}, bf, Q, nbrs.view(), dists.view());
    gt = to_host(res, nbrs.data_handle(), nq * topk);
  }

  for (const auto& spec : specs) {
    auto f            = split(spec, ':');
    const std::string kind = f[0];
    const uint32_t p1 = f.size() > 1 ? std::stoul(f[1]) : 1024;
    const uint32_t p2 = f.size() > 2 ? std::stoul(f[2]) : 0;
    std::string cfg   = "index=" + kind + " n=" + std::to_string(n) + " d=" + std::to_string(d) +
                      (kind == "cagra" ? " graph_degree=" : " n_lists=") + std::to_string(p1) +
                      (p2 ? " pq_dim=" + std::to_string(p2) : "");
    // Build `warmup + reps` times; after each timed build, measure recall (untimed). After the last
    // build, time `search_reps` searches.
    std::function<void(bool, bool)> run;  // (timed_build, time_search)
    auto report = [&](double ms, double rec, uint64_t h) {
      printf("RESULT build %s | variant=%s ms=%.3f recall=%.5f hash=%016lx\n",
             cfg.c_str(),
             g_variant.c_str(),
             ms,
             rec,
             h);
      fflush(stdout);
    };
    auto search_timing = [&](auto&& search_fn) {
      search_fn();  // warm (JIT, allocations)
      std::vector<double> t;
      for (int i = 0; i < search_reps; i++) {
        t.push_back(time_ms(res, search_fn));
      }
      for (double ms : t) {
        printf("RESULT search %s probes=%d nq=%ld | variant=%s ms=%.3f qps=%.0f\n",
               cfg.c_str(),
               n_probes,
               nq,
               g_variant.c_str(),
               ms,
               nq / (ms / 1000.0));
      }
      fflush(stdout);
    };
    for (int r = 0; r < warmup + reps; r++) {
      const bool timed = r >= warmup;
      const bool last  = r == warmup + reps - 1;
      if (kind == "flat") {
        cuvs::neighbors::ivf_flat::index_params ip;
        ip.n_lists = p1;
        std::optional<cuvs::neighbors::ivf_flat::index<float, int64_t>> idx;
        double ms = time_ms(res, [&] { idx.emplace(cuvs::neighbors::ivf_flat::build(res, ip, X)); });
        if (!timed) { continue; }
        cuvs::neighbors::ivf_flat::search_params sp;
        sp.n_probes = n_probes;
        auto s      = [&] {
          cuvs::neighbors::ivf_flat::search(res, sp, *idx, Q, nbrs.view(), dists.view());
        };
        s();
        report(ms,
               recall(to_host(res, nbrs.data_handle(), nq * topk), gt, nq, topk),
               fnv(to_host(res, idx->centers().data_handle(), idx->centers().size())));
        if (last) { search_timing(s); }
      } else if (kind == "pq" || kind == "pqc") {
        cuvs::neighbors::ivf_pq::index_params ip;
        ip.n_lists = p1;
        ip.pq_dim  = p2;
        if (kind == "pqc") { ip.codebook_kind = cuvs::neighbors::ivf_pq::codebook_gen::PER_CLUSTER; }
        std::optional<cuvs::neighbors::ivf_pq::index<int64_t>> idx;
        double ms = time_ms(res, [&] { idx.emplace(cuvs::neighbors::ivf_pq::build(res, ip, X)); });
        if (!timed) { continue; }
        cuvs::neighbors::ivf_pq::search_params sp;
        sp.n_probes = n_probes;
        auto s      = [&] {
          cuvs::neighbors::ivf_pq::search(res, sp, *idx, Q, nbrs.view(), dists.view());
        };
        s();
        auto hc = to_host(res, idx->centers().data_handle(), idx->centers().size());
        auto hp = to_host(res, idx->pq_centers().data_handle(), idx->pq_centers().size());
        report(ms, recall(to_host(res, nbrs.data_handle(), nq * topk), gt, nq, topk), fnv(hp, fnv(hc)));
        if (last) { search_timing(s); }
      } else if (kind == "sq") {
        cuvs::neighbors::ivf_sq::index_params ip;
        ip.n_lists = p1;
        std::optional<cuvs::neighbors::ivf_sq::index<uint8_t>> idx;
        double ms = time_ms(res, [&] { idx.emplace(cuvs::neighbors::ivf_sq::build(res, ip, X)); });
        if (!timed) { continue; }
        cuvs::neighbors::ivf_sq::search_params sp;
        sp.n_probes = n_probes;
        cuvs::neighbors::ivf_sq::search(res, sp, *idx, Q, nbrs.view(), dists.view());
        auto c = idx->centers();
        report(ms,
               recall(to_host(res, nbrs.data_handle(), nq * topk), gt, nq, topk),
               fnv(to_host(res, c.data_handle(), c.size())));
      } else if (kind == "rbq") {
        cuvs::neighbors::ivf_rabitq::index_params ip;
        ip.n_lists = p1;
        std::optional<cuvs::neighbors::ivf_rabitq::index<int64_t>> idx;
        double ms =
          time_ms(res, [&] { idx.emplace(cuvs::neighbors::ivf_rabitq::build(res, ip, X)); });
        if (!timed) { continue; }
        cuvs::neighbors::ivf_rabitq::search_params sp;
        sp.n_probes = n_probes;
        cuvs::neighbors::ivf_rabitq::search(res, sp, *idx, Q, nbrs.view(), dists.view());
        auto hn = to_host(res, nbrs.data_handle(), nq * topk);
        report(ms, recall(hn, gt, nq, topk), fnv(hn));
      } else if (kind == "cagra") {
        cuvs::neighbors::cagra::index_params ip;
        ip.graph_degree              = p1;
        ip.intermediate_graph_degree = 2 * p1;
        ip.graph_build_params =
          cuvs::neighbors::cagra::graph_build_params::ivf_pq_params(X.extents());
        auto view = cuvs::neighbors::make_device_padded_dataset_view(res, X);
        std::optional<cuvs::neighbors::cagra::device_padded_index<float, uint32_t>> idx;
        double ms = time_ms(res, [&] { idx.emplace(cuvs::neighbors::cagra::build(res, ip, view)); });
        if (!timed) { continue; }
        cuvs::neighbors::cagra::search_params sp;
        sp.itopk_size = 64;
        auto n32      = raft::make_device_matrix<uint32_t, int64_t>(res, nq, topk);
        cuvs::neighbors::cagra::search(res, sp, *idx, Q, n32.view(), dists.view());
        auto h32 = to_host(res, n32.data_handle(), nq * topk);
        std::vector<int64_t> hn(h32.begin(), h32.end());
        auto g = idx->graph();
        report(ms, recall(hn, gt, nq, topk), fnv(to_host(res, g.data_handle(), g.size())));
      } else {
        fprintf(stderr, "unknown index %s\n", kind.c_str());
        exit(2);
      }
    }
  }
}

int main(int argc, char** argv)
{
  if (argc < 3) {
    fprintf(stderr, "usage: %s <variant> <suite> [key=value...]\n", argv[0]);
    return 2;
  }
  g_variant = argv[1];
  std::string suite = argv[2];
  Args a;
  for (int i = 3; i < argc; i++) {
    std::string s(argv[i]);
    auto eq = s.find('=');
    if (eq != std::string::npos) { a.kv[s.substr(0, eq)] = s.substr(eq + 1); }
  }
  check_loaded_libcuvs();
  try {
    if (suite == "kmeans") {
      if (a.get("dtype", "f32") == "f64") {
        suite_kmeans<double>(a);
      } else {
        suite_kmeans<float>(a);
      }
    } else if (suite == "balanced") {
      if (a.get("dtype", "f32") == "i8") {
        suite_balanced<int8_t>(a);
      } else {
        suite_balanced<float>(a);
      }
    } else if (suite == "ivf") {
      suite_ivf(a);
    } else {
      fprintf(stderr, "unknown suite %s\n", suite.c_str());
      return 2;
    }
  } catch (std::exception& e) {
    printf("ERROR variant=%s suite=%s: %s\n", g_variant.c_str(), suite.c_str(), e.what());
    fflush(stdout);
    return 1;
  }
  return 0;
}
