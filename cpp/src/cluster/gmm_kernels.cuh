/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * CUDA kernels for the Gaussian mixture E- and M-steps, covering all four
 * covariance parameterizations (full, tied, diag, spherical).
 *
 * The mathematics mirror scikit-learn's ``GaussianMixture``. For the ``full``
 * and ``tied`` cases the precision Cholesky factor ``prec_chol`` is the
 * upper-triangular ``U`` with precision ``= U @ Uᵀ``; for ``diag``/``spherical``
 * it holds the per-feature / per-component reciprocal standard deviations.
 *
 * The full-covariance fused E-step kernels are adapted from rapids-singlecell's
 * CellCharter GMM kernels.
 */

#pragma once

#include <cuda_runtime.h>
#include <math_constants.h>

namespace cuvs::cluster::gmm::detail {

constexpr float LOG_2PI_F  = 1.8378770664093453f;
constexpr double LOG_2PI_D = 1.8378770664093453;

template <typename T>
__device__ __forceinline__ T log_2pi_const();
template <>
__device__ __forceinline__ float log_2pi_const<float>()
{
  return LOG_2PI_F;
}
template <>
__device__ __forceinline__ double log_2pi_const<double>()
{
  return LOG_2PI_D;
}

__device__ __forceinline__ int upper_tri_col_offset(int col) { return (col * (col + 1)) / 2; }

// ===========================================================================
// E-step: log Gaussian probability
// ===========================================================================

// ---------------------------------------------------------------------------
// full / tied, d <= 64. ``prec_per_component`` is 1 for full (prec_chol has a
// d×d matrix per component) and 0 for tied (a single shared d×d matrix).
// ---------------------------------------------------------------------------
template <typename T, int D = 0>
__global__ void e_step_log_prob_small_kernel(const T* __restrict__ X,
                                             const T* __restrict__ weights,
                                             const T* __restrict__ means,
                                             const T* __restrict__ prec_chol,
                                             const T* __restrict__ log_det,
                                             int n,
                                             int d,
                                             int K,
                                             int prec_per_component,
                                             T* __restrict__ log_prob)
{
  static_assert(D >= 0 && D <= 64, "GMM small E-step supports runtime d or fixed D <= 64");
  constexpr bool fixed_d = D != 0;
  int dim                = fixed_d ? D : d;
  int k                  = blockIdx.y;
  int n_idx              = blockIdx.x * blockDim.x + threadIdx.x;
  int tid                = threadIdx.x;
  int kpc                = prec_per_component ? k : 0;

  extern __shared__ unsigned char smem_raw[];
  T* sh_mean = reinterpret_cast<T*>(smem_raw);
  T* sh_pc   = sh_mean + dim;

  for (int i = tid; i < dim; i += blockDim.x)
    sh_mean[i] = means[(size_t)k * dim + i];
  int pc_size_dense = dim * dim;
  for (int i = tid; i < pc_size_dense; i += blockDim.x) {
    int row = i / dim;
    int col = i - row * dim;
    if (row <= col) {
      sh_pc[upper_tri_col_offset(col) + row] = prec_chol[(size_t)kpc * pc_size_dense + i];
    }
  }

  __shared__ T sh_const;
  if (tid == 0) { sh_const = T(-0.5) * T(dim) * log_2pi_const<T>() + log_det[k] + log(weights[k]); }

  __syncthreads();

  if (n_idx >= n) return;

  T centered_vals[fixed_d ? D : 64];
  if constexpr (fixed_d) {
#pragma unroll
    for (int dd = 0; dd < D; ++dd)
      centered_vals[dd] = X[(size_t)n_idx * D + dd] - sh_mean[dd];
  } else {
    for (int dd = 0; dd < dim; ++dd)
      centered_vals[dd] = X[(size_t)n_idx * dim + dd] - sh_mean[dd];
  }

  T mahal = T(0);
  if constexpr (fixed_d) {
#pragma unroll
    for (int j = 0; j < D; ++j) {
      T y        = T(0);
      int pc_col = upper_tri_col_offset(j);
#pragma unroll
      for (int dd = 0; dd <= j; ++dd) {
        y += centered_vals[dd] * sh_pc[pc_col + dd];
      }
      mahal += y * y;
    }
  } else {
    for (int j = 0; j < dim; ++j) {
      T y        = T(0);
      int pc_col = upper_tri_col_offset(j);
      for (int dd = 0; dd <= j; ++dd) {
        y += centered_vals[dd] * sh_pc[pc_col + dd];
      }
      mahal += y * y;
    }
  }
  log_prob[(size_t)n_idx * K + k] = sh_const - T(0.5) * mahal;
}

// ---------------------------------------------------------------------------
// full / tied, d > 64, tiled over 64-column blocks.
// ---------------------------------------------------------------------------
template <typename T, int TILE_D>
__global__ void e_step_log_prob_large_d_thread64_kernel(const T* __restrict__ X,
                                                        const T* __restrict__ weights,
                                                        const T* __restrict__ means,
                                                        const T* __restrict__ prec_chol,
                                                        const T* __restrict__ log_det,
                                                        int n,
                                                        int d,
                                                        int K,
                                                        int prec_per_component,
                                                        T* __restrict__ log_prob)
{
  static_assert(TILE_D == 64, "GMM thread64 E-step expects a 64-column precision tile");

  int k   = blockIdx.y;
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;
  int kpc = prec_per_component ? k : 0;

  extern __shared__ unsigned char smem_raw[];
  T* sh_mean = reinterpret_cast<T*>(smem_raw);  // (64,)
  T* sh_pc   = sh_mean + TILE_D;                // (64, 64)

  __shared__ T sh_const;
  if (tid == 0) { sh_const = T(-0.5) * T(d) * log_2pi_const<T>() + log_det[k] + log(weights[k]); }

  T local_mahal = T(0);
  const T* pc   = prec_chol + (size_t)kpc * d * d;

  for (int j_base = 0; j_base < d; j_base += TILE_D) {
    int cols_in_tile = min(TILE_D, d - j_base);
    int dd_limit     = min(d, j_base + TILE_D);
    T y[TILE_D];
#pragma unroll
    for (int col = 0; col < TILE_D; ++col)
      y[col] = T(0);

    for (int dd_base = 0; dd_base < dd_limit; dd_base += TILE_D) {
      int feats_in_tile = min(TILE_D, dd_limit - dd_base);

      for (int idx = tid; idx < TILE_D; idx += blockDim.x) {
        sh_mean[idx] = (idx < feats_in_tile) ? means[(size_t)k * d + dd_base + idx] : T(0);
      }

      constexpr int pc_tile_elems = TILE_D * TILE_D;
      for (int idx = tid; idx < pc_tile_elems; idx += blockDim.x) {
        int feat      = idx / TILE_D;
        int col_local = idx - feat * TILE_D;
        int dd        = dd_base + feat;
        int col       = j_base + col_local;
        T val         = T(0);
        if (feat < feats_in_tile && col_local < cols_in_tile && dd <= col) {
          val = pc[(size_t)dd * d + col];
        }
        sh_pc[feat * TILE_D + col_local] = val;
      }

      __syncthreads();

      if (row < n) {
#pragma unroll
        for (int feat = 0; feat < TILE_D; ++feat) {
          if (feat >= feats_in_tile) break;
          T diff = X[(size_t)row * d + dd_base + feat] - sh_mean[feat];
#pragma unroll
          for (int col = 0; col < TILE_D; ++col) {
            if (col >= cols_in_tile) break;
            y[col] += diff * sh_pc[feat * TILE_D + col];
          }
        }
      }

      __syncthreads();
    }

    if (row < n) {
#pragma unroll
      for (int col = 0; col < TILE_D; ++col) {
        if (col >= cols_in_tile) break;
        local_mahal += y[col] * y[col];
      }
    }
  }

  if (row < n) log_prob[(size_t)row * K + k] = sh_const - T(0.5) * local_mahal;
}

// ---------------------------------------------------------------------------
// diagonal covariance: prec_chol is (K, d) reciprocal std-devs.
//
// One thread per sample: each thread reads its X row once (kept hot in L1
// across the K components) and ``means``/``prec_chol`` are cached in shared
// memory when they fit. This avoids the K-fold re-read of X that a
// thread-per-(sample, component) layout incurs.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void e_step_log_prob_diag_kernel(const T* __restrict__ X,
                                            const T* __restrict__ weights,
                                            const T* __restrict__ means,
                                            const T* __restrict__ prec_chol,
                                            const T* __restrict__ log_det,
                                            int n,
                                            int d,
                                            int K,
                                            int use_smem,
                                            T* __restrict__ log_prob)
{
  extern __shared__ unsigned char smem_raw[];
  T* sh_means = reinterpret_cast<T*>(smem_raw);
  T* sh_pc    = sh_means + (size_t)K * d;
  if (use_smem) {
    for (int i = threadIdx.x; i < K * d; i += blockDim.x) {
      sh_means[i] = means[i];
      sh_pc[i]    = prec_chol[i];
    }
    __syncthreads();
  }
  const T* M = use_smem ? sh_means : means;
  const T* P = use_smem ? sh_pc : prec_chol;

  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) return;
  const T* x = X + (size_t)row * d;

  for (int k = 0; k < K; ++k) {
    const T* mu = M + (size_t)k * d;
    const T* pc = P + (size_t)k * d;
    T mahal     = T(0);
    for (int dd = 0; dd < d; ++dd) {
      T y = (x[dd] - mu[dd]) * pc[dd];
      mahal += y * y;
    }
    T constant = T(-0.5) * T(d) * log_2pi_const<T>() + log_det[k] + log(weights[k]);
    log_prob[(size_t)row * K + k] = constant - T(0.5) * mahal;
  }
}

// ---------------------------------------------------------------------------
// spherical covariance: prec_chol is (K,) reciprocal std-devs. One thread per
// sample; ``means`` cached in shared memory when it fits.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void e_step_log_prob_spherical_kernel(const T* __restrict__ X,
                                                 const T* __restrict__ weights,
                                                 const T* __restrict__ means,
                                                 const T* __restrict__ prec_chol,
                                                 const T* __restrict__ log_det,
                                                 int n,
                                                 int d,
                                                 int K,
                                                 int use_smem,
                                                 T* __restrict__ log_prob)
{
  extern __shared__ unsigned char smem_raw[];
  T* sh_means = reinterpret_cast<T*>(smem_raw);
  if (use_smem) {
    for (int i = threadIdx.x; i < K * d; i += blockDim.x)
      sh_means[i] = means[i];
    __syncthreads();
  }
  const T* M = use_smem ? sh_means : means;

  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) return;
  const T* x = X + (size_t)row * d;

  for (int k = 0; k < K; ++k) {
    const T* mu = M + (size_t)k * d;
    T pc        = prec_chol[k];
    T sq        = T(0);
    for (int dd = 0; dd < d; ++dd) {
      T diff = x[dd] - mu[dd];
      sq += diff * diff;
    }
    T mahal    = pc * pc * sq;
    T constant = T(-0.5) * T(d) * log_2pi_const<T>() + log_det[k] + log(weights[k]);
    log_prob[(size_t)row * K + k] = constant - T(0.5) * mahal;
  }
}

// ---------------------------------------------------------------------------
// cuBLAS E-step (full/tied, wide d): center one component's data, X - means[k].
// ---------------------------------------------------------------------------
template <typename T>
__global__ void e_step_center_kernel(const T* __restrict__ X,
                                     const T* __restrict__ means,
                                     int n,
                                     int d,
                                     int k,
                                     T* __restrict__ centered)
{
  size_t idx   = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = (size_t)n * d;
  if (idx >= total) return;
  int col       = idx % d;
  centered[idx] = X[idx] - means[(size_t)k * d + col];
}

// Projected centers c[k] = M_k @ mu_k for the un-centered E-step, where M_k is
// the (column-major, lda=d) prec-chol matrix used by the cuBLAS GEMM. Lets the
// fused fold compute ||M_k x - c_k||^2 without first materializing (X - mu_k).
template <typename T>
__global__ void fused_center_proj_kernel(const T* __restrict__ prec_chol,
                                         const T* __restrict__ means,
                                         int d,
                                         int K,
                                         int prec_pc,
                                         T* __restrict__ c)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= K * d) return;
  int k       = idx / d;
  int i       = idx % d;
  const T* M  = prec_chol + (size_t)(prec_pc ? k : 0) * d * d;
  const T* mu = means + (size_t)k * d;
  T acc       = T(0);
  for (int j = 0; j < d; ++j)
    acc += M[(size_t)i + (size_t)j * d] * mu[j];
  c[(size_t)k * d + i] = acc;
}

// cuBLAS E-step: from y = (X - means[k]) @ prec_chol[k] (n, d), compute
// log_prob[:, k] = const_k - 0.5 * sum_j y[:, j]^2 (Kahan-compensated).
template <typename T>
__global__ void e_step_log_prob_from_y_kernel(const T* __restrict__ y,
                                              const T* __restrict__ weights,
                                              const T* __restrict__ log_det,
                                              int n,
                                              int d,
                                              int K,
                                              int k,
                                              T* __restrict__ log_prob)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) return;

  T mahal        = T(0);
  T compensation = T(0);
  for (int col = 0; col < d; ++col) {
    T v          = y[(size_t)row * d + col];
    T term       = v * v - compensation;
    T next       = mahal + term;
    compensation = (next - mahal) - term;
    mahal        = next;
  }
  T constant = T(-0.5) * T(d) * log_2pi_const<T>() + log_det[k] + log(weights[k]);
  log_prob[(size_t)row * K + k] = constant - T(0.5) * mahal;
}

// ---------------------------------------------------------------------------
// Per-row log-sum-exp normalize: resp = exp(log_prob - logsumexp); also writes
// the per-sample log-likelihood (= logsumexp) into log_prob_norm.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void e_step_normalize_kernel(
  const T* __restrict__ log_prob, int n, int K, T* __restrict__ resp, T* __restrict__ log_prob_norm)
{
  int n_idx = blockIdx.x;
  if (n_idx >= n) return;
  int tid = threadIdx.x;
  // 64-bit row offset: n_idx * K overflows int32 once n * K exceeds 2^31.
  size_t row = (size_t)n_idx * K;

  __shared__ T sh_max;
  __shared__ T sh_sum;

  T local_max = -CUDART_INF_F;
  for (int k = tid; k < K; k += blockDim.x) {
    T v = log_prob[row + k];
    if (v > local_max) local_max = v;
  }
  for (int off = 16; off > 0; off >>= 1) {
    T other = __shfl_down_sync(0xffffffff, local_max, off);
    if (other > local_max) local_max = other;
  }
  if (tid == 0) sh_max = local_max;
  __syncthreads();
  T mx = sh_max;

  T local_sum = T(0);
  for (int k = tid; k < K; k += blockDim.x) {
    local_sum += exp(log_prob[row + k] - mx);
  }
  for (int off = 16; off > 0; off >>= 1)
    local_sum += __shfl_down_sync(0xffffffff, local_sum, off);
  if (tid == 0) {
    sh_sum               = local_sum;
    log_prob_norm[n_idx] = log(local_sum) + mx;
  }
  __syncthreads();
  T log_total = log(sh_sum) + mx;

  for (int k = tid; k < K; k += blockDim.x) {
    resp[row + k] = exp(log_prob[row + k] - log_total);
  }
}

// ===========================================================================
// M-step
// ===========================================================================

// means[k] = num[k] / N_k[k]; weights[k] = N_k[k] / n.
template <typename T>
__global__ void m_step_finalize_means_kernel(const T* __restrict__ N_k,
                                             const T* __restrict__ num,
                                             T* __restrict__ weights,
                                             T* __restrict__ means,
                                             T eps,
                                             int n,
                                             int d,
                                             int K)
{
  int k   = blockIdx.x;
  int tid = threadIdx.x;
  if (k >= K) return;

  T Nk     = N_k[k] + T(10) * eps;
  T inv_Nk = T(1) / Nk;
  if (tid == 0) weights[k] = Nk / T(n);

  for (int i = tid; i < d; i += blockDim.x)
    means[k * d + i] = num[k * d + i] * inv_Nk;
}

// sqrt(resp) * (X - means[k]) for one component k. Output feeds a GEMM forming
// the (unnormalized) full covariance.
template <typename T>
__global__ void weighted_center_kernel(const T* __restrict__ X,
                                       const T* __restrict__ resp,
                                       const T* __restrict__ means,
                                       int n,
                                       int d,
                                       int K,
                                       int k,
                                       T* __restrict__ centered)
{
  size_t idx   = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = (size_t)n * d;
  if (idx >= total) return;

  int row       = idx / d;
  int col       = idx - (size_t)row * d;
  T r           = resp[row * K + k];
  centered[idx] = sqrt(r) * (X[idx] - means[k * d + col]);
}

// Finalize one full covariance matrix: divide by N_k, symmetrize the row-major
// result the column-major GEMM produced, and add reg_covar to the diagonal.
template <typename T>
__global__ void m_step_finalize_cov_full_kernel(
  const T* __restrict__ N_k, T* __restrict__ covariances, T reg_covar, T eps, int d, int K)
{
  int k   = blockIdx.x;
  int tid = threadIdx.x;
  if (k >= K) return;

  T Nk      = N_k[k] + T(10) * eps;
  T inv_Nk  = T(1) / Nk;
  int total = d * d;
  T* cov    = covariances + (size_t)k * d * d;

  for (int idx = tid; idx < total; idx += blockDim.x) {
    int i = idx / d;
    int j = idx % d;
    if (i > j) continue;

    T v = cov[j * d + i] * inv_Nk;
    if (i == j) v += reg_covar;
    cov[i * d + j] = v;
    if (i != j) cov[j * d + i] = v;
  }
}

// Square every element of X into Xsq (used by the diagonal M-step GEMM).
template <typename T>
__global__ void elementwise_square_kernel(const T* __restrict__ X,
                                          size_t total,
                                          T* __restrict__ out)
{
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  T v      = X[idx];
  out[idx] = v * v;
}

// diag covariance: var[k,dd] = num2[k,dd]/N_k[k] - means[k,dd]^2 + reg_covar,
// where num2 = respᵀ @ (X*X).
template <typename T>
__global__ void m_step_finalize_diag_kernel(const T* __restrict__ N_k,
                                            const T* __restrict__ num2,
                                            const T* __restrict__ means,
                                            T reg_covar,
                                            T eps,
                                            int d,
                                            int K,
                                            T* __restrict__ variances)
{
  int k   = blockIdx.x;
  int tid = threadIdx.x;
  if (k >= K) return;
  T Nk     = N_k[k] + T(10) * eps;
  T inv_Nk = T(1) / Nk;
  for (int dd = tid; dd < d; dd += blockDim.x) {
    T mu                          = means[(size_t)k * d + dd];
    variances[(size_t)k * d + dd] = num2[(size_t)k * d + dd] * inv_Nk - mu * mu + reg_covar;
  }
}

// spherical covariance: var[k] = mean over features of the diagonal variances.
template <typename T>
__global__ void m_step_spherical_from_diag_kernel(const T* __restrict__ diag_var,
                                                  int d,
                                                  int K,
                                                  T* __restrict__ variances)
{
  int k   = blockIdx.x;
  int tid = threadIdx.x;
  if (k >= K) return;

  __shared__ T sh[256];
  T local = T(0);
  for (int dd = tid; dd < d; dd += blockDim.x)
    local += diag_var[(size_t)k * d + dd];
  sh[tid] = local;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (tid < off) sh[tid] += sh[tid + off];
    __syncthreads();
  }
  if (tid == 0) variances[k] = sh[0] / T(d);
}

// Scale each component mean by N_k (for the tied covariance weighted outer
// product (N_k * means)ᵀ @ means).
template <typename T>
__global__ void scale_rows_by_kernel(
  const T* __restrict__ in, const T* __restrict__ scale, T eps, int d, int K, T* __restrict__ out)
{
  int k   = blockIdx.x;
  int tid = threadIdx.x;
  if (k >= K) return;
  T s = scale[k] + T(10) * eps;
  for (int dd = tid; dd < d; dd += blockDim.x)
    out[(size_t)k * d + dd] = in[(size_t)k * d + dd] * s;
}

// tied covariance: cov = (XtX - weighted_means_outer) / sum(N_k) + reg_covar*I.
template <typename T>
__global__ void m_step_finalize_tied_kernel(const T* __restrict__ XtX,
                                            const T* __restrict__ weighted_outer,
                                            T sum_Nk,
                                            T reg_covar,
                                            int d,
                                            T* __restrict__ covariance)
{
  int idx   = blockIdx.x * blockDim.x + threadIdx.x;
  int total = d * d;
  if (idx >= total) return;
  int i = idx / d;
  int j = idx % d;
  if (i > j) return;  // upper triangle drives a symmetric write
  T inv = T(1) / sum_Nk;
  T v   = (XtX[idx] - weighted_outer[idx]) * inv;
  if (i == j) v += reg_covar;
  covariance[(size_t)i * d + j] = v;
  if (i != j) covariance[(size_t)j * d + i] = v;
}

// ===========================================================================
// Precision Cholesky helpers (diag / spherical) and log-determinants
// ===========================================================================

// diag/spherical precision Cholesky: 1 / sqrt(variance), element-wise.
template <typename T>
__global__ void recip_sqrt_kernel(const T* __restrict__ var, size_t total, T* __restrict__ out)
{
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  out[idx] = T(1) / sqrt(var[idx]);
}

// log-det of the precision Cholesky for full covariance:
// log_det[k] = sum_i log(prec_chol[k,i,i]).
template <typename T>
__global__ void log_det_full_kernel(const T* __restrict__ prec_chol,
                                    int d,
                                    int K,
                                    T* __restrict__ log_det)
{
  int k   = blockIdx.x;
  int tid = threadIdx.x;
  if (k >= K) return;
  __shared__ T sh[256];
  T local       = T(0);
  const T* pc_k = prec_chol + (size_t)k * d * d;
  for (int i = tid; i < d; i += blockDim.x)
    local += log(pc_k[(size_t)i * d + i]);
  sh[tid] = local;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (tid < off) sh[tid] += sh[tid + off];
    __syncthreads();
  }
  if (tid == 0) log_det[k] = sh[0];
}

// log-det for tied covariance: a single value broadcast to all K components.
template <typename T>
__global__ void log_det_tied_kernel(const T* __restrict__ prec_chol,
                                    int d,
                                    int K,
                                    T* __restrict__ log_det)
{
  int tid = threadIdx.x;
  __shared__ T sh[256];
  T local = T(0);
  for (int i = tid; i < d; i += blockDim.x)
    local += log(prec_chol[(size_t)i * d + i]);
  sh[tid] = local;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (tid < off) sh[tid] += sh[tid + off];
    __syncthreads();
  }
  if (tid == 0) {
    for (int k = 0; k < K; ++k)
      log_det[k] = sh[0];
  }
}

// log-det for diagonal covariance: log_det[k] = sum_dd log(prec_chol[k,dd]).
template <typename T>
__global__ void log_det_diag_kernel(const T* __restrict__ prec_chol,
                                    int d,
                                    int K,
                                    T* __restrict__ log_det)
{
  int k   = blockIdx.x;
  int tid = threadIdx.x;
  if (k >= K) return;
  __shared__ T sh[256];
  T local = T(0);
  for (int dd = tid; dd < d; dd += blockDim.x)
    local += log(prec_chol[(size_t)k * d + dd]);
  sh[tid] = local;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (tid < off) sh[tid] += sh[tid + off];
    __syncthreads();
  }
  if (tid == 0) log_det[k] = sh[0];
}

// log-det for spherical covariance: log_det[k] = d * log(prec_chol[k]).
template <typename T>
__global__ void log_det_spherical_kernel(const T* __restrict__ prec_chol,
                                         int d,
                                         int K,
                                         T* __restrict__ log_det)
{
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= K) return;
  log_det[k] = T(d) * log(prec_chol[k]);
}

// ===========================================================================
// Miscellaneous
// ===========================================================================

// Identity matrix (d×d), row-major.
template <typename T>
__global__ void set_identity_kernel(T* __restrict__ A, int d)
{
  int idx   = blockIdx.x * blockDim.x + threadIdx.x;
  int total = d * d;
  if (idx >= total) return;
  int i  = idx / d;
  int j  = idx % d;
  A[idx] = (i == j) ? T(1) : T(0);
}

// Batched identity: K stacked d×d identity matrices, row-major (K, d, d).
template <typename T>
__global__ void set_identity_batched_kernel(T* __restrict__ A, int d, int K)
{
  size_t idx   = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = (size_t)K * d * d;
  if (idx >= total) return;
  size_t within = idx % ((size_t)d * d);
  int i         = (int)(within / d);
  int j         = (int)(within % d);
  A[idx]        = (i == j) ? T(1) : T(0);
}

// Argmax over components: labels[n] = argmax_k resp[n,k].
template <typename T>
__global__ void argmax_kernel(const T* __restrict__ resp, int n, int K, int* __restrict__ labels)
{
  int n_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (n_idx >= n) return;
  const T* r = resp + (size_t)n_idx * K;
  int best   = 0;
  T best_v   = r[0];
  for (int k = 1; k < K; ++k) {
    if (r[k] > best_v) {
      best_v = r[k];
      best   = k;
    }
  }
  labels[n_idx] = best;
}

// Scatter hard labels into one-hot responsibilities (n×K).
template <typename T>
__global__ void labels_to_onehot_kernel(const int* __restrict__ labels,
                                        int n,
                                        int K,
                                        T* __restrict__ resp)
{
  int n_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (n_idx >= n) return;
  int lab = labels[n_idx];
  for (int k = 0; k < K; ++k)
    resp[(size_t)n_idx * K + k] = (k == lab) ? T(1) : T(0);
}

// One-hot from a set of chosen sample indices (resp[indices[k], k] = 1), used by
// 'random_from_data'. resp must be pre-zeroed.
template <typename T>
__global__ void scatter_onehot_indices_kernel(const int* __restrict__ indices,
                                              int K,
                                              T* __restrict__ resp)
{
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= K) return;
  resp[(size_t)indices[k] * K + k] = T(1);
}

// Normalize each row of resp to sum to one (used by 'random' init).
template <typename T>
__global__ void normalize_rows_kernel(T* __restrict__ resp, int n, int K)
{
  int n_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (n_idx >= n) return;
  T* r  = resp + (size_t)n_idx * K;
  T sum = T(0);
  for (int k = 0; k < K; ++k)
    sum += r[k];
  T inv = T(1) / sum;
  for (int k = 0; k < K; ++k)
    r[k] *= inv;
}

// precisions = prec_chol^2 for diag/spherical.
template <typename T>
__global__ void square_kernel(const T* __restrict__ in, size_t total, T* __restrict__ out)
{
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  T v      = in[idx];
  out[idx] = v * v;
}

// ===========================================================================
// Fused E-step (no N×K materialization): per-sample online log-sum-exp.
// Produces log_prob_norm[n] = log p(x_n) directly, and optionally the argmax
// component label[n]. Used by score_samples / predict for all covariance
// types — eliminates the (N, K) log_prob/resp buffers the standard E-step
// allocates. ``is_diag`` selects spherical (prec_chol is (K,)) vs diagonal
// (prec_chol is (K, d)); full / tied feed the cuBLAS-fold path below.
// ===========================================================================
template <typename T>
__global__ void fused_estep_diag_sph_kernel(const T* __restrict__ X,
                                            const T* __restrict__ weights,
                                            const T* __restrict__ means,
                                            const T* __restrict__ prec_chol,
                                            const T* __restrict__ log_det,
                                            int n,
                                            int d,
                                            int K,
                                            int is_diag,
                                            T* __restrict__ log_prob_norm,
                                            int* __restrict__ labels)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) return;
  const T* x = X + (size_t)row * d;

  T rmax = -CUDART_INF_F, rsum = T(0), best_lp = -CUDART_INF_F;
  int best = 0;
  for (int k = 0; k < K; ++k) {
    const T* mu = means + (size_t)k * d;
    T mahal     = T(0);
    if (is_diag) {
      const T* pc = prec_chol + (size_t)k * d;
      for (int dd = 0; dd < d; ++dd) {
        T y = (x[dd] - mu[dd]) * pc[dd];
        mahal += y * y;
      }
    } else {
      T sq = T(0);
      for (int dd = 0; dd < d; ++dd) {
        T diff = x[dd] - mu[dd];
        sq += diff * diff;
      }
      mahal = prec_chol[k] * prec_chol[k] * sq;
    }
    T lp = T(-0.5) * T(d) * log_2pi_const<T>() + log_det[k] + log(weights[k]) - T(0.5) * mahal;
    if (lp > best_lp) {
      best_lp = lp;
      best    = k;
    }
    T nm = fmax(rmax, lp);
    rsum = rsum * exp(rmax - nm) + exp(lp - nm);
    rmax = nm;
  }
  log_prob_norm[row] = rmax + log(rsum);
  if (labels) labels[row] = best;
}

// Fast tiled fused E-step (diag / spherical): thread-per-sample with the
// sample's feature vector held in registers and component means (and, for
// diag, reciprocal std-devs) streamed through shared memory in tiles of CELL
// components. Online log-sum-exp -> log_prob_norm (+ optional argmax label),
// no N×K materialization. ``MAXD`` is the register footprint for the feature
// vector (caller picks the smallest bucket >= d); ``DIAG`` selects the
// covariance form at compile time so the shared buffers are sized exactly.
// const_k[k] = -0.5 d log(2pi) + log_det[k] + log(weights[k]), precomputed once
// per component so the hot E-step loop never calls log() per (sample, k).
template <typename T>
__global__ void fused_const_kernel(const T* __restrict__ weights,
                                   const T* __restrict__ log_det,
                                   int d,
                                   int K,
                                   T* __restrict__ const_k)
{
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= K) return;
  const_k[k] = T(-0.5) * T(d) * log_2pi_const<T>() + log_det[k] + log(weights[k]);
}

template <typename T, int MAXD, bool DIAG>
__global__ void fused_estep_tiled_kernel(const T* __restrict__ X,
                                         const T* __restrict__ const_k,
                                         const T* __restrict__ means,
                                         const T* __restrict__ prec_chol,
                                         int n,
                                         int d,
                                         int K,
                                         T* __restrict__ log_prob_norm,
                                         int* __restrict__ labels)
{
  constexpr int CELL = 32;
  int q              = blockIdx.x * blockDim.x + threadIdx.x;
  T xq[MAXD];
  if (q < n) {
    for (int dd = 0; dd < d; ++dd)
      xq[dd] = X[(size_t)q * d + dd];
  }
  __shared__ T smu[CELL * MAXD];
  __shared__ T spc[DIAG ? CELL * MAXD : 1];

  T rmax = -CUDART_INF_F, rsum = T(0), best_lp = -CUDART_INF_F;
  int best = 0;
  for (int k0 = 0; k0 < K; k0 += CELL) {
    int cells = min(CELL, K - k0);
    for (int i = threadIdx.x; i < cells * d; i += blockDim.x) {
      smu[i] = means[(size_t)k0 * d + i];
      if constexpr (DIAG) spc[i] = prec_chol[(size_t)k0 * d + i];
    }
    __syncthreads();
    if (q < n) {
      for (int c = 0; c < cells; ++c) {
        int k       = k0 + c;
        const T* mu = smu + c * d;
        T mahal     = T(0);
        if constexpr (DIAG) {
          const T* pc = spc + c * d;
          for (int dd = 0; dd < d; ++dd) {
            T y = (xq[dd] - mu[dd]) * pc[dd];
            mahal += y * y;
          }
        } else {
          T sq = T(0);
          for (int dd = 0; dd < d; ++dd) {
            T df = xq[dd] - mu[dd];
            sq += df * df;
          }
          mahal = prec_chol[k] * prec_chol[k] * sq;
        }
        T lp = const_k[k] - T(0.5) * mahal;
        if (lp > best_lp) {
          best_lp = lp;
          best    = k;
        }
        T nm = fmax(rmax, lp);
        rsum = rsum * exp(rmax - nm) + exp(lp - nm);
        rmax = nm;
      }
    }
    __syncthreads();
  }
  if (q < n) {
    log_prob_norm[q] = rmax + log(rsum);
    if (labels) labels[q] = best;
  }
}

// Initialize the running (max, sum) accumulators for the cuBLAS fold path.
template <typename T>
__global__ void fused_lse_init_kernel(T* __restrict__ rmax,
                                      T* __restrict__ rsum,
                                      T* __restrict__ best_lp,
                                      int* __restrict__ labels,
                                      int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  rmax[i] = -CUDART_INF_F;
  rsum[i] = T(0);
  if (best_lp) best_lp[i] = -CUDART_INF_F;
  if (labels) labels[i] = 0;
}

// Fold component k (full/tied) into the running log-sum-exp from
// y = (x - mu_k) @ prec_chol_k, with Kahan-compensated ‖y‖² (matches the
// standard cuBLAS E-step). Optionally tracks the argmax component.
template <typename T>
__global__ void fused_fold_from_y_kernel(const T* __restrict__ y,
                                         const T* __restrict__ const_k,
                                         const T* __restrict__ c,
                                         int n,
                                         int d,
                                         int k,
                                         T* __restrict__ rmax,
                                         T* __restrict__ rsum,
                                         T* __restrict__ best_lp,
                                         int* __restrict__ labels)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) return;
  // y = M_k @ x (un-centered). mahal = ||M_k(x - mu_k)||^2 = ||y - c_k||^2
  // with the projected center c_k = M_k @ mu_k precomputed once per component.
  const T* ck = c + (size_t)k * d;
  T mahal = T(0), comp = T(0);
  for (int col = 0; col < d; ++col) {
    T v    = y[(size_t)row * d + col] - ck[col];
    T term = v * v - comp;
    T next = mahal + term;
    comp   = (next - mahal) - term;
    mahal  = next;
  }
  T lp = const_k[k] - T(0.5) * mahal;
  if (labels && lp > best_lp[row]) {
    best_lp[row] = lp;
    labels[row]  = k;
  }
  T pm      = rmax[row];
  T nm      = fmax(pm, lp);
  rsum[row] = rsum[row] * exp(pm - nm) + exp(lp - nm);
  rmax[row] = nm;
}

// log_prob_norm[n] = rmax[n] + log(rsum[n]).
template <typename T>
__global__ void fused_lse_finalize_kernel(const T* __restrict__ rmax,
                                          const T* __restrict__ rsum,
                                          int n,
                                          T* __restrict__ log_prob_norm)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  log_prob_norm[i] = rmax[i] + log(rsum[i]);
}

}  // namespace cuvs::cluster::gmm::detail
