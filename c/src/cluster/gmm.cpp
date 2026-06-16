/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>

#include <dlpack/dlpack.h>

#include <cuvs/cluster/gmm.h>
#include <cuvs/cluster/gmm.hpp>
#include <cuvs/core/c_api.h>

#include "../core/exceptions.hpp"
#include "../core/interop.hpp"

namespace {

cuvs::cluster::gmm::params convert_params(const cuvsGMMParams& params)
{
  auto gmm_params         = cuvs::cluster::gmm::params();
  gmm_params.n_components = params.n_components;
  gmm_params.cov_type  = static_cast<cuvs::cluster::gmm::covariance_type>(params.covariance_type);
  gmm_params.tol       = params.tol;
  gmm_params.reg_covar = params.reg_covar;
  gmm_params.max_iter  = params.max_iter;
  gmm_params.n_init    = params.n_init;
  gmm_params.init      = static_cast<cuvs::cluster::gmm::init_method>(params.init);
  gmm_params.seed      = params.seed;
  return gmm_params;
}

// Total number of elements of a (device) DLPack tensor, used for the
// covariance-shaped buffers whose rank depends on the covariance type.
int64_t tensor_numel(DLManagedTensor* tensor, const char* name)
{
  auto dl = tensor->dl_tensor;
  if (!cuvs::core::is_dlpack_device_compatible(dl)) {
    RAFT_FAIL("%s must be on device memory", name);
  }
  int64_t total = 1;
  for (int i = 0; i < dl.ndim; ++i)
    total *= dl.shape[i];
  return total;
}

// The flat (covariance-shaped) buffers are reinterpret_cast to T, so their
// element type must match the dtype the call was dispatched on (X's dtype).
// from_dlpack enforces this for matrix/vector views; do the same here.
template <typename T>
void check_flat_dtype(DLManagedTensor* tensor, const char* name)
{
  auto dt = tensor->dl_tensor.dtype;
  RAFT_EXPECTS(dt.code == kDLFloat && dt.bits == sizeof(T) * 8,
               "%s must be a %d-bit float buffer matching the dtype of X",
               name,
               static_cast<int>(sizeof(T) * 8));
}

template <typename T>
raft::device_vector_view<T, int64_t> flat_device_view(DLManagedTensor* tensor, const char* name)
{
  check_flat_dtype<T>(tensor, name);
  return raft::make_device_vector_view<T, int64_t>(reinterpret_cast<T*>(tensor->dl_tensor.data),
                                                   tensor_numel(tensor, name));
}

template <typename T>
raft::device_vector_view<const T, int64_t> flat_device_view_const(DLManagedTensor* tensor,
                                                                  const char* name)
{
  check_flat_dtype<T>(tensor, name);
  return raft::make_device_vector_view<const T, int64_t>(
    reinterpret_cast<const T*>(tensor->dl_tensor.data), tensor_numel(tensor, name));
}

template <typename T>
void _fit(cuvsResources_t res,
          const cuvsGMMParams& params,
          DLManagedTensor* X_tensor,
          DLManagedTensor* weights_tensor,
          DLManagedTensor* means_tensor,
          DLManagedTensor* covariances_tensor,
          DLManagedTensor* precisions_chol_tensor,
          DLManagedTensor* precisions_tensor,
          DLManagedTensor* labels_tensor,
          double* lower_bound,
          int* n_iter,
          bool* converged,
          bool warm_start)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);
  if (!cuvs::core::is_dlpack_device_compatible(X_tensor->dl_tensor)) {
    RAFT_FAIL("X dataset must be accessible on device memory");
  }

  using const_matrix_type = raft::device_matrix_view<const T, int64_t, raft::row_major>;
  using matrix_type       = raft::device_matrix_view<T, int64_t, raft::row_major>;
  using labels_type       = raft::device_vector_view<int32_t, int64_t, raft::row_major>;

  T lower_bound_temp;
  int n_iter_temp;
  bool converged_temp;

  auto gmm_params = convert_params(params);
  cuvs::cluster::gmm::fit(*res_ptr,
                          gmm_params,
                          cuvs::core::from_dlpack<const_matrix_type>(X_tensor),
                          flat_device_view<T>(weights_tensor, "weights"),
                          cuvs::core::from_dlpack<matrix_type>(means_tensor),
                          flat_device_view<T>(covariances_tensor, "covariances"),
                          flat_device_view<T>(precisions_chol_tensor, "precisions_chol"),
                          flat_device_view<T>(precisions_tensor, "precisions"),
                          cuvs::core::from_dlpack<labels_type>(labels_tensor),
                          raft::make_host_scalar_view<T>(&lower_bound_temp),
                          raft::make_host_scalar_view<int>(&n_iter_temp),
                          raft::make_host_scalar_view<bool>(&converged_temp),
                          warm_start);

  *lower_bound = lower_bound_temp;
  *n_iter      = n_iter_temp;
  *converged   = converged_temp;
}

template <typename T>
void _predict(cuvsResources_t res,
              const cuvsGMMParams& params,
              DLManagedTensor* X_tensor,
              DLManagedTensor* weights_tensor,
              DLManagedTensor* means_tensor,
              DLManagedTensor* precisions_chol_tensor,
              DLManagedTensor* labels_tensor)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);
  if (!cuvs::core::is_dlpack_device_compatible(X_tensor->dl_tensor)) {
    RAFT_FAIL("X dataset must be accessible on device memory");
  }

  using const_matrix_type = raft::device_matrix_view<const T, int64_t, raft::row_major>;
  using labels_type       = raft::device_vector_view<int32_t, int64_t, raft::row_major>;

  auto gmm_params = convert_params(params);
  cuvs::cluster::gmm::predict(*res_ptr,
                              gmm_params,
                              cuvs::core::from_dlpack<const_matrix_type>(X_tensor),
                              flat_device_view_const<T>(weights_tensor, "weights"),
                              cuvs::core::from_dlpack<const_matrix_type>(means_tensor),
                              flat_device_view_const<T>(precisions_chol_tensor, "precisions_chol"),
                              cuvs::core::from_dlpack<labels_type>(labels_tensor));
}

template <typename T>
void _predict_proba(cuvsResources_t res,
                    const cuvsGMMParams& params,
                    DLManagedTensor* X_tensor,
                    DLManagedTensor* weights_tensor,
                    DLManagedTensor* means_tensor,
                    DLManagedTensor* precisions_chol_tensor,
                    DLManagedTensor* resp_tensor)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);
  if (!cuvs::core::is_dlpack_device_compatible(X_tensor->dl_tensor)) {
    RAFT_FAIL("X dataset must be accessible on device memory");
  }

  using const_matrix_type = raft::device_matrix_view<const T, int64_t, raft::row_major>;
  using matrix_type       = raft::device_matrix_view<T, int64_t, raft::row_major>;

  auto gmm_params = convert_params(params);
  cuvs::cluster::gmm::predict_proba(
    *res_ptr,
    gmm_params,
    cuvs::core::from_dlpack<const_matrix_type>(X_tensor),
    flat_device_view_const<T>(weights_tensor, "weights"),
    cuvs::core::from_dlpack<const_matrix_type>(means_tensor),
    flat_device_view_const<T>(precisions_chol_tensor, "precisions_chol"),
    cuvs::core::from_dlpack<matrix_type>(resp_tensor));
}

template <typename T>
void _score_samples(cuvsResources_t res,
                    const cuvsGMMParams& params,
                    DLManagedTensor* X_tensor,
                    DLManagedTensor* weights_tensor,
                    DLManagedTensor* means_tensor,
                    DLManagedTensor* precisions_chol_tensor,
                    DLManagedTensor* log_prob_norm_tensor)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);
  if (!cuvs::core::is_dlpack_device_compatible(X_tensor->dl_tensor)) {
    RAFT_FAIL("X dataset must be accessible on device memory");
  }

  using const_matrix_type = raft::device_matrix_view<const T, int64_t, raft::row_major>;

  auto gmm_params = convert_params(params);
  cuvs::cluster::gmm::score_samples(
    *res_ptr,
    gmm_params,
    cuvs::core::from_dlpack<const_matrix_type>(X_tensor),
    flat_device_view_const<T>(weights_tensor, "weights"),
    cuvs::core::from_dlpack<const_matrix_type>(means_tensor),
    flat_device_view_const<T>(precisions_chol_tensor, "precisions_chol"),
    flat_device_view<T>(log_prob_norm_tensor, "log_prob_norm"));
}

}  // namespace

extern "C" cuvsError_t cuvsGMMParamsCreate(cuvsGMMParams_t* params)
{
  return cuvs::core::translate_exceptions([=] {
    cuvs::cluster::gmm::params cpp_params;
    *params =
      new cuvsGMMParams{.n_components    = cpp_params.n_components,
                        .covariance_type = static_cast<cuvsGMMCovarianceType>(cpp_params.cov_type),
                        .tol             = cpp_params.tol,
                        .reg_covar       = cpp_params.reg_covar,
                        .max_iter        = cpp_params.max_iter,
                        .n_init          = cpp_params.n_init,
                        .init            = static_cast<cuvsGMMInitMethod>(cpp_params.init),
                        .seed            = cpp_params.seed};
  });
}

extern "C" cuvsError_t cuvsGMMParamsDestroy(cuvsGMMParams_t params)
{
  return cuvs::core::translate_exceptions([=] { delete params; });
}

extern "C" cuvsError_t cuvsGMMFit(cuvsResources_t res,
                                  cuvsGMMParams_t params,
                                  DLManagedTensor* X,
                                  DLManagedTensor* weights,
                                  DLManagedTensor* means,
                                  DLManagedTensor* covariances,
                                  DLManagedTensor* precisions_chol,
                                  DLManagedTensor* precisions,
                                  DLManagedTensor* labels,
                                  double* lower_bound,
                                  int* n_iter,
                                  bool* converged,
                                  bool warm_start)
{
  return cuvs::core::translate_exceptions([=] {
    auto dataset = X->dl_tensor;
    if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
      _fit<float>(res,
                  *params,
                  X,
                  weights,
                  means,
                  covariances,
                  precisions_chol,
                  precisions,
                  labels,
                  lower_bound,
                  n_iter,
                  converged,
                  warm_start);
    } else if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 64) {
      _fit<double>(res,
                   *params,
                   X,
                   weights,
                   means,
                   covariances,
                   precisions_chol,
                   precisions,
                   labels,
                   lower_bound,
                   n_iter,
                   converged,
                   warm_start);
    } else {
      RAFT_FAIL("Unsupported dataset DLtensor dtype: %d and bits: %d",
                dataset.dtype.code,
                dataset.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsGMMPredict(cuvsResources_t res,
                                      cuvsGMMParams_t params,
                                      DLManagedTensor* X,
                                      DLManagedTensor* weights,
                                      DLManagedTensor* means,
                                      DLManagedTensor* precisions_chol,
                                      DLManagedTensor* labels)
{
  return cuvs::core::translate_exceptions([=] {
    auto dataset = X->dl_tensor;
    if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
      _predict<float>(res, *params, X, weights, means, precisions_chol, labels);
    } else if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 64) {
      _predict<double>(res, *params, X, weights, means, precisions_chol, labels);
    } else {
      RAFT_FAIL("Unsupported dataset DLtensor dtype: %d and bits: %d",
                dataset.dtype.code,
                dataset.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsGMMPredictProba(cuvsResources_t res,
                                           cuvsGMMParams_t params,
                                           DLManagedTensor* X,
                                           DLManagedTensor* weights,
                                           DLManagedTensor* means,
                                           DLManagedTensor* precisions_chol,
                                           DLManagedTensor* resp)
{
  return cuvs::core::translate_exceptions([=] {
    auto dataset = X->dl_tensor;
    if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
      _predict_proba<float>(res, *params, X, weights, means, precisions_chol, resp);
    } else if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 64) {
      _predict_proba<double>(res, *params, X, weights, means, precisions_chol, resp);
    } else {
      RAFT_FAIL("Unsupported dataset DLtensor dtype: %d and bits: %d",
                dataset.dtype.code,
                dataset.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsGMMScoreSamples(cuvsResources_t res,
                                           cuvsGMMParams_t params,
                                           DLManagedTensor* X,
                                           DLManagedTensor* weights,
                                           DLManagedTensor* means,
                                           DLManagedTensor* precisions_chol,
                                           DLManagedTensor* log_prob_norm)
{
  return cuvs::core::translate_exceptions([=] {
    auto dataset = X->dl_tensor;
    if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
      _score_samples<float>(res, *params, X, weights, means, precisions_chol, log_prob_norm);
    } else if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 64) {
      _score_samples<double>(res, *params, X, weights, means, precisions_chol, log_prob_norm);
    } else {
      RAFT_FAIL("Unsupported dataset DLtensor dtype: %d and bits: %d",
                dataset.dtype.code,
                dataset.dtype.bits);
    }
  });
}
