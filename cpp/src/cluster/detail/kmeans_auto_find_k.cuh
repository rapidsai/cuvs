/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "kmeans.cuh"
#include <cuvs/cluster/kmeans.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/stats/dispersion.cuh>

#include <thrust/host_vector.h>

namespace cuvs::cluster::kmeans::detail {

template <typename ValueT, typename IdxT>  // NOLINT(readability-identifier-naming)
void compute_dispersion(raft::resources const& handle,
                        raft::device_matrix_view<const ValueT, IdxT> X,
                        cuvs::cluster::kmeans::params& params,
                        raft::device_matrix_view<ValueT, IdxT> centroids_view,
                        raft::device_vector_view<IdxT, IdxT> labels,
                        raft::device_vector_view<IdxT, IdxT> cluster_sizes,
                        rmm::device_uvector<char>& workspace,
                        raft::host_vector_view<ValueT> cluster_dispertion_view,
                        raft::host_vector_view<ValueT> results_view,
                        raft::host_scalar_view<ValueT> residual,
                        raft::host_scalar_view<IdxT> n_iter,
                        int val,
                        IdxT n,
                        IdxT d)
{
  auto centroids_const_view =
    raft::make_device_matrix_view<const ValueT, IdxT>(centroids_view.data_handle(), val, d);

  IdxT* cluster_sizes_ptr = cluster_sizes.data_handle();
  auto cluster_sizes_view = raft::make_device_vector_view<const IdxT, IdxT>(cluster_sizes_ptr, val);

  params.n_clusters = val;

  cuvs::cluster::kmeans::fit_predict(
    handle, params, X, std::nullopt, std::make_optional(centroids_view), labels, residual, n_iter);

  detail::count_labels(
    handle, labels.data_handle(), cluster_sizes.data_handle(), n, val, workspace);

  results_view[val]            = residual[0];
  cluster_dispertion_view[val] = raft::stats::cluster_dispersion(
    handle, centroids_const_view, cluster_sizes_view, std::nullopt, n);
}

template <typename IdxT, typename ValueT>  // NOLINT(readability-identifier-naming)
void find_k(raft::resources const& handle,
            raft::device_matrix_view<const ValueT, IdxT> X,
            raft::host_scalar_view<IdxT> best_k,
            raft::host_scalar_view<ValueT> residual,
            raft::host_scalar_view<IdxT> n_iter,
            IdxT kmax,
            IdxT kmin    = 1,
            IdxT maxiter = 100,
            ValueT tol   = 1e-2)
{
  IdxT n = X.extent(0);
  IdxT d = X.extent(1);

  RAFT_EXPECTS(n >= 1, "n must be >= 1");
  RAFT_EXPECTS(d >= 1, "d must be >= 1");
  RAFT_EXPECTS(kmin >= 1, "kmin must be >= 1");
  RAFT_EXPECTS(kmax <= n, "kmax must be <= number of data samples in X");
  RAFT_EXPECTS(tol >= 0, "tolerance must be >= 0");
  RAFT_EXPECTS(maxiter >= 0, "maxiter must be >= 0");
  // Allocate memory
  // Device memory

  auto centroids     = raft::make_device_matrix<ValueT, IdxT>(handle, kmax, X.extent(1));
  auto cluster_sizes = raft::make_device_vector<IdxT>(handle, kmax);
  auto labels        = raft::make_device_vector<IdxT>(handle, n);

  rmm::device_uvector<char> workspace(0, raft::resource::get_cuda_stream(handle));

  IdxT* cluster_sizes_ptr = cluster_sizes.data_handle();

  // Host memory
  auto results            = raft::make_host_vector<ValueT>(kmax + 1);
  auto cluster_dispersion = raft::make_host_vector<ValueT>(kmax + 1);

  auto cluster_dispertion_view = cluster_dispersion.view();
  auto results_view            = results.view();

  // Loop to find *best* k
  // Perform k-means in binary search
  int left   = kmin;  // must be at least 2
  int right  = kmax;  // int(floor(len(data)/2)) #assumption of clusters of size 2 at least
  int mid    = (static_cast<unsigned int>(left) + static_cast<unsigned int>(right)) >> 1;
  int oldmid = mid;
  int tests  = 0;
  double objective[3];      // 0= left of mid, 1= right of mid
  if (left == 1) left = 2;  // at least do 2 clusters

  cuvs::cluster::kmeans::params params;
  params.max_iter = maxiter;
  params.tol      = tol;

  auto centroids_view =
    raft::make_device_matrix_view<ValueT, IdxT>(centroids.data_handle(), left, d);
  auto labels_view = raft::make_device_vector_view<IdxT, IdxT>(labels.data_handle(), n);
  auto cluster_sizes_view =
    raft::make_device_vector_view<IdxT, IdxT>(cluster_sizes.data_handle(), kmax);
  compute_dispersion<ValueT, IdxT>(handle,
                                   X,
                                   params,
                                   centroids_view,
                                   labels_view,
                                   cluster_sizes_view,
                                   workspace,
                                   cluster_dispertion_view,
                                   results_view,
                                   residual,
                                   n_iter,
                                   left,
                                   n,
                                   d);

  // eval right edge0
  results_view[right] = 1e20;
  while (results_view[right] > results_view[left] && tests < 3) {
    centroids_view = raft::make_device_matrix_view<ValueT, IdxT>(centroids.data_handle(), right, d);
    compute_dispersion<ValueT, IdxT>(handle,
                                     X,
                                     params,
                                     centroids_view,
                                     labels_view,
                                     cluster_sizes_view,
                                     workspace,
                                     cluster_dispertion_view,
                                     results_view,
                                     residual,
                                     n_iter,
                                     right,
                                     n,
                                     d);

    tests += 1;
  }

  objective[0] = (n - left) / (left - 1) * cluster_dispertion_view[left] / results_view[left];
  objective[1] = (n - right) / (right - 1) * cluster_dispertion_view[right] / results_view[right];
  while (left < right - 1) {
    results_view[mid] = 1e20;
    tests             = 0;
    while (results_view[mid] > results_view[left] && tests < 3) {
      centroids_view = raft::make_device_matrix_view<ValueT, IdxT>(centroids.data_handle(), mid, d);
      compute_dispersion<ValueT, IdxT>(handle,
                                       X,
                                       params,
                                       centroids_view,
                                       labels_view,
                                       cluster_sizes_view,
                                       workspace,
                                       cluster_dispertion_view,
                                       results_view,
                                       residual,
                                       n_iter,
                                       mid,
                                       n,
                                       d);

      if (results_view[mid] > results_view[left] && (mid + 1) < right) {
        mid += 1;
        results_view[mid] = 1e20;
      } else if (results_view[mid] > results_view[left] && (mid - 1) > left) {
        mid -= 1;
        results_view[mid] = 1e20;
      }
      tests += 1;
    }

    // maximize Calinski-Harabasz Index, minimize resid/ cluster
    objective[0] = (n - left) / (left - 1) * cluster_dispertion_view[left] / results_view[left];
    objective[1] = (n - right) / (right - 1) * cluster_dispertion_view[right] / results_view[right];
    objective[2] = (n - mid) / (mid - 1) * cluster_dispertion_view[mid] / results_view[mid];
    objective[0] = (objective[2] - objective[0]) / (mid - left);
    objective[1] = (objective[1] - objective[2]) / (right - mid);

    if (objective[0] > 0 && objective[1] < 0) {
      // our point is in the left-of-mid side
      right = mid;
    } else {
      left = mid;
    }
    oldmid = mid;
    mid    = (static_cast<unsigned int>(right) + static_cast<unsigned int>(left)) >> 1;
  }

  best_k[0]    = right;
  objective[0] = (n - left) / (left - 1) * cluster_dispertion_view[left] / results_view[left];
  objective[1] =
    (n - oldmid) / (oldmid - 1) * cluster_dispertion_view[oldmid] / results_view[oldmid];
  if (objective[1] < objective[0]) { best_k[0] = left; }

  // if best_k isn't what we just ran, re-run to get correct centroids and dist data on return->
  // this saves memory
  if (best_k[0] != oldmid) {
    auto centroids_view =
      raft::make_device_matrix_view<ValueT, IdxT>(centroids.data_handle(), best_k[0], d);

    params.n_clusters = best_k[0];
    cuvs::cluster::kmeans::fit_predict(handle,
                                       params,
                                       X,
                                       std::nullopt,
                                       std::make_optional(centroids_view),
                                       labels.view(),
                                       residual,
                                       n_iter);
  }
}
}  // namespace  cuvs::cluster::kmeans::detail
