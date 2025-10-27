/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/add.cuh>
#include <raft/linalg/coalesced_reduction.cuh>
#include <raft/linalg/dot.cuh>
#include <raft/linalg/gemm.hpp>
#include <raft/linalg/gemv.cuh>
#include <raft/linalg/linalg_types.hpp>
#include <raft/linalg/map.cuh>
#include <raft/linalg/matrix_vector.cuh>
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/linalg/multiply.cuh>
#include <raft/linalg/normalize.cuh>
#include <raft/linalg/power.cuh>
#include <raft/linalg/reduce.cuh>
#include <raft/linalg/transpose.cuh>
#include <raft/matrix/argmin.cuh>
#include <raft/matrix/copy.cuh>
#include <raft/matrix/diagonal.cuh>
#include <raft/matrix/init.cuh>

/**
 * @brief Compute SOAR labels for each dataset vector
 *
 * Compute a second, spilled cluster for each dataset vector by minimizing
 * the loss function in Theorem 3.1 of https://arxiv.org/abs/2404.00774
 *
 * @tparam T
 * @tparam LavelT
 * @param res raft resources
 * @param dataset the dataset, size [n_rows, dim]
 * @param residuals the residual vectors r, size [n_rows, dim]
 * @param centers the cluster centers, size [n_clusters, dim]
 * @param labels the cluster assignments, size [n_rows]
 * @param soar_labels the computed soar labels
 * @param lambda the weight for the projection of a residual r' onto r in the SOAR loss
 */
template <typename T, typename LabelT>
void compute_soar_labels(raft::resources const& dev_resources,
                         raft::device_matrix_view<const T, int64_t> dataset,
                         raft::device_matrix_view<const T, int64_t> residuals,
                         raft::device_matrix_view<T, int64_t> centers,
                         raft::device_vector_view<const LabelT, int64_t> labels,
                         raft::device_vector_view<LabelT, int64_t> soar_labels,
                         float lambda)
{
  auto dim = dataset.extent(1);

  // compute SOAR metric for each center
  auto soar_scores =
    raft::make_device_matrix<float, int64_t>(dev_resources, dataset.extent(0), centers.extent(0));
  auto n_centers = centers.extent(0);

  auto residuals_norm = raft::make_device_matrix<float, int64_t>(
    dev_resources, residuals.extent(0), residuals.extent(1));

  raft::linalg::row_normalize(dev_resources,
                              residuals,
                              residuals_norm.view(),
                              0.0f,
                              raft::sq_op(),
                              raft::add_op(),
                              raft::sqrt_op());

  int64_t n_rows = dataset.extent(0);

  auto cublas_handle = raft::resource::get_cublas_handle(dev_resources);

  auto alpha = raft::make_host_scalar<float>(1.0);
  auto beta  = raft::make_host_scalar<float>(0.0);

  auto dataset_dot_residual = raft::make_device_vector<float, int64_t>(dev_resources, n_rows);

  RAFT_CUBLAS_TRY(cublasSgemvStridedBatched(cublas_handle,
                                            CUBLAS_OP_N,
                                            1,
                                            dim,
                                            alpha.data_handle(),
                                            dataset.data_handle(),
                                            1,
                                            dim,
                                            residuals_norm.data_handle(),
                                            1,
                                            dim,
                                            beta.data_handle(),
                                            dataset_dot_residual.data_handle(),
                                            1,
                                            1,
                                            n_rows));

  auto centers_norm = raft::make_device_vector<float, int64_t>(dev_resources, centers.extent(0));
  auto centers_transpose =
    raft::make_device_matrix<T, int64_t>(dev_resources, centers.extent(1), centers.extent(0));

  raft::linalg::reduce<true, true>(centers_norm.data_handle(),
                                   centers.data_handle(),
                                   centers.extent(1),
                                   centers.extent(0),
                                   0.0f,
                                   raft::resource::get_cuda_stream(dev_resources),
                                   false,
                                   raft::sq_op(),
                                   raft::add_op(),
                                   raft::identity_op());

  raft::linalg::transpose(dev_resources, centers, centers_transpose.view());

  raft::linalg::gemm(
    dev_resources, residuals_norm.view(), centers_transpose.view(), soar_scores.view());

  raft::linalg::matrix_vector_op<raft::Apply::ALONG_COLUMNS>(
    dev_resources,
    raft::make_const_mdspan(soar_scores.view()),
    raft::make_const_mdspan(dataset_dot_residual.view()),
    soar_scores.view(),
    raft::sub_op());

  raft::linalg::map(
    dev_resources, raft::make_const_mdspan(soar_scores.view()), soar_scores.view(), raft::sq_op());

  raft::linalg::map(dev_resources,
                    raft::make_const_mdspan(soar_scores.view()),
                    soar_scores.view(),
                    raft::mul_const_op<float>(lambda));

  auto nc_dataset = raft::make_device_matrix_view<float, int64_t>(
    const_cast<float*>(dataset.data_handle()), dataset.extent(0), dataset.extent(1));

  alpha(0) = -2.0;
  beta(0)  = 1.0;

  raft::linalg::gemm<float,
                     int64_t,
                     raft::row_major,
                     raft::row_major,
                     raft::row_major,
                     uint32_t,
                     raft::host_scalar_view<float, uint32_t>>(dev_resources,
                                                              nc_dataset,
                                                              centers_transpose.view(),
                                                              soar_scores.view(),
                                                              alpha.view(),
                                                              beta.view());

  raft::linalg::matrix_vector_op<raft::Apply::ALONG_ROWS>(
    dev_resources,
    raft::make_const_mdspan(soar_scores.view()),
    raft::make_const_mdspan(centers_norm.view()),
    soar_scores.view(),
    raft::add_op());

  raft::matrix::argmin(dev_resources, raft::make_const_mdspan(soar_scores.view()), soar_labels);
}
