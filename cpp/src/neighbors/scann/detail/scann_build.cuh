/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "../../../cluster/kmeans_balanced.cuh"
#include "../../detail/ann_utils.cuh"
#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/neighbors/scann.hpp>

#include <cub/cub.cuh>
#include <cuda_bf16.h>
#include <nvtx3/nvtx3.hpp>
#include <raft/cluster/kmeans.cuh>
#include <raft/cluster/kmeans_types.hpp>
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
#include <raft/linalg/unary_op.cuh>
#include <raft/matrix/argmin.cuh>
#include <raft/matrix/copy.cuh>
#include <raft/matrix/diagonal.cuh>
#include <raft/matrix/init.cuh>
#include <raft/matrix/sample_rows.cuh>
#include <raft/matrix/slice.cuh>
#include <raft/random/make_blobs.cuh>
#include <raft/random/rng.cuh>
#include <raft/sparse/neighbors/cross_component_nn.cuh>
#include <raft/util/memory_type_dispatcher.cuh>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/unique.h>

#include <cuvs/distance/distance.hpp>

#include "scann_avq.cuh"
#include "scann_common.cuh"
#include "scann_quantize.cuh"
#include "scann_soar.cuh"

#include <chrono>
#include <cstdio>
#include <vector>

namespace cuvs::neighbors::experimental::scann::detail {
using namespace cuvs::spatial::knn::detail;  // NOLINT

/* @defgroup scann_build_detail scann build
 * @{
 */

static const std::string RAFT_NAME = "raft";

template <typename T,
          typename IdxT     = int64_t,
          typename Accessor = raft::host_device_accessor<std::experimental::default_accessor<T>,
                                                         raft::memory_type::host>>
index<T, IdxT> build(
  raft::resources const& res,
  const index_params& params,
  raft::mdspan<const T, raft::matrix_extent<IdxT>, raft::row_major, Accessor> dataset)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(res);
  IdxT dim            = dataset.extent(1);

  RAFT_LOG_DEBUG("Creating empty index");

  index<T, IdxT> idx(res, params, dataset.extent(0), dataset.extent(1));
  raft::device_matrix_view<T, int64_t> centroids_view = idx.centers();
  raft::random::RngState random_state{137};
  // sample/project

  cuvs::cluster::kmeans::balanced_params kmeans_params;
  kmeans_params.n_iters = params.kmeans_n_iters;

  {
    auto trainset = raft::make_device_matrix<T, int64_t>(res, 0, 0);

    try {
      trainset =
        raft::make_device_matrix<T, int64_t>(res, params.kmeans_n_rows_train, dataset.extent(1));
    } catch (raft::logic_error& e) {
      RAFT_LOG_ERROR(
        "Insufficient device memory for kmeans training set allocation. Please "
        "reduce kmeans_n_rows_train.");
      throw;
    }

    RAFT_LOG_DEBUG("Sampling rows.\n");
    raft::matrix::sample_rows(res, random_state, dataset, trainset.view());

    raft::resource::sync_stream(res);

    // fit kmean

    RAFT_LOG_DEBUG("Fitting Kmeans");
    cuvs::cluster::kmeans_balanced::fit(
      res, kmeans_params, raft::make_const_mdspan(trainset.view()), centroids_view);
  }
  raft::resource::sync_stream(res);

  raft::device_vector_view<uint32_t, int64_t> labels_view = idx.labels();

  // setup batching for kmeans prediction + quantization
  auto* device_memory = raft::resource::get_workspace_resource(res);

  constexpr size_t kReasonableMaxBatchSize = 65536;
  size_t max_batch_size = std::min<size_t>(dataset.extent(0), kReasonableMaxBatchSize);
  auto copy_stream      = raft::resource::get_cuda_stream(res);
  bool enable_prefetch  = false;

  if (res.has_resource_factory(raft::resource::resource_type::CUDA_STREAM_POOL)) {
    if (raft::resource::get_stream_pool_size(res) >= 1) {
      enable_prefetch = true;

      copy_stream = raft::resource::get_stream_from_stream_pool(res);
    }
  }

  utils::batch_load_iterator<T> dataset_vec_batches(dataset.data_handle(),
                                                    dataset.extent(0),
                                                    dataset.extent(1),
                                                    max_batch_size,
                                                    copy_stream,
                                                    device_memory,
                                                    enable_prefetch);

  dataset_vec_batches.reset();
  dataset_vec_batches.prefetch_next_batch();

  RAFT_LOG_DEBUG("Batched kmeans prediction.\n");
  for (const auto& batch : dataset_vec_batches) {
    auto batch_view = raft::make_device_matrix_view<const T, int64_t>(
      batch.data(), batch.size(), dataset.extent(1));

    auto batch_labels_view = raft::make_device_vector_view<uint32_t, int64_t>(
      labels_view.data_handle() + batch.offset(), batch.size());

    cuvs::cluster::kmeans_balanced::predict(
      res, kmeans_params, batch_view, raft::make_const_mdspan(centroids_view), batch_labels_view);

    dataset_vec_batches.prefetch_next_batch();

    // Make sure work on device is finished before swapping buffers
    raft::resource::sync_stream(res);
  }

  // AVQ update of KMeans centroids
  apply_avq(res,
            dataset,
            centroids_view,
            raft::make_const_mdspan(labels_view),
            params.partitioning_eta,
            copy_stream);

  raft::device_vector_view<uint32_t, int64_t> soar_labels_view = idx.soar_labels();

  // Train PQ codebooks
  RAFT_LOG_DEBUG("Train PQ Codebooks\n");

  // Limit PQ training rows to 100k. IVF-PQ/VPQ limit to 65k as recall
  // doesn't increase significantly. ScaNN configs in common benchmarks use 100k,
  // so staying consistent with that.
  constexpr int64_t kMaxPQTrainRows = 100000;
  int64_t pq_n_rows_train           = std::min(params.pq_n_rows_train, kMaxPQTrainRows);

  auto trainset_residuals = sample_training_residuals(res,
                                                      random_state,
                                                      dataset,
                                                      raft::make_const_mdspan(centroids_view),
                                                      raft::make_const_mdspan(labels_view),
                                                      pq_n_rows_train);

  int num_subspaces    = dataset.extent(1) / params.pq_dim;
  int dim_per_subspace = params.pq_dim;
  int num_clusters     = 1 << params.pq_bits;

  auto full_codebook =
    raft::make_device_matrix<T, uint32_t>(res, num_clusters * num_subspaces, dim_per_subspace);

  // Loop each subspace, training codebooks for each
  for (int subspace = 0; subspace < num_subspaces; subspace++) {
    int sub_dim_start = subspace * dim_per_subspace;
    int sub_dim_end   = (subspace + 1) * dim_per_subspace;

    auto sub_trainset = raft::make_device_matrix<T, int64_t>(
      res, trainset_residuals.extent(0), (int64_t)dim_per_subspace);
    raft::matrix::slice_coordinates<int64_t> avq_sub_coords(
      0, sub_dim_start, trainset_residuals.extent(0), sub_dim_end);
    raft::matrix::slice(
      res, raft::make_const_mdspan(trainset_residuals.view()), sub_trainset.view(), avq_sub_coords);

    // Set up quantization bits and params
    cuvs::neighbors::vpq_params pq_params;
    pq_params.pq_bits = params.pq_bits;
    // For VPQ, pq_dim is the number of subspaces, not the dimension of the subspaces
    pq_params.pq_dim = 1;
    // We handle sampling/training set construction above, so use the full set in VPQ
    pq_params.pq_kmeans_trainset_fraction = 1.0;
    pq_params.kmeans_n_iters              = params.pq_train_iters;

    // Create pq codebook for this subspace
    auto sub_pq_codebook =
      create_pq_codebook<T>(res, raft::make_const_mdspan(sub_trainset.view()), pq_params);

    raft::copy(full_codebook.data_handle() + (subspace * sub_pq_codebook.size()),
               sub_pq_codebook.data_handle(),
               sub_pq_codebook.size(),
               stream);
  }
  raft::resource::sync_stream(res);

  // Set up quantization bits and params
  cuvs::neighbors::vpq_params pq_params;
  pq_params.pq_bits = params.pq_bits;
  pq_params.pq_dim  = dataset.extent(1) / params.pq_dim;

  dataset_vec_batches.reset();
  dataset_vec_batches.prefetch_next_batch();

  // Quantize residuals for both normal and SOAR assignments
  RAFT_LOG_DEBUG("Quantize residuals");
  for (const auto& batch : dataset_vec_batches) {
    auto batch_view = raft::make_device_matrix_view<const T, int64_t>(
      batch.data(), batch.size(), dataset.extent(1));

    auto batch_labels_view = raft::make_device_vector_view<const uint32_t, int64_t>(
      labels_view.data_handle() + batch.offset(), batch.size());

    auto batch_soar_labels_view = raft::make_device_vector_view<uint32_t, int64_t>(
      soar_labels_view.data_handle() + batch.offset(), batch.size());

    // Compute residuals for use in SOAR + quantization
    auto avq_residuals = compute_residuals<T, uint32_t>(
      res, batch_view, raft::make_const_mdspan(centroids_view), batch_labels_view);

    // Compute SOAR labels.
    // We compute SOAR labels in this loop to eliminate one HtoD copy of the full dataset.
    compute_soar_labels<T, uint32_t>(res,
                                     batch_view,
                                     raft::make_const_mdspan(avq_residuals.view()),
                                     centroids_view,
                                     batch_labels_view,
                                     batch_soar_labels_view,
                                     params.soar_lambda);

    // Compute and quantize residuals
    auto avq_quant = quantize_residuals<T, IdxT, uint32_t>(
      res, raft::make_const_mdspan(avq_residuals.view()), full_codebook.view(), pq_params);

    // Compute and quantize SOAR residuals
    auto soar_residuals =
      compute_residuals<T, uint32_t>(res,
                                     batch_view,
                                     raft::make_const_mdspan(centroids_view),
                                     raft::make_const_mdspan(batch_soar_labels_view));

    auto soar_quant = quantize_residuals<T, IdxT, uint32_t>(
      res, raft::make_const_mdspan(soar_residuals.view()), full_codebook.view(), pq_params);

    // unpack codes
    auto quantized_residuals =
      raft::make_device_matrix<uint8_t, IdxT>(res, batch.size(), num_subspaces);
    auto quantized_soar_residuals =
      raft::make_device_matrix<uint8_t, IdxT>(res, batch.size(), num_subspaces);

    unpack_codes(res,
                 quantized_residuals.view(),
                 raft::make_const_mdspan(avq_quant.view()),
                 params.pq_bits,
                 num_subspaces);
    unpack_codes(res,
                 quantized_soar_residuals.view(),
                 raft::make_const_mdspan(soar_quant.view()),
                 params.pq_bits,
                 num_subspaces);

    // quantize dataset to bfloat16, if enabled. Similar to SOAR, quantization
    // is performed in this loop to improve locality
    // TODO (rmaschal): Might be more efficient to do on CPU, to avoid DtoH copy
    auto bf16_dataset = raft::make_device_matrix<int16_t, int64_t>(res, batch_view.extent(0), dim);

    if (params.bf16_enabled) {
      raft::linalg::unaryOp(
        bf16_dataset.data_handle(),
        batch_view.data_handle(),
        batch_view.size(),
        [] __device__(T x) {
          nv_bfloat16 val = __float2bfloat16(x);
          return reinterpret_cast<int16_t&>(val);
        },
        resource::get_cuda_stream(res));
    }

    // Prefetch next batch
    dataset_vec_batches.prefetch_next_batch();

    // Copy unpacked codes to host
    // TODO (rmaschal): these copies are blocking and not overlapped
    raft::copy(idx.quantized_residuals().data_handle() + batch.offset() * num_subspaces,
               quantized_residuals.data_handle(),
               quantized_residuals.size(),
               stream);

    raft::copy(idx.quantized_soar_residuals().data_handle() + batch.offset() * num_subspaces,
               quantized_soar_residuals.data_handle(),
               quantized_soar_residuals.size(),
               stream);

    if (params.bf16_enabled) {
      raft::copy(idx.bf16_dataset().data_handle() + batch.offset() * dim,
                 bf16_dataset.data_handle(),
                 bf16_dataset.size(),
                 stream);
    }

    // Make sure work on device is finished before swapping buffers
    raft::resource::sync_stream(res);
  }

  // Codebooks from VPQ have the shape [subspace idx, subspace dim, code]
  // This converts the codebook into matrix format for easy interoperability
  // with open-source ScaNN search
  auto full_codebook_view = full_codebook.view();

  raft::linalg::map_offset(
    res,
    idx.pq_codebook(),
    [full_codebook_view, num_subspaces, dim_per_subspace, num_clusters] __device__(size_t i) {
      int codebook_dim   = num_subspaces * dim_per_subspace;
      int row_idx        = i / codebook_dim;
      int el_idx         = i % codebook_dim;
      int subspace_id    = el_idx / dim_per_subspace;
      int id_in_subspace = el_idx % dim_per_subspace;

      return full_codebook_view(row_idx + subspace_id * num_clusters, id_in_subspace);
    });

  raft::resource::sync_stream(res);

  return idx;
}

/**
 * @}
 */

}  // namespace cuvs::neighbors::experimental::scann::detail
