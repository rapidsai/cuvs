/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

#include "../../core/nvtx.hpp"
#include "../ivf_common.cuh"
#include "../ivf_list.cuh"
#include "ivf_pq_codepacking.cuh"
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>

#include "../detail/ann_utils.cuh"  // utils::mapping

// TODO (cjnolet): This should be using an exposed API instead of circumventing the public APIs.
#include "../../cluster/kmeans_balanced.cuh"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/cuda_stream_pool.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/add.cuh>
#include <raft/linalg/detail/qr.cuh>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/map.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/norm_types.hpp>
#include <raft/linalg/normalize.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/matrix/gather.cuh>
#include <raft/matrix/linewise_op.cuh>
#include <raft/matrix/sample_rows.cuh>
#include <raft/random/rng.cuh>
#include <raft/stats/histogram.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/device_atomics.cuh>
#include <raft/util/integer_utils.hpp>
#include <raft/util/pow2_utils.cuh>
#include <raft/util/vectorized.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>

#include <cuda_fp16.h>
#include <thrust/extrema.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>

#include <memory>
#include <variant>

namespace cuvs::neighbors::ivf_pq::detail {
using namespace cuvs::spatial::knn::detail;  // NOLINT

using internal_extents_t = int64_t;  // The default mdspan extent type used internally.

/**
 * @brief Compute residual vectors from the source dataset given by selected indices.
 *
 * The residual has the form `rotation_matrix %* (dataset[row_ids, :] - center)`
 *
 */
template <typename T, typename IdxT>
void select_residuals(raft::resources const& handle,
                      float* residuals,
                      IdxT n_rows,
                      uint32_t dim,
                      uint32_t rot_dim,
                      const float* rotation_matrix,  // [rot_dim, dim]
                      const float* center,           // [dim]
                      const T* dataset,              // [.., dim]
                      const IdxT* row_ids,           // [n_rows]
                      rmm::mr::device_memory_resource* device_memory

)
{
  auto stream = raft::resource::get_cuda_stream(handle);
  rmm::device_uvector<float> tmp(size_t(n_rows) * size_t(dim), stream, device_memory);
  // Note: the number of rows of the input dataset isn't actually n_rows, but matrix::gather doesn't
  // need to know it, any strictly positive number would work.
  thrust::transform_iterator<utils::mapping<float>, const T*, thrust::use_default, float>
    mapping_itr(dataset, utils::mapping<float>{});
  raft::matrix::gather(mapping_itr, (IdxT)dim, n_rows, row_ids, n_rows, tmp.data(), stream);

  raft::matrix::linewise_op<raft::Apply::ALONG_ROWS>(
    handle,
    raft::make_device_matrix_view<const T, IdxT>(tmp.data(), n_rows, dim),
    raft::make_device_matrix_view<T, IdxT>(tmp.data(), n_rows, dim),
    raft::sub_op{},
    raft::make_device_vector_view<const T, IdxT>(center, dim));

  float alpha = 1.0;
  float beta  = 0.0;
  raft::linalg::gemm(handle,
                     true,
                     false,
                     rot_dim,
                     n_rows,
                     dim,
                     &alpha,
                     rotation_matrix,
                     dim,
                     tmp.data(),
                     dim,
                     &beta,
                     residuals,
                     rot_dim,
                     stream);
}

/**
 * @brief Compute residual vectors from the source dataset given by selected indices.
 *
 * The residual has the form
 *  `rotation_matrix %* (dataset[:, :] - centers[labels[:], 0:dim])`
 *
 */
template <typename T, typename IdxT>
void flat_compute_residuals(
  raft::resources const& handle,
  float* residuals,  // [n_rows, rot_dim]
  IdxT n_rows,
  raft::device_matrix_view<const float, uint32_t, raft::row_major>
    rotation_matrix,                                                         // [rot_dim, dim]
  raft::device_matrix_view<const float, uint32_t, raft::row_major> centers,  // [n_lists, dim_ext]
  const T* dataset,                                                          // [n_rows, dim]
  std::variant<uint32_t, const uint32_t*> labels,                            // [n_rows]
  rmm::device_async_resource_ref device_memory)
{
  auto stream  = raft::resource::get_cuda_stream(handle);
  auto dim     = rotation_matrix.extent(1);
  auto rot_dim = rotation_matrix.extent(0);
  rmm::device_uvector<float> tmp(n_rows * dim, stream, device_memory);
  auto tmp_view = raft::make_device_vector_view<float, IdxT>(tmp.data(), tmp.size());
  raft::linalg::map_offset(handle, tmp_view, [centers, dataset, labels, dim] __device__(size_t i) {
    auto row_ix = i / dim;
    auto el_ix  = i % dim;
    auto label  = std::holds_alternative<uint32_t>(labels)
                    ? std::get<uint32_t>(labels)
                    : std::get<const uint32_t*>(labels)[row_ix];
    return utils::mapping<float>{}(dataset[i]) - centers(label, el_ix);
  });

  float alpha = 1.0f;
  float beta  = 0.0f;
  raft::linalg::gemm(handle,
                     true,
                     false,
                     rot_dim,
                     n_rows,
                     dim,
                     &alpha,
                     rotation_matrix.data_handle(),
                     dim,
                     tmp.data(),
                     dim,
                     &beta,
                     residuals,
                     rot_dim,
                     stream);
}

template <uint32_t BlockDim, typename IdxT>
__launch_bounds__(BlockDim) static __global__ void fill_indices_kernel(IdxT n_rows,
                                                                       IdxT* data_indices,
                                                                       IdxT* data_offsets,
                                                                       const uint32_t* labels)
{
  const auto i = IdxT(BlockDim) * IdxT(blockIdx.x) + IdxT(threadIdx.x);
  if (i >= n_rows) { return; }
  data_indices[atomicAdd<IdxT>(data_offsets + labels[i], 1)] = i;
}

/**
 * @brief Calculate cluster offsets and arrange data indices into clusters.
 *
 * @param n_rows
 * @param n_lists
 * @param[in] labels output of k-means prediction [n_rows]
 * @param[in] cluster_sizes [n_lists]
 * @param[out] cluster_offsets [n_lists+1]
 * @param[out] data_indices [n_rows]
 *
 * @return size of the largest cluster
 */
template <typename IdxT>
auto calculate_offsets_and_indices(IdxT n_rows,
                                   uint32_t n_lists,
                                   const uint32_t* labels,
                                   const uint32_t* cluster_sizes,
                                   IdxT* cluster_offsets,
                                   IdxT* data_indices,
                                   rmm::cuda_stream_view stream) -> uint32_t
{
  auto exec_policy = rmm::exec_policy(stream);
  // Calculate the offsets
  IdxT cumsum = 0;
  raft::update_device(cluster_offsets, &cumsum, 1, stream);
  thrust::inclusive_scan(
    exec_policy, cluster_sizes, cluster_sizes + n_lists, cluster_offsets + 1, raft::add_op{});
  raft::update_host(&cumsum, cluster_offsets + n_lists, 1, stream);
  uint32_t max_cluster_size =
    *thrust::max_element(exec_policy, cluster_sizes, cluster_sizes + n_lists);
  stream.synchronize();
  RAFT_EXPECTS(cumsum == n_rows, "cluster sizes do not add up.");
  RAFT_LOG_DEBUG("Max cluster size %d", max_cluster_size);
  rmm::device_uvector<IdxT> data_offsets_buf(n_lists, stream);
  auto data_offsets = data_offsets_buf.data();
  raft::copy(data_offsets, cluster_offsets, n_lists, stream);
  constexpr uint32_t n_threads = 128;  // NOLINT
  const IdxT n_blocks          = raft::div_rounding_up_unsafe(n_rows, n_threads);
  fill_indices_kernel<n_threads>
    <<<n_blocks, n_threads, 0, stream>>>(n_rows, data_indices, data_offsets, labels);
  return max_cluster_size;
}

template <typename IdxT>
void set_centers(raft::resources const& handle, index<IdxT>* index, const float* cluster_centers)
{
  auto stream         = raft::resource::get_cuda_stream(handle);
  auto* device_memory = raft::resource::get_workspace_resource(handle);

  // Make sure to have trailing zeroes between dim and dim_ext;
  // We rely on this to enable padded tensor gemm kernels during coarse search.
  cuvs::spatial::knn::detail::utils::memzero(
    index->centers().data_handle(), index->centers().size(), stream);
  // combine cluster_centers and their norms
  RAFT_CUDA_TRY(cudaMemcpy2DAsync(index->centers().data_handle(),
                                  sizeof(float) * index->dim_ext(),
                                  cluster_centers,
                                  sizeof(float) * index->dim(),
                                  sizeof(float) * index->dim(),
                                  index->n_lists(),
                                  cudaMemcpyDefault,
                                  stream));

  rmm::device_uvector<float> center_norms(index->n_lists(), stream, device_memory);
  raft::linalg::rowNorm<raft::linalg::L2Norm, true>(
    center_norms.data(), cluster_centers, index->dim(), index->n_lists(), stream);
  RAFT_CUDA_TRY(cudaMemcpy2DAsync(index->centers().data_handle() + index->dim(),
                                  sizeof(float) * index->dim_ext(),
                                  center_norms.data(),
                                  sizeof(float),
                                  sizeof(float),
                                  index->n_lists(),
                                  cudaMemcpyDefault,
                                  stream));

  //     Rotate cluster_centers
  float alpha = 1.0;
  float beta  = 0.0;
  raft::linalg::gemm(handle,
                     true,
                     false,
                     index->rot_dim(),
                     index->n_lists(),
                     index->dim(),
                     &alpha,
                     index->rotation_matrix().data_handle(),
                     index->dim(),
                     cluster_centers,
                     index->dim(),
                     &beta,
                     index->centers_rot().data_handle(),
                     index->rot_dim(),
                     raft::resource::get_cuda_stream(handle));
}

template <typename IdxT>
void transpose_pq_centers(const raft::resources& handle,
                          index<IdxT>& index,
                          const float* pq_centers_source)
{
  auto stream  = raft::resource::get_cuda_stream(handle);
  auto extents = index.pq_centers().extents();
  static_assert(extents.rank() == 3);
  auto extents_source =
    raft::make_extents<uint32_t>(extents.extent(0), extents.extent(2), extents.extent(1));
  auto span_source = raft::make_mdspan<const float, uint32_t, raft::row_major, false, true>(
    pq_centers_source, extents_source);
  auto pq_centers_view = raft::make_device_vector_view<float, IdxT>(
    index.pq_centers().data_handle(), index.pq_centers().size());
  raft::linalg::map_offset(handle, pq_centers_view, [span_source, extents] __device__(size_t i) {
    uint32_t ii[3];
    for (int r = 2; r > 0; r--) {
      ii[r] = i % extents.extent(r);
      i /= extents.extent(r);
    }
    ii[0] = i;
    return span_source(ii[0], ii[2], ii[1]);
  });
}

template <typename IdxT>
void train_per_subset(raft::resources const& handle,
                      index<IdxT>& index,
                      size_t n_rows,
                      const float* trainset,   // [n_rows, dim]
                      const uint32_t* labels,  // [n_rows]
                      uint32_t kmeans_n_iters,
                      uint32_t max_train_points_per_pq_code)
{
  auto stream        = raft::resource::get_cuda_stream(handle);
  auto device_memory = raft::resource::get_workspace_resource(handle);

  rmm::device_uvector<float> pq_centers_tmp(index.pq_centers().size(), stream, device_memory);
  // Subsampling the train set for codebook generation based on max_train_points_per_pq_code.
  size_t big_enough = max_train_points_per_pq_code * size_t(index.pq_book_size());
  auto pq_n_rows    = uint32_t(std::min(big_enough, n_rows));
  rmm::device_uvector<float> sub_trainset(
    pq_n_rows * size_t(index.pq_len()), stream, device_memory);
  rmm::device_uvector<uint32_t> sub_labels(pq_n_rows, stream, device_memory);

  rmm::device_uvector<uint32_t> pq_cluster_sizes(index.pq_book_size(), stream, device_memory);

  for (uint32_t j = 0; j < index.pq_dim(); j++) {
    raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> pq_per_subspace_scope(
      "ivf_pq::build::per_subspace[%u]", j);

    // Get the rotated cluster centers for each training vector.
    // This will be subtracted from the input vectors afterwards.
    utils::copy_selected<float, float, size_t, uint32_t>(
      pq_n_rows,
      index.pq_len(),
      index.centers_rot().data_handle() + index.pq_len() * j,
      labels,
      index.rot_dim(),
      sub_trainset.data(),
      index.pq_len(),
      stream);

    // sub_trainset is the slice of: rotate(trainset) - centers_rot
    float alpha = 1.0;
    float beta  = -1.0;
    raft::linalg::gemm(handle,
                       true,
                       false,
                       index.pq_len(),
                       pq_n_rows,
                       index.dim(),
                       &alpha,
                       index.rotation_matrix().data_handle() + index.dim() * index.pq_len() * j,
                       index.dim(),
                       trainset,
                       index.dim(),
                       &beta,
                       sub_trainset.data(),
                       index.pq_len(),
                       stream);

    // train PQ codebook for this subspace
    auto sub_trainset_view = raft::make_device_matrix_view<const float, internal_extents_t>(
      sub_trainset.data(), pq_n_rows, index.pq_len());
    auto centers_tmp_view = raft::make_device_matrix_view<float, internal_extents_t>(
      pq_centers_tmp.data() + index.pq_book_size() * index.pq_len() * j,
      index.pq_book_size(),
      index.pq_len());
    auto sub_labels_view =
      raft::make_device_vector_view<uint32_t, internal_extents_t>(sub_labels.data(), pq_n_rows);
    auto cluster_sizes_view = raft::make_device_vector_view<uint32_t, internal_extents_t>(
      pq_cluster_sizes.data(), index.pq_book_size());
    cuvs::cluster::kmeans::balanced_params kmeans_params;
    kmeans_params.n_iters = kmeans_n_iters;
    kmeans_params.metric  = cuvs::distance::DistanceType::L2Expanded;
    cuvs::cluster::kmeans_balanced::helpers::build_clusters(handle,
                                                            kmeans_params,
                                                            sub_trainset_view,
                                                            centers_tmp_view,
                                                            sub_labels_view,
                                                            cluster_sizes_view,
                                                            utils::mapping<float>{});
  }
  transpose_pq_centers(handle, index, pq_centers_tmp.data());
}

template <typename IdxT>
void train_per_cluster(raft::resources const& handle,
                       index<IdxT>& index,
                       size_t n_rows,
                       const float* trainset,   // [n_rows, dim]
                       const uint32_t* labels,  // [n_rows]
                       uint32_t kmeans_n_iters,
                       uint32_t max_train_points_per_pq_code)
{
  auto stream        = raft::resource::get_cuda_stream(handle);
  auto device_memory = raft::resource::get_workspace_resource(handle);
  // NB: Managed memory is used for small arrays accessed from both device and host. There's no
  // performance reasoning behind this, just avoiding the boilerplate of explicit copies.
  rmm::mr::managed_memory_resource managed_memory;

  rmm::device_uvector<float> pq_centers_tmp(index.pq_centers().size(), stream, device_memory);
  rmm::device_uvector<uint32_t> cluster_sizes(index.n_lists(), stream, managed_memory);
  rmm::device_uvector<IdxT> indices_buf(n_rows, stream, device_memory);
  rmm::device_uvector<IdxT> offsets_buf(index.n_lists() + 1, stream, managed_memory);

  raft::stats::histogram<uint32_t, size_t>(raft::stats::HistTypeAuto,
                                           reinterpret_cast<int32_t*>(cluster_sizes.data()),
                                           index.n_lists(),
                                           labels,
                                           n_rows,
                                           1,
                                           stream);

  auto cluster_offsets      = offsets_buf.data();
  auto indices              = indices_buf.data();
  uint32_t max_cluster_size = calculate_offsets_and_indices(
    IdxT(n_rows), index.n_lists(), labels, cluster_sizes.data(), cluster_offsets, indices, stream);

  rmm::device_uvector<uint32_t> pq_labels(
    size_t(max_cluster_size) * size_t(index.pq_dim()), stream, device_memory);
  rmm::device_uvector<uint32_t> pq_cluster_sizes(index.pq_book_size(), stream, device_memory);
  rmm::device_uvector<float> rot_vectors(
    size_t(max_cluster_size) * size_t(index.rot_dim()), stream, device_memory);

  raft::resource::sync_stream(handle);  // make sure cluster offsets are up-to-date
  for (uint32_t l = 0; l < index.n_lists(); l++) {
    auto cluster_size = cluster_sizes.data()[l];
    if (cluster_size == 0) continue;
    raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> pq_per_cluster_scope(
      "ivf_pq::build::per_cluster[%u](size = %u)", l, cluster_size);

    select_residuals(handle,
                     rot_vectors.data(),
                     IdxT(cluster_size),
                     index.dim(),
                     index.rot_dim(),
                     index.rotation_matrix().data_handle(),
                     index.centers().data_handle() + size_t(l) * size_t(index.dim_ext()),
                     trainset,
                     indices + cluster_offsets[l],
                     device_memory);

    // limit the cluster size to bound the training time based on max_train_points_per_pq_code
    // If pq_book_size is less than pq_dim, use max_train_points_per_pq_code per pq_dim instead
    // [sic] we interpret the data as pq_len-dimensional
    size_t big_enough =
      max_train_points_per_pq_code * std::max<size_t>(index.pq_book_size(), index.pq_dim());
    size_t available_rows = size_t(cluster_size) * size_t(index.pq_dim());
    auto pq_n_rows        = uint32_t(std::min(big_enough, available_rows));
    // train PQ codebook for this cluster
    auto rot_vectors_view = raft::make_device_matrix_view<const float, internal_extents_t>(
      rot_vectors.data(), pq_n_rows, index.pq_len());
    auto centers_tmp_view = raft::make_device_matrix_view<float, internal_extents_t>(
      pq_centers_tmp.data() + static_cast<size_t>(index.pq_book_size()) *
                                static_cast<size_t>(index.pq_len()) * static_cast<size_t>(l),
      index.pq_book_size(),
      index.pq_len());
    auto pq_labels_view =
      raft::make_device_vector_view<uint32_t, internal_extents_t>(pq_labels.data(), pq_n_rows);
    auto pq_cluster_sizes_view = raft::make_device_vector_view<uint32_t, internal_extents_t>(
      pq_cluster_sizes.data(), index.pq_book_size());
    cuvs::cluster::kmeans::balanced_params kmeans_params;
    kmeans_params.n_iters = kmeans_n_iters;
    kmeans_params.metric  = cuvs::distance::DistanceType::L2Expanded;
    cuvs::cluster::kmeans_balanced::helpers::build_clusters(handle,
                                                            kmeans_params,
                                                            rot_vectors_view,
                                                            centers_tmp_view,
                                                            pq_labels_view,
                                                            pq_cluster_sizes_view,
                                                            utils::mapping<float>{});
  }
  transpose_pq_centers(handle, index, pq_centers_tmp.data());
}

/**
 * A helper function: given the dataset in the rotated space
 *  [n_rows, rot_dim] = [n_rows, pq_dim * pq_len],
 * reinterpret the last dimension as two: [n_rows, pq_dim, pq_len]
 *
 * @tparam T
 * @tparam IdxT
 *
 * @param vectors input data [n_rows, rot_dim]
 * @param pq_centers codebook (used to infer the structure - pq_len)
 * @return reinterpreted vectors [n_rows, pq_dim, pq_len]
 */
template <typename T, typename IdxT>
static __device__ auto reinterpret_vectors(
  raft::device_matrix_view<T, IdxT, raft::row_major> vectors,
  raft::device_mdspan<const float, raft::extent_3d<uint32_t>, raft::row_major> pq_centers)
  -> raft::device_mdspan<T, raft::extent_3d<IdxT>, raft::row_major>
{
  const uint32_t pq_len = pq_centers.extent(1);
  const uint32_t pq_dim = vectors.extent(1) / pq_len;
  using layout_t        = typename decltype(vectors)::layout_type;
  using accessor_t      = typename decltype(vectors)::accessor_type;
  return raft::mdspan<T, raft::extent_3d<IdxT>, layout_t, accessor_t>(
    vectors.data_handle(), raft::extent_3d<IdxT>{vectors.extent(0), pq_dim, pq_len});
}

/**
 * A consumer for the `run_on_list` and `run_on_vector` that just flattens PQ codes
 * one-per-byte. That is, independent of the code width (pq_bits), one code uses
 * the whole byte, hence one vectors uses pq_dim bytes.
 */
struct unpack_codes {
  raft::device_matrix_view<uint8_t, uint32_t, raft::row_major> out_codes;

  /**
   * Create a callable to be passed to `run_on_list`.
   *
   * @param[out] out_codes the destination for the read codes.
   */
  __device__ inline unpack_codes(
    raft::device_matrix_view<uint8_t, uint32_t, raft::row_major> out_codes)
    : out_codes{out_codes}
  {
  }

  /**  Write j-th component (code) of the i-th vector into the output array. */
  __device__ inline void operator()(uint8_t code, uint32_t i, uint32_t j)
  {
    out_codes(i, j) = code;
  }
};

template <uint32_t BlockSize, uint32_t PqBits>
__launch_bounds__(BlockSize) static __global__ void unpack_list_data_kernel(
  raft::device_matrix_view<uint8_t, uint32_t, raft::row_major> out_codes,
  raft::device_mdspan<const uint8_t, list_spec<uint32_t, uint32_t>::list_extents, raft::row_major>
    in_list_data,
  std::variant<uint32_t, const uint32_t*> offset_or_indices)
{
  const uint32_t pq_dim = out_codes.extent(1);
  auto unpack_action    = unpack_codes{out_codes};
  run_on_list<PqBits>(in_list_data, offset_or_indices, out_codes.extent(0), pq_dim, unpack_action);
}

/**
 * Unpack flat PQ codes from an existing list by the given offset.
 *
 * @param[out] codes flat PQ codes, one code per byte [n_rows, pq_dim]
 * @param[in] list_data the packed ivf::list data.
 * @param[in] offset_or_indices how many records in the list to skip or the exact indices.
 * @param[in] pq_bits codebook size (1 << pq_bits)
 * @param[in] stream
 */
inline void unpack_list_data(
  raft::device_matrix_view<uint8_t, uint32_t, raft::row_major> codes,
  raft::device_mdspan<const uint8_t, list_spec<uint32_t, uint32_t>::list_extents, raft::row_major>
    list_data,
  std::variant<uint32_t, const uint32_t*> offset_or_indices,
  uint32_t pq_bits,
  rmm::cuda_stream_view stream)
{
  auto n_rows = codes.extent(0);
  if (n_rows == 0) { return; }

  constexpr uint32_t kBlockSize = 256;
  dim3 blocks(raft::div_rounding_up_safe<uint32_t>(n_rows, kBlockSize), 1, 1);
  dim3 threads(kBlockSize, 1, 1);
  auto kernel = [pq_bits]() {
    switch (pq_bits) {
      case 4: return unpack_list_data_kernel<kBlockSize, 4>;
      case 5: return unpack_list_data_kernel<kBlockSize, 5>;
      case 6: return unpack_list_data_kernel<kBlockSize, 6>;
      case 7: return unpack_list_data_kernel<kBlockSize, 7>;
      case 8: return unpack_list_data_kernel<kBlockSize, 8>;
      default: RAFT_FAIL("Invalid pq_bits (%u), the value must be within [4, 8]", pq_bits);
    }
  }();
  kernel<<<blocks, threads, 0, stream>>>(codes, list_data, offset_or_indices);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/** Unpack the list data; see the public interface for the api and usage. */
template <typename IdxT>
void unpack_list_data(raft::resources const& res,
                      const index<IdxT>& index,
                      raft::device_matrix_view<uint8_t, uint32_t, raft::row_major> out_codes,
                      uint32_t label,
                      std::variant<uint32_t, const uint32_t*> offset_or_indices)
{
  unpack_list_data(out_codes,
                   index.lists()[label]->data.view(),
                   offset_or_indices,
                   index.pq_bits(),
                   raft::resource::get_cuda_stream(res));
}

/**
 * A consumer for the `run_on_vector` that just flattens PQ codes
 * into a tightly packed matrix. That is, the codes are not expanded to one code-per-byte.
 */
template <uint32_t PqBits>
struct unpack_contiguous {
  uint8_t* codes;
  uint32_t code_size;

  /**
   * Create a callable to be passed to `run_on_vector`.
   *
   * @param[in] codes flat compressed PQ codes
   */
  __host__ __device__ inline unpack_contiguous(uint8_t* codes, uint32_t pq_dim)
    : codes{codes}, code_size{raft::ceildiv<uint32_t>(pq_dim * PqBits, 8)}
  {
  }

  /**  Write j-th component (code) of the i-th vector into the output array. */
  __host__ __device__ inline void operator()(uint8_t code, uint32_t i, uint32_t j)
  {
    bitfield_view_t<PqBits> code_view{codes + i * code_size};
    code_view[j] = code;
  }
};

template <uint32_t BlockSize, uint32_t PqBits>
__launch_bounds__(BlockSize) static __global__ void unpack_contiguous_list_data_kernel(
  uint8_t* out_codes,
  raft::device_mdspan<const uint8_t, list_spec<uint32_t, uint32_t>::list_extents, raft::row_major>
    in_list_data,
  uint32_t n_rows,
  uint32_t pq_dim,
  std::variant<uint32_t, const uint32_t*> offset_or_indices)
{
  run_on_list<PqBits>(
    in_list_data, offset_or_indices, n_rows, pq_dim, unpack_contiguous<PqBits>(out_codes, pq_dim));
}

/**
 * Unpack flat PQ codes from an existing list by the given offset.
 *
 * @param[out] codes flat compressed PQ codes [n_rows, ceildiv(pq_dim * pq_bits, 8)]
 * @param[in] list_data the packed ivf::list data.
 * @param[in] offset_or_indices how many records in the list to skip or the exact indices.
 * @param[in] pq_bits codebook size (1 << pq_bits)
 * @param[in] stream
 */
inline void unpack_contiguous_list_data(
  uint8_t* codes,
  raft::device_mdspan<const uint8_t, list_spec<uint32_t, uint32_t>::list_extents, raft::row_major>
    list_data,
  uint32_t n_rows,
  uint32_t pq_dim,
  std::variant<uint32_t, const uint32_t*> offset_or_indices,
  uint32_t pq_bits,
  rmm::cuda_stream_view stream)
{
  if (n_rows == 0) { return; }

  constexpr uint32_t kBlockSize = 256;
  dim3 blocks(raft::div_rounding_up_safe<uint32_t>(n_rows, kBlockSize), 1, 1);
  dim3 threads(kBlockSize, 1, 1);
  auto kernel = [pq_bits]() {
    switch (pq_bits) {
      case 4: return unpack_contiguous_list_data_kernel<kBlockSize, 4>;
      case 5: return unpack_contiguous_list_data_kernel<kBlockSize, 5>;
      case 6: return unpack_contiguous_list_data_kernel<kBlockSize, 6>;
      case 7: return unpack_contiguous_list_data_kernel<kBlockSize, 7>;
      case 8: return unpack_contiguous_list_data_kernel<kBlockSize, 8>;
      default: RAFT_FAIL("Invalid pq_bits (%u), the value must be within [4, 8]", pq_bits);
    }
  }();
  kernel<<<blocks, threads, 0, stream>>>(codes, list_data, n_rows, pq_dim, offset_or_indices);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/** Unpack the list data; see the public interface for the api and usage. */
template <typename IdxT>
void unpack_contiguous_list_data(raft::resources const& res,
                                 const index<IdxT>& index,
                                 uint8_t* out_codes,
                                 uint32_t n_rows,
                                 uint32_t label,
                                 std::variant<uint32_t, const uint32_t*> offset_or_indices)
{
  unpack_contiguous_list_data(out_codes,
                              index.lists()[label]->data.view(),
                              n_rows,
                              index.pq_dim(),
                              offset_or_indices,
                              index.pq_bits(),
                              raft::resource::get_cuda_stream(res));
}

/** A consumer for the `run_on_list` and `run_on_vector` that approximates the original input data.
 */
struct reconstruct_vectors {
  codebook_gen codebook_kind;
  uint32_t cluster_ix;
  uint32_t pq_len;
  raft::device_mdspan<const float, raft::extent_3d<uint32_t>, raft::row_major> pq_centers;
  raft::device_mdspan<const float, raft::extent_3d<uint32_t>, raft::row_major> centers_rot;
  raft::device_mdspan<float, raft::extent_3d<uint32_t>, raft::row_major> out_vectors;

  /**
   * Create a callable to be passed to `run_on_list`.
   *
   * @param[out] out_vectors the destination for the decoded vectors.
   * @param[in] pq_centers the codebook
   * @param[in] centers_rot
   * @param[in] codebook_kind
   * @param[in] cluster_ix label/id of the cluster.
   */
  __device__ inline reconstruct_vectors(
    raft::device_matrix_view<float, uint32_t, raft::row_major> out_vectors,
    raft::device_mdspan<const float, raft::extent_3d<uint32_t>, raft::row_major> pq_centers,
    raft::device_matrix_view<const float, uint32_t, raft::row_major> centers_rot,
    codebook_gen codebook_kind,
    uint32_t cluster_ix)
    : codebook_kind{codebook_kind},
      cluster_ix{cluster_ix},
      pq_len{pq_centers.extent(1)},
      pq_centers{pq_centers},
      centers_rot{reinterpret_vectors(centers_rot, pq_centers)},
      out_vectors{reinterpret_vectors(out_vectors, pq_centers)}
  {
  }

  /**
   * Decode j-th component of the i-th vector by its code and write it into a chunk of the output
   * vectors (pq_len elements).
   */
  __device__ inline void operator()(uint8_t code, uint32_t i, uint32_t j)
  {
    uint32_t partition_ix;
    switch (codebook_kind) {
      case codebook_gen::PER_CLUSTER: {
        partition_ix = cluster_ix;
      } break;
      case codebook_gen::PER_SUBSPACE: {
        partition_ix = j;
      } break;
      default: __builtin_unreachable();
    }
    for (uint32_t k = 0; k < pq_len; k++) {
      out_vectors(i, j, k) = pq_centers(partition_ix, k, code) + centers_rot(cluster_ix, j, k);
    }
  }
};

template <uint32_t BlockSize, uint32_t PqBits>
__launch_bounds__(BlockSize) static __global__ void reconstruct_list_data_kernel(
  raft::device_matrix_view<float, uint32_t, raft::row_major> out_vectors,
  raft::device_mdspan<const uint8_t, list_spec<uint32_t, uint32_t>::list_extents, raft::row_major>
    in_list_data,
  raft::device_mdspan<const float, raft::extent_3d<uint32_t>, raft::row_major> pq_centers,
  raft::device_matrix_view<const float, uint32_t, raft::row_major> centers_rot,
  codebook_gen codebook_kind,
  uint32_t cluster_ix,
  std::variant<uint32_t, const uint32_t*> offset_or_indices)
{
  const uint32_t pq_dim = out_vectors.extent(1) / pq_centers.extent(1);
  auto reconstruct_action =
    reconstruct_vectors{out_vectors, pq_centers, centers_rot, codebook_kind, cluster_ix};
  run_on_list<PqBits>(
    in_list_data, offset_or_indices, out_vectors.extent(0), pq_dim, reconstruct_action);
}

/** Decode the list data; see the public interface for the api and usage. */
template <typename T, typename IdxT>
void reconstruct_list_data(raft::resources const& res,
                           const index<IdxT>& index,
                           raft::device_matrix_view<T, uint32_t, raft::row_major> out_vectors,
                           uint32_t label,
                           std::variant<uint32_t, const uint32_t*> offset_or_indices)
{
  auto n_rows = out_vectors.extent(0);
  if (n_rows == 0) { return; }
  auto& list = index.lists()[label];
  if (std::holds_alternative<uint32_t>(offset_or_indices)) {
    auto n_skip = std::get<uint32_t>(offset_or_indices);
    // sic! I'm using the upper bound `list.size` instead of exact `list_sizes(label)`
    // to avoid an extra device-host data raft::copy and the stream sync.
    RAFT_EXPECTS(n_skip + n_rows <= list->size.load(),
                 "offset + output size must be not bigger than the cluster size.");
  }

  auto tmp =
    raft::make_device_mdarray<float>(res,
                                     raft::resource::get_workspace_resource(res),
                                     raft::make_extents<uint32_t>(n_rows, index.rot_dim()));

  constexpr uint32_t kBlockSize = 256;
  dim3 blocks(raft::div_rounding_up_safe<uint32_t>(n_rows, kBlockSize), 1, 1);
  dim3 threads(kBlockSize, 1, 1);
  auto kernel = [](uint32_t pq_bits) {
    switch (pq_bits) {
      case 4: return reconstruct_list_data_kernel<kBlockSize, 4>;
      case 5: return reconstruct_list_data_kernel<kBlockSize, 5>;
      case 6: return reconstruct_list_data_kernel<kBlockSize, 6>;
      case 7: return reconstruct_list_data_kernel<kBlockSize, 7>;
      case 8: return reconstruct_list_data_kernel<kBlockSize, 8>;
      default: RAFT_FAIL("Invalid pq_bits (%u), the value must be within [4, 8]", pq_bits);
    }
  }(index.pq_bits());
  kernel<<<blocks, threads, 0, raft::resource::get_cuda_stream(res)>>>(tmp.view(),
                                                                       list->data.view(),
                                                                       index.pq_centers(),
                                                                       index.centers_rot(),
                                                                       index.codebook_kind(),
                                                                       label,
                                                                       offset_or_indices);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  float* out_float_ptr = nullptr;
  rmm::device_uvector<float> out_float_buf(
    0, raft::resource::get_cuda_stream(res), raft::resource::get_workspace_resource(res));
  if constexpr (std::is_same_v<T, float>) {
    out_float_ptr = out_vectors.data_handle();
  } else {
    out_float_buf.resize(size_t{n_rows} * size_t{index.dim()},
                         raft::resource::get_cuda_stream(res));
    out_float_ptr = out_float_buf.data();
  }
  // Rotate the results back to the original space
  float alpha = 1.0;
  float beta  = 0.0;
  raft::linalg::gemm(res,
                     false,
                     false,
                     index.dim(),
                     n_rows,
                     index.rot_dim(),
                     &alpha,
                     index.rotation_matrix().data_handle(),
                     index.dim(),
                     tmp.data_handle(),
                     index.rot_dim(),
                     &beta,
                     out_float_ptr,
                     index.dim(),
                     raft::resource::get_cuda_stream(res));
  // Transform the data to the original type, if necessary
  if constexpr (!std::is_same_v<T, float>) {
    raft::linalg::map(
      res,
      out_vectors,
      utils::mapping<T>{},
      raft::make_device_matrix_view<const float>(out_float_ptr, n_rows, index.dim()));
  }
}

/**
 * A producer for the `write_list` and `write_vector` reads the codes byte-by-byte. That is,
 * independent of the code width (pq_bits), one code uses the whole byte, hence one vectors uses
 * pq_dim bytes.
 */
struct pass_codes {
  raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major> codes;

  /**
   * Create a callable to be passed to `run_on_list`.
   *
   * @param[in] codes the source codes.
   */
  __device__ inline pass_codes(
    raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major> codes)
    : codes{codes}
  {
  }

  /** Read j-th component (code) of the i-th vector from the source. */
  __device__ inline auto operator()(uint32_t i, uint32_t j) const -> uint8_t { return codes(i, j); }
};

template <uint32_t BlockSize, uint32_t PqBits>
__launch_bounds__(BlockSize) static __global__ void pack_list_data_kernel(
  raft::device_mdspan<uint8_t, list_spec<uint32_t, uint32_t>::list_extents, raft::row_major>
    list_data,
  raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major> codes,
  std::variant<uint32_t, const uint32_t*> offset_or_indices)
{
  write_list<PqBits, 1>(
    list_data, offset_or_indices, codes.extent(0), codes.extent(1), pass_codes{codes});
}

/**
 * Write flat PQ codes into an existing list by the given offset.
 *
 * NB: no memory allocation happens here; the list must fit the data (offset + n_rows).
 *
 * @param[out] list_data the packed ivf::list data.
 * @param[in] codes flat PQ codes, one code per byte [n_rows, pq_dim]
 * @param[in] offset_or_indices how many records in the list to skip or the exact indices.
 * @param[in] pq_bits codebook size (1 << pq_bits)
 * @param[in] stream
 */
inline void pack_list_data(
  raft::device_mdspan<uint8_t, list_spec<uint32_t, uint32_t>::list_extents, raft::row_major>
    list_data,
  raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major> codes,
  std::variant<uint32_t, const uint32_t*> offset_or_indices,
  uint32_t pq_bits,
  rmm::cuda_stream_view stream)
{
  auto n_rows = codes.extent(0);
  if (n_rows == 0) { return; }

  constexpr uint32_t kBlockSize = 256;
  dim3 blocks(raft::div_rounding_up_safe<uint32_t>(n_rows, kBlockSize), 1, 1);
  dim3 threads(kBlockSize, 1, 1);
  auto kernel = [pq_bits]() {
    switch (pq_bits) {
      case 4: return pack_list_data_kernel<kBlockSize, 4>;
      case 5: return pack_list_data_kernel<kBlockSize, 5>;
      case 6: return pack_list_data_kernel<kBlockSize, 6>;
      case 7: return pack_list_data_kernel<kBlockSize, 7>;
      case 8: return pack_list_data_kernel<kBlockSize, 8>;
      default: RAFT_FAIL("Invalid pq_bits (%u), the value must be within [4, 8]", pq_bits);
    }
  }();
  kernel<<<blocks, threads, 0, stream>>>(list_data, codes, offset_or_indices);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename IdxT>
void pack_list_data(raft::resources const& res,
                    index<IdxT>* index,
                    raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major> new_codes,
                    uint32_t label,
                    std::variant<uint32_t, const uint32_t*> offset_or_indices)
{
  pack_list_data(index->lists()[label]->data.view(),
                 new_codes,
                 offset_or_indices,
                 index->pq_bits(),
                 raft::resource::get_cuda_stream(res));
}

/**
 * A producer for the `write_vector` reads tightly packed flat codes. That is,
 * the codes are not expanded to one code-per-byte.
 */
template <uint32_t PqBits>
struct pack_contiguous {
  const uint8_t* codes;
  uint32_t code_size;

  /**
   * Create a callable to be passed to `write_vector`.
   *
   * @param[in] codes flat compressed PQ codes
   */
  __host__ __device__ inline pack_contiguous(const uint8_t* codes, uint32_t pq_dim)
    : codes{codes}, code_size{raft::ceildiv<uint32_t>(pq_dim * PqBits, 8)}
  {
  }

  /** Read j-th component (code) of the i-th vector from the source. */
  __host__ __device__ inline auto operator()(uint32_t i, uint32_t j) -> uint8_t
  {
    bitfield_view_t<PqBits> code_view{const_cast<uint8_t*>(codes + i * code_size)};
    return uint8_t(code_view[j]);
  }
};

template <uint32_t BlockSize, uint32_t PqBits>
__launch_bounds__(BlockSize) static __global__ void pack_contiguous_list_data_kernel(
  raft::device_mdspan<uint8_t, list_spec<uint32_t, uint32_t>::list_extents, raft::row_major>
    list_data,
  const uint8_t* codes,
  uint32_t n_rows,
  uint32_t pq_dim,
  std::variant<uint32_t, const uint32_t*> offset_or_indices)
{
  write_list<PqBits, 1>(
    list_data, offset_or_indices, n_rows, pq_dim, pack_contiguous<PqBits>(codes, pq_dim));
}

/**
 * Write flat PQ codes into an existing list by the given offset.
 *
 * NB: no memory allocation happens here; the list must fit the data (offset + n_rows).
 *
 * @param[out] list_data the packed ivf::list data.
 * @param[in] codes flat compressed PQ codes [n_rows, ceildiv(pq_dim * pq_bits, 8)]
 * @param[in] offset_or_indices how many records in the list to skip or the exact indices.
 * @param[in] pq_bits codebook size (1 << pq_bits)
 * @param[in] stream
 */
inline void pack_contiguous_list_data(
  raft::device_mdspan<uint8_t, list_spec<uint32_t, uint32_t>::list_extents, raft::row_major>
    list_data,
  const uint8_t* codes,
  uint32_t n_rows,
  uint32_t pq_dim,
  std::variant<uint32_t, const uint32_t*> offset_or_indices,
  uint32_t pq_bits,
  rmm::cuda_stream_view stream)
{
  if (n_rows == 0) { return; }

  constexpr uint32_t kBlockSize = 256;
  dim3 blocks(raft::div_rounding_up_safe<uint32_t>(n_rows, kBlockSize), 1, 1);
  dim3 threads(kBlockSize, 1, 1);
  auto kernel = [pq_bits]() {
    switch (pq_bits) {
      case 4: return pack_contiguous_list_data_kernel<kBlockSize, 4>;
      case 5: return pack_contiguous_list_data_kernel<kBlockSize, 5>;
      case 6: return pack_contiguous_list_data_kernel<kBlockSize, 6>;
      case 7: return pack_contiguous_list_data_kernel<kBlockSize, 7>;
      case 8: return pack_contiguous_list_data_kernel<kBlockSize, 8>;
      default: RAFT_FAIL("Invalid pq_bits (%u), the value must be within [4, 8]", pq_bits);
    }
  }();
  kernel<<<blocks, threads, 0, stream>>>(list_data, codes, n_rows, pq_dim, offset_or_indices);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename IdxT>
void pack_contiguous_list_data(raft::resources const& res,
                               index<IdxT>* index,
                               const uint8_t* new_codes,
                               uint32_t n_rows,
                               uint32_t label,
                               std::variant<uint32_t, const uint32_t*> offset_or_indices)
{
  pack_contiguous_list_data(index->lists()[label]->data.view(),
                            new_codes,
                            n_rows,
                            index->pq_dim(),
                            offset_or_indices,
                            index->pq_bits(),
                            raft::resource::get_cuda_stream(res));
}

/**
 *
 * A producer for the `write_list` and `write_vector` that encodes level-1 input vector residuals
 * into lvl-2 PQ codes.
 * Computing a PQ code means finding the closest cluster in a pq_dim-subspace.
 *
 * @tparam SubWarpSize
 *   how many threads work on a single vector;
 *   bounded by either WarpSize or pq_book_size.
 *
 * @param pq_centers
 *   - codebook_gen::PER_SUBSPACE: [pq_dim , pq_len, pq_book_size]
 *   - codebook_gen::PER_CLUSTER:  [n_lists, pq_len, pq_book_size]
 * @param new_vector a single input of length rot_dim, reinterpreted as [pq_dim, pq_len].
 *   the input must be already transformed to floats, rotated, and the level 1 cluster
 *   center must be already substructed (i.e. this is the residual of a single input vector).
 * @param codebook_kind
 * @param j index along pq_dim "dimension"
 * @param cluster_ix is used for PER_CLUSTER codebooks.
 */
/**
 */
template <uint32_t SubWarpSize, typename IdxT>
struct encode_vectors {
  codebook_gen codebook_kind;
  uint32_t cluster_ix;
  raft::device_mdspan<const float, raft::extent_3d<uint32_t>, raft::row_major> pq_centers;
  raft::device_mdspan<const float, raft::extent_3d<IdxT>, raft::row_major> in_vectors;

  __device__ inline encode_vectors(
    raft::device_mdspan<const float, raft::extent_3d<uint32_t>, raft::row_major> pq_centers,
    raft::device_matrix_view<const float, IdxT, raft::row_major> in_vectors,
    codebook_gen codebook_kind,
    uint32_t cluster_ix)
    : codebook_kind{codebook_kind},
      cluster_ix{cluster_ix},
      pq_centers{pq_centers},
      in_vectors{reinterpret_vectors(in_vectors, pq_centers)}
  {
  }

  /**
   * Decode j-th component of the i-th vector by its code and write it into a chunk of the output
   * vectors (pq_len elements).
   */
  __device__ inline auto operator()(IdxT i, uint32_t j) -> uint8_t
  {
    uint32_t lane_id = raft::Pow2<SubWarpSize>::mod(raft::laneId());
    uint32_t partition_ix;
    switch (codebook_kind) {
      case codebook_gen::PER_CLUSTER: {
        partition_ix = cluster_ix;
      } break;
      case codebook_gen::PER_SUBSPACE: {
        partition_ix = j;
      } break;
      default: __builtin_unreachable();
    }

    const uint32_t pq_book_size = pq_centers.extent(2);
    const uint32_t pq_len       = pq_centers.extent(1);
    float min_dist              = std::numeric_limits<float>::infinity();
    uint8_t code                = 0;
    // calculate the distance for each PQ cluster, find the minimum for each thread
    for (uint32_t l = lane_id; l < pq_book_size; l += SubWarpSize) {
      // NB: the L2 quantifiers on residuals are always trained on L2 metric.
      float d = 0.0f;
      for (uint32_t k = 0; k < pq_len; k++) {
        auto t = in_vectors(i, j, k) - pq_centers(partition_ix, k, l);
        d += t * t;
      }
      if (d < min_dist) {
        min_dist = d;
        code     = uint8_t(l);
      }
    }
    // reduce among threads
#pragma unroll
    for (uint32_t stride = SubWarpSize >> 1; stride > 0; stride >>= 1) {
      const auto other_dist = raft::shfl_xor(min_dist, stride, SubWarpSize);
      const auto other_code = raft::shfl_xor(code, stride, SubWarpSize);
      if (other_dist < min_dist) {
        min_dist = other_dist;
        code     = other_code;
      }
    }
    return code;
  }
};

template <uint32_t BlockSize, uint32_t PqBits, typename IdxT>
__launch_bounds__(BlockSize) static __global__ void process_and_fill_codes_kernel(
  raft::device_matrix_view<const float, IdxT, raft::row_major> new_vectors,
  std::variant<IdxT, const IdxT*> src_offset_or_indices,
  const uint32_t* new_labels,
  raft::device_vector_view<uint32_t, uint32_t, raft::row_major> list_sizes,
  raft::device_vector_view<IdxT*, uint32_t, raft::row_major> inds_ptrs,
  raft::device_vector_view<uint8_t*, uint32_t, raft::row_major> data_ptrs,
  raft::device_mdspan<const float, raft::extent_3d<uint32_t>, raft::row_major> pq_centers,
  codebook_gen codebook_kind)
{
  constexpr uint32_t kSubWarpSize = std::min<uint32_t>(raft::WarpSize, 1u << PqBits);
  using subwarp_align             = raft::Pow2<kSubWarpSize>;
  const uint32_t lane_id          = subwarp_align::mod(threadIdx.x);
  const IdxT row_ix = subwarp_align::div(IdxT{threadIdx.x} + IdxT{BlockSize} * IdxT{blockIdx.x});
  if (row_ix >= new_vectors.extent(0)) { return; }

  const uint32_t cluster_ix = new_labels[row_ix];
  uint32_t out_ix;
  if (lane_id == 0) { out_ix = atomicAdd(&list_sizes(cluster_ix), 1); }
  out_ix = raft::shfl(out_ix, 0, kSubWarpSize);

  // write the label  (one record per subwarp)
  auto pq_indices = inds_ptrs(cluster_ix);
  if (lane_id == 0) {
    if (std::holds_alternative<IdxT>(src_offset_or_indices)) {
      pq_indices[out_ix] = std::get<IdxT>(src_offset_or_indices) + row_ix;
    } else {
      pq_indices[out_ix] = std::get<const IdxT*>(src_offset_or_indices)[row_ix];
    }
  }

  // write the codes (one record per subwarp):
  const uint32_t pq_dim = new_vectors.extent(1) / pq_centers.extent(1);
  auto pq_extents = list_spec<uint32_t, IdxT>{PqBits, pq_dim, true}.make_list_extents(out_ix + 1);
  auto pq_dataset = raft::make_mdspan<uint8_t, uint32_t, raft::row_major, false, true>(
    data_ptrs[cluster_ix], pq_extents);
  write_vector<PqBits, kSubWarpSize>(
    pq_dataset,
    out_ix,
    row_ix,
    pq_dim,
    encode_vectors<kSubWarpSize, IdxT>{pq_centers, new_vectors, codebook_kind, cluster_ix});
}

template <uint32_t BlockSize, uint32_t PqBits>
__launch_bounds__(BlockSize) static __global__ void encode_list_data_kernel(
  raft::device_mdspan<uint8_t, list_spec<uint32_t, uint32_t>::list_extents, raft::row_major>
    list_data,
  raft::device_matrix_view<const float, uint32_t, raft::row_major> new_vectors,
  raft::device_mdspan<const float, raft::extent_3d<uint32_t>, raft::row_major> pq_centers,
  codebook_gen codebook_kind,
  uint32_t cluster_ix,
  std::variant<uint32_t, const uint32_t*> offset_or_indices)
{
  constexpr uint32_t kSubWarpSize = std::min<uint32_t>(raft::WarpSize, 1u << PqBits);
  const uint32_t pq_dim           = new_vectors.extent(1) / pq_centers.extent(1);
  auto encode_action =
    encode_vectors<kSubWarpSize, uint32_t>{pq_centers, new_vectors, codebook_kind, cluster_ix};
  write_list<PqBits, kSubWarpSize>(
    list_data, offset_or_indices, new_vectors.extent(0), pq_dim, encode_action);
}

template <typename T, typename IdxT>
void encode_list_data(raft::resources const& res,
                      index<IdxT>* index,
                      raft::device_matrix_view<const T, uint32_t, raft::row_major> new_vectors,
                      uint32_t label,
                      std::variant<uint32_t, const uint32_t*> offset_or_indices)
{
  auto n_rows = new_vectors.extent(0);
  if (n_rows == 0) { return; }

  auto mr = raft::resource::get_workspace_resource(res);

  auto new_vectors_residual = raft::make_device_mdarray<float>(
    res, mr, raft::make_extents<uint32_t>(n_rows, index->rot_dim()));

  flat_compute_residuals<T, uint32_t>(res,
                                      new_vectors_residual.data_handle(),
                                      n_rows,
                                      index->rotation_matrix(),
                                      index->centers(),
                                      new_vectors.data_handle(),
                                      label,
                                      mr);

  constexpr uint32_t kBlockSize  = 256;
  const uint32_t threads_per_vec = std::min<uint32_t>(raft::WarpSize, index->pq_book_size());
  dim3 blocks(raft::div_rounding_up_safe<uint32_t>(n_rows, kBlockSize / threads_per_vec), 1, 1);
  dim3 threads(kBlockSize, 1, 1);
  auto kernel = [](uint32_t pq_bits) {
    switch (pq_bits) {
      case 4: return encode_list_data_kernel<kBlockSize, 4>;
      case 5: return encode_list_data_kernel<kBlockSize, 5>;
      case 6: return encode_list_data_kernel<kBlockSize, 6>;
      case 7: return encode_list_data_kernel<kBlockSize, 7>;
      case 8: return encode_list_data_kernel<kBlockSize, 8>;
      default: RAFT_FAIL("Invalid pq_bits (%u), the value must be within [4, 8]", pq_bits);
    }
  }(index->pq_bits());
  kernel<<<blocks, threads, 0, raft::resource::get_cuda_stream(res)>>>(
    index->lists()[label]->data.view(),
    new_vectors_residual.view(),
    index->pq_centers(),
    index->codebook_kind(),
    label,
    offset_or_indices);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/**
 * Assuming the index already has some data and allocated the space for more, write more data in it.
 * There must be enough free space in `pq_dataset()` and `indices()`, as computed using
 * `list_offsets()` and `list_sizes()`.
 *
 * NB: Since the pq_dataset is stored in the interleaved blocked format (see ivf_pq_types.hpp), one
 * cannot just concatenate the old and the new codes; the positions for the codes are determined the
 * same way as in the ivfpq_compute_similarity_kernel (see ivf_pq_search.cuh).
 *
 * @tparam T
 * @tparam IdxT
 *
 * @param handle
 * @param index
 * @param[in] new_vectors
 *    a pointer to a row-major device array [index.dim(), n_rows];
 * @param[in] src_offset_or_indices
 *    references for the new data:
 *      either a starting index for the auto-indexing
 *      or a pointer to a device array of explicit indices [n_rows];
 * @param[in] new_labels
 *    cluster ids (first-level quantization) - a device array [n_rows];
 * @param n_rows
 *    the number of records to write in.
 * @param mr
 *    a memory resource to use for device allocations
 */
template <typename T, typename IdxT>
void process_and_fill_codes(raft::resources const& handle,
                            index<IdxT>& index,
                            const T* new_vectors,
                            std::variant<IdxT, const IdxT*> src_offset_or_indices,
                            const uint32_t* new_labels,
                            IdxT n_rows,
                            rmm::device_async_resource_ref mr)
{
  auto new_vectors_residual =
    raft::make_device_mdarray<float>(handle, mr, raft::make_extents<IdxT>(n_rows, index.rot_dim()));

  flat_compute_residuals<T, IdxT>(handle,
                                  new_vectors_residual.data_handle(),
                                  n_rows,
                                  index.rotation_matrix(),
                                  index.centers(),
                                  new_vectors,
                                  new_labels,
                                  mr);

  constexpr uint32_t kBlockSize  = 256;
  const uint32_t threads_per_vec = std::min<uint32_t>(raft::WarpSize, index.pq_book_size());
  dim3 blocks(raft::div_rounding_up_safe<IdxT>(n_rows, kBlockSize / threads_per_vec), 1, 1);
  dim3 threads(kBlockSize, 1, 1);
  auto kernel = [](uint32_t pq_bits) {
    switch (pq_bits) {
      case 4: return process_and_fill_codes_kernel<kBlockSize, 4, IdxT>;
      case 5: return process_and_fill_codes_kernel<kBlockSize, 5, IdxT>;
      case 6: return process_and_fill_codes_kernel<kBlockSize, 6, IdxT>;
      case 7: return process_and_fill_codes_kernel<kBlockSize, 7, IdxT>;
      case 8: return process_and_fill_codes_kernel<kBlockSize, 8, IdxT>;
      default: RAFT_FAIL("Invalid pq_bits (%u), the value must be within [4, 8]", pq_bits);
    }
  }(index.pq_bits());
  kernel<<<blocks, threads, 0, raft::resource::get_cuda_stream(handle)>>>(
    new_vectors_residual.view(),
    src_offset_or_indices,
    new_labels,
    index.list_sizes(),
    index.inds_ptrs(),
    index.data_ptrs(),
    index.pq_centers(),
    index.codebook_kind());
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/**
 * Helper function: allocate enough space in the list, compute the offset, at which to start
 * writing, and fill-in indices.
 *
 * @return offset for writing the data
 */
template <typename IdxT>
auto extend_list_prepare(
  raft::resources const& res,
  index<IdxT>* index,
  raft::device_vector_view<const IdxT, uint32_t, raft::row_major> new_indices,
  uint32_t label) -> uint32_t
{
  uint32_t n_rows = new_indices.extent(0);
  uint32_t offset;
  // Allocate the lists to fit the new data
  raft::copy(
    &offset, index->list_sizes().data_handle() + label, 1, raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);
  uint32_t new_size = offset + n_rows;
  raft::copy(
    index->list_sizes().data_handle() + label, &new_size, 1, raft::resource::get_cuda_stream(res));
  auto spec = list_spec<uint32_t, IdxT>{
    index->pq_bits(), index->pq_dim(), index->conservative_memory_allocation()};
  auto& list = index->lists()[label];
  ivf::resize_list(res, list, spec, new_size, offset);
  raft::copy(list->indices.data_handle() + offset,
             new_indices.data_handle(),
             n_rows,
             raft::resource::get_cuda_stream(res));
  return offset;
}

/**
 * Extend one list of the index in-place, by the list label, skipping the classification and
 * encoding steps.
 * See the public interface for the api and usage.
 */
template <typename IdxT>
void extend_list_with_codes(
  raft::resources const& res,
  index<IdxT>* index,
  raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major> new_codes,
  raft::device_vector_view<const IdxT, uint32_t, raft::row_major> new_indices,
  uint32_t label)
{
  // Allocate memory and write indices
  auto offset = extend_list_prepare(res, index, new_indices, label);
  // Pack the data
  pack_list_data(res, index, new_codes, label, offset);
  // Update the pointers and the sizes
  ivf::detail::recompute_internal_state(res, *index);
}

/**
 * Extend one list of the index in-place, by the list label, skipping the classification step.
 * See the public interface for the api and usage.
 */
template <typename T, typename IdxT>
void extend_list(raft::resources const& res,
                 index<IdxT>* index,
                 raft::device_matrix_view<const T, uint32_t, raft::row_major> new_vectors,
                 raft::device_vector_view<const IdxT, uint32_t, raft::row_major> new_indices,
                 uint32_t label)
{
  // Allocate memory and write indices
  auto offset = extend_list_prepare(res, index, new_indices, label);
  // Encode the data
  encode_list_data<T, IdxT>(res, index, new_vectors, label, offset);
  // Update the pointers and the sizes
  ivf::detail::recompute_internal_state(res, *index);
}

/**
 * Remove all data from a single list.
 * See the public interface for the api and usage.
 */
template <typename IdxT>
void erase_list(raft::resources const& res, index<IdxT>* index, uint32_t label)
{
  uint32_t zero = 0;
  raft::copy(
    index->list_sizes().data_handle() + label, &zero, 1, raft::resource::get_cuda_stream(res));
  index->lists()[label].reset();
  ivf::detail::recompute_internal_state(res, *index);
}

/** raft::copy the state of an index into a new index, but share the list data among the two. */
template <typename IdxT>
auto clone(const raft::resources& res, const index<IdxT>& source) -> index<IdxT>
{
  auto stream = raft::resource::get_cuda_stream(res);

  // Allocate the new index
  index<IdxT> target(res,
                     source.metric(),
                     source.codebook_kind(),
                     source.n_lists(),
                     source.dim(),
                     source.pq_bits(),
                     source.pq_dim());

  // raft::copy the independent parts
  raft::copy(target.list_sizes().data_handle(),
             source.list_sizes().data_handle(),
             source.list_sizes().size(),
             stream);
  raft::copy(target.rotation_matrix().data_handle(),
             source.rotation_matrix().data_handle(),
             source.rotation_matrix().size(),
             stream);
  raft::copy(target.pq_centers().data_handle(),
             source.pq_centers().data_handle(),
             source.pq_centers().size(),
             stream);
  raft::copy(target.centers().data_handle(),
             source.centers().data_handle(),
             source.centers().size(),
             stream);
  raft::copy(target.centers_rot().data_handle(),
             source.centers_rot().data_handle(),
             source.centers_rot().size(),
             stream);

  // raft::copy shared pointers
  target.lists() = source.lists();

  // Make sure the device pointers point to the new lists
  ivf::detail::recompute_internal_state(res, target);

  return target;
}

/**
 * Extend the index in-place.
 * See raft::spatial::knn::ivf_pq::extend docs.
 */
template <typename T, typename IdxT>
void extend(raft::resources const& handle,
            index<IdxT>* index,
            const T* new_vectors,
            const IdxT* new_indices,
            IdxT n_rows)
{
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "ivf_pq::extend(%zu, %u)", size_t(n_rows), index->dim());

  auto stream           = raft::resource::get_cuda_stream(handle);
  const auto n_clusters = index->n_lists();

  RAFT_EXPECTS(new_indices != nullptr || index->size() == 0,
               "You must pass data indices when the index is non-empty.");

  static_assert(std::is_same_v<T, float> || std::is_same_v<T, half> || std::is_same_v<T, uint8_t> ||
                  std::is_same_v<T, int8_t>,
                "Unsupported data type");

  if (index->metric() == distance::DistanceType::CosineExpanded) {
    if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>)
      RAFT_FAIL(
        "CosineExpanded distance metric is currently not supported for uint8_t and int8_t data "
        "type");
  }

  rmm::device_async_resource_ref device_memory = raft::resource::get_workspace_resource(handle);
  rmm::device_async_resource_ref large_memory =
    raft::resource::get_large_workspace_resource(handle);

  // The spec defines how the clusters look like
  auto spec = list_spec<uint32_t, IdxT>{
    index->pq_bits(), index->pq_dim(), index->conservative_memory_allocation()};
  // Try to allocate an index with the same parameters and the projected new size
  // (which can be slightly larger than index->size() + n_rows, due to padding).
  // If this fails, the index would be too big to fit in the device anyway.
  std::optional<list_data<IdxT, size_t>> placeholder_list(
    std::in_place_t{},
    handle,
    list_spec<size_t, IdxT>{spec},
    n_rows + (kIndexGroupSize - 1) * std::min<IdxT>(n_clusters, n_rows));

  // Available device memory
  size_t free_mem = raft::resource::get_workspace_free_bytes(handle);

  // We try to use the workspace memory by default here.
  // If the workspace limit is too small, we change the resource for batch data to the
  // `large_workspace_resource`, which does not have the explicit allocation limit. The user may opt
  // to populate the `large_workspace_resource` memory resource with managed memory for easier
  // scaling.
  rmm::device_async_resource_ref labels_mr  = device_memory;
  rmm::device_async_resource_ref batches_mr = device_memory;
  if (n_rows * (index->dim() * sizeof(T) + index->pq_dim() + sizeof(IdxT) + sizeof(uint32_t)) >
      free_mem) {
    labels_mr = large_memory;
  }
  // Allocate a buffer for the new labels (classifying the new data)
  rmm::device_uvector<uint32_t> new_data_labels(n_rows, stream, labels_mr);
  free_mem = raft::resource::get_workspace_free_bytes(handle);

  // Calculate the batch size for the input data if it's not accessible directly from the device
  constexpr size_t kReasonableMaxBatchSize = 65536;
  size_t max_batch_size                    = std::min<size_t>(n_rows, kReasonableMaxBatchSize);
  {
    size_t size_factor = 0;
    // we'll use two temporary buffers for converted inputs when computing the codes.
    size_factor += (index->dim() + index->rot_dim()) * sizeof(float);
    // ...and another buffer for indices
    size_factor += sizeof(IdxT);
    // if the input data is not accessible on device, we'd need a buffer for it.
    switch (utils::check_pointer_residency(new_vectors)) {
      case utils::pointer_residency::device_only:
      case utils::pointer_residency::host_and_device: break;
      default: size_factor += index->dim() * sizeof(T);
    }
    // the same with indices
    if (new_indices != nullptr) {
      switch (utils::check_pointer_residency(new_indices)) {
        case utils::pointer_residency::device_only:
        case utils::pointer_residency::host_and_device: break;
        default: size_factor += sizeof(IdxT);
      }
    }
    // make the batch size fit into the remaining memory
    while (size_factor * max_batch_size > free_mem && max_batch_size > 128) {
      max_batch_size >>= 1;
    }
    if (size_factor * max_batch_size > free_mem) {
      // if that still doesn't fit, resort to the UVM
      batches_mr     = large_memory;
      max_batch_size = kReasonableMaxBatchSize;
    } else {
      // If we're keeping the batches in device memory, update the available mem tracker.
      free_mem -= size_factor * max_batch_size;
    }
  }

  // Determine if a stream pool exist and make sure there is at least one stream in it so we
  // could use the stream for kernel/copy overlapping by enabling prefetch.
  auto copy_stream = raft::resource::get_cuda_stream(handle);  // Using the main stream by default
  bool enable_prefetch = false;
  if (handle.has_resource_factory(raft::resource::resource_type::CUDA_STREAM_POOL)) {
    if (raft::resource::get_stream_pool_size(handle) >= 1) {
      enable_prefetch = true;
      copy_stream     = raft::resource::get_stream_from_stream_pool(handle);
    }
  }
  // Predict the cluster labels for the new data, in batches if necessary
  utils::batch_load_iterator<T> vec_batches(
    new_vectors, n_rows, index->dim(), max_batch_size, copy_stream, device_memory, enable_prefetch);
  // Release the placeholder memory, because we don't intend to allocate any more long-living
  // temporary buffers before we allocate the index data.
  // This memory could potentially speed up UVM accesses, if any.
  placeholder_list.reset();
  {
    // The cluster centers in the index are stored padded, which is not acceptable by
    // the kmeans_balanced::predict. Thus, we need the restructuring raft::copy.
    rmm::device_uvector<float> cluster_centers(
      size_t(n_clusters) * size_t(index->dim()), stream, device_memory);
    RAFT_CUDA_TRY(cudaMemcpy2DAsync(cluster_centers.data(),
                                    sizeof(float) * index->dim(),
                                    index->centers().data_handle(),
                                    sizeof(float) * index->dim_ext(),
                                    sizeof(float) * index->dim(),
                                    n_clusters,
                                    cudaMemcpyDefault,
                                    stream));
    vec_batches.prefetch_next_batch();
    for (const auto& batch : vec_batches) {
      auto batch_data_view = raft::make_device_matrix_view<const T, internal_extents_t>(
        batch.data(), batch.size(), index->dim());
      auto batch_labels_view = raft::make_device_vector_view<uint32_t, internal_extents_t>(
        new_data_labels.data() + batch.offset(), batch.size());
      auto centers_view = raft::make_device_matrix_view<const float, internal_extents_t>(
        cluster_centers.data(), n_clusters, index->dim());
      cuvs::cluster::kmeans::balanced_params kmeans_params;
      kmeans_params.metric = static_cast<cuvs::distance::DistanceType>((int)index->metric());
      cuvs::cluster::kmeans_balanced::predict(handle,
                                              kmeans_params,
                                              batch_data_view,
                                              centers_view,
                                              batch_labels_view,
                                              utils::mapping<float>{});
      vec_batches.prefetch_next_batch();
      // User needs to make sure kernel finishes its work before we overwrite batch in the next
      // iteration if different streams are used for kernel and copy.
      raft::resource::sync_stream(handle);
    }
  }

  auto list_sizes = index->list_sizes().data_handle();
  // store the current cluster sizes, because we'll need them later
  rmm::device_uvector<uint32_t> orig_list_sizes(n_clusters, stream, device_memory);
  raft::copy(orig_list_sizes.data(), list_sizes, n_clusters, stream);

  // Get the combined cluster sizes
  raft::stats::histogram<uint32_t, IdxT>(raft::stats::HistTypeAuto,
                                         reinterpret_cast<int32_t*>(list_sizes),
                                         IdxT(n_clusters),
                                         new_data_labels.data(),
                                         n_rows,
                                         1,
                                         stream);
  raft::linalg::add(list_sizes, list_sizes, orig_list_sizes.data(), n_clusters, stream);

  // Allocate the lists to fit the new data
  {
    std::vector<uint32_t> new_cluster_sizes(n_clusters);
    std::vector<uint32_t> old_cluster_sizes(n_clusters);
    raft::copy(new_cluster_sizes.data(), list_sizes, n_clusters, stream);
    raft::copy(old_cluster_sizes.data(), orig_list_sizes.data(), n_clusters, stream);
    raft::resource::sync_stream(handle);
    for (uint32_t label = 0; label < n_clusters; label++) {
      ivf::resize_list(
        handle, index->lists()[label], spec, new_cluster_sizes[label], old_cluster_sizes[label]);
    }
  }

  // Update the pointers and the sizes
  ivf::detail::recompute_internal_state(handle, *index);

  // Recover old cluster sizes: they are used as counters in the fill-codes kernel
  raft::copy(list_sizes, orig_list_sizes.data(), n_clusters, stream);

  // By this point, the index state is updated and valid except it doesn't contain the new data
  // Fill the extended index with the new data (possibly, in batches)
  utils::batch_load_iterator<IdxT> idx_batches(
    new_indices, n_rows, 1, max_batch_size, stream, batches_mr);
  vec_batches.reset();
  vec_batches.prefetch_next_batch();
  for (const auto& vec_batch : vec_batches) {
    const auto& idx_batch = *idx_batches++;
    if (index->metric() == CosineExpanded) {
      auto vec_batch_view = raft::make_device_matrix_view<T, internal_extents_t>(
        const_cast<T*>(vec_batch.data()), vec_batch.size(), index->dim());
      raft::linalg::row_normalize<raft::linalg::L2Norm>(
        handle, raft::make_const_mdspan(vec_batch_view), vec_batch_view);
    }
    process_and_fill_codes(handle,
                           *index,
                           vec_batch.data(),
                           new_indices != nullptr
                             ? std::variant<IdxT, const IdxT*>(idx_batch.data())
                             : std::variant<IdxT, const IdxT*>(IdxT(idx_batch.offset())),
                           new_data_labels.data() + vec_batch.offset(),
                           IdxT(vec_batch.size()),
                           batches_mr);
    vec_batches.prefetch_next_batch();
    // User needs to make sure kernel finishes its work before we overwrite batch in the next
    // iteration if different streams are used for kernel and copy.
    raft::resource::sync_stream(handle);
  }
}

/**
 * Create a new index that contains more data.
 * See raft::spatial::knn::ivf_pq::extend docs.
 */
template <typename T, typename IdxT>
auto extend(raft::resources const& handle,
            const index<IdxT>& orig_index,
            const T* new_vectors,
            const IdxT* new_indices,
            IdxT n_rows) -> index<IdxT>
{
  auto ext_index = clone(handle, orig_index);
  detail::extend(handle, &ext_index, new_vectors, new_indices, n_rows);
  return ext_index;
}

template <typename T, typename IdxT, typename accessor>
auto build(raft::resources const& handle,
           const index_params& params,
           raft::mdspan<const T, raft::matrix_extent<IdxT>, raft::row_major, accessor> dataset)
  -> index<IdxT>
{
  IdxT n_rows = dataset.extent(0);
  IdxT dim    = dataset.extent(1);
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "ivf_pq::build(%zu, %u)", size_t(n_rows), dim);
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, half> || std::is_same_v<T, uint8_t> ||
                  std::is_same_v<T, int8_t>,
                "Unsupported data type");

  RAFT_EXPECTS(n_rows > 0 && dim > 0, "empty dataset");
  RAFT_EXPECTS(n_rows >= params.n_lists, "number of rows can't be less than n_lists");
  if (params.metric == distance::DistanceType::CosineExpanded) {
    // TODO: support int8_t and uint8_t types (https://github.com/rapidsai/cuvs/issues/389)
    if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>)
      RAFT_FAIL(
        "CosineExpanded distance metric is currently not supported for uint8_t and int8_t data "
        "type");
  }

  auto stream = raft::resource::get_cuda_stream(handle);

  index<IdxT> index(handle, params, dim);
  utils::memzero(
    index.accum_sorted_sizes().data_handle(), index.accum_sorted_sizes().size(), stream);
  utils::memzero(index.list_sizes().data_handle(), index.list_sizes().size(), stream);
  utils::memzero(index.data_ptrs().data_handle(), index.data_ptrs().size(), stream);
  utils::memzero(index.inds_ptrs().data_handle(), index.inds_ptrs().size(), stream);

  {
    raft::random::RngState random_state{137};
    auto trainset_ratio = std::max<size_t>(
      1,
      size_t(n_rows) / std::max<size_t>(params.kmeans_trainset_fraction * n_rows, index.n_lists()));
    size_t n_rows_train = n_rows / trainset_ratio;

    rmm::device_async_resource_ref device_memory = raft::resource::get_workspace_resource(handle);

    // If the trainset is small enough to comfortably fit into device memory, put it there.
    // Otherwise, use the managed memory.
    constexpr size_t kTolerableRatio = 4;
    rmm::device_async_resource_ref big_memory_resource =
      raft::resource::get_large_workspace_resource(handle);
    if (sizeof(float) * n_rows_train * index.dim() * kTolerableRatio <
        raft::resource::get_workspace_free_bytes(handle)) {
      big_memory_resource = device_memory;
    }

    // Besides just sampling, we transform the input dataset into floats to make it easier
    // to use gemm operations from cublas.
    auto trainset = raft::make_device_mdarray<float>(
      handle, big_memory_resource, raft::make_extents<int64_t>(0, 0));
    try {
      trainset = raft::make_device_mdarray<float>(
        handle, big_memory_resource, raft::make_extents<int64_t>(n_rows_train, dim));
    } catch (raft::logic_error& e) {
      RAFT_LOG_ERROR(
        "Insufficient memory for kmeans training set allocation. Please decrease "
        "kmeans_trainset_fraction, or set large_workspace_resource appropriately.");
      throw;
    }
    // TODO: a proper sampling
    if constexpr (std::is_same_v<T, float>) {
      raft::matrix::sample_rows<T, int64_t>(handle, random_state, dataset, trainset.view());
    } else {
      raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
        "   ivf_pq::build(%zu, %zu)/sample rows with tmp trainset (%zu rows).",
        size_t(n_rows),
        size_t(dim),
        size_t(n_rows_train));

      // TODO(tfeher): Enable codebook generation with any type T, and then remove trainset tmp.
      auto trainset_tmp = raft::make_device_mdarray<T>(
        handle, big_memory_resource, raft::make_extents<int64_t>(n_rows_train, dim));

      raft::matrix::sample_rows<T, int64_t>(handle, random_state, dataset, trainset_tmp.view());

      raft::linalg::unaryOp(trainset.data_handle(),
                            trainset_tmp.data_handle(),
                            trainset.size(),
                            utils::mapping<float>{},
                            raft::resource::get_cuda_stream(handle));
    }

    // NB: here cluster_centers is used as if it is [n_clusters, data_dim] not [n_clusters,
    // dim_ext]!
    rmm::device_uvector<float> cluster_centers_buf(
      index.n_lists() * index.dim(), stream, device_memory);
    auto cluster_centers = cluster_centers_buf.data();

    // Train balanced hierarchical kmeans clustering
    auto trainset_const_view = raft::make_const_mdspan(trainset.view());
    auto centers_view        = raft::make_device_matrix_view<float, internal_extents_t>(
      cluster_centers, index.n_lists(), index.dim());
    cuvs::cluster::kmeans::balanced_params kmeans_params;
    kmeans_params.n_iters = params.kmeans_n_iters;
    kmeans_params.metric  = static_cast<cuvs::distance::DistanceType>((int)index.metric());

    if (index.metric() == distance::DistanceType::CosineExpanded) {
      raft::linalg::row_normalize<raft::linalg::L2Norm>(
        handle, trainset_const_view, trainset.view());
    }
    cuvs::cluster::kmeans_balanced::fit(
      handle, kmeans_params, trainset_const_view, centers_view, utils::mapping<float>{});

    // Trainset labels are needed for training PQ codebooks
    rmm::device_uvector<uint32_t> labels(n_rows_train, stream, big_memory_resource);
    auto centers_const_view = raft::make_device_matrix_view<const float, internal_extents_t>(
      cluster_centers, index.n_lists(), index.dim());
    if (index.metric() == distance::DistanceType::CosineExpanded) {
      raft::linalg::row_normalize<raft::linalg::L2Norm>(handle, centers_const_view, centers_view);
    }
    auto labels_view =
      raft::make_device_vector_view<uint32_t, internal_extents_t>(labels.data(), n_rows_train);
    cuvs::cluster::kmeans_balanced::predict(handle,
                                            kmeans_params,
                                            trainset_const_view,
                                            centers_const_view,
                                            labels_view,
                                            utils::mapping<float>());

    // Make rotation matrix
    helpers::make_rotation_matrix(handle, &index, params.force_random_rotation);

    helpers::set_centers(handle, &index, raft::make_const_mdspan(centers_view));

    // Train PQ codebooks
    switch (index.codebook_kind()) {
      case codebook_gen::PER_SUBSPACE:
        train_per_subset(handle,
                         index,
                         n_rows_train,
                         trainset.data_handle(),
                         labels.data(),
                         params.kmeans_n_iters,
                         params.max_train_points_per_pq_code);
        break;
      case codebook_gen::PER_CLUSTER:
        train_per_cluster(handle,
                          index,
                          n_rows_train,
                          trainset.data_handle(),
                          labels.data(),
                          params.kmeans_n_iters,
                          params.max_train_points_per_pq_code);
        break;
      default: RAFT_FAIL("Unreachable code");
    }
  }

  // add the data if necessary
  if (params.add_data_on_build) {
    detail::extend<T, IdxT>(handle, &index, dataset.data_handle(), nullptr, n_rows);
  }
  return index;
}

template <typename T, typename IdxT, typename accessor>
void build(raft::resources const& handle,
           const index_params& params,
           raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, accessor> dataset,
           index<IdxT>* index)
{
  *index = build(handle, params, dataset);
}

template <typename T, typename IdxT, typename accessor, typename accessor2>
auto extend(
  raft::resources const& handle,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, accessor> new_vectors,
  std::optional<raft::mdspan<const IdxT, raft::vector_extent<int64_t>, raft::row_major, accessor2>>
    new_indices,
  const cuvs::neighbors::ivf_pq::index<IdxT>& orig_index) -> index<IdxT>
{
  ASSERT(new_vectors.extent(1) == orig_index.dim(),
         "new_vectors should have the same dimension as the index");

  IdxT n_rows = new_vectors.extent(0);
  if (new_indices.has_value()) {
    ASSERT(n_rows == new_indices.value().extent(0),
           "new_vectors and new_indices have different number of rows");
  }

  return extend(handle,
                orig_index,
                new_vectors.data_handle(),
                new_indices.has_value() ? new_indices.value().data_handle() : nullptr,
                n_rows);
}

template <typename T, typename IdxT, typename accessor, typename accessor2>
void extend(
  raft::resources const& handle,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, accessor> new_vectors,
  std::optional<raft::mdspan<const IdxT, raft::vector_extent<int64_t>, raft::row_major, accessor2>>
    new_indices,
  index<IdxT>* index)
{
  ASSERT(new_vectors.extent(1) == index->dim(),
         "new_vectors should have the same dimension as the index");

  IdxT n_rows = new_vectors.extent(0);
  if (new_indices.has_value()) {
    ASSERT(n_rows == new_indices.value().extent(0),
           "new_vectors and new_indices have different number of rows");
  }

  *index = extend(handle,
                  *index,
                  new_vectors.data_handle(),
                  new_indices.has_value() ? new_indices.value().data_handle() : nullptr,
                  n_rows);
}

template <typename output_mdspan_type>
inline void extract_centers(raft::resources const& res,
                            const cuvs::neighbors::ivf_pq::index<int64_t>& index,
                            output_mdspan_type cluster_centers)
{
  RAFT_EXPECTS(cluster_centers.extent(0) == index.n_lists(),
               "Number of rows in the output buffer for cluster centers must be equal to the "
               "number of IVF lists");
  RAFT_EXPECTS(
    cluster_centers.extent(1) == index.dim(),
    "Number of columns in the output buffer for cluster centers and index dim are different");
  auto stream = raft::resource::get_cuda_stream(res);
  RAFT_CUDA_TRY(cudaMemcpy2DAsync(cluster_centers.data_handle(),
                                  sizeof(float) * index.dim(),
                                  index.centers().data_handle(),
                                  sizeof(float) * index.dim_ext(),
                                  sizeof(float) * index.dim(),
                                  index.n_lists(),
                                  cudaMemcpyDefault,
                                  stream));
}
}  // namespace cuvs::neighbors::ivf_pq::detail
