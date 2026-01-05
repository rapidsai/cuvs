/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuvs/neighbors/common.hpp>

#include "../../cluster/kmeans_balanced.cuh"
#include "../../preprocessing/quantize/detail/pq_codepacking.cuh"  // pq_bits-bitfield
#include "ann_utils.cuh"                                           // utils::mapping etc

#include <cuvs/cluster/kmeans.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/cuda_stream_pool.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/map.cuh>
#include <raft/util/integer_utils.hpp>
#include <raft/util/pow2_utils.cuh>
#include <raft/util/vectorized.cuh>

// A temporary stub till https://github.com/rapidsai/raft/pull/2077 is re-merged
namespace cuvs::util {

/**
 * Subsample the dataset to create a training set.
 *
 * @tparam DatasetT a row-major mdspan or mdarray (device or host)
 *
 * @param res raft handle
 * @param dataset input row-major mdspan or mdarray (device or host)
 * @param n_samples the size of the output mdarray
 *
 * @return a newly allocated subset of the dataset.
 */
template <typename DatasetT>
auto subsample(raft::resources const& res,
               const DatasetT& dataset,
               typename DatasetT::index_type n_samples)
  -> raft::device_matrix<typename DatasetT::value_type, typename DatasetT::index_type>
{
  using value_type = typename DatasetT::value_type;
  using index_type = typename DatasetT::index_type;
  static_assert(std::is_same_v<typename DatasetT::layout_type, raft::row_major>,
                "Only row-major layout is supported at the moment");
  RAFT_EXPECTS(n_samples <= dataset.extent(0),
               "The number of samples must be smaller than the number of input rows in the current "
               "implementation.");
  size_t dim  = dataset.extent(1);
  auto result = raft::make_device_matrix<value_type, index_type>(res, n_samples, dataset.extent(1));

  RAFT_CUDA_TRY(cudaMemcpy2DAsync(result.data_handle(),
                                  sizeof(value_type) * dim,
                                  dataset.data_handle(),
                                  sizeof(value_type) * dim,
                                  sizeof(value_type) * dim,
                                  n_samples,
                                  cudaMemcpyDefault,
                                  raft::resource::get_cuda_stream(res)));
  return result;
}

}  // namespace cuvs::util

namespace cuvs::neighbors::detail {

template <typename MathT, typename IdxT>
void train_pq_centers(
  const raft::resources& res,
  const cuvs::neighbors::vpq_params& params,
  const raft::device_matrix_view<const MathT, IdxT, raft::row_major> pq_trainset_view,
  const raft::device_matrix_view<MathT, uint32_t, raft::row_major> pq_centers_view,
  std::optional<raft::device_vector_view<uint32_t, IdxT>> sub_labels_view       = std::nullopt,
  std::optional<raft::device_vector_view<uint32_t, IdxT>> pq_cluster_sizes_view = std::nullopt)
{
  if (params.pq_kmeans_type == cuvs::cluster::kmeans::kmeans_type::KMeansBalanced) {
    std::optional<raft::device_vector<uint32_t, IdxT>> sub_labels_opt       = std::nullopt;
    std::optional<raft::device_vector<uint32_t, IdxT>> pq_cluster_sizes_opt = std::nullopt;
    if (!sub_labels_view) {
      sub_labels_opt = std::make_optional(
        raft::make_device_vector<uint32_t, IdxT>(res, pq_trainset_view.extent(0)));
      sub_labels_view = std::make_optional(sub_labels_opt.value().view());
    }
    if (!pq_cluster_sizes_view) {
      pq_cluster_sizes_opt = std::make_optional(
        raft::make_device_vector<uint32_t, IdxT>(res, pq_centers_view.extent(0)));
      pq_cluster_sizes_view = std::make_optional(pq_cluster_sizes_opt.value().view());
    }
    cuvs::cluster::kmeans::balanced_params kmeans_params;
    kmeans_params.n_iters = params.kmeans_n_iters;
    kmeans_params.metric  = cuvs::distance::DistanceType::L2Expanded;

    cuvs::cluster::kmeans_balanced::helpers::build_clusters<
      MathT,
      MathT,
      IdxT,
      uint32_t,
      uint32_t,
      cuvs::spatial::knn::detail::utils::mapping<MathT>>(
      res,
      kmeans_params,
      pq_trainset_view,
      pq_centers_view,
      sub_labels_view.value(),
      pq_cluster_sizes_view.value(),
      cuvs::spatial::knn::detail::utils::mapping<MathT>{});
  } else {
    const auto pq_n_centers = pq_centers_view.extent(0);
    cuvs::cluster::kmeans::params kmeans_params;
    kmeans_params.n_clusters = pq_n_centers;
    kmeans_params.max_iter   = params.kmeans_n_iters;
    kmeans_params.metric     = cuvs::distance::DistanceType::L2Expanded;
    kmeans_params.init       = cuvs::cluster::kmeans::params::InitMethod::Random;

    std::optional<raft::device_vector_view<const MathT, IdxT>> sample_weight = std::nullopt;
    MathT inertia;
    IdxT n_iter;
    cuvs::cluster::kmeans::fit(res,
                               kmeans_params,
                               pq_trainset_view,
                               sample_weight,
                               pq_centers_view,
                               raft::make_host_scalar_view<MathT>(&inertia),
                               raft::make_host_scalar_view<IdxT>(&n_iter));
  }
}

template <typename DatasetT>
auto fill_missing_params_heuristics(const vpq_params& params, const DatasetT& dataset) -> vpq_params
{
  vpq_params r  = params;
  double n_rows = dataset.extent(0);
  size_t dim    = dataset.extent(1);
  if (r.pq_dim == 0) { r.pq_dim = raft::div_rounding_up_safe(dim, size_t{4}); }
  if (r.pq_bits == 0) { r.pq_bits = 8; }
  if (r.vq_n_centers == 0) { r.vq_n_centers = raft::round_up_safe<uint32_t>(std::sqrt(n_rows), 8); }
  if (r.vq_kmeans_trainset_fraction == 0) {
    double vq_trainset_size       = 100.0 * r.vq_n_centers;
    r.vq_kmeans_trainset_fraction = std::min(1.0, vq_trainset_size / n_rows);
  }
  if (r.pq_kmeans_trainset_fraction == 0) {
    // NB: we'll have actually `pq_dim` times more samples than this
    //     (because the dataset is reinterpreted as `[n_rows * pq_dim, pq_len]`)
    double pq_trainset_size       = 1000.0 * (1u << r.pq_bits);
    r.pq_kmeans_trainset_fraction = std::min(1.0, pq_trainset_size / n_rows);
  }
  r.max_train_points_per_pq_code = params.max_train_points_per_pq_code;
  return r;
}

template <typename T, typename DatasetT>
auto transform_data(const raft::resources& res, DatasetT dataset)
  -> raft::device_mdarray<T, typename DatasetT::extents_type, typename DatasetT::layout_type>
{
  using index_type       = typename DatasetT::index_type;
  using extents_type     = typename DatasetT::extents_type;
  using layout_type      = typename DatasetT::layout_type;
  using out_mdarray_type = raft::device_mdarray<T, extents_type, layout_type>;
  if constexpr (std::is_same_v<out_mdarray_type, std::decay<DatasetT>>) { return dataset; }

  auto result = raft::make_device_mdarray<T, index_type, layout_type>(res, dataset.extents());

  raft::linalg::map(res,
                    result.view(),
                    cuvs::spatial::knn::detail::utils::mapping<T>{},
                    raft::make_const_mdspan(dataset.view()));

  return result;
}

/** Fix the internal indexing type to avoid integer underflows/overflows */
using ix_t = int64_t;

template <typename MathT, typename DatasetT>
auto train_vq(const raft::resources& res, const vpq_params& params, const DatasetT& dataset)
  -> raft::device_matrix<MathT, uint32_t, raft::row_major>
{
  const ix_t n_rows       = dataset.extent(0);
  const ix_t vq_n_centers = params.vq_n_centers;
  const ix_t dim          = dataset.extent(1);
  const ix_t n_rows_train = n_rows * params.vq_kmeans_trainset_fraction;

  // Subsample the dataset and transform into the required type if necessary
  auto vq_trainset = util::subsample(res, dataset, n_rows_train);
  auto vq_centers =
    raft::make_device_matrix<MathT, uint32_t, raft::row_major>(res, vq_n_centers, dim);

  using kmeans_in_type = typename DatasetT::value_type;
  cuvs::cluster::kmeans::balanced_params kmeans_params;
  kmeans_params.n_iters = params.kmeans_n_iters;
  kmeans_params.metric  = cuvs::distance::DistanceType::L2Expanded;
  auto vq_centers_view =
    raft::make_device_matrix_view<MathT, ix_t>(vq_centers.data_handle(), vq_n_centers, dim);
  auto vq_trainset_view = raft::make_device_matrix_view<const kmeans_in_type, ix_t>(
    vq_trainset.data_handle(), n_rows_train, dim);
  cuvs::cluster::kmeans::fit(res, kmeans_params, vq_trainset_view, vq_centers_view);

  return vq_centers;
}

template <typename LabelT, typename DatasetT, typename VqCentersT>
auto predict_vq(const raft::resources& res, const DatasetT& dataset, const VqCentersT& vq_centers)
  -> raft::device_vector<LabelT, typename DatasetT::index_type>
{
  using kmeans_data_type = typename DatasetT::value_type;
  using kmeans_math_type = typename VqCentersT::value_type;
  using index_type       = typename DatasetT::index_type;
  using label_type       = LabelT;

  auto vq_labels = raft::make_device_vector<label_type, index_type>(res, dataset.extent(0));

  cuvs::cluster::kmeans::balanced_params kmeans_params;
  kmeans_params.metric = cuvs::distance::DistanceType::L2Expanded;

  auto vq_centers_view = raft::make_device_matrix_view<const kmeans_math_type, index_type>(
    vq_centers.data_handle(), vq_centers.extent(0), vq_centers.extent(1));

  auto vq_dataset_view = raft::make_device_matrix_view<const kmeans_data_type, index_type>(
    dataset.data_handle(), dataset.extent(0), dataset.extent(1));

  cuvs::cluster::kmeans::predict(
    res, kmeans_params, vq_dataset_view, vq_centers_view, vq_labels.view());

  return vq_labels;
}

template <typename MathT, typename DatasetT>
auto train_pq(
  const raft::resources& res,
  const vpq_params& params,
  const DatasetT& dataset,
  std::optional<const raft::device_matrix_view<const MathT, uint32_t, raft::row_major>> vq_centers)
  -> raft::device_matrix<MathT, uint32_t, raft::row_major>
{
  const ix_t n_rows       = dataset.extent(0);
  const ix_t dim          = dataset.extent(1);
  const ix_t pq_dim       = params.pq_dim;
  const ix_t pq_bits      = params.pq_bits;
  const ix_t pq_n_centers = ix_t{1} << pq_bits;
  const ix_t pq_len       = raft::div_rounding_up_safe(dim, pq_dim);
  const ix_t n_rows_train = std::min((ix_t)(n_rows * params.pq_kmeans_trainset_fraction),
                                     params.max_train_points_per_pq_code * pq_n_centers);
  RAFT_EXPECTS(
    n_rows_train >= pq_n_centers,
    "The number of training samples must be greater than or equal to the number of PQ centers");

  // Subsample the dataset and transform into the required type if necessary
  auto pq_trainset = transform_data<MathT>(res, util::subsample(res, dataset, n_rows_train));

  // Subtract VQ centers
  if (vq_centers) {
    auto vq_labels   = predict_vq<uint32_t>(res, pq_trainset, vq_centers.value());
    using index_type = typename DatasetT::index_type;
    raft::linalg::map_offset(
      res,
      pq_trainset.view(),
      [labels = vq_labels.view(), centers = vq_centers.value(), dim] __device__(index_type off,
                                                                                MathT x) {
        index_type i = off / dim;
        index_type j = off % dim;
        return x - centers(labels(i), j);
      },
      raft::make_const_mdspan(pq_trainset.view()));
  }

  auto pq_centers =
    raft::make_device_matrix<MathT, uint32_t, raft::row_major>(res, pq_n_centers, pq_len);
  auto pq_trainset_view = raft::make_device_matrix_view<const MathT, ix_t>(
    pq_trainset.data_handle(), n_rows_train * pq_dim, pq_len);
  train_pq_centers<MathT, ix_t>(res, params, pq_trainset_view, pq_centers.view());

  return pq_centers;
}

template <uint32_t SubWarpSize,
          typename CodeT,
          typename DataT,
          typename MathT,
          typename IdxT,
          typename LabelT>
__device__ auto compute_code(
  raft::device_matrix_view<const DataT, IdxT, raft::row_major> dataset,
  std::optional<raft::device_matrix_view<const MathT, uint32_t, raft::row_major>> vq_centers,
  std::optional<raft::device_matrix_view<const MathT, uint32_t, raft::row_major>> pq_centers_smem,
  raft::device_matrix_view<const MathT, uint32_t, raft::row_major> pq_centers,
  IdxT i,
  uint32_t j,
  LabelT vq_label) -> CodeT
{
  static_assert(std::is_same_v<CodeT, uint8_t> || std::is_same_v<CodeT, uint16_t>,
                "CodeT must be uint8_t or uint16_t");
  auto data_mapping      = cuvs::spatial::knn::detail::utils::mapping<MathT>{};
  const uint32_t lane_id = raft::Pow2<SubWarpSize>::mod(raft::laneId());

  const uint32_t pq_book_size = pq_centers.extent(0);
  const uint32_t pq_len       = pq_centers.extent(1);
  float min_dist              = std::numeric_limits<float>::infinity();
  CodeT code                  = 0;
  // calculate the distance for each PQ cluster, find the minimum for each thread
  uint32_t l = lane_id;
  if (pq_centers_smem) {
    for (; l < pq_centers_smem.value().extent(0); l += SubWarpSize) {
      // NB: the L2 quantifiers on residuals are always trained on L2 metric.
      float d = 0.0f;
      for (uint32_t k = 0; k < pq_len; k++) {
        auto jk = j * pq_len + k;
        auto x  = data_mapping(dataset(i, jk));
        if (vq_centers) { x -= vq_centers.value()(vq_label, jk); }
        auto t = x - pq_centers_smem.value()(l, k);
        d += t * t;
      }
      if (d < min_dist) {
        min_dist = d;
        code     = CodeT(l);
      }
    }
  }
  // if there are more rows than the shared memory size, compute the distance for the remaining rows
  for (; l < pq_book_size; l += SubWarpSize) {
    // NB: the L2 quantifiers on residuals are always trained on L2 metric.
    float d = 0.0f;
    for (uint32_t k = 0; k < pq_len; k++) {
      auto jk = j * pq_len + k;
      auto x  = data_mapping(dataset(i, jk));
      if (vq_centers) { x -= vq_centers.value()(vq_label, jk); }
      auto t = x - pq_centers(l, k);
      d += t * t;
    }
    if (d < min_dist) {
      min_dist = d;
      code     = CodeT(l);
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

template <int VecBytes = 16, typename T>
__device__ inline void copy_vectorized(T* out, const T* in, uint32_t n)
{
  constexpr int VecElems = VecBytes / sizeof(T);  // NOLINT
  using align_bytes      = raft::Pow2<(size_t)VecBytes>;
  if constexpr (VecElems > 1) {
    using align_elems = raft::Pow2<VecElems>;
    if (!align_bytes::areSameAlignOffsets(out, in)) {
      return copy_vectorized<(VecBytes >> 1), T>(out, in, n);
    }
    {  // process unaligned head
      const uint32_t head = align_bytes::roundUp(in) - in;
      if (head > 0) {
        copy_vectorized<sizeof(T), T>(out, in, head);
        n -= head;
        in += head;
        out += head;
      }
    }
    {  // process main part vectorized
      using vec_t = typename raft::IOType<T, VecElems>::Type;
      copy_vectorized<sizeof(vec_t), vec_t>(
        reinterpret_cast<vec_t*>(out), reinterpret_cast<const vec_t*>(in), align_elems::div(n));
    }
    {  // process unaligned tail
      const uint32_t tail = align_elems::mod(n);
      if (tail > 0) {
        n -= tail;
        copy_vectorized<sizeof(T), T>(out + n, in + n, tail);
      }
    }
  }
  if constexpr (VecElems <= 1) {
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
      out[i] = in[i];
    }
  }
}

template <uint32_t BlockSize,
          uint32_t SubWarpSize,
          typename CodeT,
          typename DataT,
          typename MathT,
          typename IdxT,
          typename LabelT>
__launch_bounds__(BlockSize) RAFT_KERNEL process_and_fill_codes_kernel(
  raft::device_matrix_view<uint8_t, IdxT, raft::row_major> out_codes,
  raft::device_matrix_view<const DataT, IdxT, raft::row_major> dataset,
  std::optional<raft::device_matrix_view<const MathT, uint32_t, raft::row_major>> vq_centers,
  std::optional<raft::device_vector_view<const LabelT, IdxT, raft::row_major>> vq_labels,
  raft::device_matrix_view<const MathT, uint32_t, raft::row_major> pq_centers,
  const uint32_t shared_memory_size,
  const uint32_t pq_bits)
{
  extern __shared__ __align__(256) uint8_t pq_centers_smem[];
  using subwarp_align = raft::Pow2<SubWarpSize>;
  const IdxT row_ix   = subwarp_align::div(IdxT{threadIdx.x} + IdxT{BlockSize} * IdxT{blockIdx.x});
  if (row_ix >= out_codes.extent(0)) { return; }

  const uint32_t pq_dim = raft::div_rounding_up_unsafe(dataset.extent(1), pq_centers.extent(1));

  // Copy the pq_centers into shared memory for faster processing
  const IdxT rows_in_shared_memory = shared_memory_size / (sizeof(MathT) * pq_centers.extent(1));
  copy_vectorized<4>(reinterpret_cast<MathT*>(pq_centers_smem),
                     pq_centers.data_handle(),
                     rows_in_shared_memory * pq_centers.extent(1));
  __syncthreads();
  auto pq_centers_smem_view = raft::make_device_matrix_view<const MathT, uint32_t, raft::row_major>(
    reinterpret_cast<const MathT*>(pq_centers_smem), rows_in_shared_memory, pq_centers.extent(1));

  const uint32_t lane_id = subwarp_align::mod(threadIdx.x);
  const LabelT vq_label  = vq_labels ? vq_labels.value()(row_ix) : 0;
  auto* out_codes_ptr    = &out_codes(row_ix, 0);

  // write label
  if (vq_centers) {
    auto* out_label_ptr = reinterpret_cast<LabelT*>(out_codes_ptr);
    if (lane_id == 0) { *out_label_ptr = vq_label; }
    out_codes_ptr += sizeof(LabelT);
  }

  cuvs::preprocessing::quantize::detail::bitfield_view_t code_view{out_codes_ptr, pq_bits};
  for (uint32_t j = 0; j < pq_dim; j++) {
    // find PQ label
    CodeT code = compute_code<SubWarpSize, CodeT>(dataset,
                                                  vq_centers,
                                                  std::make_optional(pq_centers_smem_view),
                                                  pq_centers,
                                                  row_ix,
                                                  j,
                                                  vq_label);
    // TODO: this writes in global memory one byte per warp, which is very slow.
    //  It's better to keep the codes in the shared memory or registers and dump them at once.
    if (lane_id == 0) { code_view[j] = code; }
  }
}

template <typename MathT, typename IdxT, typename DatasetT>
void process_and_fill_codes(
  const raft::resources& res,
  const vpq_params& params,
  const DatasetT& dataset,
  std::optional<raft::device_matrix_view<const MathT, uint32_t, raft::row_major>> vq_centers,
  raft::device_matrix_view<const MathT, uint32_t, raft::row_major> pq_centers,
  raft::device_matrix_view<uint8_t, IdxT, raft::row_major> codes)
{
  using data_t     = typename DatasetT::value_type;
  using cdataset_t = vpq_dataset<MathT, IdxT>;
  using label_t    = uint32_t;

  const ix_t n_rows       = dataset.extent(0);
  const ix_t dim          = dataset.extent(1);
  const ix_t pq_dim       = params.pq_dim;
  const ix_t pq_bits      = params.pq_bits;
  const ix_t pq_n_centers = ix_t{1} << pq_bits;
  ix_t codes_rowlen       = 0;
  if (vq_centers.has_value()) {
    // NB: codes must be aligned at least to sizeof(label_t) to be able to read labels.
    codes_rowlen = sizeof(label_t) *
                   (1 + raft::div_rounding_up_safe<ix_t>(pq_dim * pq_bits, 8 * sizeof(label_t)));
  } else {
    codes_rowlen = raft::div_rounding_up_safe<ix_t>(pq_dim * pq_bits, 8);
  }
  RAFT_EXPECTS(codes.extent(0) == n_rows,
               "Codes matrix must have the same number of rows as the input dataset");
  RAFT_EXPECTS(codes.extent(1) == codes_rowlen,
               "Codes matrix must have the same number of columns as the input dataset");

  auto stream = raft::resource::get_cuda_stream(res);

  // TODO: with scaling workspace we could choose the batch size dynamically
  constexpr ix_t kReasonableMaxBatchSize = 65536;
  constexpr ix_t kBlockSize              = 256;
  constexpr ix_t kMaxSharedMemorySize    = 16384;
  const ix_t kSharedMemorySize           = std::min<ix_t>(
    kMaxSharedMemorySize, pq_centers.extent(0) * pq_centers.extent(1) * sizeof(MathT));
  const ix_t threads_per_vec = std::min<ix_t>(raft::WarpSize, pq_n_centers);
  dim3 threads(kBlockSize, 1, 1);
  ix_t max_batch_size = std::min<ix_t>(n_rows, kReasonableMaxBatchSize);
  auto kernel         = [](uint32_t pq_bits) {
    if (pq_bits == 4) {
      return process_and_fill_codes_kernel<kBlockSize, 16, uint8_t, data_t, MathT, IdxT, label_t>;
    } else if (pq_bits <= 8) {
      return process_and_fill_codes_kernel<kBlockSize,
                                                   raft::WarpSize,
                                                   uint8_t,
                                                   data_t,
                                                   MathT,
                                                   IdxT,
                                                   label_t>;
    } else if (pq_bits <= 16) {
      return process_and_fill_codes_kernel<kBlockSize,
                                                   raft::WarpSize,
                                                   uint16_t,
                                                   data_t,
                                                   MathT,
                                                   IdxT,
                                                   label_t>;
    } else {
      RAFT_FAIL("Invalid pq_bits (%u), the value must be within [4, 16]", pq_bits);
    }
  }(pq_bits);
  for (const auto& batch : cuvs::spatial::knn::detail::utils::batch_load_iterator(
         dataset.data_handle(),
         n_rows,
         dim,
         max_batch_size,
         stream,
         rmm::mr::get_current_device_resource())) {
    auto batch_view = raft::make_device_matrix_view(batch.data(), ix_t(batch.size()), dim);
    std::optional<raft::device_vector<label_t, IdxT, raft::row_major>> labels = std::nullopt;
    std::optional<raft::device_vector_view<const label_t, IdxT, raft::row_major>> labels_view =
      std::nullopt;
    if (vq_centers) {
      labels      = predict_vq<label_t>(res, batch_view, vq_centers.value());
      labels_view = raft::make_const_mdspan(labels->view());
    }
    dim3 blocks(raft::div_rounding_up_safe<ix_t>(n_rows, kBlockSize / threads_per_vec), 1, 1);
    kernel<<<blocks, threads, kSharedMemorySize, stream>>>(
      raft::make_device_matrix_view<uint8_t, IdxT>(
        codes.data_handle() + batch.offset() * codes_rowlen, batch.size(), codes_rowlen),
      batch_view,
      vq_centers,
      labels_view,
      pq_centers,
      kSharedMemorySize,
      pq_bits);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
}

template <typename NewMathT, typename OldMathT, typename IdxT>
auto vpq_convert_math_type(const raft::resources& res, vpq_dataset<OldMathT, IdxT>&& src)
  -> vpq_dataset<NewMathT, IdxT>
{
  auto vq_code_book = raft::make_device_mdarray<NewMathT>(res, src.vq_code_book.extents());
  auto pq_code_book = raft::make_device_mdarray<NewMathT>(res, src.pq_code_book.extents());

  raft::linalg::map(res,
                    vq_code_book.view(),
                    cuvs::spatial::knn::detail::utils::mapping<NewMathT>{},
                    raft::make_const_mdspan(src.vq_code_book.view()));
  raft::linalg::map(res,
                    pq_code_book.view(),
                    cuvs::spatial::knn::detail::utils::mapping<NewMathT>{},
                    raft::make_const_mdspan(src.pq_code_book.view()));
  return vpq_dataset<NewMathT, IdxT>{
    std::move(vq_code_book), std::move(pq_code_book), std::move(src.data)};
}

template <uint32_t SubWarpSize,
          typename CodeT,
          typename DataT,
          typename MathT,
          typename IdxT,
          typename LabelT>
__device__ auto compute_code_subspaces(
  raft::device_matrix_view<const DataT, IdxT, raft::row_major> dataset,
  std::optional<raft::device_matrix_view<const MathT, uint32_t, raft::row_major>> vq_centers,
  raft::device_matrix_view<const MathT, uint32_t, raft::row_major> pq_centers,
  MathT* smem_dataset,
  IdxT i,
  uint32_t j,
  LabelT vq_label) -> CodeT
{
  static_assert(std::is_same_v<CodeT, uint8_t> || std::is_same_v<CodeT, uint16_t>,
                "CodeT must be uint8_t or uint16_t");
  auto data_mapping      = cuvs::spatial::knn::detail::utils::mapping<MathT>{};
  const uint32_t lane_id = raft::Pow2<SubWarpSize>::mod(raft::laneId());

  const uint32_t pq_book_size           = pq_centers.extent(0);
  const uint32_t pq_len                 = pq_centers.extent(1);
  const MathT* __restrict__ centers_ptr = pq_centers.data_handle();
  const DataT* __restrict__ dataset_ptr = dataset.data_handle() + i * dataset.extent(1);
  MathT min_dist                        = std::numeric_limits<MathT>::infinity();
  CodeT code                            = 0;

  if (pq_len <= SubWarpSize) {
    // Each thread loads one dataset element and will share to others using shuffle
    MathT my_dataset_val = 0.0f;
    if (lane_id < pq_len) {
      auto jk = j * pq_len + lane_id;
      auto x  = data_mapping(dataset_ptr[jk]);
      if (vq_centers) { x -= vq_centers.value()(vq_label, jk); }
      my_dataset_val = x;
    }

    const uint32_t subwarp_idx  = raft::Pow2<SubWarpSize>::div(raft::laneId());
    const uint32_t subwarp_mask = (SubWarpSize == 32)
                                    ? 0xffffffffu
                                    : (((1u << SubWarpSize) - 1) << (subwarp_idx * SubWarpSize));
    // Each thread evaluates by chunks of 4 PQ centers
    uint32_t l = lane_id;

    for (; l + 3 * SubWarpSize < pq_book_size; l += 4 * SubWarpSize) {
      MathT d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;

#pragma unroll 4
      for (uint32_t k = 0; k < pq_len; k++) {
        MathT x  = __shfl_sync(subwarp_mask, my_dataset_val, k, SubWarpSize);
        MathT c0 = __ldg(&centers_ptr[l * pq_len + k]);
        MathT c1 = __ldg(&centers_ptr[(l + SubWarpSize) * pq_len + k]);
        MathT c2 = __ldg(&centers_ptr[(l + 2 * SubWarpSize) * pq_len + k]);
        MathT c3 = __ldg(&centers_ptr[(l + 3 * SubWarpSize) * pq_len + k]);

        d0 = __fmaf_rn(x - c0, x - c0, d0);
        d1 = __fmaf_rn(x - c1, x - c1, d1);
        d2 = __fmaf_rn(x - c2, x - c2, d2);
        d3 = __fmaf_rn(x - c3, x - c3, d3);
      }

      MathT best_dist_01      = fminf(d0, d1);
      MathT best_dist_23      = fminf(d2, d3);
      MathT best_dist         = fminf(best_dist_01, best_dist_23);
      const uint32_t idx_01   = (d1 < d0) ? (l + SubWarpSize) : l;
      const uint32_t idx_23   = (d3 < d2) ? (l + 3 * SubWarpSize) : (l + 2 * SubWarpSize);
      const uint32_t best_idx = (best_dist_23 < best_dist_01) ? idx_23 : idx_01;

      if (best_dist < min_dist) {
        min_dist = best_dist;
        code     = CodeT(best_idx);
      }
    }

    // Tail loop
    for (; l < pq_book_size; l += SubWarpSize) {
      MathT d0 = 0.0f;
#pragma unroll
      for (uint32_t k = 0; k < pq_len; k++) {
        MathT x          = __shfl_sync(subwarp_mask, my_dataset_val, k, SubWarpSize);
        MathT center_val = __ldg(&centers_ptr[l * pq_len + k]);
        d0               = __fmaf_rn(x - center_val, x - center_val, d0);
      }

      if (d0 < min_dist) {
        min_dist = d0;
        code     = CodeT(l);
      }
    }
  } else {
    // Cooperatively load dataset into shared memory if pq_len > SubWarpSize
    for (uint32_t k = lane_id; k < pq_len; k += SubWarpSize) {
      auto jk = j * pq_len + k;
      auto x  = data_mapping(dataset_ptr[jk]);
      if (vq_centers) { x -= vq_centers.value()(vq_label, jk); }
      smem_dataset[k] = x;
    }
    __syncwarp();

    // Each thread evaluates by chunks of 4 PQ centers
    uint32_t l = lane_id;
    for (; l + 3 * SubWarpSize < pq_book_size; l += 4 * SubWarpSize) {
      MathT d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;

#pragma unroll 4
      for (uint32_t k = 0; k < pq_len; k++) {
        MathT x  = smem_dataset[k];
        MathT c0 = __ldg(&centers_ptr[l * pq_len + k]);
        MathT c1 = __ldg(&centers_ptr[(l + SubWarpSize) * pq_len + k]);
        MathT c2 = __ldg(&centers_ptr[(l + 2 * SubWarpSize) * pq_len + k]);
        MathT c3 = __ldg(&centers_ptr[(l + 3 * SubWarpSize) * pq_len + k]);
        d0       = __fmaf_rn(x - c0, x - c0, d0);
        d1       = __fmaf_rn(x - c1, x - c1, d1);
        d2       = __fmaf_rn(x - c2, x - c2, d2);
        d3       = __fmaf_rn(x - c3, x - c3, d3);
      }

      MathT best_dist_01      = fminf(d0, d1);
      MathT best_dist_23      = fminf(d2, d3);
      MathT best_dist         = fminf(best_dist_01, best_dist_23);
      const uint32_t idx_01   = (d1 < d0) ? (l + SubWarpSize) : l;
      const uint32_t idx_23   = (d3 < d2) ? (l + 3 * SubWarpSize) : (l + 2 * SubWarpSize);
      const uint32_t best_idx = (best_dist_23 < best_dist_01) ? idx_23 : idx_01;

      if (best_dist < min_dist) {
        min_dist = best_dist;
        code     = CodeT(best_idx);
      }
    }

    // Tail loop
    for (; l < pq_book_size; l += SubWarpSize) {
      MathT d0 = 0.0f;

#pragma unroll 4
      for (uint32_t k = 0; k < pq_len; k++) {
        MathT x          = smem_dataset[k];
        MathT center_val = __ldg(&centers_ptr[l * pq_len + k]);
        d0               = __fmaf_rn(x - center_val, x - center_val, d0);
      }

      if (d0 < min_dist) {
        min_dist = d0;
        code     = CodeT(l);
      }
    }
  }

// Reduce among threads
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

template <uint32_t BlockSize,
          uint32_t SubWarpSize,
          typename CodeT,
          typename DataT,
          typename MathT,
          typename IdxT,
          typename LabelT>
__launch_bounds__(BlockSize) RAFT_KERNEL process_and_fill_codes_subspaces_kernel(
  raft::device_matrix_view<uint8_t, IdxT, raft::row_major> out_codes,
  raft::device_matrix_view<const DataT, IdxT, raft::row_major> dataset,
  std::optional<raft::device_matrix_view<const MathT, uint32_t, raft::row_major>> vq_centers,
  std::optional<raft::device_vector_view<const LabelT, IdxT, raft::row_major>> vq_labels,
  raft::device_matrix_view<const MathT, uint32_t, raft::row_major> pq_centers,
  const uint32_t pq_bits)
{
  extern __shared__ __align__(256) uint8_t smem_buffer[];
  using subwarp_align = raft::Pow2<SubWarpSize>;
  const IdxT row_ix   = subwarp_align::div(IdxT{threadIdx.x} + IdxT{BlockSize} * IdxT{blockIdx.x});
  if (row_ix >= out_codes.extent(0)) { return; }

  const uint32_t subwarp_id = subwarp_align::div(threadIdx.x);
  MathT* smem_dataset   = reinterpret_cast<MathT*>(smem_buffer) + subwarp_id * pq_centers.extent(1);
  const uint32_t pq_dim = raft::div_rounding_up_unsafe(dataset.extent(1), pq_centers.extent(1));

  const uint32_t lane_id = subwarp_align::mod(threadIdx.x);
  const LabelT vq_label  = vq_labels ? vq_labels.value()(row_ix) : 0;
  auto* out_codes_ptr    = &out_codes(row_ix, 0);

  // write label
  if (vq_centers) {
    auto* out_label_ptr = reinterpret_cast<LabelT*>(out_codes_ptr);
    if (lane_id == 0) { *out_label_ptr = vq_label; }
    out_codes_ptr += sizeof(LabelT);
  }

  cuvs::preprocessing::quantize::detail::bitfield_view_t code_view{out_codes_ptr, pq_bits};
  for (uint32_t j = 0; j < pq_dim; j++) {
    // find PQ label
    uint32_t subspace_offset = j * pq_centers.extent(1) * (1 << pq_bits);
    auto pq_subspace_view    = raft::make_device_matrix_view(
      pq_centers.data_handle() + subspace_offset, (uint32_t)(1 << pq_bits), pq_centers.extent(1));
    CodeT code = compute_code_subspaces<SubWarpSize, CodeT>(
      dataset, vq_centers, pq_subspace_view, smem_dataset, row_ix, j, vq_label);
    if (lane_id == 0) { code_view[j] = code; }
  }
}

template <typename MathT, typename IdxT, typename DatasetT>
void process_and_fill_codes_subspaces(
  const raft::resources& res,
  const vpq_params& params,
  const DatasetT& dataset,
  std::optional<raft::device_matrix_view<const MathT, uint32_t, raft::row_major>> vq_centers,
  raft::device_matrix_view<const MathT, uint32_t, raft::row_major> pq_centers,
  raft::device_matrix_view<uint8_t, IdxT, raft::row_major> codes)
{
  using data_t     = typename DatasetT::value_type;
  using cdataset_t = vpq_dataset<MathT, IdxT>;
  using label_t    = uint32_t;

  const ix_t n_rows       = dataset.extent(0);
  const ix_t dim          = dataset.extent(1);
  const ix_t pq_dim       = params.pq_dim;
  const ix_t pq_bits      = params.pq_bits;
  const ix_t pq_n_centers = ix_t{1} << pq_bits;
  ix_t codes_rowlen       = 0;
  if (vq_centers.has_value()) {
    // NB: codes must be aligned at least to sizeof(label_t) to be able to read labels.
    codes_rowlen = sizeof(label_t) *
                   (1 + raft::div_rounding_up_safe<ix_t>(pq_dim * pq_bits, 8 * sizeof(label_t)));
  } else {
    codes_rowlen = raft::div_rounding_up_safe<ix_t>(pq_dim * pq_bits, 8);
  }
  RAFT_EXPECTS(codes.extent(0) == n_rows,
               "Codes matrix must have the same number of rows as the input dataset");
  RAFT_EXPECTS(codes.extent(1) == codes_rowlen,
               "Codes matrix must have the same number of columns as the input dataset");

  auto stream = raft::resource::get_cuda_stream(res);

  constexpr ix_t kBlockSize              = 512;
  constexpr ix_t kReasonableMaxBatchSize = 131072;  // 128K rows per batch
  const ix_t threads_per_vec             = std::min<ix_t>(raft::WarpSize, pq_n_centers);
  const uint32_t num_subwarps            = kBlockSize / threads_per_vec;
  const uint32_t pq_len                  = pq_centers.extent(1);

  // Minimal shared memory allocation
  const uint32_t kSharedMemorySize =
    pq_len <= threads_per_vec ? 0 : num_subwarps * pq_len * sizeof(MathT);

  dim3 threads(kBlockSize, 1, 1);
  auto kernel = [](uint32_t pq_bits) {
    if (pq_bits == 4) {
      return process_and_fill_codes_subspaces_kernel<kBlockSize,
                                                     16,
                                                     uint8_t,
                                                     data_t,
                                                     MathT,
                                                     IdxT,
                                                     label_t>;
    } else if (pq_bits <= 8) {
      return process_and_fill_codes_subspaces_kernel<kBlockSize,
                                                     raft::WarpSize,
                                                     uint8_t,
                                                     data_t,
                                                     MathT,
                                                     IdxT,
                                                     label_t>;
    } else if (pq_bits <= 16) {
      return process_and_fill_codes_subspaces_kernel<kBlockSize,
                                                     raft::WarpSize,
                                                     uint16_t,
                                                     data_t,
                                                     MathT,
                                                     IdxT,
                                                     label_t>;
    } else {
      RAFT_FAIL("Invalid pq_bits (%u), the value must be within [4, 16]", pq_bits);
    }
  }(pq_bits);

  ix_t max_batch_size  = std::min<ix_t>(n_rows, kReasonableMaxBatchSize);
  auto copy_stream     = raft::resource::get_cuda_stream(res);  // Using the main stream by default
  bool enable_prefetch = false;
  if (res.has_resource_factory(raft::resource::resource_type::CUDA_STREAM_POOL)) {
    if (raft::resource::get_stream_pool_size(res) >= 1) {
      enable_prefetch = true;
      copy_stream     = raft::resource::get_stream_from_stream_pool(res);
    }
  }
  auto vec_batches = cuvs::spatial::knn::detail::utils::batch_load_iterator(
    dataset.data_handle(),
    n_rows,
    dim,
    max_batch_size,
    copy_stream,
    raft::resource::get_workspace_resource(res),
    enable_prefetch);
  vec_batches.prefetch_next_batch();
  for (const auto& batch : vec_batches) {
    auto batch_view = raft::make_device_matrix_view(batch.data(), ix_t(batch.size()), dim);
    std::optional<raft::device_vector<label_t, IdxT, raft::row_major>> labels = std::nullopt;
    std::optional<raft::device_vector_view<const label_t, IdxT, raft::row_major>> labels_view =
      std::nullopt;
    if (vq_centers) {
      labels      = predict_vq<label_t>(res, batch_view, vq_centers.value());
      labels_view = raft::make_const_mdspan(labels->view());
    }
    dim3 blocks(raft::div_rounding_up_safe<ix_t>(batch.size(), kBlockSize / threads_per_vec), 1, 1);
    kernel<<<blocks, threads, kSharedMemorySize, stream>>>(
      raft::make_device_matrix_view<uint8_t, IdxT>(
        codes.data_handle() + batch.offset() * codes_rowlen, batch.size(), codes_rowlen),
      batch_view,
      vq_centers,
      labels_view,
      pq_centers,
      pq_bits);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    vec_batches.prefetch_next_batch();
    raft::resource::sync_stream(res);
  }
}

template <typename DatasetT, typename MathT, typename IdxT>
auto vpq_build(const raft::resources& res, const vpq_params& params, const DatasetT& dataset)
  -> vpq_dataset<MathT, IdxT>
{
  using label_t = uint32_t;
  // Use a heuristic to impute missing parameters.
  auto ps = fill_missing_params_heuristics(params, dataset);

  // Train codes
  auto vq_code_book = train_vq<MathT>(res, ps, dataset);
  auto pq_code_book = train_pq<MathT>(
    res, ps, dataset, std::make_optional(raft::make_const_mdspan(vq_code_book.view())));

  // Encode dataset
  const IdxT n_rows       = dataset.extent(0);
  const IdxT codes_rowlen = sizeof(label_t) * (1 + raft::div_rounding_up_safe<IdxT>(
                                                     ps.pq_dim * ps.pq_bits, 8 * sizeof(label_t)));

  auto codes = raft::make_device_matrix<uint8_t, IdxT, raft::row_major>(res, n_rows, codes_rowlen);
  process_and_fill_codes<MathT, IdxT>(res,
                                      ps,
                                      dataset,
                                      raft::make_const_mdspan(vq_code_book.view()),
                                      raft::make_const_mdspan(pq_code_book.view()),
                                      codes.view());

  return vpq_dataset<MathT, IdxT>{
    std::move(vq_code_book), std::move(pq_code_book), std::move(codes)};
}

}  // namespace cuvs::neighbors::detail
