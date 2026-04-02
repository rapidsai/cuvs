/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../core/nvtx.hpp"
#include "../ivf_common.cuh"
#include "../ivf_list.cuh"

#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_sq.hpp>

#include "../detail/ann_utils.cuh"
#include <cuvs/distance/distance.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/cuda_stream_pool.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/add.cuh>
#include <raft/linalg/map.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/matrix/sample_rows.cuh>
#include <raft/random/rng.cuh>
#include <raft/stats/histogram.cuh>
#include <raft/util/pow2_utils.cuh>

#include <cub/block/block_reduce.cuh>

#include <rmm/cuda_stream_view.hpp>

#include <cstdint>
#include <limits>

namespace cuvs::neighbors::ivf_sq {
using namespace cuvs::spatial::knn::detail;  // NOLINT

namespace detail {

struct ColMinMaxPair {
  float min_val;
  float max_val;
};

struct ColMinMaxOp {
  __device__ __forceinline__ ColMinMaxPair operator()(const ColMinMaxPair& a,
                                                      const ColMinMaxPair& b) const
  {
    return {fminf(a.min_val, b.min_val), fmaxf(a.max_val, b.max_val)};
  }
};

/**
 * Fused per-column min+max in a single pass (2x less DRAM traffic than two
 * separate reductions).  One thread block per column; threads stride over
 * rows and feed CUB BlockReduce with a combined min/max pair.
 *
 * Row-loop is manually 4x-unrolled so the compiler can overlap four
 * independent read-only loads in the memory pipeline.
 */
template <int BlockSize, typename T>
__launch_bounds__(BlockSize) RAFT_KERNEL fused_column_minmax_kernel(const T* __restrict__ data,
                                                                    float* __restrict__ col_min,
                                                                    float* __restrict__ col_max,
                                                                    int64_t n_rows,
                                                                    uint32_t dim)
{
  using BlockReduce = cub::BlockReduce<ColMinMaxPair, BlockSize>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  const uint32_t col = blockIdx.x;
  if (col >= dim) return;

  ColMinMaxPair agg = {std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest()};

  const int64_t stride = static_cast<int64_t>(BlockSize);
  int64_t row          = static_cast<int64_t>(threadIdx.x);

  for (; row + 3 * stride < n_rows; row += 4 * stride) {
    float v0    = float(data[row * dim + col]);
    float v1    = float(data[(row + stride) * dim + col]);
    float v2    = float(data[(row + 2 * stride) * dim + col]);
    float v3    = float(data[(row + 3 * stride) * dim + col]);
    agg.min_val = fminf(agg.min_val, fminf(fminf(v0, v1), fminf(v2, v3)));
    agg.max_val = fmaxf(agg.max_val, fmaxf(fmaxf(v0, v1), fmaxf(v2, v3)));
  }
  for (; row < n_rows; row += stride) {
    float val   = float(data[row * dim + col]);
    agg.min_val = fminf(agg.min_val, val);
    agg.max_val = fmaxf(agg.max_val, val);
  }

  agg = BlockReduce(temp_storage).Reduce(agg, ColMinMaxOp());

  if (threadIdx.x == 0) {
    col_min[col] = agg.min_val;
    col_max[col] = agg.max_val;
  }
}

template <typename IdxT>
auto clone(const raft::resources& res, const index<IdxT>& source) -> index<IdxT>
{
  auto stream = raft::resource::get_cuda_stream(res);

  index<IdxT> target(
    res, source.metric(), source.n_lists(), source.dim(), source.conservative_memory_allocation());

  raft::copy(target.list_sizes().data_handle(),
             source.list_sizes().data_handle(),
             source.list_sizes().size(),
             stream);
  raft::copy(target.centers().data_handle(),
             source.centers().data_handle(),
             source.centers().size(),
             stream);
  if (source.center_norms().has_value()) {
    target.allocate_center_norms(res);
    raft::copy(target.center_norms()->data_handle(),
               source.center_norms()->data_handle(),
               source.center_norms()->size(),
               stream);
  }
  raft::copy(target.sq_vmin().data_handle(),
             source.sq_vmin().data_handle(),
             source.sq_vmin().size(),
             stream);
  raft::copy(target.sq_delta().data_handle(),
             source.sq_delta().data_handle(),
             source.sq_delta().size(),
             stream);
  target.lists() = source.lists();
  ivf::detail::recompute_internal_state(res, target);
  return target;
}

/**
 * Kernel to encode float residuals to uint8_t SQ codes and write them interleaved.
 *
 * Uses warp-per-vector parallelism: each warp cooperatively encodes one vector
 * so that reads from residuals/vmin/delta are coalesced across the 32 lanes.
 * Lane 0 handles the atomic position assignment and the index write.
 */
template <int BlockSize>
__launch_bounds__(BlockSize) RAFT_KERNEL encode_and_fill_kernel(const uint32_t* labels,
                                                                const float* residuals,
                                                                const int64_t* source_ixs,
                                                                uint8_t** list_data_ptrs,
                                                                int64_t** list_index_ptrs,
                                                                uint32_t* list_sizes_ptr,
                                                                const float* vmin,
                                                                const float* delta,
                                                                int64_t n_rows,
                                                                uint32_t dim,
                                                                int64_t batch_offset)
{
  constexpr uint32_t kWarpSize      = kIndexGroupSize;
  constexpr uint32_t kWarpsPerBlock = BlockSize / kWarpSize;

  const uint32_t lane_id = threadIdx.x % kWarpSize;
  const int64_t row_id =
    int64_t(threadIdx.x / kWarpSize) + int64_t(blockIdx.x) * int64_t(kWarpsPerBlock);
  if (row_id >= n_rows) return;

  uint32_t list_id   = 0;
  uint32_t inlist_id = 0;
  if (lane_id == 0) {
    auto source_ix = source_ixs == nullptr ? row_id + batch_offset : source_ixs[row_id];
    list_id        = labels[row_id];
    inlist_id      = atomicAdd(list_sizes_ptr + list_id, 1);
    list_index_ptrs[list_id][inlist_id] = source_ix;
  }
  list_id   = __shfl_sync(0xFFFFFFFF, list_id, 0);
  inlist_id = __shfl_sync(0xFFFFFFFF, inlist_id, 0);

  using interleaved_group = raft::Pow2<kIndexGroupSize>;
  auto group_offset       = interleaved_group::roundDown(inlist_id);
  auto ingroup_id         = interleaved_group::mod(inlist_id);

  constexpr uint32_t veclen = list_spec<uint32_t, uint8_t, int64_t>::kVecLen;
  uint32_t padded_dim       = ((dim + veclen - 1) / veclen) * veclen;
  auto* list_dat   = list_data_ptrs[list_id] + static_cast<size_t>(group_offset) * padded_dim;
  const float* src = residuals + row_id * dim;

  for (uint32_t d = lane_id; d < padded_dim; d += kWarpSize) {
    uint8_t out;
    if (d < dim) {
      float val  = src[d];
      float dv   = delta[d];
      float v    = vmin[d];
      float code = (dv > 0.0f) ? roundf((val - v) / dv) : 0.0f;
      out        = static_cast<uint8_t>(fminf(fmaxf(code, 0.0f), 255.0f));
    } else {
      out = 0;
    }
    uint32_t l                                              = (d / veclen) * veclen;
    uint32_t j                                              = d % veclen;
    list_dat[l * kIndexGroupSize + ingroup_id * veclen + j] = out;
  }
}

/** In-place variant: dataset[i] = cast<T>(cast<float>(dataset[i]) - centers[labels[i]]) */
template <typename T>
RAFT_KERNEL compute_residuals_inplace_kernel(
  T* dataset, const float* centers, const uint32_t* labels, int64_t n_rows, uint32_t dim)
{
  int64_t i = blockIdx.x;
  if (i >= n_rows) return;
  uint32_t c = labels[i];
  for (uint32_t j = threadIdx.x; j < dim; j += blockDim.x) {
    float val            = utils::mapping<float>{}(dataset[i * dim + j]);
    dataset[i * dim + j] = utils::mapping<T>{}(val - centers[c * dim + j]);
  }
}

template <typename T, typename IdxT>
void extend(raft::resources const& handle,
            index<IdxT>* index,
            const T* new_vectors,
            const int64_t* new_indices,
            int64_t n_rows)
{
  using LabelT = uint32_t;
  RAFT_EXPECTS(index != nullptr, "index cannot be empty.");
  if (n_rows == 0) return;

  auto stream  = raft::resource::get_cuda_stream(handle);
  auto n_lists = index->n_lists();
  auto dim     = index->dim();
  list_spec<uint32_t, IdxT, int64_t> list_device_spec{index->dim(),
                                                      index->conservative_memory_allocation()};
  cuvs::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "ivf_sq::extend(%zu, %u)", size_t(n_rows), dim);

  RAFT_EXPECTS(new_indices != nullptr || index->size() == 0,
               "You must pass data indices when the index is non-empty.");

  auto new_labels =
    raft::make_device_mdarray<LabelT>(handle,
                                      raft::resource::get_large_workspace_resource(handle),
                                      raft::make_extents<int64_t>(n_rows));
  cuvs::cluster::kmeans::balanced_params kmeans_params;
  kmeans_params.metric     = index->metric();
  auto orig_centroids_view = raft::make_device_matrix_view<const float, int64_t>(
    index->centers().data_handle(), n_lists, dim);

  constexpr size_t kReasonableMaxBatchSize = 65536;
  size_t max_batch_size                    = std::min<size_t>(n_rows, kReasonableMaxBatchSize);

  auto copy_stream     = raft::resource::get_cuda_stream(handle);
  bool enable_prefetch = false;
  if (handle.has_resource_factory(raft::resource::resource_type::CUDA_STREAM_POOL)) {
    if (raft::resource::get_stream_pool_size(handle) >= 1) {
      enable_prefetch = true;
      copy_stream     = raft::resource::get_stream_from_stream_pool(handle);
    }
  }

  utils::batch_load_iterator<T> vec_batches(new_vectors,
                                            n_rows,
                                            index->dim(),
                                            max_batch_size,
                                            copy_stream,
                                            raft::resource::get_workspace_resource(handle),
                                            enable_prefetch);
  vec_batches.prefetch_next_batch();

  for (const auto& batch : vec_batches) {
    auto batch_data_view =
      raft::make_device_matrix_view<const T, int64_t>(batch.data(), batch.size(), index->dim());
    auto batch_labels_view = raft::make_device_vector_view<LabelT, int64_t>(
      new_labels.data_handle() + batch.offset(), batch.size());
    cuvs::cluster::kmeans::predict(
      handle, kmeans_params, batch_data_view, orig_centroids_view, batch_labels_view);
    vec_batches.prefetch_next_batch();
    raft::resource::sync_stream(handle);
  }

  auto* list_sizes_ptr    = index->list_sizes().data_handle();
  auto old_list_sizes_dev = raft::make_device_vector<uint32_t, int64_t>(handle, n_lists);
  raft::copy(old_list_sizes_dev.data_handle(), list_sizes_ptr, n_lists, stream);

  raft::stats::histogram<uint32_t, int64_t>(raft::stats::HistTypeAuto,
                                            reinterpret_cast<int32_t*>(list_sizes_ptr),
                                            int64_t(n_lists),
                                            new_labels.data_handle(),
                                            n_rows,
                                            1,
                                            stream);
  raft::linalg::add(
    list_sizes_ptr, list_sizes_ptr, old_list_sizes_dev.data_handle(), n_lists, stream);

  std::vector<uint32_t> new_list_sizes(n_lists);
  std::vector<uint32_t> old_list_sizes(n_lists);
  {
    raft::copy(old_list_sizes.data(), old_list_sizes_dev.data_handle(), n_lists, stream);
    raft::copy(new_list_sizes.data(), list_sizes_ptr, n_lists, stream);
    raft::resource::sync_stream(handle);
    auto& lists = index->lists();
    for (uint32_t label = 0; label < n_lists; label++) {
      ivf::resize_list(handle,
                       lists[label],
                       list_device_spec,
                       new_list_sizes[label],
                       raft::Pow2<kIndexGroupSize>::roundUp(old_list_sizes[label]));
    }
  }
  ivf::detail::recompute_internal_state(handle, *index);
  raft::copy(list_sizes_ptr, old_list_sizes_dev.data_handle(), n_lists, stream);

  utils::batch_load_iterator<int64_t> vec_indices(
    new_indices, n_rows, 1, max_batch_size, stream, raft::resource::get_workspace_resource(handle));
  vec_batches.reset();
  vec_batches.prefetch_next_batch();
  utils::batch_load_iterator<int64_t> idx_batch = vec_indices.begin();

  auto residuals_buf = raft::make_device_vector<float>(handle, max_batch_size * dim);

  size_t next_report_offset = 0;
  size_t d_report_offset    = n_rows * 5 / 100;

  for (const auto& batch : vec_batches) {
    int64_t bs = batch.size();

    {
      auto batch_view = raft::make_device_matrix_view<const T, int64_t>(batch.data(), bs, dim);
      auto residuals_view =
        raft::make_device_matrix_view<float, int64_t>(residuals_buf.data_handle(), bs, dim);

      const float* centers_ptr   = index->centers().data_handle();
      const uint32_t* labels_ptr = new_labels.data_handle() + batch.offset();

      raft::linalg::map_offset(
        handle,
        residuals_view,
        [centers_ptr, labels_ptr, dim] __device__(auto idx, T x) {
          auto i = idx / dim;
          auto j = idx % dim;
          return utils::mapping<float>{}(x)-centers_ptr[labels_ptr[i] * dim + j];
        },
        batch_view);
    }

    {
      constexpr int kEncodeBlockSize   = 256;
      constexpr int kEncodeWarpsPerBlk = kEncodeBlockSize / kIndexGroupSize;
      const dim3 block_dim(kEncodeBlockSize);
      const dim3 grid_dim(raft::ceildiv<int64_t>(bs, int64_t(kEncodeWarpsPerBlk)));
      encode_and_fill_kernel<kEncodeBlockSize>
        <<<grid_dim, block_dim, 0, stream>>>(new_labels.data_handle() + batch.offset(),
                                             residuals_buf.data_handle(),
                                             idx_batch->data(),
                                             index->data_ptrs().data_handle(),
                                             index->inds_ptrs().data_handle(),
                                             list_sizes_ptr,
                                             index->sq_vmin().data_handle(),
                                             index->sq_delta().data_handle(),
                                             bs,
                                             dim,
                                             batch.offset());
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    }

    vec_batches.prefetch_next_batch();
    raft::resource::sync_stream(handle);
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    if (batch.offset() > next_report_offset) {
      float progress = batch.offset() * 100.0f / n_rows;
      RAFT_LOG_DEBUG("ivf_sq::extend added vectors %zu, %6.1f%% complete",
                     static_cast<size_t>(batch.offset()),
                     progress);
      next_report_offset += d_report_offset;
    }
    ++idx_batch;
  }

  auto compute_center_norms = [&]() {
    if (index->metric() == cuvs::distance::DistanceType::CosineExpanded) {
      raft::linalg::rowNorm<raft::linalg::L2Norm, true>(index->center_norms()->data_handle(),
                                                        index->centers().data_handle(),
                                                        dim,
                                                        n_lists,
                                                        stream,
                                                        raft::sqrt_op{});
    } else {
      raft::linalg::rowNorm<raft::linalg::L2Norm, true>(
        index->center_norms()->data_handle(), index->centers().data_handle(), dim, n_lists, stream);
    }
  };

  if (!index->center_norms().has_value()) {
    index->allocate_center_norms(handle);
    if (index->center_norms().has_value()) { compute_center_norms(); }
  }
}

template <typename T, typename IdxT>
auto extend(raft::resources const& handle,
            const index<IdxT>& orig_index,
            const T* new_vectors,
            const int64_t* new_indices,
            int64_t n_rows) -> index<IdxT>
{
  auto ext_index = clone(handle, orig_index);
  detail::extend(handle, &ext_index, new_vectors, new_indices, n_rows);
  return ext_index;
}

template <typename T, typename IdxT, typename accessor>
inline auto build(
  raft::resources const& handle,
  const index_params& params,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, accessor> dataset)
  -> index<IdxT>
{
  int64_t n_rows = dataset.extent(0);
  uint32_t dim   = dataset.extent(1);
  auto stream    = raft::resource::get_cuda_stream(handle);
  cuvs::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "ivf_sq::build(%zu, %u)", size_t(n_rows), dim);
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, half>, "unsupported data type");
  RAFT_EXPECTS(n_rows > 0 && dim > 0, "empty dataset");
  RAFT_EXPECTS(n_rows >= params.n_lists, "number of rows can't be less than n_lists");
  RAFT_EXPECTS(params.metric != cuvs::distance::DistanceType::CosineExpanded || dim > 1,
               "Cosine metric requires more than one dim");

  index<IdxT> idx(handle, params, dim);

  // Train k-means centroids and SQ parameters on the same training subset.
  // This mirrors IVF-PQ, which also trains its codebook on a subset of the data.
  {
    raft::random::RngState random_state{137};
    auto trainset_ratio = std::max<size_t>(
      1, n_rows / std::max<size_t>(params.kmeans_trainset_fraction * n_rows, idx.n_lists()));
    auto n_rows_train = n_rows / trainset_ratio;
    auto trainset =
      raft::make_device_mdarray<T>(handle,
                                   raft::resource::get_large_workspace_resource(handle),
                                   raft::make_extents<int64_t>(n_rows_train, idx.dim()));
    raft::matrix::sample_rows<T, int64_t>(handle, random_state, dataset, trainset.view());
    auto trainset_const_view = raft::make_const_mdspan(trainset.view());
    auto centers_view        = raft::make_device_matrix_view<float, int64_t>(
      idx.centers().data_handle(), idx.n_lists(), idx.dim());
    cuvs::cluster::kmeans::balanced_params kmeans_params;
    kmeans_params.n_iters = params.kmeans_n_iters;
    kmeans_params.metric  = idx.metric();
    cuvs::cluster::kmeans::fit(handle, kmeans_params, trainset_const_view, centers_view);

    // Train SQ: predict labels for the training subset, compute residuals in-place,
    // and derive per-dimension vmin/delta from them.
    {
      auto train_labels = raft::make_device_vector<uint32_t, int64_t>(handle, n_rows_train);
      cuvs::cluster::kmeans::balanced_params pred_params;
      pred_params.metric      = idx.metric();
      auto centers_const_view = raft::make_device_matrix_view<const float, int64_t>(
        idx.centers().data_handle(), idx.n_lists(), dim);
      cuvs::cluster::kmeans::predict(
        handle, pred_params, trainset_const_view, centers_const_view, train_labels.view());

      constexpr int kResidualBlockSize = 256;
      compute_residuals_inplace_kernel<T>
        <<<n_rows_train, kResidualBlockSize, 0, stream>>>(trainset.data_handle(),
                                                          idx.centers().data_handle(),
                                                          train_labels.data_handle(),
                                                          n_rows_train,
                                                          dim);
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    }

    // After the in-place kernel, trainset now contains residuals.
    auto& residuals = trainset;

    {
      auto vmax_buf  = raft::make_device_vector<float, uint32_t>(handle, dim);
      auto* vmin_ptr = idx.sq_vmin().data_handle();
      auto* vmax_ptr = vmax_buf.data_handle();

      constexpr int kMinMaxBlockSize = 256;
      fused_column_minmax_kernel<kMinMaxBlockSize><<<dim, kMinMaxBlockSize, 0, stream>>>(
        residuals.data_handle(), vmin_ptr, vmax_ptr, n_rows_train, dim);
      RAFT_CUDA_TRY(cudaPeekAtLastError());

      // Expand the observed range by a small margin to reduce clipping on unseen data,
      // since the SQ parameters are trained on a subset rather than the full dataset.
      constexpr float kMargin = 0.05f;
      auto* delta_ptr         = idx.sq_delta().data_handle();
      raft::linalg::map_offset(
        handle, idx.sq_vmin(), [vmin_ptr, vmax_ptr, delta_ptr, kMargin] __device__(uint32_t j) {
          float range  = vmax_ptr[j] - vmin_ptr[j];
          float margin = range * kMargin;
          delta_ptr[j] = (range > 0.0f) ? (range + 2.0f * margin) / 255.0f : 1.0f;
          return vmin_ptr[j] - margin;
        });
    }
  }

  if (params.add_data_on_build) {
    detail::extend<T, IdxT>(handle, &idx, dataset.data_handle(), nullptr, n_rows);
  }

  return idx;
}

template <typename T, typename IdxT>
void build(raft::resources const& handle,
           const index_params& params,
           raft::device_matrix_view<const T, int64_t, raft::row_major> dataset,
           index<IdxT>& idx)
{
  idx = build<T, IdxT>(handle, params, dataset);
}

template <typename T, typename IdxT>
void build(raft::resources const& handle,
           const index_params& params,
           raft::host_matrix_view<const T, int64_t, raft::row_major> dataset,
           index<IdxT>& idx)
{
  idx = build<T, IdxT>(handle, params, dataset);
}

template <typename T, typename IdxT>
auto extend(raft::resources const& handle,
            raft::device_matrix_view<const T, int64_t, raft::row_major> new_vectors,
            std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
            const index<IdxT>& orig_index) -> index<IdxT>
{
  RAFT_EXPECTS(new_vectors.extent(1) == orig_index.dim(),
               "new_vectors should have the same dimension as the index");
  if (new_indices.has_value()) {
    RAFT_EXPECTS(new_indices.value().extent(0) == new_vectors.extent(0),
                 "new_vectors and new_indices have different number of rows");
  }
  int64_t n_rows = new_vectors.extent(0);
  return extend<T, IdxT>(handle,
                         orig_index,
                         new_vectors.data_handle(),
                         new_indices.has_value() ? new_indices.value().data_handle() : nullptr,
                         n_rows);
}

template <typename T, typename IdxT>
auto extend(raft::resources const& handle,
            raft::host_matrix_view<const T, int64_t, raft::row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
            const index<IdxT>& orig_index) -> index<IdxT>
{
  RAFT_EXPECTS(new_vectors.extent(1) == orig_index.dim(),
               "new_vectors should have the same dimension as the index");
  if (new_indices.has_value()) {
    RAFT_EXPECTS(new_indices.value().extent(0) == new_vectors.extent(0),
                 "new_vectors and new_indices have different number of rows");
  }
  int64_t n_rows = new_vectors.extent(0);
  return extend<T, IdxT>(handle,
                         orig_index,
                         new_vectors.data_handle(),
                         new_indices.has_value() ? new_indices.value().data_handle() : nullptr,
                         n_rows);
}

template <typename T, typename IdxT>
void extend(raft::resources const& handle,
            raft::device_matrix_view<const T, int64_t, raft::row_major> new_vectors,
            std::optional<raft::device_vector_view<const int64_t, int64_t>> new_indices,
            index<IdxT>* idx)
{
  RAFT_EXPECTS(new_vectors.extent(1) == idx->dim(),
               "new_vectors should have the same dimension as the index");
  if (new_indices.has_value()) {
    RAFT_EXPECTS(new_indices.value().extent(0) == new_vectors.extent(0),
                 "new_vectors and new_indices have different number of rows");
  }
  detail::extend(handle,
                 idx,
                 new_vectors.data_handle(),
                 new_indices.has_value() ? new_indices.value().data_handle() : nullptr,
                 new_vectors.extent(0));
}

template <typename T, typename IdxT>
void extend(raft::resources const& handle,
            raft::host_matrix_view<const T, int64_t, raft::row_major> new_vectors,
            std::optional<raft::host_vector_view<const int64_t, int64_t>> new_indices,
            index<IdxT>* idx)
{
  RAFT_EXPECTS(new_vectors.extent(1) == idx->dim(),
               "new_vectors should have the same dimension as the index");
  if (new_indices.has_value()) {
    RAFT_EXPECTS(new_indices.value().extent(0) == new_vectors.extent(0),
                 "new_vectors and new_indices have different number of rows");
  }
  detail::extend(handle,
                 idx,
                 new_vectors.data_handle(),
                 new_indices.has_value() ? new_indices.value().data_handle() : nullptr,
                 new_vectors.extent(0));
}

}  // namespace detail
}  // namespace cuvs::neighbors::ivf_sq
