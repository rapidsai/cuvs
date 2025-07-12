/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/binary_ivf.hpp>
#include <../ivf_pq/ivf_pq_build.cuh>

#include <cuvs/preprocessing/quantize/binary.hpp>

#include "../../cluster/kmeans_balanced.cuh"
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
#include <raft/stats/histogram.cuh>
#include <raft/util/pow2_utils.cuh>

#include <rmm/cuda_stream_view.hpp>

#include <cstdint>

namespace cuvs::neighbors::binary_ivf {
using namespace cuvs::spatial::knn::detail;  // NOLINT

namespace detail {

template <typename IdxT>
auto clone(const raft::resources& res, const index<IdxT>& source) -> index<IdxT>
{
  auto stream = raft::resource::get_cuda_stream(res);

  // Allocate the new index
  index<IdxT> target(res,
                        source.n_lists(),
                        source.adaptive_centers(),
                        source.conservative_memory_allocation(),
                        source.dim());

  // Copy the independent parts
  raft::copy(target.list_sizes().data_handle(),
             source.list_sizes().data_handle(),
             source.list_sizes().size(),
             stream);
  raft::copy(target.centers().data_handle(),
             source.centers().data_handle(),
             source.centers().size(),
             stream);
  // Copy shared pointers
  target.lists() = source.lists();

  // Make sure the device pointers point to the new lists
  ivf::detail::recompute_internal_state(res, target);

  return target;
}


/** See raft::neighbors::binary_ivf::extend docs */
template <typename IdxT>
void extend(raft::resources const& handle,
            index<IdxT>* index,
            const uint8_t* new_vectors,
            const IdxT* new_indices,
            IdxT n_rows)
{
  using LabelT = uint32_t;
  RAFT_EXPECTS(index != nullptr, "index cannot be empty.");

  auto stream  = raft::resource::get_cuda_stream(handle);
  auto n_lists = index->n_lists();
  auto dim     = index->dim();
  cuvs::neighbors::ivf_pq::list_spec<uint32_t, IdxT> list_device_spec{8, dim, index->conservative_memory_allocation()};
  cuvs::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "binary_ivf::extend(%zu, %u)", size_t(n_rows), dim);

  RAFT_EXPECTS(new_indices != nullptr || index->size() == 0,
               "You must pass data indices when the index is non-empty.");

  auto new_labels = raft::make_device_mdarray<LabelT>(
    handle, raft::resource::get_large_workspace_resource(handle), raft::make_extents<IdxT>(n_rows));
  cuvs::cluster::kmeans::balanced_params kmeans_params;
  kmeans_params.metric = cuvs::distance::DistanceType::BitwiseHamming;
  auto orig_centroids_view =
    raft::make_device_matrix_view<const uint8_t, IdxT>(index->centers().data_handle(), n_lists, dim);
  // Calculate the batch size for the input data if it's not accessible directly from the device
  constexpr size_t kReasonableMaxBatchSize = 65536;
  size_t max_batch_size                    = std::min<size_t>(n_rows, kReasonableMaxBatchSize);

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
  utils::batch_load_iterator<uint8_t> vec_batches(new_vectors,
                                            n_rows,
                                            index->dim(),
                                            max_batch_size,
                                            copy_stream,
                                            raft::resource::get_workspace_resource(handle),
                                            enable_prefetch);
  vec_batches.prefetch_next_batch();

  for (const auto& batch : vec_batches) {
    auto batch_data_view =
      raft::make_device_matrix_view<const uint8_t, IdxT>(batch.data(), batch.size(), index->dim());
    auto batch_labels_view = raft::make_device_vector_view<LabelT, IdxT>(
      new_labels.data_handle() + batch.offset(), batch.size());
    cuvs::cluster::kmeans_balanced::predict(handle,
                                            kmeans_params,
                                            batch_data_view,
                                            orig_centroids_view,
                                            batch_labels_view,
                                            utils::mapping<uint8_t>{});
    vec_batches.prefetch_next_batch();
    // User needs to make sure kernel finishes its work before we overwrite batch in the next
    // iteration if different streams are used for kernel and copy.
    raft::resource::sync_stream(handle);
  }

  auto* list_sizes_ptr    = index->list_sizes().data_handle();
  auto old_list_sizes_dev = raft::make_device_mdarray<uint32_t>(
    handle, raft::resource::get_workspace_resource(handle), raft::make_extents<IdxT>(n_lists));
  raft::copy(old_list_sizes_dev.data_handle(), list_sizes_ptr, n_lists, stream);

  // Calculate the centers and sizes on the new data, starting from the original values
  if (index->adaptive_centers()) {
    auto centroids_view = raft::make_device_matrix_view<float, IdxT>(
      index->centers().data_handle(), index->centers().extent(0), index->centers().extent(1));
    auto list_sizes_view =
      raft::make_device_vector_view<std::remove_pointer_t<decltype(list_sizes_ptr)>, IdxT>(
        list_sizes_ptr, n_lists);
    for (const auto& batch : vec_batches) {
      auto batch_data_view =
        raft::make_device_matrix_view<const uint8_t, IdxT>(batch.data(), batch.size(), index->dim());
      auto batch_labels_view = raft::make_device_vector_view<const LabelT, IdxT>(
        new_labels.data_handle() + batch.offset(), batch.size());
      cuvs::cluster::kmeans_balanced::helpers::calc_centers_and_sizes(handle,
                                                                      batch_data_view,
                                                                      batch_labels_view,
                                                                      centroids_view,
                                                                      list_sizes_view,
                                                                      false,
                                                                      utils::mapping<float>{});
    }
  } else {
    raft::stats::histogram<uint32_t, IdxT>(raft::stats::HistTypeAuto,
                                           reinterpret_cast<int32_t*>(list_sizes_ptr),
                                           IdxT(n_lists),
                                           new_labels.data_handle(),
                                           n_rows,
                                           1,
                                           stream);
    raft::linalg::add(
      list_sizes_ptr, list_sizes_ptr, old_list_sizes_dev.data_handle(), n_lists, stream);
  }

  // Calculate and allocate new list data
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
  // Update the pointers and the sizes
  ivf::detail::recompute_internal_state(handle, *index);
  // Copy the old sizes, so we can start from the current state of the index;
  // we'll rebuild the `list_sizes_ptr` in the following kernel, using it as an atomic counter.
  raft::copy(list_sizes_ptr, old_list_sizes_dev.data_handle(), n_lists, stream);

  utils::batch_load_iterator<IdxT> vec_indices(
    new_indices, n_rows, 1, max_batch_size, stream, raft::resource::get_workspace_resource(handle));
  vec_batches.reset();
  vec_batches.prefetch_next_batch();
  utils::batch_load_iterator<IdxT> idx_batch = vec_indices.begin();
  size_t next_report_offset                  = 0;
  size_t d_report_offset                     = n_rows * 5 / 100;
  for (const auto& batch : vec_batches) {
    auto batch_data_view =
      raft::make_device_matrix_view<const T, IdxT>(batch.data(), batch.size(), index->dim());
    // Kernel to insert the new vectors
    const dim3 block_dim(256);
    const dim3 grid_dim(raft::ceildiv<IdxT>(batch.size(), block_dim.x));
    build_index_kernel<T, IdxT, LabelT>
      <<<grid_dim, block_dim, 0, stream>>>(new_labels.data_handle() + batch.offset(),
                                           batch_data_view.data_handle(),
                                           idx_batch->data(),
                                           index->data_ptrs().data_handle(),
                                           index->inds_ptrs().data_handle(),
                                           list_sizes_ptr,
                                           batch.size(),
                                           dim,
                                           index->veclen(),
                                           batch.offset());
    vec_batches.prefetch_next_batch();
    // User needs to make sure kernel finishes its work before we overwrite batch in the next
    // iteration if different streams are used for kernel and copy.
    raft::resource::sync_stream(handle);
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    if (batch.offset() > next_report_offset) {
      float progress = batch.offset() * 100.0f / n_rows;
      RAFT_LOG_DEBUG("ivf_flat::extend added vectors %zu, %6.1f%% complete",
                     static_cast<size_t>(batch.offset()),
                     progress);
      next_report_offset += d_report_offset;
    }
    ++idx_batch;
  }
}


/** See raft::neighbors::ivf_flat::extend docs */
template <typename IdxT>
auto extend(raft::resources const& handle,
            const index<IdxT>& orig_index,
            const uint8_t* new_vectors,
            const IdxT* new_indices,
            IdxT n_rows) -> index<IdxT>
{
  auto ext_index = clone(handle, orig_index);
  detail::extend(handle, &ext_index, new_vectors, new_indices, n_rows);
  return ext_index;
}

/** See raft::neighbors::ivf_flat::build docs */
template <typename IdxT>
inline auto build(raft::resources const& handle,
                  const index_params& params,
                  const uint8_t* dataset,
                  IdxT n_rows,
                  uint32_t dim) -> index<IdxT>
{
  auto stream = raft::resource::get_cuda_stream(handle);
  cuvs::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "binary_ivf::build(%zu, %u)", size_t(n_rows), dim);
  RAFT_EXPECTS(n_rows > 0 && dim > 0, "empty dataset");
  RAFT_EXPECTS(n_rows >= params.n_lists, "number of rows can't be less than n_lists");
  index<IdxT> index(handle, params, dim);
  utils::memzero(
    index.accum_sorted_sizes().data_handle(), index.accum_sorted_sizes().size(), stream);
  utils::memzero(index.list_sizes().data_handle(), index.list_sizes().size(), stream);
  utils::memzero(index.data_ptrs().data_handle(), index.data_ptrs().size(), stream);
  utils::memzero(index.inds_ptrs().data_handle(), index.inds_ptrs().size(), stream);

  // Train the kmeans clustering
  {
    auto trainset_ratio = std::max<size_t>(
      1, n_rows / std::max<size_t>(params.kmeans_trainset_fraction * n_rows, index.n_lists()));
    auto n_rows_train = n_rows / trainset_ratio;
    rmm::device_uvector<uint8_t> trainset(
      n_rows_train * index.dim(), stream, raft::resource::get_large_workspace_resource(handle));
    // TODO: a proper sampling
    RAFT_CUDA_TRY(cudaMemcpy2DAsync(trainset.data(),
                                    sizeof(T) * index.dim(),
                                    dataset,
                                    sizeof(T) * index.dim() * trainset_ratio,
                                    sizeof(T) * index.dim(),
                                    n_rows_train,
                                    cudaMemcpyDefault,
                                    stream));

    cuvs::cluster::kmeans::balanced_params kmeans_params;
    kmeans_params.n_iters = params.kmeans_n_iters;
    kmeans_params.metric = cuvs::distance::DistanceType::L2Expanded;
    rmm::device_uvector<int8_t> decoded_trainset(
      n_rows_train * index.dim() * 8, stream, raft::resource::get_large_workspace_resource(handle));
    auto decoded_trainset_view = raft::make_device_matrix_view<int8_t, IdxT>(reinterpret_cast<int8_t*>(decoded_trainset.data()), n_rows_train, index.dim() * 8);
    raft::linalg::map_offset(handle, decoded_trainset_view, bitwise_decode_op<IdxT>(trainset.data(), index.dim()));
    trainset.clear();
    rmm::device_uvector<float> decoded_centers(
      index.n_lists() * index.dim() * 8, stream, raft::resource::get_workspace_resource(handle));
    auto decoded_centers_view = raft::make_device_matrix_view<float, IdxT>(decoded_centers.data(), index.n_lists(), index.dim() * 8);
    cuvs::cluster::kmeans_balanced::fit(
      handle, kmeans_params, raft::make_const_mdspan(decoded_trainset_view), decoded_centers_view);
    cuvs::preprocessing::quantize::binary::params binary_params;
    auto quantizer = cuvs::preprocessing::quantize::binary::train(handle, binary_params, decoded_centers_view);
    cuvs::preprocessing::quantize::binary::transform(handle, quantizer, decoded_centers_view, index.centers());
  }

  // add the data if necessary
  if (params.add_data_on_build) {
    detail::extend<IdxT>(handle, &index, dataset, nullptr, n_rows);
  }
  return index;
}

template <typename IdxT>
auto build(raft::resources const& handle,
           const index_params& params,
           raft::device_matrix_view<const uint8_t, IdxT, raft::row_major> dataset) -> index<IdxT>
{
  IdxT n_rows = dataset.extent(0);
  IdxT dim    = dataset.extent(1);
  return build(handle, params, dataset.data_handle(), n_rows, dim);
}

template <typename IdxT>
auto build(raft::resources const& handle,
           const index_params& params,
           raft::host_matrix_view<const uint8_t, IdxT, raft::row_major> dataset) -> index<IdxT>
{
  IdxT n_rows = dataset.extent(0);
  IdxT dim    = dataset.extent(1);
  return build(handle, params, dataset.data_handle(), n_rows, dim);
}

template <typename IdxT>
void build(raft::resources const& handle,
           const index_params& params,
           raft::device_matrix_view<const uint8_t, IdxT, raft::row_major> dataset,
           index<IdxT>& index)
{
  IdxT n_rows = dataset.extent(0);
  IdxT dim    = dataset.extent(1);
  index       = build(handle, params, dataset.data_handle(), n_rows, dim);
}

template <typename IdxT>
void build(raft::resources const& handle,
           const index_params& params,
           raft::host_matrix_view<const uint8_t, IdxT, raft::row_major> dataset,
           index<IdxT>& index)
{
  IdxT n_rows = dataset.extent(0);
  IdxT dim    = dataset.extent(1);
  index       = build(handle, params, dataset.data_handle(), n_rows, dim);
}

template <typename IdxT>
auto extend(raft::resources const& handle,
            raft::device_matrix_view<const uint8_t, IdxT, raft::row_major> new_vectors,
            std::optional<raft::device_vector_view<const IdxT, IdxT>> new_indices,
            const cuvs::neighbors::binary_ivf::index<IdxT>& orig_index) -> index<IdxT>
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

template <typename IdxT>
auto extend(raft::resources const& handle,
            raft::host_matrix_view<const uint8_t, IdxT, raft::row_major> new_vectors,
            std::optional<raft::host_vector_view<const IdxT, IdxT>> new_indices,
            const cuvs::neighbors::binary_ivf::index<IdxT>& orig_index) -> index<IdxT>
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

template <typename IdxT>
void extend(raft::resources const& handle,
            raft::device_matrix_view<const uint8_t, IdxT, raft::row_major> new_vectors,
            std::optional<raft::device_vector_view<const IdxT, IdxT>> new_indices,
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

template <typename IdxT>
void extend(raft::resources const& handle,
            raft::host_matrix_view<const uint8_t, IdxT, raft::row_major> new_vectors,
            std::optional<raft::host_vector_view<const IdxT, IdxT>> new_indices,
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

// Example: Using IVF-PQ's pack_list_data to write binary codes into IVF lists
// This can be used in your binary_ivf build process to directly pack uint8_t codes
// without any PQ processing.

template <typename IdxT>
void pack_binary_codes_into_ivf_lists(
  raft::resources const& handle,
  // Your binary IVF index - you'll need to adapt this to your index type
  auto* binary_index,  // Replace with your actual binary IVF index type
  const uint8_t* binary_codes,        // Your binary codes [n_rows, dim_bytes]
  const uint32_t* cluster_labels,     // Cluster assignment for each vector [n_rows]
  IdxT n_rows,
  IdxT dim_bytes,                     // Number of bytes per vector (e.g., dim/8 for binary)
  uint32_t cluster_id)                // Which cluster/list to write to
{
  // Create a device matrix view of your binary codes
  auto codes_view = raft::make_device_matrix_view<const uint8_t, uint32_t, raft::row_major>(
    binary_codes, n_rows, dim_bytes);
  
  // For binary IVF, we don't need PQ encoding, so pq_vectors = null
  // The binary codes are already in the format we want (uint8_t per byte)
  
  // Call IVF-PQ's pack_list_data function
  // Note: You'll need to include the IVF-PQ header and adapt the index type
  cuvs::neighbors::ivf_pq::detail::pack_list_data(
    handle,
    binary_index,        // Your binary IVF index (adapt to your index type)
    codes_view,         // Your binary codes
    cluster_id,         // Which cluster/list to write to
    uint32_t(0)         // Offset in the list (start from beginning)
  );
}

// Alternative: If you want to write to multiple lists based on cluster labels
template <typename IdxT>
void pack_binary_codes_into_multiple_ivf_lists(
  raft::resources const& handle,
  auto* binary_index,                 // Your binary IVF index
  const uint8_t* binary_codes,        // Your binary codes [n_rows, dim_bytes]  
  const uint32_t* cluster_labels,     // Cluster assignment for each vector [n_rows]
  IdxT n_rows,
  IdxT dim_bytes,
  uint32_t n_lists)
{
  // Process each cluster/list
  for (uint32_t cluster_id = 0; cluster_id < n_lists; cluster_id++) {
    // Count vectors in this cluster
    uint32_t cluster_size = 0;
    for (IdxT i = 0; i < n_rows; i++) {
      if (cluster_labels[i] == cluster_id) cluster_size++;
    }
    
    if (cluster_size == 0) continue;
    
    // Allocate temporary buffer for this cluster's codes
    auto cluster_codes = raft::make_device_matrix<uint8_t, uint32_t, raft::row_major>(
      handle, cluster_size, dim_bytes);
    
    // Copy codes for this cluster
    uint32_t cluster_offset = 0;
    for (IdxT i = 0; i < n_rows; i++) {
      if (cluster_labels[i] == cluster_id) {
        raft::copy(cluster_codes.data_handle() + cluster_offset * dim_bytes,
                  binary_codes + i * dim_bytes,
                  dim_bytes,
                  raft::resource::get_cuda_stream(handle));
        cluster_offset++;
      }
    }
    
    // Pack codes into this cluster's list
    cuvs::neighbors::ivf_pq::detail::pack_list_data(
      handle,
      binary_index,
      cluster_codes.view(),
      cluster_id,
      uint32_t(0)  // Start from beginning of list
    );
  }
}

}  // namespace detail
}  // namespace cuvs::neighbors::ivf_flat
