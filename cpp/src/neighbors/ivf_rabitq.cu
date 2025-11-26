/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/neighbors/ivf_rabitq.hpp>
#include <cuvs/neighbors/ivf_rabitq/gpu_index/searcher_gpu.cuh>

#include "../cluster/kmeans_balanced.cuh"
#include "../core/nvtx.hpp"
#include "detail/ann_utils.cuh"

#include <raft/core/operators.hpp>
#include <raft/linalg/map.cuh>
#include <raft/linalg/reduce.cuh>

#include <raft/util/cudart_utils.hpp>

namespace cuvs::neighbors::ivf_rabitq {

namespace detail {
using namespace cuvs::spatial::knn::detail;  // NOLINT

template <typename T, typename IdxT, typename accessor>
void build(raft::resources const& handle,
           const index_params& params,
           raft::mdspan<const T, raft::matrix_extent<IdxT>, raft::row_major, accessor> dataset,
           cuvs::neighbors::ivf_rabitq::index<int64_t>* index)
{
  IdxT n_rows = dataset.extent(0);
  IdxT dim    = dataset.extent(1);
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "ivf_rabitq::build(%zu, %u)", size_t(n_rows), dim);
  static_assert(std::is_same_v<T, float>, "Unsupported data type");

  RAFT_EXPECTS(n_rows > 0 && dim > 0, "empty dataset");
  RAFT_EXPECTS(n_rows >= params.n_lists, "number of rows can't be less than n_lists");

  rmm::device_async_resource_ref device_memory = raft::resource::get_workspace_resource(handle);
  // If the dataset is small enough to comfortably fit into device memory, put it there.
  // Otherwise, use the managed memory.
  constexpr size_t kTolerableRatio = 4;
  rmm::device_async_resource_ref big_memory_resource =
    raft::resource::get_large_workspace_resource(handle);
  if (sizeof(T) * n_rows * dim * kTolerableRatio <
      raft::resource::get_workspace_free_bytes(handle)) {
    big_memory_resource = device_memory;
  }

  // create device view of dataset
  auto d_dataset_array =
    raft::make_device_mdarray<T>(handle, big_memory_resource, raft::make_extents<int64_t>(0, 0));
  auto d_dataset_view =
    raft::make_mdspan(dataset.data_handle(), raft::make_extents<int64_t>(n_rows, dim));
  auto stream = raft::resource::get_cuda_stream(handle);
  if (utils::check_pointer_residency(dataset.data_handle()) ==
      utils::pointer_residency::device_only) {
    d_dataset_view = dataset;
  } else {
    try {
      d_dataset_array = raft::make_device_mdarray<T>(
        handle, big_memory_resource, raft::make_extents<int64_t>(n_rows, dim));
    } catch (raft::logic_error& e) {
      RAFT_LOG_ERROR(
        "Insufficient memory for kmeans clustering. Please decrease "
        "dataset size, or set large_workspace_resource appropriately.");
      throw;
    }
    d_dataset_view = d_dataset_array.view();
    raft::copy(d_dataset_array.view().data_handle(), dataset.data_handle(), n_rows * dim, stream);
  }

  // perform k-means clustering (currently using the entire dataset)
  // NB: here cluster_centers is used as if it is [n_clusters, data_dim] not [n_clusters,
  // dim_ext]!
  rmm::device_uvector<float> cluster_centers_buf(params.n_lists * dim, stream, device_memory);
  auto cluster_centers      = cluster_centers_buf.data();
  auto d_dataset_const_view = raft::make_const_mdspan(d_dataset_view);
  auto centers_view =
    raft::make_device_matrix_view<float, int64_t>(cluster_centers, params.n_lists, dim);
  cuvs::cluster::kmeans::balanced_params kmeans_params;
  kmeans_params.n_iters = params.kmeans_n_iters;
  kmeans_params.metric  = cuvs::distance::DistanceType::L2Expanded;
  cuvs::cluster::kmeans_balanced::fit(
    handle, kmeans_params, d_dataset_const_view, centers_view, utils::mapping<float>{});
  // find cluster labels for dataset vectors
  rmm::device_uvector<uint32_t> labels(n_rows, stream, big_memory_resource);
  auto centers_const_view =
    raft::make_device_matrix_view<const float, int64_t>(cluster_centers, params.n_lists, dim);
  auto labels_view = raft::make_device_vector_view<uint32_t, int64_t>(labels.data(), n_rows);
  cuvs::cluster::kmeans_balanced::predict(handle,
                                          kmeans_params,
                                          d_dataset_const_view,
                                          centers_const_view,
                                          labels_view,
                                          utils::mapping<float>());

  // TODO: make IVFGPU::construct work on device data only
  T* h_dataset_ptr     = nullptr;
  auto h_dataset_array = raft::make_host_mdarray<T>(raft::make_extents<int64_t>(0, 0));
  if constexpr (raft::is_host_mdspan_v<decltype(dataset)>) {
    h_dataset_ptr = dataset.data_handle();
  } else {
    h_dataset_array = raft::make_host_mdarray<T>(raft::make_extents<int64_t>(n_rows, dim));
    raft::copy(h_dataset_array.view().data_handle(), dataset.data_handle(), n_rows * dim, stream);
    h_dataset_ptr = h_dataset_array.data_handle();
  }
  auto h_centers_array =
    raft::make_host_mdarray<float>(raft::make_extents<int64_t>(params.n_lists, dim));
  raft::copy(h_centers_array.view().data_handle(), cluster_centers, params.n_lists * dim, stream);
  auto h_labels_array = raft::make_host_mdarray<uint32_t>(raft::make_extents<int64_t>(n_rows));
  raft::copy(h_labels_array.view().data_handle(), labels_view.data_handle(), n_rows, stream);
  // Call RaBitQ index construct
  index->rabitq_index()->construct(h_dataset_ptr,
                                   h_centers_array.view().data_handle(),
                                   h_labels_array.view().data_handle(),
                                   params.fast_quantize_flag);
}

template <typename T, typename IdxT>
void search(raft::resources const& handle,
            const search_params& params,
            index<IdxT>& idx,
            raft::device_matrix_view<const T, IdxT, raft::row_major> queries,
            raft::device_matrix_view<IdxT, IdxT, raft::row_major> neighbors,
            raft::device_matrix_view<float, IdxT, raft::row_major> distances)
{
  auto stream = raft::resource::get_cuda_stream(handle).value();

  size_t NQ           = queries.extent(0);
  size_t dim          = queries.extent(1);
  auto rabitq_idx     = idx.rabitq_index();
  auto padded_dim     = rabitq_idx->padded_dim();
  auto padded_queries = raft::make_device_matrix<T, int64_t>(handle, NQ, padded_dim);
  RAFT_CUDA_TRY(cudaMemcpy2DAsync(padded_queries.data_handle(),
                                  sizeof(T) * padded_dim,
                                  queries.data_handle(),
                                  sizeof(T) * dim,
                                  sizeof(T) * dim,
                                  NQ,
                                  cudaMemcpyDefault,
                                  stream));
  raft::resource::sync_stream(handle);

  auto rotated_queries = raft::make_device_matrix<T, int64_t>(handle, NQ, padded_dim);

  // TODO: replace RotatorGPU::rotate with cuVS/RAFT primitives
  rabitq_idx->rotator().rotate(padded_queries.data_handle(), rotated_queries.data_handle(), NQ);

  auto search_mode_to_string = [](search_mode mode) -> std::string {
    switch (mode) {
      case search_mode::LUT16: return std::string("lut16");
      case search_mode::LUT32: return std::string("lut32");
      case search_mode::QUANT4: return std::string("quant4");
      case search_mode::QUANT8: return std::string("quant8");
      default: RAFT_FAIL("Invalid search mode");
    }
  };
  detail::SearcherGPU searcher(handle,
                               rotated_queries.data_handle(),
                               padded_dim,
                               rabitq_idx->ex_bits,
                               search_mode_to_string(params.mode),
                               /* rabitq_quantize_flag = */ true);

  // find the longest cluster to allocate space
  size_t max_cluster_length = 0;
  for (auto i : rabitq_idx->h_cluster_meta) {
    max_cluster_length = max(max_cluster_length, i.num);
  }
  // TODO: this should be part of the load function
  rabitq_idx->max_cluster_length = max_cluster_length;

  auto k = neighbors.extent(1);
  searcher.AllocateSearcherSpace(*rabitq_idx, NQ, k, params.n_probes, max_cluster_length);

  float* d_topk_dists;
  uint32_t *d_topk_ids, *d_final_ids;
  RAFT_CUDA_TRY(cudaMallocAsync(&d_topk_dists, NQ * params.n_probes * k * sizeof(float), stream));
  RAFT_CUDA_TRY(cudaMallocAsync(&d_topk_ids, NQ * params.n_probes * k * sizeof(uint32_t), stream));
  RAFT_CUDA_TRY(cudaMallocAsync(&d_final_ids, NQ * k * sizeof(uint32_t), stream));

  if (params.mode == search_mode::LUT32) {
    rabitq_idx->BatchClusterSearch(rotated_queries.data_handle(),
                                   k,
                                   params.n_probes,
                                   &searcher,
                                   NQ,
                                   d_topk_dists,
                                   distances.data_handle(),
                                   d_topk_ids,
                                   d_final_ids);
  } else if (params.mode == search_mode::LUT16) {
    // test v3 lut using fp16
    rabitq_idx->BatchClusterSearchLUT16(rotated_queries.data_handle(),
                                        k,
                                        params.n_probes,
                                        &searcher,
                                        NQ,
                                        d_topk_dists,
                                        distances.data_handle(),
                                        d_topk_ids,
                                        d_final_ids);
  } else if (params.mode == search_mode::QUANT8) {
    rabitq_idx->BatchClusterSearchQuantizeQuery(rotated_queries.data_handle(),
                                                k,
                                                params.n_probes,
                                                &searcher,
                                                NQ,
                                                d_topk_dists,
                                                distances.data_handle(),
                                                d_topk_ids,
                                                d_final_ids,
                                                8);
  } else if (params.mode == search_mode::QUANT4) {
    rabitq_idx->BatchClusterSearchQuantizeQuery(rotated_queries.data_handle(),
                                                k,
                                                params.n_probes,
                                                &searcher,
                                                NQ,
                                                d_topk_dists,
                                                distances.data_handle(),
                                                d_topk_ids,
                                                d_final_ids,
                                                4);
  }

  // cast data in d_final_ids to array of IdxT in neighbors
  raft::linalg::map(handle,
                    neighbors,
                    raft::cast_op<IdxT>{},
                    raft::make_device_vector_view<const uint32_t, IdxT>(d_final_ids, NQ * k));

  RAFT_CUDA_TRY(cudaFreeAsync(d_topk_dists, stream));
  RAFT_CUDA_TRY(cudaFreeAsync(d_topk_ids, stream));
  RAFT_CUDA_TRY(cudaFreeAsync(d_final_ids, stream));
}

template <typename IdxT>
void serialize(raft::resources const& handle, const std::string& filename, index<IdxT>& index)
{
  // Save the index to a file.
  index.rabitq_index()->save(filename.c_str(), /* save_batch_flag = */ true);
}

template <typename IdxT>
void deserialize(raft::resources const& handle,
                 const std::string& filename,
                 cuvs::neighbors::ivf_rabitq::index<IdxT>* index)
{
  index->rabitq_index()->load_transposed(filename.c_str());
}

}  // namespace detail

template <typename IdxT>
index<IdxT>::index(raft::resources const& handle)
  // this constructor is just for a temporary index, for use in the deserialization
  // api. all the parameters here will get replaced with loaded values - that aren't
  // necessarily known ahead of time before deserialization.
  // TODO: do we even need a handle here - could just construct one?
  : rabitq_index_(handle)
{
}

template <typename IdxT>
index<IdxT>::index(raft::resources const& handle,
                   size_t n_rows,
                   uint32_t dim,
                   uint32_t n_lists,
                   uint32_t bits_per_dim)
  : rabitq_index_(handle, n_rows, dim, n_lists, bits_per_dim, /* batch_flag = */ true)
{
  RAFT_EXPECTS(bits_per_dim >= 2 && bits_per_dim <= 9, "Unsupported bits_per_dim");
}

template <typename IdxT>
index<IdxT>::index(raft::resources const& handle, const index_params& params, uint32_t dim)
  : index(handle)
{
}

template struct index<int64_t>;

template <typename IdxT>
detail::IVFGPU* index<IdxT>::rabitq_index() noexcept
{
  return &rabitq_index_;
}

template <typename IdxT>
uint32_t index<IdxT>::dim() const noexcept
{
  return rabitq_index_.num_dimensions;
}

void build(raft::resources const& handle,
           const cuvs::neighbors::ivf_rabitq::index_params& index_params,
           raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
           cuvs::neighbors::ivf_rabitq::index<int64_t>* idx)
{
  cuvs::neighbors::ivf_rabitq::detail::build(handle, index_params, dataset, idx);
}

void search(raft::resources const& handle,
            const cuvs::neighbors::ivf_rabitq::search_params& search_params,
            cuvs::neighbors::ivf_rabitq::index<int64_t>& index,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances)
{
  cuvs::neighbors::ivf_rabitq::detail::search(
    handle, search_params, index, queries, neighbors, distances);
}

void serialize(raft::resources const& handle,
               const std::string& filename,
               cuvs::neighbors::ivf_rabitq::index<int64_t>& index)
{
  cuvs::neighbors::ivf_rabitq::detail::serialize(handle, filename, index);
}

void deserialize(raft::resources const& handle,
                 const std::string& filename,
                 cuvs::neighbors::ivf_rabitq::index<int64_t>* index)
{
  if (!index) { RAFT_FAIL("Invalid index pointer"); }
  cuvs::neighbors::ivf_rabitq::detail::deserialize(handle, filename, index);
}

}  // namespace cuvs::neighbors::ivf_rabitq
