/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <fstream>
#include <raft/core/copy.cuh>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/util/cudart_utils.hpp>

#include <fstream>
#include <mutex>

#include <raft/core/device_mdspan.hpp>

namespace cuvs::neighbors {

using namespace raft;

namespace iface_detail {
template <typename>
inline constexpr bool is_raft_host_device_accessor_v = false;
template <typename AccessorPolicy, raft::memory_type M>
inline constexpr bool
  is_raft_host_device_accessor_v<raft::host_device_accessor<AccessorPolicy, M>> = true;

/**
 * @brief `make_padded_dataset` rejects a buffer that is already CAGRA row-padded on the device; use
 * a non-owning padded view instead. That applies to true device or managed global memory, not
 * pinned host: the latter can report a non-null \p devicePointer while
 * \p type == \p cudaMemoryTypeHost.
 */
template <typename T, typename Accessor>
bool host_mds_uses_padded_device_view(
  raft::mdspan<const T, matrix_extent<int64_t>, row_major, Accessor> mds)
{
  using value_type = T;
  uint32_t const required_stride =
    cagra_required_row_width<value_type>(static_cast<uint32_t>(mds.extent(1)));
  uint32_t const src_stride =
    mds.stride(0) > 0 ? static_cast<uint32_t>(mds.stride(0)) : static_cast<uint32_t>(mds.extent(1));
  cudaPointerAttributes a{};
  RAFT_CUDA_TRY(cudaPointerGetAttributes(&a, mds.data_handle()));
  bool const device_src =
    (a.type == cudaMemoryTypeDevice) || (a.type == cudaMemoryTypeManaged);
  return device_src && (src_stride == required_stride);
}

/**
 * @brief Build CAGRA on a "host" mdspan for the non-ACE path: own a padded copy when a copy (or
 * padding) is required; otherwise use an in-place padded `device` view to the same storage.
 */
template <typename T, typename IdxT, typename Accessor>
void cagra_from_host_padded(raft::resources const& h,
                            cagra::index_params const& cagra_params,
                            raft::mdspan<const T, matrix_extent<int64_t>, row_major, Accessor> m,
                            cuvs::neighbors::iface<cagra::index<T, IdxT>, T, IdxT>& interface)
{
  if (host_mds_uses_padded_device_view(m)) {
    cudaPointerAttributes a{};
    RAFT_CUDA_TRY(cudaPointerGetAttributes(&a, m.data_handle()));
    T const* devp = reinterpret_cast<T const*>(a.devicePointer);
    uint32_t const s_stride =
      m.stride(0) > 0 ? static_cast<uint32_t>(m.stride(0)) : static_cast<uint32_t>(m.extent(1));
    auto d_m = raft::make_device_strided_matrix_view<T const, int64_t, row_major>(
      devp, m.extent(0), m.extent(1), s_stride);
    auto padded  = cuvs::neighbors::make_padded_dataset_view(h, d_m);
    auto build_r = cuvs::neighbors::cagra::build(h, cagra_params, padded);
    RAFT_EXPECTS(!build_r.vpq.has_value(),
                 "CAGRA VPQ build from host is not supported through neighbors::build for MG.");
    interface.cagra_owned_dataset_.reset();
    interface.index_.emplace(std::move(build_r.idx));
  } else {
    auto padded_r = cuvs::neighbors::make_padded_dataset(h, m);
    auto build_r  = cuvs::neighbors::cagra::build(h, cagra_params, padded_r->as_dataset_view());
    RAFT_EXPECTS(!build_r.vpq.has_value(),
                 "CAGRA VPQ build from host is not supported through neighbors::build for MG.");
    interface.cagra_owned_dataset_ =
      std::unique_ptr<cuvs::neighbors::dataset<int64_t>>(padded_r.release());
    interface.index_.emplace(std::move(build_r.idx));
  }
}
}  // namespace iface_detail

template <typename AnnIndexType, typename T, typename IdxT, typename Accessor>
void build(const raft::resources& handle,
           cuvs::neighbors::iface<AnnIndexType, T, IdxT>& interface,
           const cuvs::neighbors::index_params* index_params,
           raft::mdspan<const T, matrix_extent<int64_t>, row_major, Accessor> index_dataset)
{
  std::lock_guard lock(*interface.mutex_);

  if constexpr (std::is_same<AnnIndexType, ivf_flat::index<T, IdxT>>::value) {
    auto idx = cuvs::neighbors::ivf_flat::build(
      handle, *static_cast<const ivf_flat::index_params*>(index_params), index_dataset);
    interface.index_.emplace(std::move(idx));
  } else if constexpr (std::is_same<AnnIndexType, ivf_pq::index<IdxT>>::value) {
    auto idx = cuvs::neighbors::ivf_pq::build(
      handle, *static_cast<const ivf_pq::index_params*>(index_params), index_dataset);
    interface.index_.emplace(std::move(idx));
  } else if constexpr (std::is_same<AnnIndexType, cagra::index<T, IdxT>>::value) {
    const auto& cagra_params = *static_cast<const cagra::index_params*>(index_params);
    // Use compile-time routing for raft::host_device_accessor: a runtime `if (host vs device)`
    // still type-checks both branches; device mdspan + ACE host code then fails (build returns
    // index, not ace_build_result). Pointer fallback remains for other accessor types.
    if constexpr (iface_detail::is_raft_host_device_accessor_v<Accessor>) {
      if constexpr (Accessor::mem_type == raft::memory_type::device) {
        auto idx = cuvs::neighbors::cagra::build(handle, cagra_params, index_dataset);
        interface.index_.emplace(std::move(idx));
      } else {
        // Host mdspan is only accepted on the ACE build path; non-ACE requires dataset_view.
        if (std::holds_alternative<cagra::graph_build_params::ace_params>(
              cagra_params.graph_build_params)) {
          auto result = cuvs::neighbors::cagra::build(handle, cagra_params, index_dataset);
          interface.cagra_build_dataset_ = std::move(result.dataset);
          interface.index_.emplace(std::move(result.idx));
        } else {
          iface_detail::cagra_from_host_padded(handle, cagra_params, index_dataset, interface);
        }
      }
    } else {
      const bool dataset_on_host =
        (raft::get_device_for_address(index_dataset.data_handle()) == -1);
      if (dataset_on_host) {
        if (std::holds_alternative<cagra::graph_build_params::ace_params>(
              cagra_params.graph_build_params)) {
          auto result = cuvs::neighbors::cagra::build(handle, cagra_params, index_dataset);
          interface.cagra_build_dataset_ = std::move(result.dataset);
          interface.index_.emplace(std::move(result.idx));
        } else {
          iface_detail::cagra_from_host_padded(handle, cagra_params, index_dataset, interface);
        }
      } else {
        auto idx = cuvs::neighbors::cagra::build(handle, cagra_params, index_dataset);
        interface.index_.emplace(std::move(idx));
      }
    }
  }
  resource::sync_stream(handle);
}

template <typename AnnIndexType, typename T, typename IdxT, typename Accessor1, typename Accessor2>
void extend(
  const raft::resources& handle,
  cuvs::neighbors::iface<AnnIndexType, T, IdxT>& interface,
  raft::mdspan<const T, matrix_extent<int64_t>, row_major, Accessor1> new_vectors,
  std::optional<raft::mdspan<const IdxT, vector_extent<int64_t>, layout_c_contiguous, Accessor2>>
    new_indices)
{
  std::lock_guard lock(*interface.mutex_);

  if constexpr (std::is_same<AnnIndexType, ivf_flat::index<T, IdxT>>::value) {
    auto idx =
      cuvs::neighbors::ivf_flat::extend(handle, new_vectors, new_indices, interface.index_.value());
    interface.index_.emplace(std::move(idx));
  } else if constexpr (std::is_same<AnnIndexType, ivf_pq::index<IdxT>>::value) {
    auto idx =
      cuvs::neighbors::ivf_pq::extend(handle, new_vectors, new_indices, interface.index_.value());
    interface.index_.emplace(std::move(idx));
  } else if constexpr (std::is_same<AnnIndexType, cagra::index<T, IdxT>>::value) {
    RAFT_FAIL("CAGRA does not implement the extend method");
  }
  resource::sync_stream(handle);
}

template <typename AnnIndexType, typename T, typename IdxT, typename searchIdxT>
void search(const raft::resources& handle,
            const cuvs::neighbors::iface<AnnIndexType, T, IdxT>& interface,
            const cuvs::neighbors::search_params* search_params,
            raft::device_matrix_view<const T, int64_t, row_major> queries,
            raft::device_matrix_view<searchIdxT, int64_t, row_major> neighbors,
            raft::device_matrix_view<float, int64_t, row_major> distances)
{
  // std::lock_guard(*interface.mutex_);
  if constexpr (std::is_same<AnnIndexType, ivf_flat::index<T, int64_t>>::value) {
    cuvs::neighbors::ivf_flat::search(
      handle,
      *reinterpret_cast<const ivf_flat::search_params*>(search_params),
      interface.index_.value(),
      queries,
      neighbors,
      distances);
  } else if constexpr (std::is_same<AnnIndexType, ivf_pq::index<int64_t>>::value) {
    cuvs::neighbors::ivf_pq::search(handle,
                                    *reinterpret_cast<const ivf_pq::search_params*>(search_params),
                                    interface.index_.value(),
                                    queries,
                                    neighbors,
                                    distances);
  } else if constexpr (std::is_same<AnnIndexType, cagra::index<T, uint32_t>>::value) {
    cuvs::neighbors::cagra::search(handle,
                                   *reinterpret_cast<const cagra::search_params*>(search_params),
                                   interface.index_.value(),
                                   queries,
                                   neighbors,
                                   distances);
  }
  // resource::sync_stream(handle);
}

// for MG ANN only
template <typename AnnIndexType, typename T, typename IdxT, typename searchIdxT>
void search(const raft::resources& handle,
            const cuvs::neighbors::iface<AnnIndexType, T, IdxT>& interface,
            const cuvs::neighbors::search_params* search_params,
            raft::host_matrix_view<const T, int64_t, row_major> h_queries,
            raft::device_matrix_view<searchIdxT, int64_t, row_major> d_neighbors,
            raft::device_matrix_view<float, int64_t, row_major> d_distances)
{
  // std::lock_guard(*interface.mutex_);

  int64_t n_rows = h_queries.extent(0);
  int64_t n_dims = h_queries.extent(1);
  auto d_queries = raft::make_device_matrix<T, int64_t, row_major>(handle, n_rows, n_dims);
  raft::copy(handle, d_queries.view(), h_queries);
  auto d_query_view = raft::make_const_mdspan(d_queries.view());

  search(handle, interface, search_params, d_query_view, d_neighbors, d_distances);
}

template <typename AnnIndexType, typename T, typename IdxT>
void serialize(const raft::resources& handle,
               const cuvs::neighbors::iface<AnnIndexType, T, IdxT>& interface,
               std::ostream& os)
{
  std::lock_guard lock(*interface.mutex_);

  if constexpr (std::is_same<AnnIndexType, ivf_flat::index<T, IdxT>>::value) {
    ivf_flat::serialize(handle, os, interface.index_.value());
  } else if constexpr (std::is_same<AnnIndexType, ivf_pq::index<IdxT>>::value) {
    ivf_pq::serialize(handle, os, interface.index_.value());
  } else if constexpr (std::is_same<AnnIndexType, cagra::index<T, IdxT>>::value) {
    cagra::serialize(handle, os, interface.index_.value(), true);
  }

  resource::sync_stream(handle);
}

template <typename AnnIndexType, typename T, typename IdxT>
void deserialize(const raft::resources& handle,
                 cuvs::neighbors::iface<AnnIndexType, T, IdxT>& interface,
                 std::istream& is)
{
  std::lock_guard lock(*interface.mutex_);

  if constexpr (std::is_same<AnnIndexType, ivf_flat::index<T, IdxT>>::value) {
    ivf_flat::index<T, IdxT> idx(handle);
    ivf_flat::deserialize(handle, is, &idx);
    resource::sync_stream(handle);
    interface.index_.emplace(std::move(idx));
  } else if constexpr (std::is_same<AnnIndexType, ivf_pq::index<IdxT>>::value) {
    ivf_pq::index<IdxT> idx(handle);
    ivf_pq::deserialize(handle, is, &idx);
    resource::sync_stream(handle);
    interface.index_.emplace(std::move(idx));
  } else if constexpr (std::is_same<AnnIndexType, cagra::index<T, IdxT>>::value) {
    cagra::index<T, IdxT> idx(handle);
    std::unique_ptr<cuvs::neighbors::dataset<int64_t>> out_dataset;
    cagra::deserialize(handle, is, &idx, &out_dataset);
    if (out_dataset) { interface.cagra_owned_dataset_ = std::move(out_dataset); }
    resource::sync_stream(handle);
    interface.index_.emplace(std::move(idx));
  }
}

template <typename AnnIndexType, typename T, typename IdxT>
void deserialize(const raft::resources& handle,
                 cuvs::neighbors::iface<AnnIndexType, T, IdxT>& interface,
                 const std::string& filename)
{
  std::lock_guard lock(*interface.mutex_);

  std::ifstream is(filename, std::ios::in | std::ios::binary);
  if (!is) { RAFT_FAIL("Cannot open file %s", filename.c_str()); }

  if constexpr (std::is_same<AnnIndexType, ivf_flat::index<T, IdxT>>::value) {
    ivf_flat::index<T, IdxT> idx(handle);
    ivf_flat::deserialize(handle, is, &idx);
    resource::sync_stream(handle);
    interface.index_.emplace(std::move(idx));
  } else if constexpr (std::is_same<AnnIndexType, ivf_pq::index<IdxT>>::value) {
    ivf_pq::index<IdxT> idx(handle);
    ivf_pq::deserialize(handle, is, &idx);
    resource::sync_stream(handle);
    interface.index_.emplace(std::move(idx));
  } else if constexpr (std::is_same<AnnIndexType, cagra::index<T, IdxT>>::value) {
    cagra::index<T, IdxT> idx(handle);
    std::unique_ptr<cuvs::neighbors::dataset<int64_t>> out_dataset;
    cagra::deserialize(handle, is, &idx, &out_dataset);
    if (out_dataset) { interface.cagra_owned_dataset_ = std::move(out_dataset); }
    resource::sync_stream(handle);
    interface.index_.emplace(std::move(idx));
  }

  is.close();
}

};  // namespace cuvs::neighbors
