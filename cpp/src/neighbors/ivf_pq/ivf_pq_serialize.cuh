/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../ivf_common.cuh"
#include "../ivf_list.cuh"
#include "../ivf_pq_impl.hpp"
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <raft/core/copy.cuh>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/serialize.hpp>

#include <fstream>
#include <memory>

namespace cuvs::neighbors::ivf_pq::detail {

// Serialization version
// Version 4 adds codes_layout field
constexpr int kSerializationVersion = 4;

/**
 * Write the index to an output stream
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @param[in] handle the raft handle
 * @param[in] os output stream
 * @param[in] index IVF-PQ index
 *
 */
template <typename IdxT>
void serialize(raft::resources const& handle_, std::ostream& os, const index<IdxT>& index)
{
  RAFT_LOG_DEBUG("Size %zu, dim %d, pq_dim %d, pq_bits %d",
                 static_cast<size_t>(index.size()),
                 static_cast<int>(index.dim()),
                 static_cast<int>(index.pq_dim()),
                 static_cast<int>(index.pq_bits()));

  raft::serialize_scalar(handle_, os, kSerializationVersion);
  raft::serialize_scalar(handle_, os, index.size());
  raft::serialize_scalar(handle_, os, index.dim());
  raft::serialize_scalar(handle_, os, index.pq_bits());
  raft::serialize_scalar(handle_, os, index.pq_dim());
  raft::serialize_scalar(handle_, os, index.conservative_memory_allocation());

  raft::serialize_scalar(handle_, os, index.metric());
  raft::serialize_scalar(handle_, os, index.codebook_kind());
  raft::serialize_scalar(handle_, os, index.codes_layout());
  raft::serialize_scalar(handle_, os, index.n_lists());

  raft::serialize_mdspan(handle_, os, index.pq_centers());
  raft::serialize_mdspan(handle_, os, index.centers());
  raft::serialize_mdspan(handle_, os, index.centers_rot());
  raft::serialize_mdspan(handle_, os, index.rotation_matrix());

  auto sizes_host =
    raft::make_host_mdarray<uint32_t, uint32_t, raft::row_major>(index.list_sizes().extents());
  raft::copy(handle_, sizes_host.view(), index.list_sizes());
  raft::resource::sync_stream(handle_);
  raft::serialize_mdspan(handle_, os, sizes_host.view());
  // NOTE: We use static_cast here because serialize_list requires the concrete list type
  // to access the spec_type for determining the serialized data layout.
  if (index.codes_layout() == list_layout::FLAT) {
    auto list_store_spec = list_spec_flat<uint32_t, IdxT>{index.pq_bits(), index.pq_dim(), true};
    for (uint32_t label = 0; label < index.n_lists(); label++) {
      auto& typed_list = static_cast<const list_data_flat<IdxT>&>(*index.lists()[label]);
      ivf::serialize_list(handle_, os, typed_list, list_store_spec, sizes_host(label));
    }
  } else {
    auto list_store_spec =
      list_spec_interleaved<uint32_t, IdxT>{index.pq_bits(), index.pq_dim(), true};
    for (uint32_t label = 0; label < index.n_lists(); label++) {
      auto& typed_list = static_cast<const list_data_interleaved<IdxT>&>(*index.lists()[label]);
      ivf::serialize_list(handle_, os, typed_list, list_store_spec, sizes_host(label));
    }
  }
}

/**
 * Save the index to file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @param[in] handle the raft handle
 * @param[in] filename the file name for saving the index
 * @param[in] index IVF-PQ index
 *
 */
template <typename IdxT>
void serialize(raft::resources const& handle_,
               const std::string& filename,
               const index<IdxT>& index)
{
  std::ofstream of(filename, std::ios::out | std::ios::binary);
  if (!of) { RAFT_FAIL("Cannot open file %s", filename.c_str()); }

  detail::serialize(handle_, of, index);

  of.close();
  if (!of) { RAFT_FAIL("Error writing output %s", filename.c_str()); }
  return;
}

/**
 * Load index from input stream
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @param[in] handle the raft handle
 * @param[in] is input stream
 *
 */
template <typename IdxT>
auto deserialize(raft::resources const& handle_, std::istream& is) -> index<IdxT>
{
  auto ver = raft::deserialize_scalar<int>(handle_, is);
  if (ver != kSerializationVersion) {
    RAFT_FAIL("serialization version mismatch %d vs. %d", ver, kSerializationVersion);
  }
  auto n_rows  = raft::deserialize_scalar<IdxT>(handle_, is);
  auto dim     = raft::deserialize_scalar<std::uint32_t>(handle_, is);
  auto pq_bits = raft::deserialize_scalar<std::uint32_t>(handle_, is);
  auto pq_dim  = raft::deserialize_scalar<std::uint32_t>(handle_, is);
  auto cma     = raft::deserialize_scalar<bool>(handle_, is);

  auto metric        = raft::deserialize_scalar<cuvs::distance::DistanceType>(handle_, is);
  auto codebook_kind = raft::deserialize_scalar<cuvs::neighbors::ivf_pq::codebook_gen>(handle_, is);
  auto codes_layout  = raft::deserialize_scalar<cuvs::neighbors::ivf_pq::list_layout>(handle_, is);
  auto n_lists       = raft::deserialize_scalar<std::uint32_t>(handle_, is);

  RAFT_LOG_DEBUG("n_rows %zu, dim %d, pq_dim %d, pq_bits %d, n_lists %d",
                 static_cast<std::size_t>(n_rows),
                 static_cast<int>(dim),
                 static_cast<int>(pq_dim),
                 static_cast<int>(pq_bits),
                 static_cast<int>(n_lists));

  // Create owning_impl directly to get mutable access for deserialization
  auto impl = std::make_unique<owning_impl<IdxT>>(
    handle_, metric, codebook_kind, n_lists, dim, pq_bits, pq_dim, cma, codes_layout);

  // Deserialize center/matrix data using mutable accessors
  raft::deserialize_mdspan(handle_, is, impl->pq_centers());
  raft::deserialize_mdspan(handle_, is, impl->centers());
  raft::deserialize_mdspan(handle_, is, impl->centers_rot());
  raft::deserialize_mdspan(handle_, is, impl->rotation_matrix());
  raft::deserialize_mdspan(handle_, is, impl->list_sizes());
  if (codes_layout == list_layout::FLAT) {
    auto list_device_spec = list_spec_flat<uint32_t, IdxT>{pq_bits, pq_dim, cma};
    auto list_store_spec  = list_spec_flat<uint32_t, IdxT>{pq_bits, pq_dim, true};
    for (auto& list_data_base_ptr : impl->lists()) {
      std::shared_ptr<list_data_flat<IdxT>> typed_list;
      ivf::deserialize_list(handle_, is, typed_list, list_store_spec, list_device_spec);
      list_data_base_ptr = typed_list;
    }
  } else {
    auto list_device_spec = list_spec_interleaved<uint32_t, IdxT>{pq_bits, pq_dim, cma};
    auto list_store_spec  = list_spec_interleaved<uint32_t, IdxT>{pq_bits, pq_dim, true};
    for (auto& list_data_base_ptr : impl->lists()) {
      std::shared_ptr<list_data_interleaved<IdxT>> typed_list;
      ivf::deserialize_list(handle_, is, typed_list, list_store_spec, list_device_spec);
      list_data_base_ptr = typed_list;
    }
  }

  index<IdxT> idx(std::move(impl));

  raft::resource::sync_stream(handle_);

  ivf::detail::recompute_internal_state(handle_, idx);

  return idx;
}

/**
 * Load index from file.
 *
 * Experimental, both the API and the serialization format are subject to change.
 *
 * @param[in] handle the raft handle
 * @param[in] filename the name of the file that stores the index
 *
 */
template <typename IdxT>
auto deserialize(raft::resources const& handle_, const std::string& filename) -> index<IdxT>
{
  std::ifstream infile(filename, std::ios::in | std::ios::binary);

  if (!infile) { RAFT_FAIL("Cannot open file %s", filename.c_str()); }

  auto index = detail::deserialize<IdxT>(handle_, infile);

  infile.close();

  return index;
}

}  // namespace cuvs::neighbors::ivf_pq::detail
