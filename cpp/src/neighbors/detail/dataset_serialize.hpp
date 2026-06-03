/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuvs/neighbors/common.hpp>

#include <raft/core/host_mdarray.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/serialize.hpp>
#include <raft/matrix/copy.cuh>
#include <raft/util/cudart_utils.hpp>

#include <raft/core/logger.hpp>

#include <cuda_fp16.h>

#include <fstream>
#include <memory>

namespace cuvs::neighbors::detail {

using dataset_instance_tag                              = uint32_t;
constexpr dataset_instance_tag kSerializeEmptyDataset   = 1;
constexpr dataset_instance_tag kSerializeStridedDataset = 2;
constexpr dataset_instance_tag kSerializeVPQDataset     = 3;

// Padded: `padded_dataset_view` writes the payload.
template <typename DataT, typename IdxT>
void serialize(const raft::resources& res,
               std::ostream& os,
               const padded_dataset_view<DataT, IdxT>& dataset)
{
  auto n_rows = dataset.n_rows();
  auto dim    = dataset.dim();
  auto stride = dataset.stride();
  raft::serialize_scalar(res, os, n_rows);
  raft::serialize_scalar(res, os, dim);
  raft::serialize_scalar(res, os, stride);
  auto src = dataset.view();
  auto dst = raft::make_host_matrix<DataT, IdxT>(n_rows, dim);
  raft::copy_matrix(dst.data_handle(),
                    dim,
                    src.data_handle(),
                    stride,
                    dim,
                    n_rows,
                    raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);
  raft::serialize_mdspan(res, os, dst.view());
}

/** Write CAGRA index dataset blob (tag + element dtype + padded payload). */
template <typename DataT, typename IdxT>
void serialize_cagra_padded_dataset(const raft::resources& res,
                                    std::ostream& os,
                                    const padded_dataset_view<DataT, IdxT>& dataset)
{
  raft::serialize_scalar(res, os, kSerializeStridedDataset);
  if constexpr (std::is_same_v<DataT, float>) {
    raft::serialize_scalar(res, os, CUDA_R_32F);
  } else if constexpr (std::is_same_v<DataT, half>) {
    raft::serialize_scalar(res, os, CUDA_R_16F);
  } else if constexpr (std::is_same_v<DataT, int8_t>) {
    raft::serialize_scalar(res, os, CUDA_R_8I);
  } else if constexpr (std::is_same_v<DataT, uint8_t>) {
    raft::serialize_scalar(res, os, CUDA_R_8U);
  } else {
    static_assert(!std::is_same_v<DataT, DataT>, "unsupported element type for CAGRA serialize");
  }
  serialize(res, os, dataset);
}

template <typename IdxT>
auto deserialize_empty(raft::resources const& res, std::istream& is)
  -> std::unique_ptr<any_owning_dataset<IdxT>>
{
  auto suggested_dim = raft::deserialize_scalar<uint32_t>(res, is);
  auto v             = empty_dataset<IdxT>(suggested_dim);
  return std::make_unique<any_owning_dataset<IdxT>>(std::move(v));
}

template <typename DataT, typename IdxT>
auto deserialize_strided(raft::resources const& res, std::istream& is)
  -> std::unique_ptr<any_owning_dataset<IdxT>>
{
  auto n_rows = raft::deserialize_scalar<IdxT>(res, is);
  auto dim    = raft::deserialize_scalar<uint32_t>(res, is);
  auto stride = raft::deserialize_scalar<uint32_t>(res, is);
  RAFT_EXPECTS(dim <= stride,
               "deserialize_strided: logical dim (%u) must not exceed row stride (%u).",
               static_cast<unsigned>(dim),
               static_cast<unsigned>(stride));
  auto host_array = raft::make_host_matrix<DataT, IdxT>(n_rows, dim);
  raft::deserialize_mdspan(res, is, host_array.view());
  auto padded = cuvs::neighbors::make_padded_dataset(res, host_array.view());
  return cuvs::neighbors::wrap_any_owning(std::move(padded));
}

template <typename DataT, typename IdxT>
auto deserialize_vpq(raft::resources const& res, std::istream& is)
  -> std::unique_ptr<any_owning_dataset<IdxT>>
{
  auto n_rows             = raft::deserialize_scalar<IdxT>(res, is);
  auto dim                = raft::deserialize_scalar<uint32_t>(res, is);
  auto vq_n_centers       = raft::deserialize_scalar<uint32_t>(res, is);
  auto pq_n_centers       = raft::deserialize_scalar<uint32_t>(res, is);
  auto pq_len             = raft::deserialize_scalar<uint32_t>(res, is);
  auto encoded_row_length = raft::deserialize_scalar<uint32_t>(res, is);

  auto vq_code_book =
    raft::make_device_matrix<DataT, uint32_t, raft::row_major>(res, vq_n_centers, dim);
  auto pq_code_book =
    raft::make_device_matrix<DataT, uint32_t, raft::row_major>(res, pq_n_centers, pq_len);
  auto data =
    raft::make_device_matrix<uint8_t, IdxT, raft::row_major>(res, n_rows, encoded_row_length);

  raft::deserialize_mdspan(res, is, vq_code_book.view());
  raft::deserialize_mdspan(res, is, pq_code_book.view());
  raft::deserialize_mdspan(res, is, data.view());

  vpq_dataset<DataT, IdxT> vpq{std::move(vq_code_book), std::move(pq_code_book), std::move(data)};
  return std::make_unique<any_owning_dataset<IdxT>>(std::move(vpq));
}

template <typename IdxT>
auto deserialize_dataset(raft::resources const& res, std::istream& is)
  -> std::unique_ptr<any_owning_dataset<IdxT>>
{
  const auto tag = raft::deserialize_scalar<dataset_instance_tag>(res, is);
  switch (tag) {
    case kSerializeEmptyDataset: return deserialize_empty<IdxT>(res, is);
    case kSerializeStridedDataset: {
      const auto dtype = raft::deserialize_scalar<cudaDataType_t>(res, is);
      switch (dtype) {
        case CUDA_R_32F: return deserialize_strided<float, IdxT>(res, is);
        case CUDA_R_16F: return deserialize_strided<half, IdxT>(res, is);
        case CUDA_R_8I: return deserialize_strided<int8_t, IdxT>(res, is);
        case CUDA_R_8U: return deserialize_strided<uint8_t, IdxT>(res, is);
        default:
          RAFT_FAIL("Failed to deserialize dataset: unsupported strided dataset element type %d.",
                    static_cast<int>(dtype));
      }
    }
    case kSerializeVPQDataset: {
      const auto dtype = raft::deserialize_scalar<cudaDataType_t>(res, is);
      switch (dtype) {
        case CUDA_R_32F: return deserialize_vpq<float, IdxT>(res, is);
        case CUDA_R_16F: return deserialize_vpq<half, IdxT>(res, is);
        default:
          RAFT_FAIL("Failed to deserialize dataset: unsupported VPQ dtype %d.",
                    static_cast<int>(dtype));
      }
    }
    default:
      RAFT_FAIL("Failed to deserialize dataset: unknown instance tag %u.",
                static_cast<unsigned>(tag));
  }
}

}  // namespace cuvs::neighbors::detail
