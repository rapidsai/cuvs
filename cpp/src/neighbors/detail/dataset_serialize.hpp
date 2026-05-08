/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuvs/neighbors/common.hpp>

#include <raft/core/host_mdarray.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/serialize.hpp>
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

template <typename IdxT>
void serialize(const raft::resources& res, std::ostream& os, const empty_dataset<IdxT>& dataset)
{
  raft::serialize_scalar(res, os, dataset.suggested_dim);
}

template <typename DataT, typename IdxT>
void serialize(const raft::resources& res,
               std::ostream& os,
               const strided_dataset_view<DataT, IdxT>& dataset)
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
  RAFT_CUDA_TRY(cudaMemcpy2DAsync(dst.data_handle(),
                                  sizeof(DataT) * dim,
                                  src.data_handle(),
                                  sizeof(DataT) * stride,
                                  sizeof(DataT) * dim,
                                  n_rows,
                                  cudaMemcpyDefault,
                                  raft::resource::get_cuda_stream(res)));
  raft::resource::sync_stream(res);
  raft::serialize_mdspan(res, os, dst.view());
}

template <typename DataT, typename IdxT>
void serialize(const raft::resources& res,
               std::ostream& os,
               const vpq_dataset<DataT, IdxT>& dataset)
{
  raft::serialize_scalar(res, os, dataset.n_rows());
  raft::serialize_scalar(res, os, dataset.dim());
  raft::serialize_scalar(res, os, dataset.vq_n_centers());
  raft::serialize_scalar(res, os, dataset.pq_n_centers());
  raft::serialize_scalar(res, os, dataset.pq_len());
  raft::serialize_scalar(res, os, dataset.encoded_row_length());
  raft::serialize_mdspan(res, os, make_const_mdspan(dataset.vq_code_book.view()));
  raft::serialize_mdspan(res, os, make_const_mdspan(dataset.pq_code_book.view()));
  raft::serialize_mdspan(res, os, make_const_mdspan(dataset.data.view()));
}

template <typename IdxT>
void serialize_indirect_target(const raft::resources& res,
                               std::ostream& os,
                               indirect_dataset_view<IdxT> const& ind);

template <typename IdxT>
void serialize(const raft::resources& res,
               std::ostream& os,
               const any_owning_dataset<IdxT>& dataset)
{
  using OT      = any_owning_dataset_types<IdxT>;
  auto const& v = dataset.as_variant();
  if (std::holds_alternative<typename OT::empty_owning>(v)) {
    serialize(res, os, std::get<typename OT::empty_owning>(v));
    return;
  }
  if (std::holds_alternative<typename OT::padded_f32_owning>(v)) {
    serialize(res, os, std::get<typename OT::padded_f32_owning>(v));
    return;
  }
  if (std::holds_alternative<typename OT::padded_f16_owning>(v)) {
    serialize(res, os, std::get<typename OT::padded_f16_owning>(v));
    return;
  }
  if (std::holds_alternative<typename OT::padded_i8_owning>(v)) {
    serialize(res, os, std::get<typename OT::padded_i8_owning>(v));
    return;
  }
  if (std::holds_alternative<typename OT::padded_u8_owning>(v)) {
    serialize(res, os, std::get<typename OT::padded_u8_owning>(v));
    return;
  }
  if (std::holds_alternative<typename OT::vpq_f32_owning>(v)) {
    serialize(res, os, std::get<typename OT::vpq_f32_owning>(v));
    return;
  }
  if (std::holds_alternative<typename OT::vpq_f16_owning>(v)) {
    serialize(res, os, std::get<typename OT::vpq_f16_owning>(v));
    return;
  }
  RAFT_FAIL(
    "serialize(any_owning_dataset): unsupported owning variant "
    "(strided owning storage is not serialized — use padded or VPQ for persistence)");
}

template <typename IdxT>
void serialize_indirect_target(const raft::resources& res,
                               std::ostream& os,
                               indirect_dataset_view<IdxT> const& ind)
{
  switch (ind.get_indirect_target_type()) {
    case indirect_target_type::empty_v:
      raft::serialize_scalar(res, os, kSerializeEmptyDataset);
      raft::serialize_scalar(
        res, os, static_cast<empty_dataset<IdxT> const*>(ind.raw_target())->suggested_dim);
      return;
    case indirect_target_type::padded_f32:
      raft::serialize_scalar(res, os, kSerializeStridedDataset);
      raft::serialize_scalar(res, os, CUDA_R_32F);
      return serialize(
        res,
        os,
        static_cast<padded_dataset<float, IdxT> const*>(ind.raw_target())->as_dataset_view());
    case indirect_target_type::padded_f16:
      raft::serialize_scalar(res, os, kSerializeStridedDataset);
      raft::serialize_scalar(res, os, CUDA_R_16F);
      return serialize(
        res,
        os,
        static_cast<padded_dataset<half, IdxT> const*>(ind.raw_target())->as_dataset_view());
    case indirect_target_type::padded_i8:
      raft::serialize_scalar(res, os, kSerializeStridedDataset);
      raft::serialize_scalar(res, os, CUDA_R_8I);
      return serialize(
        res,
        os,
        static_cast<padded_dataset<int8_t, IdxT> const*>(ind.raw_target())->as_dataset_view());
    case indirect_target_type::padded_u8:
      raft::serialize_scalar(res, os, kSerializeStridedDataset);
      raft::serialize_scalar(res, os, CUDA_R_8U);
      return serialize(
        res,
        os,
        static_cast<padded_dataset<uint8_t, IdxT> const*>(ind.raw_target())->as_dataset_view());
    case indirect_target_type::vpq_f32:
      raft::serialize_scalar(res, os, kSerializeVPQDataset);
      raft::serialize_scalar(res, os, CUDA_R_32F);
      return serialize(res, os, *static_cast<vpq_dataset<float, IdxT> const*>(ind.raw_target()));
    case indirect_target_type::vpq_f16:
      raft::serialize_scalar(res, os, kSerializeVPQDataset);
      raft::serialize_scalar(res, os, CUDA_R_16F);
      return serialize(res, os, *static_cast<vpq_dataset<half, IdxT> const*>(ind.raw_target()));
    default: RAFT_FAIL("serialize_indirect_target: unsupported indirect_target_type");
  }
}

template <typename T, typename IdxT>
void serialize(const raft::resources& res,
               std::ostream& os,
               const any_dataset_view<T, IdxT>& dataset)
{
  auto write_row_element_tag = [&]() {
    if constexpr (std::is_same_v<T, float>) {
      raft::serialize_scalar(res, os, CUDA_R_32F);
    } else if constexpr (std::is_same_v<T, half>) {
      raft::serialize_scalar(res, os, CUDA_R_16F);
    } else if constexpr (std::is_same_v<T, int8_t>) {
      raft::serialize_scalar(res, os, CUDA_R_8I);
    } else if constexpr (std::is_same_v<T, uint8_t>) {
      raft::serialize_scalar(res, os, CUDA_R_8U);
    } else {
      static_assert(!std::is_same_v<T, T>, "unsupported T for CAGRA serialize");
    }
  };

  using VT        = any_dataset_view_types<T, IdxT>;
  auto const& var = dataset.as_variant();
  if (std::holds_alternative<typename VT::empty_view>(var)) {
    raft::serialize_scalar(res, os, kSerializeEmptyDataset);
    raft::serialize_scalar(res, os, std::get<typename VT::empty_view>(var).dim());
    return;
  }
  if (std::holds_alternative<typename VT::indirect_view>(var)) {
    serialize_indirect_target(res, os, std::get<typename VT::indirect_view>(var));
    return;
  }
  if (std::holds_alternative<typename VT::padded_view>(var)) {
    raft::serialize_scalar(res, os, kSerializeStridedDataset);
    write_row_element_tag();
    serialize(res, os, std::get<typename VT::padded_view>(var));
    return;
  }
  if (std::holds_alternative<typename VT::strided_view>(var)) {
    raft::serialize_scalar(res, os, kSerializeStridedDataset);
    write_row_element_tag();
    serialize(res, os, std::get<typename VT::strided_view>(var));
  }
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
  auto n_rows     = raft::deserialize_scalar<IdxT>(res, is);
  auto dim        = raft::deserialize_scalar<uint32_t>(res, is);
  auto stride     = raft::deserialize_scalar<uint32_t>(res, is);
  auto host_array = raft::make_host_matrix<DataT, IdxT>(n_rows, dim);
  raft::deserialize_mdspan(res, is, host_array.view());
  auto padded = make_padded_dataset(res, host_array.view());
  return wrap_any_owning(std::move(padded));
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
  switch (raft::deserialize_scalar<dataset_instance_tag>(res, is)) {
    case kSerializeEmptyDataset: return deserialize_empty<IdxT>(res, is);
    case kSerializeStridedDataset:
      switch (raft::deserialize_scalar<cudaDataType_t>(res, is)) {
        case CUDA_R_32F: return deserialize_strided<float, IdxT>(res, is);
        case CUDA_R_16F: return deserialize_strided<half, IdxT>(res, is);
        case CUDA_R_8I: return deserialize_strided<int8_t, IdxT>(res, is);
        case CUDA_R_8U: return deserialize_strided<uint8_t, IdxT>(res, is);
        default: break;
      }
    case kSerializeVPQDataset:
      switch (raft::deserialize_scalar<cudaDataType_t>(res, is)) {
        case CUDA_R_32F: return deserialize_vpq<float, IdxT>(res, is);
        case CUDA_R_16F: return deserialize_vpq<half, IdxT>(res, is);
        default: break;
      }
    default: break;
  }
  RAFT_FAIL("Failed to deserialize dataset: unsupported combination of instance tags.");
}

}  // namespace cuvs::neighbors::detail
