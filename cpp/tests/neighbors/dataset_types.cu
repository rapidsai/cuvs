/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Standalone tests for all dataset types in cuvs::neighbors:
 * - empty_dataset
 * - strided_dataset (owning / "padded" and non_owning / "padded view")
 * - vpq_dataset
 * - pq_dataset
 * Plus type traits: is_strided_dataset_v, is_vpq_dataset_v, is_pq_dataset_v.
 */

#include <cuvs/neighbors/common.hpp>
#include <exception>
#include <gtest/gtest.h>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/util/cudart_utils.hpp>
#include <string>

namespace cuvs::neighbors::test {

using namespace cuvs::neighbors;

// Helper: assert that ptr is device memory (for device_* dataset views).
inline void expect_device_pointer(const void* ptr)
{
  cudaPointerAttributes attr;
  RAFT_CUDA_TRY(cudaPointerGetAttributes(&attr, ptr));
  EXPECT_EQ(attr.type, cudaMemoryTypeDevice) << "Expected device memory";
}

// Helper: assert that ptr is host memory (for host_* dataset views).
inline void expect_host_pointer(const void* ptr)
{
  cudaPointerAttributes attr;
  RAFT_CUDA_TRY(cudaPointerGetAttributes(&attr, ptr));
  EXPECT_TRUE(attr.type == cudaMemoryTypeHost || attr.type == cudaMemoryTypeUnregistered)
    << "Expected host memory";
}

// Type aliases to avoid commas in GTest macro arguments (preprocessor splits on comma).
using strided_float_i64    = strided_dataset<float, int64_t>;
using non_owning_float_i64 = non_owning_dataset<float, int64_t>;
using vpq_float_i64        = vpq_dataset<float, int64_t>;

// ---------------------------------------------------------------------------
// empty_dataset
// ---------------------------------------------------------------------------
TEST(DatasetTypes, EmptyDataset)
{
  empty_dataset<int64_t> ds(128);
  EXPECT_EQ(ds.n_rows(), 0);
  EXPECT_EQ(ds.dim(), 128u);
  EXPECT_TRUE(ds.is_owning());

  empty_dataset<int32_t> ds32(64);
  EXPECT_EQ(ds32.n_rows(), 0);
  EXPECT_EQ(ds32.dim(), 64u);
  EXPECT_TRUE(ds32.is_owning());
}

// ---------------------------------------------------------------------------
// Type traits (compile-time and runtime sanity)
// ---------------------------------------------------------------------------
TEST(DatasetTypes, TypeTraits)
{
  EXPECT_TRUE((is_strided_dataset_v<strided_float_i64>));
  EXPECT_TRUE((is_strided_dataset_v<non_owning_float_i64>));
  EXPECT_FALSE((is_strided_dataset_v<empty_dataset<int64_t>>));
  EXPECT_FALSE((is_strided_dataset_v<vpq_float_i64>));
  // EXPECT_FALSE((is_strided_dataset_v<pq_dataset<float, int64_t>>));  // TODO: enable when
  // pq_dataset is in common.hpp

  EXPECT_TRUE((is_vpq_dataset_v<vpq_float_i64>));
  // EXPECT_FALSE((is_vpq_dataset_v<pq_dataset<float, int64_t>>));  // TODO: enable when pq_dataset
  // is in common.hpp
  EXPECT_FALSE((is_vpq_dataset_v<strided_float_i64>));

  // TODO: enable when pq_dataset is in common.hpp
  // EXPECT_TRUE((is_pq_dataset_v<pq_dataset<float, int64_t>>));
  // EXPECT_FALSE((is_pq_dataset_v<vpq_float_i64>));
  // EXPECT_FALSE((is_pq_dataset_v<strided_float_i64>));

  // Padded dataset type traits
  EXPECT_TRUE((is_padded_dataset_v<device_padded_dataset<float, int64_t>>));
  EXPECT_TRUE((is_padded_dataset_v<device_padded_dataset_view<float, int64_t>>));
  EXPECT_TRUE((is_padded_dataset_v<host_padded_dataset<float, int64_t>>));
  EXPECT_TRUE((is_padded_dataset_v<host_padded_dataset_view<float, int64_t>>));
  EXPECT_FALSE((is_padded_dataset_v<strided_float_i64>));
  EXPECT_FALSE((is_padded_dataset_v<empty_dataset<int64_t>>));
}

// ---------------------------------------------------------------------------
// Strided (owning / "padded dataset") and non-owning ("padded view")
// ---------------------------------------------------------------------------
TEST(DatasetTypes, StridedOwningAndNonOwning)
{
  raft::resources res;

  const int64_t n_rows = 100;
  const uint32_t dim   = 16;

  auto dev_matrix = raft::make_device_matrix<float, int64_t>(res, n_rows, dim);
  // Leave data uninitialized; we only check shape/stride/ownership.

  // Required stride equal to dim -> may get non-owning if layout matches
  auto ds_maybe_view = make_strided_dataset(res, dev_matrix.view(), dim);
  ASSERT_NE(ds_maybe_view, nullptr);
  EXPECT_EQ(ds_maybe_view->n_rows(), n_rows);
  EXPECT_EQ(ds_maybe_view->dim(), dim);

  auto* strided = ds_maybe_view.get();
  EXPECT_EQ(strided->stride(), dim);
  // With matching stride and device pointer, we expect non-owning
  EXPECT_FALSE(ds_maybe_view->is_owning());

  // Force owning by requiring a larger stride (padding)
  const uint32_t padded_stride = dim + 8;
  auto ds_owning               = make_strided_dataset(res, dev_matrix.view(), padded_stride);
  ASSERT_NE(ds_owning, nullptr);
  EXPECT_EQ(ds_owning->n_rows(), n_rows);
  EXPECT_EQ(ds_owning->dim(), dim);
  EXPECT_EQ(ds_owning->stride(), padded_stride);
  EXPECT_TRUE(ds_owning->is_owning());
}

// ---------------------------------------------------------------------------
// make_aligned_dataset (produces strided dataset with alignment; maybe owning)
// ---------------------------------------------------------------------------
// View vs copy is determined by whether row size in bytes is already aligned.
// For align_bytes=16 and float (4 bytes): row_bytes = dim * 4. When row_bytes is a multiple
// of 16, required_stride equals dim and matches the source stride -> we return a non-owning
// view. When row_bytes is not a multiple of 16, we round up to the next multiple, so
// required_stride > dim and does not match the source -> we allocate and copy (owning).
// Example: dim=32 -> 128 bytes (multiple of 16) -> view. dim=30 -> 120 bytes (not) -> copy.
//
// dim=32, align=16: row bytes 128 already aligned -> required_stride=32 matches src -> view
TEST(DatasetTypes, MakeAlignedDatasetViewWhenStrideMatches)
{
  raft::resources res;

  const int64_t n_rows = 50;
  const uint32_t dim   = 32;

  auto dev_matrix = raft::make_device_matrix<float, int64_t>(res, n_rows, dim);
  auto ds         = make_aligned_dataset(res, dev_matrix.view(), 16u);
  ASSERT_NE(ds, nullptr);
  EXPECT_EQ(ds->n_rows(), n_rows);
  EXPECT_EQ(ds->dim(), dim);
  EXPECT_GE(ds->stride(), dim);
  EXPECT_FALSE(ds->is_owning());  // stride matches -> no copy, non-owning view
}

// dim=30, align=16: row bytes 120 -> round up to 128 -> required_stride=32, src_stride=30 -> copy
TEST(DatasetTypes, MakeAlignedDatasetOwningWhenPadded)
{
  raft::resources res;

  const int64_t n_rows = 50;
  const uint32_t dim   = 30;

  auto dev_matrix = raft::make_device_matrix<float, int64_t>(res, n_rows, dim);
  auto ds         = make_aligned_dataset(res, dev_matrix.view(), 16u);
  ASSERT_NE(ds, nullptr);
  EXPECT_EQ(ds->n_rows(), n_rows);
  EXPECT_EQ(ds->dim(), dim);
  EXPECT_GE(ds->stride(), dim);  // stride will be 32 (rounded up from 30)
  EXPECT_TRUE(ds->is_owning());  // stride mismatch -> copy with padding
}

// ---------------------------------------------------------------------------
// Padded datasets (device_padded_dataset, device_padded_dataset_view, host_*)
// ---------------------------------------------------------------------------
// These tests exercise the dataset *types* (shape, stride, is_owning, view()).
// Padded construction factories are tested in cagra_padded_dataset.cu.
// Owning vs view is determined by which factory is used, not by dim/stride:
//   make_*_padded_dataset(...)  -> always allocates -> is_owning() == true
//   make_*_padded_dataset_view(...) -> wraps existing memory -> is_owning() == false
//
TEST(DatasetTypes, DevicePaddedDataset)
{
  raft::resources res;
  const int64_t n_rows = 40;
  const uint32_t dim   = 16;

  auto data = raft::make_device_matrix<float, int64_t>(res, n_rows, dim);
  auto ds   = std::make_unique<device_padded_dataset<float, int64_t>>(std::move(data), dim);
  ASSERT_NE(ds, nullptr);
  EXPECT_EQ(ds->n_rows(), n_rows);
  EXPECT_EQ(ds->dim(), dim);
  EXPECT_EQ(ds->stride(), dim);
  EXPECT_TRUE(ds->is_owning());
  expect_device_pointer(ds->view().data_handle());
  auto v = ds->view();
  EXPECT_EQ(v.extent(0), n_rows);
  EXPECT_EQ(v.extent(1), dim);

  // With explicit stride (padding)
  const uint32_t padded_stride = dim + 8;
  auto data_padded            = raft::make_device_matrix<float, int64_t>(res, n_rows, padded_stride);
  auto ds_padded =
    std::make_unique<device_padded_dataset<float, int64_t>>(std::move(data_padded), dim);
  ASSERT_NE(ds_padded, nullptr);
  EXPECT_EQ(ds_padded->n_rows(), n_rows);
  EXPECT_EQ(ds_padded->dim(), dim);
  EXPECT_EQ(ds_padded->stride(), padded_stride);
  EXPECT_TRUE(ds_padded->is_owning());
  expect_device_pointer(ds_padded->view().data_handle());
}

TEST(DatasetTypes, DevicePaddedDatasetView)
{
  raft::resources res;
  const int64_t n_rows = 20;
  const uint32_t dim   = 8;
  auto dev_matrix = raft::make_device_matrix<float, int64_t>(res, n_rows, dim);
  auto ds         = make_padded_dataset_view(res, dev_matrix.view());
  EXPECT_EQ(ds.n_rows(), n_rows);
  EXPECT_EQ(ds.dim(), dim);
  EXPECT_EQ(ds.stride(), dim);
  EXPECT_FALSE(ds.is_owning());
  expect_device_pointer(ds.view().data_handle());
  auto v = ds.view();
  EXPECT_EQ(v.extent(0), n_rows);
  EXPECT_EQ(v.extent(1), dim);
}

TEST(DatasetTypes, HostPaddedDataset)
{
  raft::resources res;
  const int64_t n_rows = 30;
  const uint32_t dim   = 12;

  auto data = raft::make_host_matrix<float, int64_t>(res, n_rows, dim);
  auto ds   = std::make_unique<host_padded_dataset<float, int64_t>>(std::move(data), dim);
  ASSERT_NE(ds, nullptr);
  EXPECT_EQ(ds->n_rows(), n_rows);
  EXPECT_EQ(ds->dim(), dim);
  EXPECT_EQ(ds->stride(), dim);
  EXPECT_TRUE(ds->is_owning());
  expect_host_pointer(ds->view().data_handle());
  auto v = ds->view();
  EXPECT_EQ(v.extent(0), n_rows);
  EXPECT_EQ(v.extent(1), dim);
}

TEST(DatasetTypes, HostPaddedDatasetView)
{
  raft::resources res;
  const int64_t n_rows = 10;
  const uint32_t dim   = 4;
  auto host_matrix     = raft::make_host_matrix<float, int64_t>(res, n_rows, dim);
  host_padded_dataset_view<float, int64_t> ds(host_matrix.view());
  EXPECT_EQ(ds.n_rows(), n_rows);
  EXPECT_EQ(ds.dim(), dim);
  EXPECT_EQ(ds.stride(), dim);
  EXPECT_FALSE(ds.is_owning());
  expect_host_pointer(ds.view().data_handle());
  auto v = ds.view();
  EXPECT_EQ(v.extent(0), n_rows);
  EXPECT_EQ(v.extent(1), dim);
}

// make_padded_dataset_view throws when stride does not match required alignment stride;
// error message tells user to use make_padded_dataset() for an owning copy.
TEST(DatasetTypes, MakePaddedDatasetViewThrowsWhenStrideMismatch)
{
  raft::resources res;
  const int64_t n_rows = 10;
  const uint32_t dim   = 30;  // float dim 30 -> required stride 32 (16-byte align)
  auto dev_matrix     = raft::make_device_matrix<float, int64_t>(res, n_rows, 32);
  auto wrong_stride_view =
    raft::make_device_matrix_view(dev_matrix.data_handle(), n_rows, static_cast<int64_t>(dim));  // stride 30
  EXPECT_THROW(
    {
      try {
        (void)make_padded_dataset_view(res, wrong_stride_view);
        FAIL() << "Expected make_padded_dataset_view to throw for incorrect stride";
      } catch (const std::exception& e) {
        std::string msg(e.what());
        EXPECT_NE(msg.find("stride"), std::string::npos)
          << "Expected error message to mention stride, got: " << msg;
        EXPECT_NE(msg.find("make_padded_dataset"), std::string::npos)
          << "Expected error message to direct user to make_padded_dataset(), got: " << msg;
        throw;
      }
    },
    std::exception);
}

// make_padded_dataset throws when source is device and stride already matches required stride;
// error message tells user to use make_padded_dataset_view() instead to avoid redundant copy.
TEST(DatasetTypes, MakePaddedDatasetThrowsWhenStrideMatchesUseViewInstead)
{
  raft::resources res;
  const int64_t n_rows = 10;
  const uint32_t dim   = 8;  // float dim 8 -> required stride 8, so no padding needed
  auto dev_matrix     = raft::make_device_matrix<float, int64_t>(res, n_rows, dim);
  auto correct_stride_view = dev_matrix.view();
  EXPECT_THROW(
    {
      try {
        (void)make_padded_dataset(res, correct_stride_view);
        FAIL() << "Expected make_padded_dataset to throw when stride already correct";
      } catch (const std::exception& e) {
        std::string msg(e.what());
        EXPECT_NE(msg.find("stride is already correct"), std::string::npos)
          << "Expected error to say stride is already correct, got: " << msg;
        EXPECT_NE(msg.find("make_padded_dataset_view"), std::string::npos)
          << "Expected error to direct user to make_padded_dataset_view(), got: " << msg;
        throw;
      }
    },
    std::exception);
}

// ---------------------------------------------------------------------------
// vpq_dataset
// ---------------------------------------------------------------------------
TEST(DatasetTypes, VpqDataset)
{
  raft::resources res;

  const uint32_t dim          = 8;
  const uint32_t vq_n_centers = 4;
  const uint32_t pq_len       = 2;
  const uint32_t pq_n_centers = 256;
  const int64_t n_rows        = 10;
  const uint32_t pq_dim       = dim / pq_len;  // 4

  auto vq_code_book = raft::make_device_matrix<float, uint32_t>(res, vq_n_centers, dim);
  auto pq_code_book = raft::make_device_matrix<float, uint32_t>(res, pq_n_centers, pq_len);
  auto data         = raft::make_device_matrix<uint8_t, int64_t>(res, n_rows, pq_dim);

  vpq_dataset<float, int64_t> vpq(
    std::move(vq_code_book), std::move(pq_code_book), std::move(data));

  EXPECT_EQ(vpq.n_rows(), n_rows);
  EXPECT_EQ(vpq.dim(), dim);
  EXPECT_TRUE(vpq.is_owning());
  EXPECT_EQ(vpq.encoded_row_length(), pq_dim);
  EXPECT_EQ(vpq.vq_n_centers(), vq_n_centers);
  EXPECT_EQ(vpq.pq_len(), pq_len);
  EXPECT_EQ(vpq.pq_n_centers(), pq_n_centers);
  EXPECT_EQ(vpq.pq_dim(), pq_dim);
  EXPECT_EQ(vpq.pq_bits(), 8u);  // 256 = 2^8
}

// ---------------------------------------------------------------------------
// pq_dataset (disabled until pq_dataset is added to common.hpp)
// ---------------------------------------------------------------------------
// TEST(DatasetTypes, PqDataset)
// {
//   raft::resources res;
//
//   const uint32_t pq_len       = 4;
//   const uint32_t pq_n_centers  = 256;
//   const int64_t n_rows         = 20;
//   const uint32_t num_subspaces = 8;  // pq_dim
//
//   auto pq_code_book =
//     raft::make_device_matrix<float, uint32_t>(res, pq_n_centers, pq_len);
//   auto data =
//     raft::make_device_matrix<uint8_t, int64_t>(res, n_rows, num_subspaces);
//
//   pq_dataset<float, int64_t> pq(std::move(pq_code_book), std::move(data));
//
//   EXPECT_EQ(pq.n_rows(), n_rows);
//   EXPECT_EQ(pq.dim(), num_subspaces * pq_len);  // 32
//   EXPECT_TRUE(pq.is_owning());
//   EXPECT_EQ(pq.encoded_row_length(), num_subspaces);
//   EXPECT_EQ(pq.pq_len(), pq_len);
//   EXPECT_EQ(pq.pq_n_centers(), pq_n_centers);
//   EXPECT_EQ(pq.pq_dim(), num_subspaces);
//   EXPECT_EQ(pq.pq_bits(), 8u);
// }

// ---------------------------------------------------------------------------
// Polymorphic access via dataset<IdxT>*
// ---------------------------------------------------------------------------
TEST(DatasetTypes, PolymorphicBaseAccess)
{
  raft::resources res;

  // empty
  empty_dataset<int64_t> empty(64);
  dataset<int64_t>* base = &empty;
  EXPECT_EQ(base->n_rows(), 0);
  EXPECT_EQ(base->dim(), 64u);
  EXPECT_TRUE(base->is_owning());

  // strided (owning)
  auto dev_matrix = raft::make_device_matrix<float, int64_t>(res, 5, 8);
  auto ds_strided = make_strided_dataset(res, dev_matrix.view(), 16u);
  base            = ds_strided.get();
  EXPECT_EQ(base->n_rows(), 5);
  EXPECT_EQ(base->dim(), 8u);
  EXPECT_TRUE(base->is_owning());

  // device padded (owning)
  auto dev_data = raft::make_device_matrix<float, int64_t>(res, 6, 4);
  auto ds_padded =
    std::make_unique<device_padded_dataset<float, int64_t>>(std::move(dev_data), 4u);
  base = ds_padded.get();
  EXPECT_EQ(base->n_rows(), 6);
  EXPECT_EQ(base->dim(), 4u);
  EXPECT_TRUE(base->is_owning());

  // vpq
  auto vq       = raft::make_device_matrix<float, uint32_t>(res, 2, 4);
  auto pq       = raft::make_device_matrix<float, uint32_t>(res, 256, 2);
  auto vpq_data = raft::make_device_matrix<uint8_t, int64_t>(res, 3, 2);
  vpq_dataset<float, int64_t> vpq(std::move(vq), std::move(pq), std::move(vpq_data));
  base = &vpq;
  EXPECT_EQ(base->n_rows(), 3);
  EXPECT_EQ(base->dim(), 4u);
  EXPECT_TRUE(base->is_owning());

  // pq (disabled until pq_dataset is in common.hpp)
  // auto pq_cb = raft::make_device_matrix<float, uint32_t>(res, 256, 2);
  // auto pq_d  = raft::make_device_matrix<uint8_t, int64_t>(res, 4, 2);
  // pq_dataset<float, int64_t> pq_ds(std::move(pq_cb), std::move(pq_d));
  // base = &pq_ds;
  // EXPECT_EQ(base->n_rows(), 4);
  // EXPECT_EQ(base->dim(), 4u);  // 2 subspaces * 2 pq_len
  // EXPECT_TRUE(base->is_owning());
}

}  // namespace cuvs::neighbors::test
