/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <cuvs/neighbors/cagra.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/managed_mdarray.hpp>
#include <raft/core/pinned_mdarray.hpp>

#include <raft/core/copy.cuh>
#include <raft/core/resources.hpp>
#include <raft/matrix/init.cuh>
#include <raft/util/cudart_utils.hpp>

#include "../../../src/neighbors/detail/cagra/utils.hpp"

#include <array>
#include <cstdint>
#include <tuple>
#include <vector>

namespace cuvs::neighbors::cagra {

using IdxT = uint32_t;
using DeviceAccessor =
  raft::host_device_accessor<cuda::std::default_accessor<IdxT>, raft::memory_type::device>;
using HostAccessor =
  raft::host_device_accessor<cuda::std::default_accessor<IdxT>, raft::memory_type::host>;
using PinnedAccessor =
  raft::host_device_accessor<cuda::std::default_accessor<IdxT>, raft::memory_type::pinned>;
using ManagedAccessor =
  raft::host_device_accessor<cuda::std::default_accessor<IdxT>, raft::memory_type::managed>;

struct BatchConfig {
  bool initialize;
  bool host_writeback;
};

struct DimsConfig {
  int64_t n_rows;
  int64_t n_cols;
  uint64_t batch_size;
};

class BatchedDeviceViewTest : public ::testing::Test {
 protected:
  void SetUp() override { raft::resource::sync_stream(res); }

  /**
   * Run batched_device_view over host data, copy device views back,
   * and verify against the input.
   */
  template <typename AccessorInputView>
  void run_and_verify_batched(
    raft::mdspan<IdxT, raft::matrix_extent<int64_t>, raft::row_major, AccessorInputView> input_view,
    uint64_t batch_size,
    bool host_writeback,
    bool initialize)
  {
    int64_t n_rows = input_view.extent(0);
    int64_t n_cols = input_view.extent(1);

    std::vector<IdxT> readback(n_rows * n_cols);

    int64_t total_processed = 0;

    {
      cagra::detail::batched_device_view<IdxT, int64_t, AccessorInputView> batched(
        res, input_view, batch_size, host_writeback, initialize);
      while (true) {
        auto dev_view = batched.next_view();
        if (dev_view.extent(0) == 0) break;

        if (initialize) {
          raft::copy(readback.data() + total_processed * n_cols,
                     dev_view.data_handle(),
                     dev_view.extent(0) * dev_view.extent(1),
                     raft::resource::get_cuda_stream(res));
        }
        if (host_writeback) {
          // Re-wrap as a plain device_matrix_view to strip the (potentially
          // layout_stride / pinned- or managed-accessor) shape that the
          // passthrough path would otherwise hand us, so raft::matrix::fill's
          // device_matrix_view overload accepts the call. dev_view is always
          // exhaustive (contiguous row range of a row-major matrix), so
          // (data_handle, extent(0), extent(1)) describes the same memory.
          // This should eventually be fixed by adding a more generic
          // overload to raft::matrix::fill.
          raft::matrix::fill(res,
                             raft::make_device_matrix_view<IdxT, int64_t>(
                               dev_view.data_handle(), dev_view.extent(0), dev_view.extent(1)),
                             IdxT(17));
        }
        total_processed += dev_view.extent(0);

        // Pair next_view() with prefetch_next(): the next batch's H2D and the
        // previous batch's D2H run on copy_stream_ concurrently with the
        // raft::copy / raft::matrix::fill kernels we just queued on res_.
        batched.prefetch_next();
      }
    }
    raft::resource::sync_stream(res);

    EXPECT_EQ(total_processed, n_rows);
    if (initialize) {
      for (int64_t i = 0; i < n_rows * n_cols; ++i) {
        EXPECT_EQ(readback[i], IdxT(13)) << "Mismatch (initialize) at index " << i;
      }
    }
    if (host_writeback) {
      auto readback_view =
        raft::make_host_matrix_view<IdxT, int64_t>(readback.data(), n_rows, n_cols);
      raft::copy(res, readback_view, input_view);
      raft::resource::sync_stream(res);
      for (int64_t i = 0; i < n_rows * n_cols; ++i) {
        EXPECT_EQ(readback[i], IdxT(17)) << "Mismatch (host_writeback) at index " << i;
      }
    }
  }

  raft::resources res;
};

TEST_F(BatchedDeviceViewTest, EmptyViewFromHost)
{
  auto host_empty = raft::make_host_matrix<IdxT, int64_t>(0, 8);
  auto host_view  = host_empty.view();
  cagra::detail::batched_device_view<IdxT, int64_t, HostAccessor> batched(
    res, host_view, /*batch_size=*/128, /*host_writeback=*/false, /*initialize=*/true);

  auto view = batched.next_view();
  EXPECT_EQ(view.extent(0), 0);
  EXPECT_EQ(view.extent(1), 8);
  EXPECT_EQ(view.data_handle(), nullptr);
}

TEST_F(BatchedDeviceViewTest, EmptyViewFromDevice)
{
  auto device_empty = raft::make_device_matrix<IdxT, int64_t>(res, 0, 8);
  auto device_view  = device_empty.view();
  cagra::detail::batched_device_view<IdxT, int64_t, DeviceAccessor> batched(
    res, device_view, /*batch_size=*/128, /*host_writeback=*/false, /*initialize=*/true);

  auto view = batched.next_view();
  EXPECT_EQ(view.extent(0), 0);
  EXPECT_EQ(view.extent(1), 8);
  EXPECT_EQ(view.data_handle(), nullptr);
}

using BatchDimsParam = std::tuple<BatchConfig, DimsConfig>;

class BatchedDeviceViewParameterizedTest : public BatchedDeviceViewTest,
                                           public ::testing::WithParamInterface<BatchDimsParam> {};

TEST_P(BatchedDeviceViewParameterizedTest, VectorHostData)
{
  auto [batch_config, dims_config]  = GetParam();
  auto [initialize, host_writeback] = batch_config;
  auto [n_rows, n_cols, batch_size] = dims_config;

  std::vector<IdxT> host_data(n_rows * n_cols);
  auto host_view = raft::make_host_matrix_view<IdxT, int64_t>(host_data.data(), n_rows, n_cols);

  std::fill(host_view.data_handle(), host_view.data_handle() + n_rows * n_cols, IdxT(13));

  run_and_verify_batched(host_view, batch_size, host_writeback, initialize);
}

TEST_P(BatchedDeviceViewParameterizedTest, PinnedMemory)
{
  auto [batch_config, dims_config]  = GetParam();
  auto [initialize, host_writeback] = batch_config;
  auto [n_rows, n_cols, batch_size] = dims_config;

  auto pinned_matrix = raft::make_pinned_matrix<IdxT, int64_t>(res, n_rows, n_cols);
  // auto pinned_view   = pinned_matrix.view();
  auto pinned_view =
    raft::mdspan<IdxT, raft::matrix_extent<int64_t>, raft::row_major, PinnedAccessor>(
      pinned_matrix.data_handle(), n_rows, n_cols);
  std::fill(pinned_view.data_handle(), pinned_view.data_handle() + n_rows * n_cols, IdxT(13));

  run_and_verify_batched(pinned_view, batch_size, host_writeback, initialize);
}

TEST_P(BatchedDeviceViewParameterizedTest, PinnedMemoryForcedToHost)
{
  auto [batch_config, dims_config]  = GetParam();
  auto [initialize, host_writeback] = batch_config;
  auto [n_rows, n_cols, batch_size] = dims_config;

  auto pinned_matrix = raft::make_pinned_matrix<IdxT, int64_t>(res, n_rows, n_cols);

  auto pinned_view =
    raft::mdspan<IdxT, raft::matrix_extent<int64_t>, raft::row_major, HostAccessor>(
      pinned_matrix.data_handle(), n_rows, n_cols);

  std::fill(pinned_view.data_handle(), pinned_view.data_handle() + n_rows * n_cols, IdxT(13));
  run_and_verify_batched(pinned_view, batch_size, host_writeback, initialize);
}

TEST_P(BatchedDeviceViewParameterizedTest, ManagedMemory)
{
  auto [batch_config, dims_config]  = GetParam();
  auto [initialize, host_writeback] = batch_config;
  auto [n_rows, n_cols, batch_size] = dims_config;

  auto managed_matrix = raft::make_managed_matrix<IdxT, int64_t>(res, n_rows, n_cols);
  auto managed_view   = managed_matrix.view();

  std::fill(managed_view.data_handle(), managed_view.data_handle() + n_rows * n_cols, IdxT(13));

  run_and_verify_batched(managed_view, batch_size, host_writeback, initialize);
}

TEST_P(BatchedDeviceViewParameterizedTest, ManagedMemoryForcedToHost)
{
  auto [batch_config, dims_config]  = GetParam();
  auto [initialize, host_writeback] = batch_config;
  auto [n_rows, n_cols, batch_size] = dims_config;

  auto managed_matrix = raft::make_managed_matrix<IdxT, int64_t>(res, n_rows, n_cols);

  auto managed_view =
    raft::mdspan<IdxT, raft::matrix_extent<int64_t>, raft::row_major, HostAccessor>(
      managed_matrix.data_handle(), n_rows, n_cols);

  std::fill(managed_view.data_handle(), managed_view.data_handle() + n_rows * n_cols, IdxT(13));

  run_and_verify_batched(managed_view, batch_size, host_writeback, initialize);
}

TEST_P(BatchedDeviceViewParameterizedTest, DeviceMemory)
{
  auto [batch_config, dims_config]  = GetParam();
  auto [initialize, host_writeback] = batch_config;
  auto [n_rows, n_cols, batch_size] = dims_config;

  auto device_matrix = raft::make_device_matrix<IdxT, int64_t>(res, n_rows, n_cols);
  auto device_view   = device_matrix.view();

  raft::matrix::fill(res, device_view, IdxT(13));

  run_and_verify_batched(device_view, batch_size, host_writeback, initialize);
}

static const std::array<BatchConfig, 3> kBatchConfigs = {{
  {/*initialize=*/true, /*host_writeback=*/false},
  {/*initialize=*/false, /*host_writeback=*/true},
  {/*initialize=*/true, /*host_writeback=*/true},
}};

static const std::array<DimsConfig, 4> kDimsConfigs = {{
  {/*n_rows=*/64, /*n_cols=*/32, /*batch_size=*/256},  // rows less than batch size, single batch
  {/*n_rows=*/64, /*n_cols=*/32, /*batch_size=*/64},   // single batch
  {/*n_rows=*/256, /*n_cols=*/32, /*batch_size=*/32},  // multiple batches
  {/*n_rows=*/500,
   /*n_cols=*/32,
   /*batch_size=*/128},  // multiple batches, partial batch in the end
}};

INSTANTIATE_TEST_SUITE_P(BatchConfigs,
                         BatchedDeviceViewParameterizedTest,
                         ::testing::Combine(::testing::ValuesIn(kBatchConfigs),
                                            ::testing::ValuesIn(kDimsConfigs)));

}  // namespace cuvs::neighbors::cagra
