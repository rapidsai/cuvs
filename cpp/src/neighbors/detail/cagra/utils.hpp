/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

// TODO: This shouldn't be invoking anything from detail outside of neighbors namespace
#include <raft/core/copy.cuh>
#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/cuda_stream_pool.hpp>
#include <raft/core/resource/device_id.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/init.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/integer_utils.hpp>
#include <rmm/cuda_stream_pool.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda.h>
#include <cuda_fp16.h>

#include <cfloat>
#include <cstdint>
#include <iostream>
#include <type_traits>

namespace cuvs::neighbors::cagra::detail {
namespace utils {
template <class DATA_T>
inline cudaDataType_t get_cuda_data_type();
template <>
inline cudaDataType_t get_cuda_data_type<float>()
{
  return CUDA_R_32F;
}
template <>
inline cudaDataType_t get_cuda_data_type<half>()
{
  return CUDA_R_16F;
}
template <>
inline cudaDataType_t get_cuda_data_type<int8_t>()
{
  return CUDA_R_8I;
}
template <>
inline cudaDataType_t get_cuda_data_type<uint8_t>()
{
  return CUDA_R_8U;
}
template <>
inline cudaDataType_t get_cuda_data_type<uint32_t>()
{
  return CUDA_R_32U;
}
template <>
inline cudaDataType_t get_cuda_data_type<uint64_t>()
{
  return CUDA_R_64U;
}

template <class T>
constexpr unsigned size_of();
template <>
_RAFT_HOST_DEVICE constexpr unsigned size_of<std::int8_t>()
{
  return 1;
}
template <>
_RAFT_HOST_DEVICE constexpr unsigned size_of<std::uint8_t>()
{
  return 1;
}
template <>
_RAFT_HOST_DEVICE constexpr unsigned size_of<std::uint16_t>()
{
  return 2;
}
template <>
_RAFT_HOST_DEVICE constexpr unsigned size_of<std::uint32_t>()
{
  return 4;
}
template <>
_RAFT_HOST_DEVICE constexpr unsigned size_of<std::uint64_t>()
{
  return 8;
}
template <>
_RAFT_HOST_DEVICE constexpr unsigned size_of<uint4>()
{
  return 16;
}
template <>
_RAFT_HOST_DEVICE constexpr unsigned size_of<ulonglong4>()
{
  return 32;
}
template <>
_RAFT_HOST_DEVICE constexpr unsigned size_of<float>()
{
  return 4;
}
template <>
_RAFT_HOST_DEVICE constexpr unsigned size_of<half>()
{
  return 2;
}
template <>
_RAFT_HOST_DEVICE constexpr unsigned size_of<half2>()
{
  return 4;
}

// max values for data types
template <class BS_T, class FP_T>
union fp_conv {
  BS_T bs;
  FP_T fp;
};
template <class T>
_RAFT_HOST_DEVICE constexpr inline T get_max_value();
template <>
_RAFT_HOST_DEVICE constexpr inline float get_max_value<float>()
{
  return FLT_MAX;
};
template <>
_RAFT_HOST_DEVICE constexpr inline half get_max_value<half>()
{
  return fp_conv<std::uint16_t, half>{.bs = 0x7aff}.fp;
};
template <>
_RAFT_HOST_DEVICE constexpr inline std::uint32_t get_max_value<std::uint32_t>()
{
  return 0xffffffffu;
};
template <>
_RAFT_HOST_DEVICE constexpr inline std::uint64_t get_max_value<std::uint64_t>()
{
  return 0xfffffffffffffffflu;
};

template <int A, int B, class = void>
struct constexpr_max {
  static const int value = A;
};

template <int A, int B>
struct constexpr_max<A, B, std::enable_if_t<(B > A), bool>> {
  static const int value = B;
};

template <class IdxT>
struct gen_index_msb_1_mask {
  static constexpr IdxT value = static_cast<IdxT>(1) << (utils::size_of<IdxT>() * 8 - 1);
};
}  // namespace utils

/**
 * Utility to sync memory from a host_matrix_view to a device_matrix_view
 *
 * In certain situations (UVM/HMM/ATS) host memory might be directly accessible on the
 * device, and no extra allocations need to be performed. This class checks
 * if the host_matrix_view is already accessible on the device, and only creates device
 * memory and copies over if necessary. In memory limited situations this is preferable
 * to having both a host and device copy
 * TODO: once the mdbuffer changes here https://github.com/wphicks/raft/blob/fea-mdbuffer
 * have been merged, we should remove this class and switch over to using mdbuffer for this
 */
template <typename T, typename IdxT>
class device_matrix_view_from_host {
 public:
  device_matrix_view_from_host(raft::resources const& res,
                               raft::host_matrix_view<T, IdxT> host_view)
    : res_(res), host_view_(host_view)
  {
    cudaPointerAttributes attr;
    RAFT_CUDA_TRY(cudaPointerGetAttributes(&attr, host_view.data_handle()));
    device_ptr      = reinterpret_cast<T*>(attr.devicePointer);
    bool needs_copy = (device_ptr == NULL) ||
                      (attr.type != cudaMemoryTypeDevice && attr.type != cudaMemoryTypeManaged);
    if (needs_copy) {
      // allocate memory and copy over
      // NB: We use the temporary "large" workspace resource here; this structure is supposed to
      // live on stack and not returned to a user.
      // The user may opt to set this resource to managed memory to allow large allocations.
      device_mem_.emplace(raft::make_device_mdarray<T, IdxT>(
        res, raft::resource::get_large_workspace_resource_ref(res), host_view.extents()));
      raft::copy(res, device_mem_->view(), host_view);
      device_ptr = device_mem_->data_handle();
    }
  }

  ~device_matrix_view_from_host() noexcept
  {
    /*
    If there's no copy, there's no allocation owned by this struct.
    If there's no allocation, there's no guarantee that the device pointer is stream-ordered.
    If there's no stream order guarantee, we must synchronize with the stream before the struct is
    destroyed to make sure all GPU operations in that stream finish earlier.
    */
    if (!allocated_memory()) { raft::resource::sync_stream(res_); }
  }

  raft::device_matrix_view<T, IdxT> view()
  {
    return raft::make_device_matrix_view<T, IdxT>(
      device_ptr, host_view_.extent(0), host_view_.extent(1));
  }

  T* data_handle() { return device_ptr; }

  [[nodiscard]] bool allocated_memory() const { return device_mem_.has_value(); }

 private:
  const raft::resources& res_;
  std::optional<raft::device_matrix<T, IdxT>> device_mem_;
  raft::host_matrix_view<T, IdxT> host_view_;
  T* device_ptr;
};

/**
 * Utility to sync memory from a device_matrix_view to a host_matrix_view
 *
 * In certain situations (UVM/HMM/ATS) device memory might be directly accessible on the
 * host, and no extra allocations need to be performed. This class checks
 * if the device_matrix_view is already accessible on the host, and only creates host
 * memory and copies over if necessary. In memory limited situations this is preferable
 * to having both a host and device copy
 * TODO: once the mdbuffer changes here https://github.com/wphicks/raft/blob/fea-mdbuffer
 * have been merged, we should remove this class and switch over to using mdbuffer for this
 */
template <typename T, typename IdxT>
class host_matrix_view_from_device {
 public:
  host_matrix_view_from_device(raft::resources const& res,
                               raft::device_matrix_view<T, IdxT> device_view)
    : device_view_(device_view)
  {
    cudaPointerAttributes attr;
    RAFT_CUDA_TRY(cudaPointerGetAttributes(&attr, device_view.data_handle()));
    host_ptr = reinterpret_cast<T*>(attr.hostPointer);
    if (host_ptr == NULL) {
      // allocate memory and copy over
      host_mem_.emplace(
        raft::make_host_matrix<T, IdxT>(device_view.extent(0), device_view.extent(1)));
      raft::copy(res, host_mem_->view(), device_view);
      host_ptr = host_mem_->data_handle();
    }
  }

  raft::host_matrix_view<T, IdxT> view()
  {
    return raft::make_host_matrix_view<T, IdxT>(
      host_ptr, device_view_.extent(0), device_view_.extent(1));
  }

  T* data_handle() { return host_ptr; }

  bool allocated_memory() const { return host_mem_.has_value(); }

 private:
  std::optional<raft::host_matrix<T, IdxT>> host_mem_;
  raft::device_matrix_view<T, IdxT> device_view_;
  T* host_ptr;
};

// Copy matrix src to dst. pad rows with 0 if necessary to make them 16 byte aligned.
template <typename T, typename data_accessor>
void copy_with_padding(
  raft::resources const& res,
  raft::device_matrix<T, int64_t, raft::row_major>& dst,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, data_accessor> src,
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref())
{
  size_t padded_dim = raft::round_up_safe<size_t>(src.extent(1) * sizeof(T), 16) / sizeof(T);

  if ((dst.extent(0) != src.extent(0)) || (static_cast<size_t>(dst.extent(1)) != padded_dim)) {
    // clear existing memory before allocating to prevent OOM errors on large datasets
    if (dst.size()) { dst = raft::make_device_matrix<T, int64_t>(res, 0, 0); }
    dst =
      raft::make_device_mdarray<T>(res, mr, raft::make_extents<int64_t>(src.extent(0), padded_dim));
  }
  if (dst.extent(1) == src.extent(1)) {
    raft::copy(res, dst.view(), src);
  } else {
    // copy with padding
    raft::matrix::fill(res, dst.view(), T(0));
    raft::copy_matrix(dst.data_handle(),
                      dst.extent(1),
                      src.data_handle(),
                      src.extent(1),
                      src.extent(1),
                      src.extent(0),
                      raft::resource::get_cuda_stream(res));
  }
}

/**
 * Iterate a 2D mdspan in row-batches with overlapping prefetch/writeback.
 *
 * The strategy is selected at compile time from
 * `AccessorInputView::is_device_accessible`:
 *   * passthrough: each batch is `cuda::std::submdspan(input_view_, ...)`.
 *     No buffering, no arithmetic on `input_view_.data_handle()`.
 *   * copy_device: each batch is staged through an internal device buffer
 *     via `cudaMemcpyAsync`. When `host_writeback` is set, every returned
 *     batch is copied back to `input_view_` on destruction (and earlier
 *     batches incrementally as iteration progresses).
 *
 * Usage:
 * ```
 * batched_device_view<float, int32_t, AccessorInputView> view(
 *   res, input_view, batch_size, host_writeback, initialize);
 * for (;;) {
 *   auto device_view = view.next_view();
 *   if (device_view.extent(0) == 0) { break; }   // sole stop condition
 *   // ... call next_view() exactly once per batch; never in the loop condition
 * }
 * ```
 *
 * Differences vs `cuvs::neighbors::detail::utils::batch_load_iterator`
 * (cpp/src/neighbors/detail/ann_utils.cuh):
 *   * Input typing: this class takes a typed mdspan and decides the strategy
 *     at compile time from the accessor; `batch_load_iterator` takes
 *     `(T const*, n_rows, row_width)` and decides at runtime via
 *     `cudaPointerGetAttributes` (also forces a copy for HMM/ATS sources).
 *   * API: this class exposes a single-pass `next_view()` that returns an
 *     mdspan; `batch_load_iterator` is an STL iterator (begin/end,
 *     operator++/--, random access) that yields a `batch` wrapping a
 *     `T const*`.
 *   * Mutability: returned views here are mutable T* and support host
 *     writeback on destruction; `batch_load_iterator` is read-only and never
 *     copies back.
 *   * Pipelining: copy_device uses 2-3 device buffers and dedicated prefetch
 *     and writeback streams for true overlap; `batch_load_iterator` uses
 *     1-2 buffers and exposes a synchronous `prefetch_next_batch()` that the
 *     caller must interleave with kernels on a separate stream.
 *
 * @tparam T                 element type
 * @tparam IdxT              index type for the input mdspan extents
 * @tparam AccessorInputView accessor of the input mdspan
 */
template <typename T, typename IdxT, typename AccessorInputView>
class batched_device_view {
  using input_view_t =
    raft::mdspan<T, raft::matrix_extent<IdxT>, raft::row_major, AccessorInputView>;

 public:
  // Compile-time strategy switch; see class-level documentation for semantics.
  static constexpr bool kPassthrough = AccessorInputView::is_device_accessible;

  static_assert(kPassthrough ||
                  cuda::std::is_convertible_v<typename AccessorInputView::data_handle_type, T*>,
                "copy_device path issues cudaMemcpyAsync against input_view_.data_handle() "
                "in both directions, so AccessorInputView::data_handle_type must be "
                "convertible to T*. To lift this, route prefetch/writeback through "
                "raft::copy on submdspans of input_view_.");

  // Result of submdspan(layout_right, tuple<IdxT,IdxT>, full_extent):
  // layout_stride with AccessorInputView preserved.
  using next_view_passthrough_type =
    decltype(cuda::std::submdspan(std::declval<input_view_t&>(),
                                  std::declval<cuda::std::tuple<IdxT, IdxT>>(),
                                  cuda::std::full_extent));

  // Internal contiguous row-major device buffer.
  using next_view_copy_type = raft::device_matrix_view<T, IdxT>;

  /**
   * @param res             raft resources (must outlive this object)
   * @param input_view      mdspan to iterate over
   * @param batch_size      rows per batch (must be > 0 if input_view is non-empty)
   * @param host_writeback  copy each batch back to input_view_ after use
   *                        (no-op for passthrough)
   * @param initialize      stage each batch's contents into the device buffer
   *                        before returning it (no-op for passthrough)
   *
   * At least one of host_writeback / initialize must be true for non-empty input.
   */
  batched_device_view(raft::resources const& res,
                      input_view_t input_view,
                      uint64_t batch_size,
                      bool host_writeback = false,
                      bool initialize     = true)
    : res_(res),
      input_view_(input_view),
      batch_size_(batch_size),
      offset_(0),
      batch_id_(-2),
      num_buffers_(2),
      host_writeback_(host_writeback),
      initialize_(initialize)
  {
    if (input_view.extent(0) == 0) { return; }

    RAFT_EXPECTS(batch_size_ > 0, "batch_size must be greater than zero for non-empty input");
    RAFT_EXPECTS(host_writeback_ || initialize_,
                 "At least one of host_writeback or initialize must be true");

    RAFT_LOG_DEBUG("Memory strategy: %s for matrix of type %s, dimensions %zu x %zu",
                   kPassthrough ? "passthrough" : "copy_device",
                   typeid(T).name(),
                   input_view.extent(0),
                   input_view.extent(1));

    // device buffers (copy_device only)
    if constexpr (!kPassthrough) {
      try {
        device_mem_[0].emplace(raft::make_device_mdarray<T, IdxT>(
          res,
          raft::resource::get_workspace_resource_ref(res),
          raft::make_extents<int64_t>(batch_size, input_view.extent(1))));
        device_ptr[0] = device_mem_[0]->data_handle();
        if (batch_size < static_cast<uint64_t>(input_view.extent(0))) {
          device_mem_[1].emplace(raft::make_device_mdarray<T, IdxT>(
            res,
            raft::resource::get_workspace_resource_ref(res),
            raft::make_extents<int64_t>(batch_size, input_view.extent(1))));
          device_ptr[1] = device_mem_[1]->data_handle();
        }
        if (host_writeback_ && initialize_ &&
            batch_size * 2 < static_cast<uint64_t>(input_view.extent(0))) {
          num_buffers_ = 3;
          device_mem_[2].emplace(raft::make_device_mdarray<T, IdxT>(
            res,
            raft::resource::get_workspace_resource_ref(res),
            raft::make_extents<int64_t>(batch_size, input_view.extent(1))));
          device_ptr[2] = device_mem_[2]->data_handle();
        }
      } catch (std::bad_alloc& e) {
        throw std::bad_alloc();
      } catch (raft::logic_error& e) {
        throw raft::logic_error("Insufficient memory for device buffers (logic error)");
      }
    }

    // ensure a stream pool with one stream per concurrent direction
    size_t required_streams = host_writeback_ && initialize_ ? 2 : 1;
    if (!res.has_resource_factory(raft::resource::resource_type::CUDA_STREAM_POOL) ||
        raft::resource::get_stream_pool_size(res) < required_streams) {
      raft::resource::set_cuda_stream_pool(res, std::make_shared<rmm::cuda_stream_pool>(2));
    }
    prefetch_stream_  = raft::resource::get_stream_from_stream_pool(res);
    writeback_stream_ = raft::resource::get_stream_from_stream_pool(res);

    // prime batch 0
    prefetch_next_batch();
  }

  ~batched_device_view() noexcept
  {
    raft::resource::sync_stream(res_);

    // Nothing was returned to the caller -> nothing to write back.
    if (batch_id_ < 0) { return; }

    // Passthrough has no internal buffer and writes through input_view_ directly.
    if constexpr (!kPassthrough) {
      if (host_writeback_) {
        uint32_t writeback_pos_last = batch_id_ % num_buffers_;
        if (batch_id_ > 0) {
          uint32_t writeback_pos    = (batch_id_ - 1) % num_buffers_;
          uint64_t writeback_offset = (batch_id_ - 1) * batch_size_;
          writeback_from_device_to_host(device_ptr[writeback_pos], writeback_offset, batch_size_);
        }
        {
          uint64_t writeback_offset_last = batch_id_ * batch_size_;
          writeback_from_device_to_host(device_ptr[writeback_pos_last],
                                        writeback_offset_last,
                                        actual_batch_size_[writeback_pos_last]);
        }
        writeback_stream_.synchronize();
      }
    }
  }

  /**
   * Advance to the next batch and return its view.
   *
   * In copy_device mode this also kicks off prefetch of the batch after it
   * and writeback of the previously returned batch (when host_writeback is
   * set). In passthrough mode the call is essentially a slice update.
   *
   * Return type is `next_view_passthrough_type` or `next_view_copy_type`;
   * both are 2D mdspans of T with the same element- and extent-API surface.
   * Iteration ends when extent(0) == 0; this is the only legal stop signal.
   *
   * Must be called exactly once per batch (never in a loop condition).
   */
  auto next_view()
  {
    bool end_of_data = static_cast<uint64_t>((batch_id_ + 1) * batch_size_) >=
                       static_cast<uint64_t>(input_view_.extent(0));

    if constexpr (kPassthrough) {
      // Slice goes through the accessor; data_handle() is never used.
      if (end_of_data) {
        auto end = static_cast<IdxT>(input_view_.extent(0));
        return cuda::std::submdspan(
          input_view_, cuda::std::tuple{end, end}, cuda::std::full_extent);
      }

      prefetch_next_batch();

      uint32_t current_pos = batch_id_ % num_buffers_;
      auto first           = static_cast<IdxT>(batch_id_ * batch_size_);
      auto last            = static_cast<IdxT>(first + actual_batch_size_[current_pos]);
      return cuda::std::submdspan(
        input_view_, cuda::std::tuple{first, last}, cuda::std::full_extent);
    } else {
      auto cols = static_cast<IdxT>(input_view_.extent(1));

      if (end_of_data) { return next_view_copy_type{nullptr, IdxT{0}, cols}; }

      prefetch_next_batch();

      uint32_t current_pos = batch_id_ % num_buffers_;
      return next_view_copy_type{
        device_ptr[current_pos], static_cast<IdxT>(actual_batch_size_[current_pos]), cols};
    }
  }

 private:
  /**
   * Advance batch_id_ to the next batch and stage I/O around it (copy_device only):
   *   * prefetch the (batch_id_ + 1)-th batch when initialize_ is set
   *   * write back the (batch_id_ - 1)-th batch when host_writeback_ is set
   * In passthrough mode this only updates batch_id_ and offset_ bookkeeping.
   *
   * @return true if more input rows remain after this advance.
   */
  bool prefetch_next_batch()
  {
    batch_id_++;

    // wait for the prior round's prefetch / writeback before reusing the slots
    if (initialize_) { prefetch_stream_.synchronize(); }
    if (host_writeback_) { writeback_stream_.synchronize(); }

    RAFT_EXPECTS(static_cast<int64_t>(offset_) <= input_view_.extent(0), "Offset out of bounds");

    bool next_batch_exists = offset_ < static_cast<uint64_t>(input_view_.extent(0));

    if (next_batch_exists) {
      // make sure all kernels touching batch_id_ - 1 (the now-writable slot)
      // are done before we either overwrite it (initialize) or read it back
      raft::resource::sync_stream(res_);

      int32_t prefetch_pos             = (batch_id_ + 1) % num_buffers_;
      actual_batch_size_[prefetch_pos] = min(batch_size_, input_view_.extent(0) - offset_);

      if constexpr (!kPassthrough) {
        if (host_writeback_ && batch_id_ > 0) {
          uint32_t writeback_pos    = (batch_id_ - 1) % num_buffers_;
          uint64_t writeback_offset = (batch_id_ - 1) * batch_size_;
          writeback_from_device_to_host(device_ptr[writeback_pos], writeback_offset, batch_size_);
        }
        if (initialize_) {
          prefetch_from_host_to_device(
            device_ptr[prefetch_pos], offset_, actual_batch_size_[prefetch_pos]);
        }
      }

      offset_ += actual_batch_size_[prefetch_pos];
    }

    return next_batch_exists;
  }

  void prefetch_from_host_to_device(T* dev_ptr, size_t src_row_offset, size_t num_rows)
  {
    const size_t n_elem  = num_rows * input_view_.extent(1);
    const size_t n_bytes = n_elem * sizeof(T);
    // use memcpy instead of raft::copy to avoid strange behavior with HMM/ATS memory
    RAFT_CUDA_TRY(
      cudaMemcpyAsync(dev_ptr,
                      input_view_.data_handle() + src_row_offset * input_view_.extent(1),
                      n_bytes,
                      cudaMemcpyHostToDevice,
                      prefetch_stream_));
  }

  void writeback_from_device_to_host(T* dev_ptr, size_t dst_row_offset, size_t num_rows)
  {
    const size_t n_elem  = num_rows * input_view_.extent(1);
    const size_t n_bytes = n_elem * sizeof(T);
    // use memcpy instead of raft::copy to avoid strange behavior with HMM/ATS memory
    RAFT_CUDA_TRY(
      cudaMemcpyAsync(input_view_.data_handle() + dst_row_offset * input_view_.extent(1),
                      dev_ptr,
                      n_bytes,
                      cudaMemcpyDeviceToHost,
                      writeback_stream_));
  }

  // streams (copy_device only; unused in passthrough)
  rmm::cuda_stream_view prefetch_stream_;
  rmm::cuda_stream_view writeback_stream_;

  // configuration
  const raft::resources& res_;
  bool initialize_;
  bool host_writeback_;

  // iteration state
  uint64_t batch_size_;
  int32_t batch_id_;  // -2 before any prefetch; >= 0 once a batch has been returned
  uint64_t offset_;   // first row of the upcoming-prefetch batch, in input_view_ rows

  input_view_t input_view_;

  // device buffers (copy_device only)
  uint64_t num_buffers_;
  std::optional<raft::device_matrix<T, IdxT>> device_mem_[3];
  T* device_ptr[3];
  uint32_t actual_batch_size_[3];
};

}  // namespace cuvs::neighbors::cagra::detail
