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
#include <raft/util/integer_utils.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda.h>
#include <cuda_fp16.h>

#include <cfloat>
#include <cstdint>
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

template <typename T>
bool is_ptr_device_accessible(T* ptr)
{
  cudaPointerAttributes attr;
  RAFT_CUDA_TRY(cudaPointerGetAttributes(&attr, ptr));
  return attr.devicePointer != nullptr;
}

template <typename T>
bool is_ptr_host_accessible(T* ptr)
{
  cudaPointerAttributes attr;
  RAFT_CUDA_TRY(cudaPointerGetAttributes(&attr, ptr));
  return attr.hostPointer != nullptr;
}

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
    device_ptr = reinterpret_cast<T*>(attr.devicePointer);
    if (device_ptr == NULL) {
      // allocate memory and copy over
      // NB: We use the temporary "large" workspace resource here; this structure is supposed to
      // live on stack and not returned to a user.
      // The user may opt to set this resource to managed memory to allow large allocations.
      device_mem_.emplace(raft::make_device_mdarray<T, IdxT>(
        res, raft::resource::get_large_workspace_resource(res), host_view.extents()));
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
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource())
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
    RAFT_CUDA_TRY(cudaMemcpy2DAsync(dst.data_handle(),
                                    sizeof(T) * dst.extent(1),
                                    src.data_handle(),
                                    sizeof(T) * src.extent(1),
                                    sizeof(T) * src.extent(1),
                                    src.extent(0),
                                    cudaMemcpyDefault,
                                    raft::resource::get_cuda_stream(res)));
  }
}

/**
 * Utility to create a batched device view from a host view
 *
 * This utility will create a batched device view from a host view and will handle the prefetch and
 * writeback of the data Each batch can be referenced exactlyonce by calling the next_view()
 * function
 *
 * @tparam T The type of the data
 * @tparam IdxT The type of the index
 * @param res The resources
 * @param host_view The host view to create the batched device view from
 * @param batch_size The batch size
 * @param read_only Whether the data is read only (only for managed memory)
 * @param host_writeback Whether to write back the data to the host (only for host memory)
 * @param initialize Whether to initialize the data (only for managed memory)
 * @param evict Whether to evict the data (only for managed memory)
 *
 * @return The batched device view
 */
template <typename T, typename IdxT>
class batched_device_view_from_host {
 public:
  batched_device_view_from_host(raft::resources const& res,
                                raft::host_matrix_view<T, IdxT> host_view,
                                uint64_t batch_size,
                                bool read_only      = false,
                                bool host_writeback = false,
                                bool initialize     = true,
                                bool evict          = false)
    : res_(res),
      host_view_(host_view),
      batch_size_(batch_size),
      offset_(0),
      batch_id_(0),
      num_buffers_(2),
      read_only_(read_only),
      host_writeback_(host_writeback),
      next_buffer_pos_(0),
      evict_(evict),
      initialize_(initialize)
  {
    cudaPointerAttributes attr;
    RAFT_CUDA_TRY(cudaPointerGetAttributes(&attr, host_view.data_handle()));
    mem_type_ = attr.type;
    // cudaMemoryTypeUnregistered = 0
    // cudaMemoryTypeHost = 1
    // cudaMemoryTypeDevice = 2
    // cudaMemoryTypeManaged = 3

    prefetch_stream_  = raft::resource::get_cuda_stream(res);
    writeback_stream_ = raft::resource::get_cuda_stream(res);
    if (res.has_resource_factory(raft::resource::resource_type::CUDA_STREAM_POOL)) {
      if (raft::resource::get_stream_pool_size(res) >= 1) {
        prefetch_stream_  = raft::resource::get_stream_from_stream_pool(res);
        writeback_stream_ = raft::resource::get_stream_from_stream_pool(res);
      }
    }

    // allocations
    if (mem_type_ == cudaMemoryTypeHost || mem_type_ == cudaMemoryTypeUnregistered) {
      device_mem_[0].emplace(raft::make_device_mdarray<T, IdxT>(
        res,
        raft::resource::get_large_workspace_resource(res),
        raft::make_extents<int64_t>(batch_size, host_view.extent(1))));
      device_ptr[0] = device_mem_[0]->data_handle();
      if (batch_size < static_cast<uint64_t>(host_view.extent(0))) {
        device_mem_[1].emplace(raft::make_device_mdarray<T, IdxT>(
          res,
          raft::resource::get_large_workspace_resource(res),
          raft::make_extents<int64_t>(batch_size, host_view.extent(1))));
        device_ptr[1] = device_mem_[1]->data_handle();
      }
      if (host_writeback_ && batch_size * 2 < static_cast<uint64_t>(host_view.extent(0))) {
        num_buffers_ = 3;
        device_mem_[2].emplace(raft::make_device_mdarray<T, IdxT>(
          res,
          raft::resource::get_large_workspace_resource(res),
          raft::make_extents<int64_t>(batch_size, host_view.extent(1))));
        device_ptr[2] = device_mem_[2]->data_handle();
      }
    }

    // if data is managed and not for_write_ we can set the attribute on the device ptr
    if (mem_type_ == cudaMemoryTypeManaged) {
      // location_.type = CU_MEM_LOCATION_TYPE_DEVICE;
      location_.type = cudaMemLocationTypeDevice;
      location_.id   = static_cast<CUdevice>(raft::resource::get_device_id(res_));
      if (read_only_) {
#if CUDA_VERSION >= 13000
        RAFT_CUDA_TRY(cudaMemAdvise(host_view_.data_handle(),
                                    host_view_.extent(0) * host_view_.extent(1) * sizeof(T),
                                    cudaMemAdviseSetReadMostly,
                                    location_));
#else
        RAFT_CUDA_TRY(cudaMemAdvise_v2(host_view_.data_handle(),
                                       host_view_.extent(0) * host_view_.extent(1) * sizeof(T),
                                       cudaMemAdviseSetReadMostly,
                                       location_));
#endif
        // TODO maybe also reset upon destruction
      }
    }

    // prefetch next batch (0)
    prefetch_next_batch();
  }

  bool prefetch_next_batch()
  {
    // this function will ensure the device_ptr [next_buffer_pos_] is pointing to the correct memory
    // after the next synchronization with the prefetch stream

    // if data is on host and we are writing to it we will have to copy it back
    // if data is on host we will have to copy it to the device_ptr

    // if data is managed and evict_ is true we can evict the data from device memory
    // if data is managed we have to prefetch it

    bool next_batch_exists = offset_ < static_cast<uint64_t>(host_view_.extent(0));

    if (next_batch_exists) {
      actual_batch_size_[next_buffer_pos_] =
        next_batch_exists ? min(batch_size_, host_view_.extent(0) - offset_) : 0;

      switch (mem_type_) {
        case cudaMemoryTypeManaged:
#if CUDA_VERSION >= 13000
          if (evict_ && batch_id_ > 1) {
            // evict last active
            CUdeviceptr dptrs[]      = {device_ptr[next_buffer_pos_]};
            size_t sizes[]           = {batch_size_ * host_view_.extent(1) * sizeof(T)};
            size_t prefetchLocIdxs[] = {0};
            RAFT_CUDA_TRY(cuMemDiscardBatchAsync(
              dptrs, sizes, 1, &location_, prefetchLocIdxs, 1, 0, prefetch_stream_));
          }
#endif
          // prefetch
          device_ptr[next_buffer_pos_] = host_view_.data_handle() + offset_ * host_view_.extent(1);
          if (initialize_) {
            // managed API call to prefetch async
#if CUDA_VERSION >= 13000
            RAFT_CUDA_TRY(cudaMemPrefetchAsync(
              device_ptr[next_buffer_pos_],
              actual_batch_size_[next_buffer_pos_] * host_view_.extent(1) * sizeof(T),
              location_,
              0,
              prefetch_stream_));
#else
            RAFT_CUDA_TRY(cudaMemPrefetchAsync_v2(
              device_ptr[next_buffer_pos_],
              actual_batch_size_[next_buffer_pos_] * host_view_.extent(1) * sizeof(T),
              location_,
              0,
              prefetch_stream_));
#endif
          } else {
            // managed API call to cuMemDiscardAndPrefetchBatchAsync (discard and prefetch batch)
#if CUDA_VERSION >= 13000
            CUdeviceptr dptrs[] = {device_ptr[next_buffer_pos_]};
            size_t sizes[]      = {actual_batch_size_[next_buffer_pos_] * host_view_.extent(1) *
                                   sizeof(T)};
            size_t prefetchLocIdxs[] = {0};
            RAFT_CUDA_TRY(cuMemDiscardAndPrefetchBatchAsync(
              dptrs, sizes, 1, &location_, prefetchLocIdxs, 1, 0, prefetch_stream_));
#endif
          }

          break;
        case cudaMemoryTypeHost:
        case cudaMemoryTypeUnregistered:
          if (host_writeback_ && batch_id_ > 1) {
            writeback_stream_.synchronize();
            // copy back last active
            uint32_t writeback_pos    = (next_buffer_pos_ + num_buffers_ - 2) % num_buffers_;
            uint64_t writeback_offset = (offset_ - 2 * batch_size_) * host_view_.extent(1);
            raft::copy(host_view_.data_handle() + writeback_offset,
                       device_ptr[writeback_pos],
                       actual_batch_size_[writeback_pos] * host_view_.extent(1),
                       writeback_stream_);
          }
          if (initialize_) {
            // prefetch next position
            raft::copy(device_ptr[next_buffer_pos_],
                       host_view_.data_handle() + offset_ * host_view_.extent(1),
                       actual_batch_size_[next_buffer_pos_] * host_view_.extent(1),
                       prefetch_stream_);
          }

          break;
        case cudaMemoryTypeDevice:
          // just move pointer to next position
          device_ptr[next_buffer_pos_] = host_view_.data_handle() + offset_ * host_view_.extent(1);
          break;
      }

      offset_ += actual_batch_size_[next_buffer_pos_];
      // swap next_buffer_pos_
      next_buffer_pos_ = (next_buffer_pos_ + 1) % num_buffers_;
    }

    return next_batch_exists;
  }

  ~batched_device_view_from_host() noexcept
  {
    prefetch_stream_.synchronize();
    writeback_stream_.synchronize();
    raft::resource::sync_stream(res_);

    // if data is on host and for_write --> make sure to copy back last active
    // if data is managed and evict --> evict last active

    // make sure to sync on prefetch & writeback stream & res
    switch (mem_type_) {
      case cudaMemoryTypeManaged:
#if CUDA_VERSION >= 13000
        if (evict_ && batch_id_ > 0) {
          // managed API call to evict 2
          uint32_t evict_pos       = (next_buffer_pos_ + num_buffers_ - 1) % num_buffers_;
          CUdeviceptr dptrs[]      = {device_ptr[evict_pos]};
          size_t sizes[]           = {batch_size_ * host_view_.extent(1) * sizeof(T)};
          size_t prefetchLocIdxs[] = {0};
          RAFT_CUDA_TRY(cuMemDiscardBatchAsync(
            dptrs, sizes, 1, &location_, prefetchLocIdxs, 1, 0, prefetch_stream_));
        }
        prefetch_stream_.synchronize();
#endif
        break;
      case cudaMemoryTypeHost:
      case cudaMemoryTypeUnregistered:
        if (host_writeback_ && batch_id_ > 0) {
          // TODO managed API call to copy back last active
          uint32_t writeback_pos = (next_buffer_pos_ + num_buffers_ - 1) % num_buffers_;
          uint64_t writeback_offset =
            (offset_ - actual_batch_size_[writeback_pos]) * host_view_.extent(1);
          raft::copy(host_view_.data_handle() + writeback_offset,
                     device_ptr[writeback_pos],
                     actual_batch_size_[writeback_pos] * host_view_.extent(1),
                     writeback_stream_);
        }
        writeback_stream_.synchronize();
        break;
      case cudaMemoryTypeDevice: break;
    }
  }

  /**
   * Returns the next view of the batch
   *
   * This function will ensure the next batch is ready and will trigger the prefetch of the
   * subsequent next batch
   *
   * @return The next view of the batch
   */
  raft::device_matrix_view<T, IdxT> next_view()
  {
    RAFT_EXPECTS(batch_id_ * batch_size_ < host_view_.extent(0), "Batch index out of bounds");

    // ensure current batch is ready
    prefetch_stream_.synchronize();

    // trigger prefetch of next batch
    bool next_batch_exists = prefetch_next_batch();

    batch_id_++;

    uint32_t current_pos =
      (next_buffer_pos_ + num_buffers_ - (next_batch_exists ? 2 : 1)) % num_buffers_;
    return raft::make_device_matrix_view<T, IdxT>(
      device_ptr[current_pos], actual_batch_size_[current_pos], host_view_.extent(1));
  }

 private:
  cudaMemoryType mem_type_;
  const raft::resources& res_;
  uint64_t batch_size_;
  uint64_t offset_;
  uint64_t num_buffers_;
  bool initialize_;
  rmm::cuda_stream_view prefetch_stream_;
  rmm::cuda_stream_view writeback_stream_;
  bool read_only_;
  bool host_writeback_;
  bool evict_;
  int32_t next_buffer_pos_;
  int32_t batch_id_;
  cudaMemLocation location_;
  std::optional<raft::device_matrix<T, IdxT>> device_mem_[3];
  raft::host_matrix_view<T, IdxT> host_view_;
  T* device_ptr[3];
  uint32_t actual_batch_size_[3];
};

}  // namespace cuvs::neighbors::cagra::detail
