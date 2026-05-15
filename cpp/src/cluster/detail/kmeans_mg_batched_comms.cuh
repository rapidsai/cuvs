/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/resource/comms.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/nccl_comm.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <nccl.h>

#include <cstddef>
#include <cstdint>

namespace cuvs::cluster::kmeans::mg::detail {

template <typename T>
ncclDataType_t nccl_dtype();

template <>
inline ncclDataType_t nccl_dtype<float>()
{
  return ncclFloat;
}

template <>
inline ncclDataType_t nccl_dtype<double>()
{
  return ncclDouble;
}

template <>
inline ncclDataType_t nccl_dtype<std::int64_t>()
{
  return ncclInt64;
}

class mnmg_comms {
 public:
  mnmg_comms(raft::resources const& dev_res, bool use_nccl, ncclComm_t nccl_comm)
    : dev_res_(dev_res),
      use_nccl_(use_nccl),
      nccl_comm_(nccl_comm),
      stream_(raft::resource::get_cuda_stream(dev_res_))
  {
  }

  cudaStream_t stream() const { return stream_; }

  template <typename T>
  void allreduce(T* sendbuf, T* recvbuf, std::size_t count) const
  {
    if (use_nccl_) {
      RAFT_NCCL_TRY(
        ncclAllReduce(sendbuf, recvbuf, count, nccl_dtype<T>(), ncclSum, nccl_comm_, stream_));
    } else {
      const auto& comm = raft::resource::get_comms(dev_res_);
      comm.allreduce(sendbuf, recvbuf, count, raft::comms::op_t::SUM, stream_);
    }
  }

  template <typename T>
  void bcast(T* buf, std::size_t count, int root) const
  {
    if (use_nccl_) {
      RAFT_NCCL_TRY(ncclBroadcast(buf, buf, count, nccl_dtype<T>(), root, nccl_comm_, stream_));
    } else {
      const auto& comm = raft::resource::get_comms(dev_res_);
      comm.bcast(buf, count, root, stream_);
    }
  }

  template <typename T>
  void allgather(T* sendbuf, T* recvbuf, std::size_t count) const
  {
    if (use_nccl_) {
      RAFT_NCCL_TRY(ncclAllGather(sendbuf, recvbuf, count, nccl_dtype<T>(), nccl_comm_, stream_));
    } else {
      const auto& comm = raft::resource::get_comms(dev_res_);
      comm.allgather(sendbuf, recvbuf, count, stream_);
    }
  }

  void group_start() const
  {
    if (use_nccl_) { RAFT_NCCL_TRY(ncclGroupStart()); }
  }

  void group_end() const
  {
    if (use_nccl_) { RAFT_NCCL_TRY(ncclGroupEnd()); }
  }

 private:
  raft::resources const& dev_res_;
  bool use_nccl_;
  ncclComm_t nccl_comm_;
  cudaStream_t stream_;
};

}  // namespace cuvs::cluster::kmeans::mg::detail
