/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuda_runtime.h>
#include <raft/util/cuda_rt_essentials.hpp>

namespace cuvs::neighbors::cagra::detail {

template <class T>
void set_value_batch(T* const dev_ptr,
                     const std::size_t ld,
                     const T val,
                     const std::size_t count,
                     const std::size_t batch_size,
                     cudaStream_t cuda_stream);

}  // namespace cuvs::neighbors::cagra::detail
