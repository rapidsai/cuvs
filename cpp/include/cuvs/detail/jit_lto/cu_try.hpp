/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/util/cuda_rt_essentials.hpp>

#include <cuda.h>

#define CU_TRY(call)                                                                      \
  do {                                                                                    \
    CUresult const status = call;                                                         \
    if (status != CUDA_SUCCESS) {                                                         \
      std::string msg{};                                                                  \
      const char* name;                                                                   \
      cuGetErrorName(status, &name);                                                      \
      const char* str;                                                                    \
      cuGetErrorString(status, &str);                                                     \
      SET_ERROR_MSG(                                                                      \
        msg, "CUDA error encountered at: ", "call='%s', Reason=%s:%s", #call, name, str); \
      throw raft::cuda_error(msg);                                                        \
    }                                                                                     \
  } while (0)
