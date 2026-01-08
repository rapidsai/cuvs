/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/util/cutlass_utils.hpp>

#include <cutlass/cutlass.h>

/**
 * @brief Error checking macro for CUTLASS functions.
 *
 * Invokes a CUTLASS function call, if the call does not return cutlass::Status::kSuccess,
 * throws an exception detailing the CUTLASS error that occurred.
 *
 */
#define CUVS_CUTLASS_TRY(call)                        \
  do {                                                \
    cutlass::Status const status = call;              \
    if (status != cutlass::Status::kSuccess) {        \
      std::string msg{};                              \
      SET_ERROR_MSG(msg,                              \
                    "CUTLASS error encountered at: ", \
                    "call='%s', Reason=%s",           \
                    #call,                            \
                    cutlassGetStatusString(status));  \
      throw cuvs::cutlass_error(msg);                 \
    }                                                 \
  } while (0)
