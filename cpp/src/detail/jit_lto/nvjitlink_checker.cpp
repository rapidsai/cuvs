/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nvjitlink_checker.hpp"

#include <memory>
#include <nvJitLink.h>
#include <string>

#include <raft/core/error.hpp>

void check_nvjitlink_result(nvJitLinkHandle handle, nvJitLinkResult result)
{
  if (result != NVJITLINK_SUCCESS) {
    RAFT_FAIL("nvJITLink failed with error %s", std::to_string(result).c_str());
    size_t log_size = 0;
    result          = nvJitLinkGetErrorLogSize(handle, &log_size);
    if (result == NVJITLINK_SUCCESS && log_size > 0) {
      std::unique_ptr<char[]> log{new char[log_size]};
      result = nvJitLinkGetErrorLog(handle, log.get());
      if (result == NVJITLINK_SUCCESS) {
        RAFT_FAIL("AlgorithmPlanner nvJITLink error log: %s", std::string(log.get()).c_str());
      }
    }
  }
}
