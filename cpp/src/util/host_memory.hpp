/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/error.hpp>

#include <fstream>
#include <string>

namespace cuvs::util {

/**
 * @brief Get available host memory from /proc/meminfo
 *
 * Queries the system for available memory by reading /proc/meminfo.
 * This is useful for determining how much host memory can be used
 * for buffering or temporary storage.
 *
 * @return Available memory in bytes
 */
inline size_t get_free_host_memory()
{
  size_t available_memory = 0;
  std::ifstream meminfo("/proc/meminfo");
  std::string line;
  while (std::getline(meminfo, line)) {
    if (line.find("MemAvailable:") != std::string::npos) {
      available_memory = std::stoi(line.substr(line.find(":") + 1));
    }
  }
  available_memory *= 1024;
  meminfo.close();
  RAFT_EXPECTS(available_memory > 0, "Failed to get available memory from /proc/meminfo");
  return available_memory;
}

}  // namespace cuvs::util
