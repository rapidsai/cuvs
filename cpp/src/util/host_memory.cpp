/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/util/host_memory.hpp>

#include <fstream>
#include <string>

namespace cuvs::util {

auto get_free_host_memory() -> size_t
{
  size_t available_memory = 0;
  std::ifstream meminfo("/proc/meminfo");
  std::string line;
  while (std::getline(meminfo, line)) {
    if (line.find("MemAvailable:") != std::string::npos) {
      available_memory = std::stoull(line.substr(line.find(":") + 1));
    }
  }
  available_memory *= 1024;
  meminfo.close();
  RAFT_EXPECTS(available_memory > 0, "Failed to get available memory from /proc/meminfo");
  return available_memory;
}

}  // namespace cuvs::util
