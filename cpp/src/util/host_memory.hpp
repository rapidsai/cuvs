/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
