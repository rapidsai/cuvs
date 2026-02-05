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
size_t get_free_host_memory();

}  // namespace cuvs::util
