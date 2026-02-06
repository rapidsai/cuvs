/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/error.hpp>

namespace cuvs {

/**
 * @brief Exception thrown when a CUTLASS error is encountered.
 */
struct cutlass_error : public raft::exception {
  explicit cutlass_error(char const* const message) : raft::exception(message) {}
  explicit cutlass_error(std::string const& message) : raft::exception(message) {}
};

}  // namespace cuvs
