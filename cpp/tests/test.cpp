/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>
#include <iostream>
#include <raft.hpp>

namespace raft {

TEST(Raft, print)
{
  std::cout << test_raft() << std::endl;
}  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)

}  // namespace raft
