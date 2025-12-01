/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>
#include <iostream>
#include <raft.hpp>

namespace raft {

TEST(Raft, print) { std::cout << test_raft() << std::endl; }

}  // namespace raft
