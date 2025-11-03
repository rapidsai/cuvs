/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuvs/core/bitset.hpp>
#include <raft/core/bitset.cuh>

template struct raft::core::bitset<uint8_t, uint32_t>;
template struct raft::core::bitset<uint16_t, uint32_t>;
template struct raft::core::bitset<uint32_t, uint32_t>;
template struct raft::core::bitset<uint32_t, int64_t>;
template struct raft::core::bitset<uint64_t, int64_t>;
