/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/host_mdarray.hpp>

#include <random>
#include <stdint.h>

#define FORCE_INLINE inline __attribute__((always_inline))
#define likely(x)    __builtin_expect(!!(x), 1)
#define unlikely(x)  __builtin_expect(!!(x), 0)
#define lowbit(x)    (x & (-x))
#define bit_id(x)    (__builtin_popcount(x - 1))

constexpr size_t FAST_SIZE = 32;

using PID         = uint32_t;
using pair_di     = std::pair<double, int>;
using FloatRowMat = raft::host_matrix<float, int64_t>;
using IntRowMat   = raft::host_matrix<int32_t, int64_t>;
using UintRowMat  = raft::host_matrix<uint32_t, int64_t>;

struct Candidate {
  PID id;
  float distance;

  Candidate() = default;
  Candidate(PID id, float distance) : id(id), distance(distance) {}

  bool operator<(const Candidate& other) const { return distance < other.distance; }

  bool operator>(const Candidate& other) const { return !(*this < other); }
};

struct ExFactor {
  float xipnorm;
};
