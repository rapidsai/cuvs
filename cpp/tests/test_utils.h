/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cmath>
#include <iostream>

#include <gtest/gtest.h>

#include <raft/core/kvp.hpp>

namespace cuvs {

template <typename T>
struct Compare {
  auto operator()(const T& a, const T& b) const -> bool { return a == b; }
};

#if CUDART_VERSION < 12040
// Workaround to support half precision on older CUDA versions. See:
// https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/#cuda-math-release-12-4
template <>
struct Compare<half> {
  bool operator()(const half& a, const half& b) const { return float{a} == float{b}; }
};
#endif

template <typename Key, typename Value>
struct Compare<raft::KeyValuePair<Key, Value>> {
  auto operator()(const raft::KeyValuePair<Key, Value>& a,
                  const raft::KeyValuePair<Key, Value>& b) const -> bool
  {
    return a.key == b.key && a.value == b.value;
  }
};

template <typename T>
struct CompareApprox {
  explicit CompareApprox(T eps_) : eps(eps_) {}
  auto operator()(const T& a, const T& b) const -> bool
  {
    T diff  = std::abs(a - b);
    T m     = std::max(std::abs(a), std::abs(b));
    T ratio = diff > eps ? diff / m : diff;

    return (ratio <= eps);
  }

 private:
  T eps;
};

template <typename Key, typename Value>
auto operator<<(::std::ostream& os, const raft::KeyValuePair<Key, Value>& kv) -> ::std::ostream&
{
  os << "{ " << kv.key << ", " << kv.value << '}';
  return os;
}

template <typename Key, typename Value>
struct CompareApprox<raft::KeyValuePair<Key, Value>> {
  explicit CompareApprox(raft::KeyValuePair<Key, Value> eps)
    : compare_keys(eps.key), compare_values(eps.value)
  {
  }
  auto operator()(const raft::KeyValuePair<Key, Value>& a,
                  const raft::KeyValuePair<Key, Value>& b) const -> bool
  {
    return compare_keys(a.key, b.key) && compare_values(a.value, b.value);
  }

 private:
  CompareApprox<Key> compare_keys;
  CompareApprox<Value> compare_values;
};

template <typename T>
struct CompareApproxAbs {
  explicit CompareApproxAbs(T eps_) : eps(eps_) {}
  auto operator()(const T& a, const T& b) const -> bool
  {
    T diff  = std::abs(std::abs(a) - std::abs(b));
    T m     = std::max(std::abs(a), std::abs(b));
    T ratio = diff >= eps ? diff / m : diff;
    return (ratio <= eps);
  }

 private:
  T eps;
};

template <typename T>
struct CompareApproxNoScaling {
  explicit CompareApproxNoScaling(T eps_) : eps(eps_) {}
  auto operator()(const T& a, const T& b) const -> bool { return (std::abs(a - b) <= eps); }

 private:
  T eps;
};

template <typename T, typename L>
auto match(const T& expected, const T& actual, L eq_compare) -> testing::AssertionResult
{
  if (!eq_compare(expected, actual)) {
    return testing::AssertionFailure() << "actual=" << actual << " != expected=" << expected;
  }
  return testing::AssertionSuccess();
}

};  // end namespace cuvs
