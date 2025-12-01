/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2025, NVIDIA CORPORATION.
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
  bool operator()(const T& a, const T& b) const { return a == b; }
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
  bool operator()(const raft::KeyValuePair<Key, Value>& a,
                  const raft::KeyValuePair<Key, Value>& b) const
  {
    return a.key == b.key && a.value == b.value;
  }
};

template <typename T>
struct CompareApprox {
  CompareApprox(T eps_) : eps(eps_) {}
  bool operator()(const T& a, const T& b) const
  {
    using compute_t     = double;
    const auto a_comp   = static_cast<compute_t>(a);
    const auto b_comp   = static_cast<compute_t>(b);
    const auto eps_comp = static_cast<compute_t>(eps);

    const auto diff  = std::abs(a_comp - b_comp);
    const auto m     = std::max(std::abs(a_comp), std::abs(b_comp));
    const auto ratio = diff > eps_comp ? diff / m : diff;

    return (ratio <= eps_comp);
  }

 private:
  T eps;
};

template <typename Key, typename Value>
::std::ostream& operator<<(::std::ostream& os, const raft::KeyValuePair<Key, Value>& kv)
{
  os << "{ " << kv.key << ", " << kv.value << '}';
  return os;
}

template <typename Key, typename Value>
struct CompareApprox<raft::KeyValuePair<Key, Value>> {
  CompareApprox(raft::KeyValuePair<Key, Value> eps)
    : compare_keys(eps.key), compare_values(eps.value)
  {
  }
  bool operator()(const raft::KeyValuePair<Key, Value>& a,
                  const raft::KeyValuePair<Key, Value>& b) const
  {
    return compare_keys(a.key, b.key) && compare_values(a.value, b.value);
  }

 private:
  CompareApprox<Key> compare_keys;
  CompareApprox<Value> compare_values;
};

template <typename T>
struct CompareApproxAbs {
  CompareApproxAbs(T eps_) : eps(eps_) {}
  bool operator()(const T& a, const T& b) const
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
  CompareApproxNoScaling(T eps_) : eps(eps_) {}
  bool operator()(const T& a, const T& b) const { return (std::abs(a - b) <= eps); }

 private:
  T eps;
};

template <typename T, typename L>
testing::AssertionResult match(const T& expected, const T& actual, L eq_compare)
{
  if (!eq_compare(expected, actual)) {
    return testing::AssertionFailure() << "actual=" << static_cast<double>(actual)
                                       << " != expected=" << static_cast<double>(expected);
  }
  return testing::AssertionSuccess();
}

};  // end namespace cuvs
