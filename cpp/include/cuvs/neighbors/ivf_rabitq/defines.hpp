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

#include <random>
#include <stdint.h>

#include <cuvs/neighbors/ivf_rabitq/third/Eigen/Dense>

#define FORCE_INLINE inline __attribute__((always_inline))
#define likely(x)    __builtin_expect(!!(x), 1)
#define unlikely(x)  __builtin_expect(!!(x), 0)
#define lowbit(x)    (x & (-x))
#define bit_id(x)    (__builtin_popcount(x - 1))

constexpr size_t FAST_SIZE = 32;

using PID          = uint32_t;
using pair_di      = std::pair<double, int>;
using FloatRowMat  = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using IntRowMat    = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using UintRowMat   = Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using DoubleRowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T>
using RowMajorMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T>
using RowMajorArray = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T>
RowMajorMatrix<T> random_gaussian_matrix(size_t rows, size_t cols)
{
  RowMajorMatrix<T> rand(rows, cols);

#ifdef DEBUG_BATCH_CONSTRUCT
  static std::mt19937 gen(42);
  std::normal_distribution<T> dist(0, 1);
#else
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::normal_distribution<T> dist(0, 1);
#endif

  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      rand(i, j) = dist(gen);
      //            if (i == 0) {
      //                std::cout << rand(i,j) ;
      //            }
    }
  }
  //    std::cout << std::endl;

  return rand;
}

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
