/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <omp.h>

#include <queue>

#include <cuvs/neighbors/ivf_rabitq/defines.hpp>

inline constexpr size_t div_rd_up(size_t x, size_t y)
{
  return (x / y) + static_cast<size_t>((x % y) != 0);
}

inline constexpr size_t rd_up_to_multiple_of(size_t x, size_t y) { return y * (div_rd_up(x, y)); }

double get_ratio(size_t numq,
                 const FloatRowMat& query,
                 const FloatRowMat& data,
                 const UintRowMat& gt,
                 PID* ann_results,
                 size_t K,
                 float (*dist_func)(const float*, const float*, size_t))
{
  std::priority_queue<float> gt_distances;
  std::priority_queue<float> ann_distances;

  for (size_t i = 0; i < K; ++i) {
    PID gt_id  = gt(numq, i);
    PID ann_id = ann_results[i];
    gt_distances.emplace(dist_func(&query(numq, 0), &data(gt_id, 0), data.cols()));
    ann_distances.emplace(dist_func(&query(numq, 0), &data(ann_id, 0), data.cols()));
  }

  double ret     = 0;
  size_t valid_k = 0;

  while (!gt_distances.empty()) {
    if (gt_distances.top() > 1e-5) {
      ret += std::sqrt((double)ann_distances.top() / gt_distances.top());
      ++valid_k;
    }
    gt_distances.pop();
    ann_distances.pop();
  }

  if (valid_k == 0) { return 1.0 * K; }
  //    printf("ret = %f, valid_k = %zu, K = %zu\n", ret, valid_k, K);
  //    fflush(stdout);
  return ret / valid_k * K;
}

template <typename T>
std::vector<T> horizontal_avg(const std::vector<std::vector<T>>& data)
{
  size_t rows = data.size();
  size_t cols = data[0].size();

  for (auto& row : data) {
    assert(row.size() == cols);
  }

  std::vector<T> avg(cols, 0);
  for (auto& row : data) {
    for (size_t j = 0; j < cols; ++j) {
      avg[j] += row[j];
    }
  }

  for (size_t j = 0; j < cols; ++j) {
    avg[j] /= rows;
  }

  return avg;
}
