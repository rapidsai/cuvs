/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Created by Stardust on 4/14/25.
//

#pragma once

#include <algorithm>
#include <cstdint>
#include <cuvs/neighbors/ivf_rabitq/utils/utils_cuda.cuh>
#include <limits>

namespace cuvs::neighbors::ivf_rabitq::detail {

// A simple candidate structure.
struct DistId {
  float est_dist;
  uint32_t id;
};

class BoundedKNN {
 public:
  explicit BoundedKNN(size_t capacity) : capacity_(capacity) {}

  // Insert a candidate in sorted order (ascending by est_dist).
  void insert(const DistId& cand)
  {
    // Find insertion position using binary search.
    auto it =
      std::upper_bound(queue_.begin(), queue_.end(), cand, [](const DistId& a, const DistId& b) {
        return a.est_dist < b.est_dist;
      });
    queue_.insert(it, cand);
    // If we exceed capacity, drop the worst candidate (largest est_dist).
    if (queue_.size() > capacity_) { queue_.pop_back(); }
  }

  // Returns the worst (largest est_dist) candidate.
  const DistId& worst() const { return queue_.back(); }

  size_t size() const { return queue_.size(); }

  const std::vector<DistId>& candidates() const { return queue_; }

 private:
  size_t capacity_;
  // Sorted in ascending order by record.est_dist so that the worst is at the back.
  std::vector<DistId> queue_;
};

/// DeviceResultPool is designed to be updated sequentially (by one thread or within a lock).
/// It stores a fixed number (capacity) of candidate results in sorted order.
/// The arrays are allocated in device memory (e.g., in global or shared memory).
struct DeviceResultPool {
  // Pointers to device arrays. For simplicity we assume they point to global memory.
  uint32_t* ids;     // array of candidate IDs
  float* distances;  // corresponding distances
  int capacity;      // maximum number of candidates
  int size = 0;      // current number of candidates

  // Device function: binary search to find the insertion index.
  __device__ int find_bsearch(float dist) const
  {
    int lo = 0;
    int hi = size;
    while (lo < hi) {
      int mid = (lo + hi) >> 1;
      if (distances[mid] < dist)
        lo = mid + 1;
      else
        hi = mid;
    }
    return lo;
  }

  // Device function: insert candidate (id, dist) into the pool, maintaining sorted order.
  // It assumes that the pool is updated sequentially.
  __device__ void insert(uint32_t id, float dist)
  {
    // If the pool is full and dist is greater than the worst (last element), do nothing.
    if (size == capacity && dist > distances[size - 1]) return;

    int idx = find_bsearch(dist);
    // Shift elements to the right.
    for (int j = size; j > idx; j--) {
      ids[j]       = ids[j - 1];
      distances[j] = distances[j - 1];
    }
    ids[idx]       = id;
    distances[idx] = dist;
    if (size < capacity) size++;
  }

  // Return the current worst distance (if full), or maximum float if not full.
  //    __device__ float worst_distance() const {
  //        return (size == capacity) ? distances[size - 1] : std::numeric_limits<float>::max();
  //    }
};

struct HostResultPool {
  // Pointers to host arrays. For simplicity we assume they point to global memory.
  uint32_t* ids;     // array of candidate IDs
  float* distances;  // corresponding distances
  int capacity;      // maximum number of candidates
  int size = 0;      // current number of candidates
};

DeviceResultPool* createDeviceResultPool(int capacity, cudaStream_t stream = 0);

// Frees both the device buffers and the host‑side wrapper.
// Safe to call with a nullptr.
inline void freeDeviceResultPool(DeviceResultPool* pool, cudaStream_t stream = 0)
{
  if (pool == nullptr) {  // nothing to do
    return;
  }

  // 1. Release device buffers (if they were allocated).
  if (pool->ids != nullptr) {
    cudaFreeAsync(pool->ids, stream);
    pool->ids = nullptr;
  }
  if (pool->distances != nullptr) {
    cudaFreeAsync(pool->distances, stream);
    pool->distances = nullptr;
  }

  // 2. Finally, delete the host‑side structure itself.
  delete pool;
}

/**
 * @brief Copy candidate IDs from a device result pool to a host array.
 *
 * @param d_pool        Pointer to the DeviceResultPool in device memory.
 * @param host_results  Host array (of size at least pool->size) where the IDs will be copied.
 */
void copy_results_from_pool(const DeviceResultPool* d_pool, uint32_t* host_results);

}  // namespace cuvs::neighbors::ivf_rabitq::detail
