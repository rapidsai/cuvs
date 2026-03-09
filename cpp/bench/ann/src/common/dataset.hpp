/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "ann_types.hpp"
#include "blob.hpp"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <optional>
#include <random>
#include <string>

namespace cuvs::bench {

template <typename CarrierT>
void generate_bernoulli(CarrierT* data, size_t words, double p)
{
  constexpr size_t kBitsPerCarrierValue = sizeof(CarrierT) * 8;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::bernoulli_distribution d(p);
  for (size_t i = 0; i < words; i++) {
    CarrierT word = 0;
    for (size_t j = 0; j < kBitsPerCarrierValue; j++) {
      word |= CarrierT{d(gen)} << j;
    }
    data[i] = word;
  }
};

template <typename DataT, typename IdxT = int32_t>
struct dataset {
 public:
  using bitset_carrier_type                           = uint32_t;
  static inline constexpr size_t kBitsPerCarrierValue = sizeof(bitset_carrier_type) * 8;
  static constexpr uint32_t kMaxQueriesForRecall      = 10'000;
  struct gt_entry {
    // Hash map of {id, neighbor_rank} for a particular query in the ground truth set
    // e.g. gt_map[i] = j means that the neighbor with id i is the j-th nearest.
    // Note that the nearest neighbor rank starts from 0.
    std::unordered_map<IdxT, IdxT> gt_map;
    uint32_t filter_pass_count = 0;
    std::once_flag once_flag;
  };

 private:
  std::string name_;
  std::string distance_;
  blob<DataT> base_set_;
  blob<DataT> query_set_;
  std::optional<blob<IdxT>> ground_truth_set_;
  std::optional<blob<bitset_carrier_type>> filter_bitset_;
  mutable std::vector<gt_entry> gt_entries_;
  uint32_t max_k_;
  mutable std::once_flag gt_entries_init_flag_;

  // Protects the lazy mutations of the blobs accessed by multiple threads
  mutable std::mutex mutex_;
  // The dim can be read either from the training set or from the query set.
  // This cache variable is filled from either of the two sets loaded first.
  mutable std::atomic<int> dim_ = -1;

  // Cache the dim value from the passed blob.
  inline void cache_dim(const blob<DataT>& blob) const
  {
    if (dim_.load(std::memory_order_relaxed) == -1) {
      dim_.store(static_cast<int>(blob.n_cols()), std::memory_order_relaxed);
    }
  }

 public:
  dataset(std::string name,
          std::string base_file,
          uint32_t subset_first_row,
          uint32_t subset_size,
          std::string query_file,
          std::string distance,
          std::optional<std::string> groundtruth_neighbors_file,
          std::optional<double> filtering_rate = std::nullopt)
    : name_{std::move(name)},
      distance_{std::move(distance)},
      base_set_{base_file, subset_first_row, subset_size},
      query_set_{query_file},
      ground_truth_set_{groundtruth_neighbors_file.has_value()
                          ? std::make_optional<blob<IdxT>>(groundtruth_neighbors_file.value())
                          : std::nullopt},
      max_k_{ground_truth_set_.has_value() ? ground_truth_set_->n_cols() : 0}
  {
    if (filtering_rate.has_value()) {
      // Generate a random bitset for filtering
      auto n_rows = static_cast<size_t>(subset_size) + static_cast<size_t>(subset_first_row);
      if (subset_size == 0) {
        // Read the base set size as a last resort only - for better laziness
        n_rows = base_set_size();
      }
      auto bitset_size = (n_rows - 1) / kBitsPerCarrierValue + 1;
      blob_file<bitset_carrier_type> bitset_blob_file{static_cast<uint32_t>(bitset_size), 1};
      blob_mmap<bitset_carrier_type> bitset_blob{
        std::move(bitset_blob_file), false, HugePages::kDisable};
      generate_bernoulli(const_cast<bitset_carrier_type*>(bitset_blob.data()),
                         bitset_size,
                         1.0 - filtering_rate.value());
      filter_bitset_.emplace(std::move(bitset_blob));
    }
  }

  [[nodiscard]] auto name() const -> std::string { return name_; }
  [[nodiscard]] auto distance() const -> std::string { return distance_; }
  [[nodiscard]] auto dim() const -> int
  {
    auto d = dim_.load(std::memory_order_relaxed);
    if (d > -1) { return d; }
    std::lock_guard<std::mutex> lock(mutex_);
    // Otherwise, try reading both (one of the two sets may be missing)
    try {
      d = static_cast<int>(query_set_.n_cols());
    } catch (const std::runtime_error& e) {
      // Any exception raised above will re-raise next time we try to access the query set.
      query_set_.reset_lazy_state();
      // If the query set is not accessible, use the base set.
      // Don't catch the exception here, because we have nothing else to do anyway.
      d = static_cast<int>(base_set_.n_cols());
    }
    dim_.store(d, std::memory_order_relaxed);
    return d;
  }
  [[nodiscard]] auto max_k() const -> uint32_t { return max_k_; }
  [[nodiscard]] auto base_set_size() const -> size_t
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto r = base_set_.n_rows();
    cache_dim(base_set_);
    return r;
  }
  [[nodiscard]] auto query_set_size() const -> size_t
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto r = query_set_.n_rows();
    cache_dim(query_set_);
    return r;
  }

  [[nodiscard]] auto gt_set() const -> const IdxT*
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (ground_truth_set_.has_value()) { return ground_truth_set_->data(); }
    return nullptr;
  }

  [[nodiscard]] auto query_set() const -> const DataT*
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto* r = query_set_.data();
    cache_dim(query_set_);
    return r;
  }
  [[nodiscard]] auto query_set(MemoryType memory_type,
                               HugePages request_hugepages_2mb = HugePages::kDisable) const
    -> const DataT*
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto* r = query_set_.data(memory_type, request_hugepages_2mb);
    cache_dim(query_set_);
    return r;
  }

  [[nodiscard]] auto base_set() const -> const DataT*
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto* r = base_set_.data();
    cache_dim(base_set_);
    return r;
  }
  [[nodiscard]] auto base_set(MemoryType memory_type,
                              HugePages request_hugepages_2mb = HugePages::kDisable) const
    -> const DataT*
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto* r = base_set_.data(memory_type, request_hugepages_2mb);
    cache_dim(base_set_);
    return r;
  }

  [[nodiscard]] auto filter_bitset() const -> const bitset_carrier_type*
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (filter_bitset_.has_value()) { return filter_bitset_->data(); }
    return nullptr;
  }

  [[nodiscard]] auto filter_bitset(MemoryType memory_type,
                                   HugePages request_hugepages_2mb = HugePages::kDisable) const
    -> const bitset_carrier_type*
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (filter_bitset_.has_value()) {
      return filter_bitset_->data(memory_type, request_hugepages_2mb);
    }
    return nullptr;
  }

  [[nodiscard]] auto ground_truth_entry(IdxT query_idx) const -> const gt_entry&
  {
    std::call_once(gt_entries_init_flag_, [this]() {
      // we can't use resize here because the std::once_flag member of gt_entry is not copyable
      // (std::vector::resize requires copy constructible type)
      gt_entries_ = std::vector<gt_entry>(std::min(query_set_.n_rows(), kMaxQueriesForRecall));
    });
    // Lazily iterate over and optionally filter the ground truth set to build gt_maps for up to
    // kMaxQueriesForRecall queries
    /* NOTE: recall correctness & filtering

    We generate the filtered ground truth values and build unordered_maps with them to
    enable O(1) lookup. We need enough ground truth values to compute recall correctly
    though. But the ground truth file only contains `max_k_` values per row; if there are
    less valid values than k among them, we overestimate the recall. Essentially, we compare
    the first `filter_pass_count` values of the algorithm output, and this counter can be
    less than `k`. In the extreme case of very high filtering rate, we may be bypassing
    entire rows of results. However, this is still better than no recall estimate at all.

    */
    if (ground_truth_set_.has_value()) {
      auto filter = [this](IdxT i) -> bool {
        if (!this->filter_bitset_.has_value()) { return true; }
        auto word = this->filter_bitset_->data()[i >> 5];
        return word & (1 << (i & 31));
      };
      std::call_once(gt_entries_[query_idx].once_flag, [this, query_idx, &filter] {
        for (std::uint32_t neighbor_rank = 0; neighbor_rank < max_k_; ++neighbor_rank) {
          auto id = ground_truth_set_->data()[query_idx * max_k_ + neighbor_rank];
          if (!filter(id)) { continue; }
          if (gt_entries_[query_idx].gt_map.count(id)) {
            throw std::invalid_argument(
              "Duplicate neighbor id found in ground truth set for query " +
              std::to_string(query_idx));
          }
          gt_entries_[query_idx].gt_map[id] = neighbor_rank;
          ++gt_entries_[query_idx].filter_pass_count;
        }
      });
    }
    return gt_entries_[query_idx];
  }
};

}  // namespace  cuvs::bench
