/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

#include "ann_types.hpp"
#include "blob.hpp"

#include <cstdint>
#include <cstdio>
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

 private:
  std::string name_;
  std::string distance_;
  blob<DataT> base_set_;
  blob<DataT> query_set_;
  std::optional<blob<IdxT>> ground_truth_set_;
  std::optional<blob<bitset_carrier_type>> filter_bitset_;

  mutable bool base_set_accessed_  = false;
  mutable bool query_set_accessed_ = false;

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
                          : std::nullopt}
  {
    if (filtering_rate.has_value()) {
      // Generate a random bitset for filtering
      auto bitset_size = (base_set_size() - 1) / kBitsPerCarrierValue + 1;
      blob_file<bitset_carrier_type> bitset_blob_file{uint32_t(bitset_size), 1};
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
    // If any of base/query set are already accessed, use those
    if (base_set_accessed_) { return static_cast<int>(base_set_.n_cols()); }
    if (query_set_accessed_) { return static_cast<int>(query_set_.n_cols()); }
    // Otherwise, try reading both (one of the two sets may be missing)
    try {
      return static_cast<int>(query_set_.n_cols());
    } catch (const std::runtime_error& e) {
      // Any exception raised above will re-raise next time we try to access the query set.
      query_set_.reset_lazy_state();
    }
    return static_cast<int>(base_set_.n_cols());
  }
  [[nodiscard]] auto max_k() const -> uint32_t
  {
    if (ground_truth_set_.has_value()) { return ground_truth_set_->n_cols(); }
    return 0;
  }
  [[nodiscard]] auto base_set_size() const -> size_t
  {
    base_set_accessed_ = true;
    return base_set_.n_rows();
  }
  [[nodiscard]] auto query_set_size() const -> size_t
  {
    query_set_accessed_ = true;
    return query_set_.n_rows();
  }

  [[nodiscard]] auto gt_set() const -> const IdxT*
  {
    if (ground_truth_set_.has_value()) { return ground_truth_set_->data(); }
    return nullptr;
  }

  [[nodiscard]] auto query_set() const -> const DataT*
  {
    query_set_accessed_ = true;
    return query_set_.data();
  }
  [[nodiscard]] auto query_set(MemoryType memory_type,
                               HugePages request_hugepages_2mb = HugePages::kDisable) const
    -> const DataT*
  {
    query_set_accessed_ = true;
    return query_set_.data(memory_type, request_hugepages_2mb);
  }

  [[nodiscard]] auto base_set() const -> const DataT*
  {
    base_set_accessed_ = true;
    return base_set_.data();
  }
  [[nodiscard]] auto base_set(MemoryType memory_type,
                              HugePages request_hugepages_2mb = HugePages::kDisable) const
    -> const DataT*
  {
    base_set_accessed_ = true;
    return base_set_.data(memory_type, request_hugepages_2mb);
  }

  [[nodiscard]] auto filter_bitset() const -> const bitset_carrier_type*
  {
    if (filter_bitset_.has_value()) { return filter_bitset_->data(); }
    return nullptr;
  }

  [[nodiscard]] auto filter_bitset(MemoryType memory_type,
                                   HugePages request_hugepages_2mb = HugePages::kDisable) const
    -> const bitset_carrier_type*
  {
    if (filter_bitset_.has_value()) {
      return filter_bitset_->data(memory_type, request_hugepages_2mb);
    }
    return nullptr;
  }
};

}  // namespace  cuvs::bench
