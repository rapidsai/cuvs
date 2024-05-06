/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <raft/core/device_container_policy.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>
#include <thrust/functional.h>

namespace cuvs::core {
/**
 * @defgroup bitset Bitset
 * @{
 */
/**
 * @brief View of a cuVS Bitset.
 *
 * This lightweight structure stores a pointer to a bitset in device memory with it's length.
 * It provides a test() device function to check if a given index is set in the bitset.
 *
 * @tparam bitset_t Underlying type of the bitset array. Default is uint32_t.
 * @tparam index_t Indexing type used. Default is uint32_t.
 */
template <typename bitset_t = uint32_t, typename index_t = uint32_t>
struct bitset_view {
  static constexpr index_t bitset_element_size = sizeof(bitset_t) * 8;

  _RAFT_HOST_DEVICE bitset_view(bitset_t* bitset_ptr, index_t bitset_len);
  /**
   * @brief Create a bitset view from a device vector view of the bitset.
   *
   * @param bitset_span Device vector view of the bitset
   * @param bitset_len Number of bits in the bitset
   */
  _RAFT_HOST_DEVICE bitset_view(raft::device_vector_view<bitset_t, index_t> bitset_span,
                                index_t bitset_len);
  /**
   * @brief Device function to test if a given index is set in the bitset.
   *
   * @param sample_index Single index to test
   * @return bool True if index has not been unset in the bitset
   */
  _RAFT_HOST_DEVICE inline bool test(const index_t sample_index) const;

  /**
   * @brief Device function to test if a given index is set in the bitset.
   *
   * @param sample_index Single index to test
   * @return bool True if index has not been unset in the bitset
   */
  _RAFT_HOST_DEVICE bool operator[](const index_t sample_index) const;

  /**
   * @brief Device function to set a given index to set_value in the bitset.
   *
   * @param sample_index index to set
   * @param set_value Value to set the bit to (true or false)
   */
  _RAFT_DEVICE void set(const index_t sample_index, bool set_value) const;

  /**
   * @brief Get the device pointer to the bitset.
   */
  _RAFT_HOST_DEVICE bitset_t* data();
  _RAFT_HOST_DEVICE const bitset_t* data() const;
  /**
   * @brief Get the number of bits of the bitset representation.
   */
  _RAFT_HOST_DEVICE index_t size() const;

  /**
   * @brief Get the number of elements used by the bitset representation.
   */
  _RAFT_HOST_DEVICE index_t n_elements() const;

  raft::device_vector_view<bitset_t, index_t> to_mdspan();
  raft::device_vector_view<const bitset_t, index_t> to_mdspan() const;

 private:
  bitset_t* bitset_ptr_;
  index_t bitset_len_;
};

/**
 * @brief cuVS Bitset.
 *
 * This structure encapsulates a bitset in device memory. It provides a view() method to get a
 * device-usable lightweight view of the bitset.
 * Each index is represented by a single bit in the bitset. The total number of bytes used is
 * ceil(bitset_len / 8).
 * @tparam bitset_t Underlying type of the bitset array. Default is uint32_t.
 * @tparam index_t Indexing type used. Default is uint32_t.
 */
template <typename bitset_t = uint32_t, typename index_t = uint32_t>
struct bitset {
  static constexpr index_t bitset_element_size = sizeof(bitset_t) * 8;

  /**
   * @brief Construct a new bitset object with a list of indices to unset.
   *
   * @param res RAFT resources
   * @param mask_index List of indices to unset in the bitset
   * @param bitset_len Length of the bitset
   * @param default_value Default value to set the bits to. Default is true.
   */
  bitset(const raft::resources& res,
         raft::device_vector_view<const index_t, index_t> mask_index,
         index_t bitset_len,
         bool default_value = true);

  /**
   * @brief Construct a new bitset object
   *
   * @param res RAFT resources
   * @param bitset_len Length of the bitset
   * @param default_value Default value to set the bits to. Default is true.
   */
  bitset(const raft::resources& res, index_t bitset_len, bool default_value = true);
  // Disable copy constructor
  bitset(const bitset&)            = delete;
  bitset(bitset&&)                 = default;
  bitset& operator=(const bitset&) = delete;
  bitset& operator=(bitset&&)      = default;

  /**
   * @brief Create a device-usable view of the bitset.
   *
   * @return bitset_view<bitset_t, index_t>
   */
  cuvs::core::bitset_view<bitset_t, index_t> view();
  cuvs::core::bitset_view<const bitset_t, index_t> view() const;

  /**
   * @brief Get the device pointer to the bitset.
   */
  bitset_t* data();
  const bitset_t* data() const;
  /**
   * @brief Get the number of bits of the bitset representation.
   */
  index_t size() const;

  /**
   * @brief Get the number of elements used by the bitset representation.
   */
  index_t n_elements() const;

  /** @brief Get an mdspan view of the current bitset */
  raft::device_vector_view<bitset_t, index_t> to_mdspan();
  raft::device_vector_view<const bitset_t, index_t> to_mdspan() const;

  /** @brief Resize the bitset. If the requested size is larger, new memory is allocated and set to
   * the default value.
   * @param res RAFT resources
   * @param new_bitset_len new size of the bitset
   * @param default_value default value to initialize the new bits to
   */
  void resize(const raft::resources& res, index_t new_bitset_len, bool default_value = true);

  /**
   * @brief Test a list of indices in a bitset.
   *
   * @tparam output_t Output type of the test. Default is bool.
   * @param res RAFT resources
   * @param queries List of indices to test
   * @param output List of outputs
   */
  template <typename output_t = bool>
  void test(const raft::resources& res,
            raft::device_vector_view<const index_t, index_t> queries,
            raft::device_vector_view<output_t, index_t> output) const;
  /**
   * @brief Set a list of indices in a bitset to set_value.
   *
   * @param res RAFT resources
   * @param mask_index indices to remove from the bitset
   * @param set_value Value to set the bits to (true or false)
   */
  void set(const raft::resources& res,
           raft::device_vector_view<const index_t, index_t> mask_index,
           bool set_value = false);
  /**
   * @brief Flip all the bits in a bitset.
   * @param res RAFT resources
   */
  void flip(const raft::resources& res);
  /**
   * @brief Reset the bits in a bitset.
   *
   * @param res RAFT resources
   * @param default_value Value to set the bits to (true or false)
   */
  void reset(const raft::resources& res, bool default_value = true);
  /**
   * @brief Returns the number of bits set to true in count_gpu_scalar.
   *
   * @param[in] res RAFT resources
   * @param[out] count_gpu_scalar Device scalar to store the count
   */
  void count(const raft::resources& res, raft::device_scalar_view<index_t> count_gpu_scalar);

  /**
   * @brief Returns the number of bits set to true.
   *
   * @param res RAFT resources
   * @return index_t Number of bits set to true
   */
  index_t count(const raft::resources& res);

  /**
   * @brief Checks if any of the bits are set to true in the bitset.
   * @param res RAFT resources
   */
  bool any(const raft::resources& res) { return count(res) > 0; }
  /**
   * @brief Checks if all of the bits are set to true in the bitset.
   * @param res RAFT resources
   */
  bool all(const raft::resources& res) { return count(res) == bitset_len_; }
  /**
   * @brief Checks if none of the bits are set to true in the bitset.
   * @param res RAFT resources
   */
  bool none(const raft::resources& res) { return count(res) == 0; }

 private:
  raft::device_uvector<bitset_t> bitset_;
  index_t bitset_len_;
};

/** @} */
}  // end namespace cuvs::core
