/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/util/cuda_utils.cuh>
#include <raft/util/integer_utils.hpp>
#include <raft/util/pow2_utils.cuh>

namespace cuvs::preprocessing::quantize::detail {

/**
 * Bitfield reference for reading from const pointers.
 * This type mimics `uint8_t` or `uint16_t` for reading bitfield values.
 */
struct bitfield_const_ref_t {
  const uint8_t* ptr;
  uint32_t offset;
  uint32_t bits;

  __host__ __device__ operator uint8_t() const  // NOLINT
  {
    const uint8_t mask = static_cast<uint8_t>((1u << bits) - 1u);
    uint32_t pair      = static_cast<uint32_t>(ptr[0]);
    if (offset + bits > 8) { pair |= static_cast<uint32_t>(ptr[1]) << 8; }
    return static_cast<uint8_t>((pair >> offset) & mask);
  }

  __host__ __device__ operator uint16_t() const  // NOLINT
  {
    const uint16_t mask = static_cast<uint16_t>((1u << bits) - 1u);
    uint32_t pair       = static_cast<uint32_t>(ptr[0]);
    if (offset + bits > 8) { pair |= static_cast<uint32_t>(ptr[1]) << 8; }
    if (offset + bits > 16) { pair |= static_cast<uint32_t>(ptr[2]) << 16; }
    return static_cast<uint16_t>((pair >> offset) & mask);
  }
};

/**
 * Bitfield reference for reading/writing from non-const pointers.
 * This type mimics `uint8_t&` or `uint16_t&` for the indexing operator.
 */
struct bitfield_ref_t {
  uint8_t* ptr;
  uint32_t offset;
  uint32_t bits;

  __host__ __device__ operator uint8_t() const  // NOLINT
  {
    const uint8_t mask = static_cast<uint8_t>((1u << bits) - 1u);
    uint32_t pair      = static_cast<uint32_t>(ptr[0]);
    if (offset + bits > 8) { pair |= static_cast<uint32_t>(ptr[1]) << 8; }
    return static_cast<uint8_t>((pair >> offset) & mask);
  }

  __host__ __device__ operator uint16_t() const  // NOLINT
  {
    const uint16_t mask = static_cast<uint16_t>((1u << bits) - 1u);
    uint32_t pair       = static_cast<uint32_t>(ptr[0]);
    if (offset + bits > 8) { pair |= static_cast<uint32_t>(ptr[1]) << 8; }
    if (offset + bits > 16) { pair |= static_cast<uint32_t>(ptr[2]) << 16; }
    return static_cast<uint16_t>((pair >> offset) & mask);
  }

  __host__ __device__ auto operator=(uint8_t code) -> bitfield_ref_t&
  {
    const uint8_t mask = static_cast<uint8_t>((1u << bits) - 1u);

    if (offset + bits > 8) {
      auto pair = static_cast<uint16_t>(ptr[0]);
      pair |= static_cast<uint16_t>(ptr[1]) << 8;
      pair &= ~(static_cast<uint16_t>(mask) << offset);
      pair |= static_cast<uint16_t>(code) << offset;
      ptr[0] = static_cast<uint8_t>(pair & 0xFF);
      ptr[1] = static_cast<uint8_t>((pair >> 8) & 0xFF);
    } else {
      ptr[0] = (ptr[0] & ~(mask << offset)) | (code << offset);
    }
    return *this;
  }

  __host__ __device__ auto operator=(uint16_t code) -> bitfield_ref_t&
  {
    const uint16_t mask = static_cast<uint16_t>((1u << bits) - 1u);

    // General case for multi-byte operations
    uint32_t pair = static_cast<uint32_t>(ptr[0]);
    pair |= static_cast<uint32_t>(ptr[1]) << 8;
    if (offset + bits > 16) { pair |= static_cast<uint32_t>(ptr[2]) << 16; }

    pair &= ~(static_cast<uint32_t>(mask) << offset);
    pair |= static_cast<uint32_t>(code) << offset;

    ptr[0] = static_cast<uint8_t>(pair & 0xFF);
    ptr[1] = static_cast<uint8_t>((pair >> 8) & 0xFF);
    if (offset + bits > 16) { ptr[2] = static_cast<uint8_t>((pair >> 16) & 0xFF); }

    return *this;
  }
};

/**
 * Bitfield view for const pointers.
 * View a byte array as an array of unsigned integers of custom small bit size.
 */
struct bitfield_const_view_t {
  const uint8_t* raw;
  uint32_t bits;

  __host__ __device__ auto operator[](uint32_t i) const -> bitfield_const_ref_t
  {
    uint32_t bit_offset = i * bits;
    return bitfield_const_ref_t{
      raw + raft::Pow2<8>::div(bit_offset), raft::Pow2<8>::mod(bit_offset), bits};
  }
};

/**
 * Bitfield view for non-const pointers.
 * View a byte array as an array of unsigned integers of custom small bit size.
 * Supports both reading and writing individual bit-packed values.
 *
 * Example usage:
 * ```
 * uint8_t data[10] = {0};
 * bitfield_view_t view{data, 6};  // 6-bit values
 * view[0] = 15;  // Write 15 to first 6-bit slot
 * uint16_t val = view[0];  // Read back the value
 * ```
 */
struct bitfield_view_t {
  uint8_t* raw;
  uint32_t bits;

  __host__ __device__ auto operator[](uint32_t i) -> bitfield_ref_t
  {
    uint32_t bit_offset = i * bits;
    return bitfield_ref_t{
      raw + raft::Pow2<8>::div(bit_offset), raft::Pow2<8>::mod(bit_offset), bits};
  }

  __host__ __device__ auto operator[](uint32_t i) const -> bitfield_const_ref_t
  {
    uint32_t bit_offset = i * bits;
    return bitfield_const_ref_t{
      raw + raft::Pow2<8>::div(bit_offset), raft::Pow2<8>::mod(bit_offset), bits};
  }
};
}  // namespace cuvs::preprocessing::quantize::detail
