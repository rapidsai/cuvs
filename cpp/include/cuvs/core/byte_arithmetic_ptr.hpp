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

#include <cstdint>

namespace cuvs::detail {

struct byte_arithmetic_ptr {
  void* data     = nullptr;
  bool is_signed = false;

  __host__ __device__ byte_arithmetic_ptr(void* ptr, bool signed_flag)
    : data(ptr), is_signed(signed_flag)
  {
  }

  // Proxy that references an element in the array
  struct byte {
    byte_arithmetic_ptr* parent = nullptr;
    int64_t idx                 = -1;
    uint8_t value               = 0;  // used for detached proxies

    // Constructor for live proxy
    __host__ __device__ byte(byte_arithmetic_ptr& p, int64_t i) : parent(&p), idx(i) {}

    // Copy constructor: detached copy stores the current value
    __host__ __device__ byte(const byte& other)
      : parent(nullptr), idx(-1), value(static_cast<uint8_t>(other))
    {
    }

    // Copy assignment: detached copy stores value
    __host__ __device__ byte& operator=(const byte& other)
    {
      parent = nullptr;
      idx    = -1;
      value  = static_cast<uint8_t>(other);
      return *this;
    }

    // Deleted move operations
    __host__ __device__ byte(byte&& other)            = delete;
    __host__ __device__ byte& operator=(byte&& other) = delete;

    // Conversion to uint8_t
    __host__ __device__ operator uint8_t() const
    {
      if (parent) {
        if (parent->is_signed) {
          int8_t val = reinterpret_cast<int8_t*>(parent->data)[idx];
          return static_cast<uint8_t>(static_cast<int16_t>(val) + 128);
        } else {
          return reinterpret_cast<uint8_t*>(parent->data)[idx];
        }
      } else {
        return value;  // return local value if detached
      }
    }

    // Assignment from uint8_t
    __host__ __device__ byte& operator=(uint8_t normalized_value)
    {
      if (parent) {
        if (parent->is_signed) {
          reinterpret_cast<int8_t*>(parent->data)[idx] =
            static_cast<int8_t>(static_cast<int16_t>(normalized_value) - 128);
        } else {
          reinterpret_cast<uint8_t*>(parent->data)[idx] = normalized_value;
        }
      } else {
        value = normalized_value;  // store in local value if detached
      }
      return *this;
    }
  };

  // Non-const index access: returns live proxy
  __host__ __device__ byte operator[](int64_t idx) { return byte(*this, idx); }

  // Const index access: returns immediate value
  __host__ __device__ uint8_t operator[](int64_t idx) const
  {
    if (is_signed) {
      int8_t val = reinterpret_cast<int8_t*>(data)[idx];
      return static_cast<uint8_t>(static_cast<int16_t>(val) + 128);
    } else {
      return reinterpret_cast<uint8_t*>(data)[idx];
    }
  }

  // Dereference (like *ptr)
  __host__ __device__ uint8_t operator*() const { return (*this)[0]; }
  __host__ __device__ byte operator*() { return byte(*this, 0); }

  // Pointer arithmetic
  __host__ __device__ byte_arithmetic_ptr operator+(int64_t offset) const
  {
    if (is_signed)
      return byte_arithmetic_ptr(static_cast<int8_t*>(data) + offset, true);
    else
      return byte_arithmetic_ptr(static_cast<uint8_t*>(data) + offset, false);
  }

  __host__ __device__ bool operator==(const byte_arithmetic_ptr& other) const
  {
    return data == other.data;
  }
  __host__ __device__ bool operator!=(const byte_arithmetic_ptr& other) const
  {
    return !(*this == other);
  }
};

}  // namespace cuvs::detail
