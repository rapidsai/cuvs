/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuvs/core/cuda_fp16.hpp>
#include <cuvs/core/export.hpp>

#include <raft/core/detail/mdspan_numpy_serializer.hpp>
#include <raft/core/error.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace CUVS_EXPORT cuvs {
namespace util {

template <typename T>
inline auto numpy_dtype_string() -> std::string
{
  if constexpr (std::is_same_v<T, half>) {
    return "<f2";
  } else {
    return raft::detail::numpy_serializer::get_numpy_dtype<T>().to_string();
  }
}

inline auto make_numpy_header_from_dtype(const std::string& dtype, const std::vector<size_t>& shape)
  -> std::string
{
  std::stringstream dict;
  dict << "{'descr': '" << dtype << "', 'fortran_order': False, 'shape': (";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) { dict << ", "; }
    dict << shape[i];
  }
  if (shape.size() == 1) { dict << ","; }
  dict << "), }";

  std::string header = dict.str();
  constexpr size_t preamble_size =
    6 + 2 + sizeof(uint16_t);  // magic string, version, v1 header length
  const size_t remainder = (preamble_size + header.size() + 1) % 16;
  if (remainder != 0) { header.append(16 - remainder, ' '); }
  header.push_back('\n');

  RAFT_EXPECTS(header.size() <= std::numeric_limits<uint16_t>::max(),
               "NumPy v1 header is too large: %zu bytes",
               header.size());

  const auto header_len = static_cast<uint16_t>(header.size());
  std::string result;
  result.reserve(preamble_size + header.size());
  result.append("\x93NUMPY", 6);
  result.push_back(1);
  result.push_back(0);
  result.push_back(static_cast<char>(header_len & 0xff));
  result.push_back(static_cast<char>((header_len >> 8) & 0xff));
  result.append(header);
  return result;
}

template <typename T>
inline auto make_numpy_header_string(const std::vector<size_t>& shape) -> std::string
{
  if constexpr (std::is_same_v<T, half>) {
    return make_numpy_header_from_dtype(numpy_dtype_string<T>(), shape);
  } else {
    const auto dtype         = raft::detail::numpy_serializer::get_numpy_dtype<T>();
    const bool fortran_order = false;
    const raft::detail::numpy_serializer::header_t header = {dtype, fortran_order, shape};

    std::stringstream ss;
    raft::detail::numpy_serializer::write_header(ss, header);
    return ss.str();
  }
}

}  // namespace util
}  // namespace CUVS_EXPORT cuvs
