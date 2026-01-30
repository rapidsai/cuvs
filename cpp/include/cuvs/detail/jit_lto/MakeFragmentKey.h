/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>
#include <typeinfo>

namespace detail {

template <typename T>
std::string type_as_string()
{
  if constexpr (std::is_reference_v<T>) {
    return std::string(typeid(T).name()) + "&";
  } else {
    return std::string(typeid(T).name());
  }
}
}  // namespace detail

template <typename... Ts>
std::string make_fragment_key()
{
  std::string result;
  ((result += detail::type_as_string<Ts>() + "_"), ...);
  if (!result.empty()) { result.pop_back(); }
  return result;
}
