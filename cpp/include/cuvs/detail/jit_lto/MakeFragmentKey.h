
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

#include <string>
#include <typeinfo>

namespace detail {
std::string nvrtc_name(std::type_info const& info);

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
  return result;
}
