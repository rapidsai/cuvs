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

struct _32_tag {};
struct _64_tag {};
struct _128_tag {};
struct _256_tag {};
struct _512_tag {};

template <unsigned N>
struct get_tag {};

template <>
struct get_tag<32> {
  using type = _32_tag;
};

template <>
struct get_tag<64> {
  using type = _64_tag;
};
template <>
struct get_tag<128> {
  using type = _128_tag;
};
template <>
struct get_tag<256> {
  using type = _256_tag;
};
template <>
struct get_tag<512> {
  using type = _512_tag;
};
