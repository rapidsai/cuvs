/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "MakeFragmentKey.hpp"

void registerFatbinFragment(std::string const& algo,
                            std::string const& params,
                            unsigned char const* blob,
                            std::size_t size);

namespace {

template <typename... Ts>
void registerAlgorithm(std::string algo, unsigned char const* blob, std::size_t size)
{
  auto key = make_fragment_key<Ts...>();
  registerFatbinFragment(algo, key, blob, size);
}

}  // namespace
