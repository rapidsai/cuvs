/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <nvJitLink.h>

#include "nvjitlink_checker.hpp"

struct FragmentEntry {
  virtual bool add_to(nvJitLinkHandle& handle) const = 0;

  virtual const char* get_key() const = 0;
};

struct FatbinFragmentEntry : FragmentEntry {
  virtual const uint8_t* get_data() const = 0;

  virtual size_t get_length() const = 0;

  bool add_to(nvJitLinkHandle& handle) const override final;
};

template <typename FragmentT>
struct StaticFatbinFragmentEntry : FatbinFragmentEntry {
  const uint8_t* get_data() const override final { return FragmentT::data; }

  size_t get_length() const override final { return FragmentT::length; }

  const char* get_key() const override final { return typeid(FragmentT).name(); }
};
