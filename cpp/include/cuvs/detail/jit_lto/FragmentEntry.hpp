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

template <typename FragmentT>
struct FatbinFragmentEntry : FragmentEntry {
  bool add_to(nvJitLinkHandle& handle) const override final
  {
    auto result = nvJitLinkAddData(
      handle, NVJITLINK_INPUT_ANY, FragmentT::data, FragmentT::length, typeid(FragmentT).name());

    check_nvjitlink_result(handle, result);
    return true;
  }

  const char* get_key() const override final { return typeid(FragmentT).name(); }
};
