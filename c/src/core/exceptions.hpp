/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/c_api.h>

#include <exception>

namespace cuvs::core {

/**
 * @brief Translates C++ exceptions into cuvs C-API error codes
 */
template <typename Fn>
cuvsError_t translate_exceptions(Fn func)
{
  cuvsError_t status;
  try {
    func();
    status = CUVS_SUCCESS;
    cuvsSetLastErrorText(NULL);
  } catch (const std::exception& e) {
    cuvsSetLastErrorText(e.what());
    status = CUVS_ERROR;
  } catch (...) {
    cuvsSetLastErrorText("unknown exception");
    status = CUVS_ERROR;
  }
  return status;
}
}  // namespace cuvs::core
