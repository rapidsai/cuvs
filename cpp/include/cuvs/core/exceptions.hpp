/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "c_api.h"

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
