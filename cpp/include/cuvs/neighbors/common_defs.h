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

/**
 * @file common_defs.h
 * @brief Minimal shared definitions for C and C++ interoperability.
 *
 * This header defines lightweight, language-neutral types (e.g., enums, structs)
 * that are used in both C and C++ code to ensure consistency and ABI compatibility.
 *
 * Design goals:
 * - Prevent duplicated definitions between C and C++.
 * - Avoid pulling in unnecessary dependencies.
 * - Safe for inclusion in both C and C++ environments.
 * - Suitable for use in public C API headers and internal C++ code.
 *
 * Only use this header for simple shared types such as:
 * - Enums (e.g., strategy selection, status codes)
 * - Plain structs (POD-style, no constructors or methods)
 *
 * Do NOT:
 * - Include other headers.
 * - Define functions or C++-specific types.
 * - Use templates, classes, or namespaces.
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Strategy for merging CAGRA indices.
 */
typedef enum {
  PHYSICAL = 0,  ///< Merge indices physically
  LOGICAL  = 1   ///< Merge indices logically (if supported)
} cuvsCagraMergeStrategy;

#ifdef __cplusplus
}
#endif
