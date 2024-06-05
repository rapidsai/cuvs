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

#include <raft/common/nvtx.hpp>

namespace cuvs::common::nvtx {
namespace domain {
/** @brief The default NVTX domain. */
struct app {
  static constexpr char const* name{"application"};
};

/** @brief This NVTX domain is supposed to be used within cuvs.  */
struct cuvs {
  static constexpr const char* name = "cuvs";
};
}  // namespace domain

/**
 * @brief Push a named NVTX range.
 *
 * @tparam Domain optional struct that defines the NVTX domain message;
 *   You can create a new domain with a custom message as follows:
 *   \code{.cpp}
 *      struct custom_domain { static constexpr char const* name{"custom message"}; }
 *   \endcode
 *   NB: make sure to use the same domain for `push_range` and `pop_range`.
 * @param format range name format (accepts printf-style arguments)
 * @param args the arguments for the printf-style formatting
 */
template <typename Domain = domain::app, typename... Args>
inline void push_range(const char* format, Args... args)
{
  raft::common::nvtx::detail::push_range<Domain, Args...>(format, args...);
}

/**
 * @brief Pop the latest range.
 *
 * @tparam Domain optional struct that defines the NVTX domain message;
 *   You can create a new domain with a custom message as follows:
 *   \code{.cpp}
 *      struct custom_domain { static constexpr char const* name{"custom message"}; }
 *   \endcode
 *   NB: make sure to use the same domain for `push_range` and `pop_range`.
 */
template <typename Domain = domain::app>
inline void pop_range()
{
  raft::common::nvtx::detail::pop_range<Domain>();
}

/**
 * @brief Push a named NVTX range that would be popped at the end of the object lifetime.
 *
 * Refer to \ref Usage for the usage examples.
 *
 * @tparam Domain optional struct that defines the NVTX domain message;
 *   You can create a new domain with a custom message as follows:
 *   \code{.cpp}
 *      struct custom_domain { static constexpr char const* name{"custom message"}; }
 *   \endcode
 */
template <typename Domain = domain::app>
class range {
 public:
  /**
   * Push a named NVTX range.
   * At the end of the object lifetime, pop the range back.
   *
   * @param format range name format (accepts printf-style arguments)
   * @param args the arguments for the printf-style formatting
   */
  template <typename... Args>
  explicit range(const char* format, Args... args)
  {
    push_range<Domain, Args...>(format, args...);
  }

  ~range() { pop_range<Domain>(); }

  /* This object is not meant to be touched. */
  range(const range&)                              = delete;
  range(range&&)                                   = delete;
  auto operator=(const range&) -> range&           = delete;
  auto operator=(range&&) -> range&                = delete;
  static auto operator new(std::size_t) -> void*   = delete;
  static auto operator new[](std::size_t) -> void* = delete;
};
};  // namespace cuvs::common::nvtx
