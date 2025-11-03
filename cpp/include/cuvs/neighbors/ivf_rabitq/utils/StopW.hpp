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

#include <chrono>

class StopW {
  std::chrono::steady_clock::time_point time_begin;

 public:
  StopW() { time_begin = std::chrono::steady_clock::now(); }

  float getElapsedTimeSec()
  {
    std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
    return (std::chrono::duration_cast<std::chrono::seconds>(time_end - time_begin).count());
  }

  float getElapsedTimeMili()
  {
    std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
    return (std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_begin).count());
  }

  float getElapsedTimeMicro()
  {
    std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
    return (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count());
  }

  float getElapsedTimeNano()
  {
    std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
    return (std::chrono::duration_cast<std::chrono::nanoseconds>(time_end - time_begin).count());
  }

  void reset() { time_begin = std::chrono::steady_clock::now(); }
};
