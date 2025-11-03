/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
