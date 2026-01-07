/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// Basic smoke test to verify non of the C API headers
// ever use a C++ only construct
#include <cuvs/core/all.h>
#include <cuvs/core/all.h> //smoke out missing include guards

int main()
{
  return 0;
}
