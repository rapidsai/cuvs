/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "select_k_int32.hpp"
#include "select_k.cuh"
instantiate_cuvs_selection_select_k(int32_t, uint32_t);
