/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "selection_ownership.hpp"

#define CUVS_SELECTION_INSTANTIATION(...) template __VA_ARGS__;
#include "selection_instantiations.inl"
#undef CUVS_SELECTION_INSTANTIATION
