/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "reduction_ownership.hpp"

#define CUVS_REDUCTION_INSTANTIATION(...) template __VA_ARGS__;
#include "reduction_instantiations.inl"
#undef CUVS_REDUCTION_INSTANTIATION
