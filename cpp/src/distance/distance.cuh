/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#ifdef CUVS_UNITY_KERNEL_OWNER
#include "distance-inl.cuh"
#else
#include "distance-ext.cuh"
#endif
