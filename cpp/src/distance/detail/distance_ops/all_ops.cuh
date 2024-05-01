/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

// Defines a named requirement "has_cutlass_op"
#include "../distance_ops/cutlass.cuh"

// The distance operations:
#include "../distance_ops/canberra.cuh"
#include "../distance_ops/correlation.cuh"
#include "../distance_ops/cosine.cuh"
#include "../distance_ops/hamming.cuh"
#include "../distance_ops/hellinger.cuh"
#include "../distance_ops/jensen_shannon.cuh"
#include "../distance_ops/kl_divergence.cuh"
#include "../distance_ops/l1.cuh"
#include "../distance_ops/l2_exp.cuh"
#include "../distance_ops/l2_unexp.cuh"
#include "../distance_ops/l_inf.cuh"
#include "../distance_ops/lp_unexp.cuh"
#include "../distance_ops/russel_rao.cuh"
