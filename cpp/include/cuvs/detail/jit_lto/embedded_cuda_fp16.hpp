/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace cuvs::detail::jit_lto {

/** Toolkit `cuda_fp16.h` embedded at libcuvs build time; null-terminated. */
extern char const k_nvrtc_embedded_cuda_fp16_h[];

}  // namespace cuvs::detail::jit_lto
