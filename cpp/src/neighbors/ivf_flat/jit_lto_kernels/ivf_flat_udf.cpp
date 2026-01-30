/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/detail/jit_lto/NVRTCLTOFragmentCompiler.cuh>
#include <cuvs/neighbors/ivf_flat.hpp>

namespace cuvs::neighbors::ivf_flat::udf {

void compile_metric(std::string const& code)
{
  NVRTCLTOFragmentCompiler compiler;
  compiler.compile("IVF_FLAT_SEARCH_METRIC_UDF", code);
}

}  // namespace cuvs::neighbors::ivf_flat::udf
