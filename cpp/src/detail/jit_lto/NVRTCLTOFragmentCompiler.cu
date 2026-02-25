/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/detail/jit_lto/FragmentDatabase.hpp>
#include <cuvs/detail/jit_lto/NVRTCLTOFragmentCompiler.hpp>

#include <raft/core/error.hpp>

#include "cuda.h"
#include <nvrtc.h>

#define NVRTC_SAFE_CALL(_call)                                                 \
  {                                                                            \
    nvrtcResult result = _call;                                                \
    std::string error_string =                                                 \
      std::string("nvrtc error: ") + std::string(nvrtcGetErrorString(result)); \
    RAFT_EXPECTS(result == NVRTC_SUCCESS, "%s", error_string.c_str());         \
  }

NVRTCLTOFragmentCompiler::NVRTCLTOFragmentCompiler()
{
  int device = 0;
  int major  = 0;
  int minor  = 0;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);

  this->standard_compile_opts.resize(7);

  std::size_t i = 0;
  // Use actual GPU architecture for optimal code generation
  this->standard_compile_opts[i++] =
    std::string{"-arch=sm_" + std::to_string((major * 10 + minor))};
  this->standard_compile_opts[i++] = std::string{"-dlto"};
  this->standard_compile_opts[i++] = std::string{"-rdc=true"};
  this->standard_compile_opts[i++] = std::string{"-default-device"};
  this->standard_compile_opts[i++] = std::string{"--gen-opt-lto"};
  // Optimization flags - NVRTC uses different syntax than nvcc
  this->standard_compile_opts[i++] = std::string{"--use_fast_math"};
  this->standard_compile_opts[i++] = std::string{"--extra-device-vectorization"};
}

void NVRTCLTOFragmentCompiler::compile(std::string const& key, std::string const& code) const
{
  // Check if this fragment is already cached - avoid expensive NVRTC compilation
  if (fragment_database().has_fragment(key)) { return; }

  nvrtcProgram prog;
  NVRTC_SAFE_CALL(
    nvrtcCreateProgram(&prog, code.c_str(), "nvrtc_lto_fragment", 0, nullptr, nullptr));

  // Convert std::vector<std::string> to std::vector<const char*> for nvrtc API
  std::vector<const char*> opts;
  opts.reserve(this->standard_compile_opts.size());
  for (const auto& opt : this->standard_compile_opts) {
    opts.push_back(opt.c_str());
  }

  nvrtcResult compileResult = nvrtcCompileProgram(prog,          // prog
                                                  opts.size(),   // numOptions
                                                  opts.data());  // options

  if (compileResult != NVRTC_SUCCESS) {
    // Obtain compilation log from the program.
    size_t log_size;
    NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &log_size));
    std::unique_ptr<char[]> log{new char[log_size]};
    NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log.get()));
    RAFT_FAIL("nvrtc compile error log: \n%s", log.get());
  }

  // Obtain generated LTO IR from the program.
  std::size_t ltoIRSize;
  NVRTC_SAFE_CALL(nvrtcGetLTOIRSize(prog, &ltoIRSize));

  std::unique_ptr<char[]> program = std::make_unique<char[]>(ltoIRSize);
  nvrtcGetLTOIR(prog, program.get());

  NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));

  registerNVRTCFragment(key, std::move(program), ltoIRSize);
}

NVRTCLTOFragmentCompiler& nvrtc_compiler()
{
  static NVRTCLTOFragmentCompiler compiler;
  return compiler;
}
