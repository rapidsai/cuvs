/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <cuvs/detail/jit_lto/AlgorithmLauncher.h>

#include <iostream>

AlgorithmLauncher::AlgorithmLauncher(CUlibrary l, CUkernel k) : library{l}, kernel{k}
{
  // Validate that we have a valid kernel
  if (kernel == nullptr) {
    std::cerr << "ERROR: AlgorithmLauncher constructed with null kernel" << std::endl;
  }
  if (library == nullptr) {
    std::cerr << "ERROR: AlgorithmLauncher constructed with null library" << std::endl;
  }
  std::cout << "AlgorithmLauncher constructed with kernel: " << kernel << ", library: " << library
            << std::endl;
}

void AlgorithmLauncher::call(
  cudaStream_t stream, dim3 grid, dim3 block, std::size_t shared_mem, void** kernel_args)
{
  std::cout << "In the launcher" << std::endl;

  // Validate inputs
  if (kernel == nullptr) {
    std::cerr << "ERROR: Cannot launch null kernel" << std::endl;
    return;
  }

  if (grid.x == 0 || grid.y == 0 || grid.z == 0) {
    std::cerr << "ERROR: Invalid grid dimensions: " << grid.x << "x" << grid.y << "x" << grid.z
              << std::endl;
    return;
  }

  if (block.x == 0 || block.y == 0 || block.z == 0) {
    std::cerr << "ERROR: Invalid block dimensions: " << block.x << "x" << block.y << "x" << block.z
              << std::endl;
    return;
  }

  std::cout << "Grid: " << grid.x << "x" << grid.y << "x" << grid.z << ", Block: " << block.x << "x"
            << block.y << "x" << block.z << ", Shared mem: " << shared_mem << std::endl;

  // Debug kernel arguments
  if (kernel_args != nullptr) {
    std::cout << "Kernel arguments pointer: " << kernel_args << std::endl;
    // Note: We can't safely dereference kernel_args without knowing the types,
    // but we can at least check if the pointer is valid
  } else {
    std::cout << "WARNING: kernel_args is null" << std::endl;
  }
  CUlaunchAttribute attribute[1];
  attribute[0].id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
  attribute[0].value.programmaticStreamSerializationAllowed = 1;

  CUlaunchConfig config{};
  config.gridDimX       = grid.x;
  config.gridDimY       = grid.y;
  config.gridDimZ       = grid.z;
  config.blockDimX      = block.x;
  config.blockDimY      = block.y;
  config.blockDimZ      = block.z;
  config.sharedMemBytes = shared_mem;
  config.hStream        = stream;
  config.attrs          = attribute;
  config.numAttrs       = 1;

  std::cout << "Launching kernel" << std::endl;

  // Check CUDA context
  CUcontext ctx;
  CUresult ctx_result = cuCtxGetCurrent(&ctx);
  if (ctx_result != CUDA_SUCCESS) {
    std::cerr << "ERROR: No active CUDA context. Error: " << ctx_result << std::endl;
    return;
  }

  // Check stream validity
  if (stream == nullptr) {
    std::cerr << "ERROR: Stream is null" << std::endl;
    return;
  }
  std::cout << "Stream: " << stream << std::endl;

  // Check device properties for debugging
  int device;
  cudaGetDevice(&device);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  std::cout << "Device: " << device << " (" << prop.name << ")" << std::endl;
  std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
  std::cout << "Max shared memory per block: " << prop.sharedMemPerBlock << " bytes" << std::endl;

  // Check if our launch parameters are within limits
  int total_threads = block.x * block.y * block.z;
  if (total_threads > prop.maxThreadsPerBlock) {
    std::cerr << "ERROR: Block size exceeds max threads per block (" << total_threads << " > "
              << prop.maxThreadsPerBlock << ")" << std::endl;
    return;
  }

  if (shared_mem > prop.sharedMemPerBlock) {
    std::cerr << "ERROR: Shared memory exceeds max per block (" << shared_mem << " > "
              << prop.sharedMemPerBlock << ")" << std::endl;
    return;
  }

  // Launch kernel and check for errors
  std::cout << "About to launch kernel with cuLaunchKernelEx..." << std::endl;
  CUresult launch_result = cuLaunchKernelEx(&config, (CUfunction)kernel, kernel_args, 0);
  if (launch_result != CUDA_SUCCESS) {
    std::cerr << "ERROR: Kernel launch failed with error: " << launch_result << std::endl;
    std::cerr << "This suggests the kernel function is invalid or there's a parameter issue"
              << std::endl;
    return;
  }
  std::cout << "cuLaunchKernelEx returned successfully" << std::endl;

  std::cout << "Kernel launched successfully, synchronizing stream..." << std::endl;

  // Check for CUDA runtime errors before synchronization
  cudaError_t cuda_err = cudaGetLastError();
  if (cuda_err != cudaSuccess) {
    std::cerr << "ERROR: CUDA error before sync: " << cudaGetErrorString(cuda_err) << std::endl;
    return;
  }

  // Add timeout mechanism for debugging
  std::cout << "Starting stream synchronization (this may hang if kernel is stuck)..." << std::endl;

  // Try to get stream status first
  cudaStreamQuery(stream);
  cuda_err = cudaGetLastError();
  if (cuda_err != cudaSuccess && cuda_err != cudaErrorNotReady) {
    std::cerr << "ERROR: Stream query failed: " << cudaGetErrorString(cuda_err) << std::endl;
    return;
  }

  std::cout << "Stream query completed, proceeding with synchronization..." << std::endl;

  // Let's try a different approach - check if the kernel is actually running
  std::cout << "About to call cudaStreamSynchronize - this is where it hangs..." << std::endl;

  // First, let's try to see if we can get any information about the kernel
  std::cout << "Checking kernel function pointer: " << kernel << std::endl;

  // Try to get kernel attributes
  int max_threads = 0;
  CUresult attr_result =
    cuFuncGetAttribute(&max_threads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, (CUfunction)kernel);
  if (attr_result == CUDA_SUCCESS) {
    std::cout << "Kernel function appears to be valid, max threads per block: " << max_threads
              << std::endl;
  } else {
    std::cerr << "WARNING: Could not get kernel attributes, error: " << attr_result << std::endl;
    std::cerr << "This suggests the kernel function might be invalid!" << std::endl;
  }

  // Try to get kernel name
  const char* kernel_name = nullptr;
  CUresult name_result    = cuFuncGetName(&kernel_name, (CUfunction)kernel);
  if (name_result == CUDA_SUCCESS && kernel_name != nullptr) {
    std::cout << "Kernel name: " << kernel_name << std::endl;
  } else {
    std::cerr << "WARNING: Could not get kernel name, error: " << name_result << std::endl;
  }

  // Now try the synchronization - this is where it hangs
  std::cout << "Calling cudaStreamSynchronize now..." << std::endl;

  // Try using CUDA Driver API instead of runtime API
  CUstream cu_stream   = (CUstream)stream;
  CUresult sync_result = cuStreamSynchronize(cu_stream);
  if (sync_result != CUDA_SUCCESS) {
    std::cerr << "ERROR: cuStreamSynchronize failed with error: " << sync_result << std::endl;
    return;
  }
  std::cout << "cuStreamSynchronize returned successfully!" << std::endl;

  // Check for errors after synchronization
  cuda_err = cudaGetLastError();
  if (cuda_err != cudaSuccess) {
    std::cerr << "ERROR: CUDA error after sync: " << cudaGetErrorString(cuda_err) << std::endl;
    return;
  }

  std::cout << "Launched kernel" << std::endl;
}

std::unordered_map<std::string, AlgorithmLauncher>& get_cached_launchers()
{
  static std::unordered_map<std::string, AlgorithmLauncher> launchers;
  return launchers;
}
