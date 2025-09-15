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
package com.nvidia.cuvs;

import java.util.List;

public interface GPUInfoProvider {

  int MIN_COMPUTE_CAPABILITY_MAJOR = 7;
  int MIN_COMPUTE_CAPABILITY_MINOR = 0;

  int MIN_DEVICE_MEMORY_IN_MB = 8192;

  /**
   * Gets all the available GPUs
   *
   * @return a list of {@link GPUInfo} objects with GPU details
   */
  List<GPUInfo> availableGPUs();

  /**
   * Get the list of compatible GPUs based on compute capability >= 7.0 and total
   * memory >= 8GB
   *
   * @return a list of compatible GPUs. See {@link GPUInfo}
   */
  List<GPUInfo> compatibleGPUs();

  /**
   * Gets memory information relative to a {@link CuVSResources}
   * @param resources from which to obtain memory information
   * @return a {@link CuVSResourcesInfo} record containing the memory information
   */
  CuVSResourcesInfo getCurrentInfo(CuVSResources resources);
}
