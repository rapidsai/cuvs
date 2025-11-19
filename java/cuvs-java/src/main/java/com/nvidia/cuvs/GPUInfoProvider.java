/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
