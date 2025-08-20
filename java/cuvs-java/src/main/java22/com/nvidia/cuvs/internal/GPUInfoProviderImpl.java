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
package com.nvidia.cuvs.internal;

import static com.nvidia.cuvs.internal.common.Util.checkCuVSError;
import static com.nvidia.cuvs.internal.common.Util.checkCudaError;
import static com.nvidia.cuvs.internal.panama.headers_h.cudaMemGetInfo;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsDeviceIdGet;
import static com.nvidia.cuvs.internal.panama.headers_h_1.*;

import com.nvidia.cuvs.CuVSMemoryInfo;
import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.GPUInfo;
import com.nvidia.cuvs.GPUInfoProvider;
import com.nvidia.cuvs.internal.common.Util;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;

public class GPUInfoProviderImpl implements GPUInfoProvider {

  @Override
  public List<GPUInfo> availableGPUs() throws Throwable {
    return Util.availableGPUs();
  }

  @Override
  public List<GPUInfo> compatibleGPUs() throws Throwable {
    return Util.compatibleGPUs(MIN_COMPUTE_CAPABILITY, MIN_DEVICE_MEMORY_IN_MB);
  }

  @Override
  public CuVSMemoryInfo getCurrentMemoryInfo(CuVSResources resources) {
    try (var resourcesAccess = resources.access()) {
      try (var localArena = Arena.ofConfined()) {
        var deviceIdPtr = localArena.allocate(C_INT);
        checkCudaError(cudaGetDevice(deviceIdPtr), "cudaGetDevice");
        var currentDeviceId = deviceIdPtr.get(C_INT, 0);

        checkCuVSError(cuvsDeviceIdGet(resourcesAccess.handle(), deviceIdPtr), "cuvsDeviceIdGet");
        var resourcesDeviceId = deviceIdPtr.get(C_INT, 0);

        if (resourcesDeviceId != currentDeviceId) {
          checkCudaError(cudaSetDevice(resourcesDeviceId), "cudaSetDevice");
        }

        MemorySegment freeMemoryPtr = localArena.allocate(size_t);
        MemorySegment totalMemoryPtr = localArena.allocate(size_t);
        checkCudaError(cudaMemGetInfo(freeMemoryPtr, totalMemoryPtr), "cudaMemGetInfo");

        if (resourcesDeviceId != currentDeviceId) {
          checkCudaError(cudaSetDevice(currentDeviceId), "cudaSetDevice");
        }

        return new CuVSMemoryInfo(freeMemoryPtr.get(size_t, 0));
      }
    }
  }
}
