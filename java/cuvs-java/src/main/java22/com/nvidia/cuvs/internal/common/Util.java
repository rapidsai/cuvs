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

package com.nvidia.cuvs.internal.common;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemoryLayout.PathElement;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.VarHandle;
import java.util.ArrayList;
import java.util.List;

import com.nvidia.cuvs.GPUInfo;
import com.nvidia.cuvs.internal.panama.GpuInfo;

import static com.nvidia.cuvs.internal.common.LinkerHelper.C_CHAR;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_FLOAT;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_INT;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_LONG;
import static com.nvidia.cuvs.internal.common.LinkerHelper.downcallHandle;
import static java.lang.foreign.ValueLayout.ADDRESS;

public class Util {

  public static final int CUVS_SUCCESS = 1;

  private static final MethodHandle getGpuInfoMethodHandle = downcallHandle("get_gpu_info",
          FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, ADDRESS));

  private static final MethodHandle getLastErrorTextMethodHandle = downcallHandle("cuvsGetLastErrorText",
          FunctionDescriptor.of(ADDRESS));

  private Util() {}

  /**
   * Checks the result value of a native method handle call.
   *
   * @param value the return value
   * @param caller the native method handle that was called
   */
  public static void checkError(int value, String caller) {
    if (value != CUVS_SUCCESS) {
      String errorMsg = getLastErrorText();
      throw new RuntimeException(caller + " returned " + value + "[" + errorMsg + "]");
    }
  }

  static final long MAX_ERROR_TEXT = 1_000_000L;

  static String getLastErrorText() {
    try {
      MemorySegment seg = (MemorySegment) getLastErrorTextMethodHandle.invokeExact();
      return seg.reinterpret(MAX_ERROR_TEXT).getString(0L);
    } catch (Throwable t) {
      throw new RuntimeException(t);
    }
  }

  /**
   * Get the list of compatible GPUs based on compute capability >= 7.0 and total
   * memory >= 8GB
   *
   * @return a list of compatible GPUs. See {@link GPUInfo}
   */
  public static List<GPUInfo> compatibleGPUs() throws Throwable {
    return compatibleGPUs(7.0, 8192);
  }

  /**
   * Get the list of compatible GPUs based on given compute capability and total
   * memory
   *
   * @param minComputeCapability the minimum compute capability
   * @param minDeviceMemoryMB    the minimum total available memory in MB
   * @return a list of compatible GPUs. See {@link GPUInfo}
   */
  public static List<GPUInfo> compatibleGPUs(double minComputeCapability, int minDeviceMemoryMB) throws Throwable {
    List<GPUInfo> compatibleGPUs = new ArrayList<GPUInfo>();
    double minDeviceMemoryB = Math.pow(2, 20) * minDeviceMemoryMB;
    for (GPUInfo gpuInfo : availableGPUs()) {
      if (gpuInfo.computeCapability() >= minComputeCapability && gpuInfo.totalMemory() >= minDeviceMemoryB) {
        compatibleGPUs.add(gpuInfo);
      }
    }
    return compatibleGPUs;
  }

  /**
   * Gets all the available GPUs
   *
   * @return a list of {@link GPUInfo} objects with GPU details
   */
  public static List<GPUInfo> availableGPUs() throws Throwable {
    List<GPUInfo> results = new ArrayList<>();
    try (var localArena = Arena.ofConfined()) {
      MemorySegment returnValueMemorySegment = localArena.allocate(C_INT);
      MemorySegment numGpuMemorySegment = localArena.allocate(C_INT);

      /*
       * Setting a value of 1024 because we cannot predict how much memory to allocate
       * before the function is invoked as cudaGetDeviceCount is inside the
       * get_gpu_info function.
       */
      MemorySegment GpuInfoArrayMemorySegment = GpuInfo.allocateArray(1024, localArena);
      getGpuInfoMethodHandle.invokeExact(returnValueMemorySegment, numGpuMemorySegment, GpuInfoArrayMemorySegment);
      int numGPUs = numGpuMemorySegment.get(ValueLayout.JAVA_INT, 0);
      MemoryLayout ml = MemoryLayout.sequenceLayout(numGPUs, GpuInfo.layout());
      for (int i = 0; i < numGPUs; i++) {
        VarHandle gpuIdVarHandle = ml.varHandle(PathElement.sequenceElement(i), PathElement.groupElement("gpu_id"));
        VarHandle freeMemoryVarHandle = ml.varHandle(PathElement.sequenceElement(i),
                PathElement.groupElement("free_memory"));
        VarHandle totalMemoryVarHandle = ml.varHandle(PathElement.sequenceElement(i),
                PathElement.groupElement("total_memory"));
        VarHandle ComputeCapabilityVarHandle = ml.varHandle(PathElement.sequenceElement(i),
                PathElement.groupElement("compute_capability"));
        StringBuilder gpuName = new StringBuilder();
        char b = 1;
        int p = 0;
        while (b != 0x00) {
          VarHandle gpuNameVarHandle = ml.varHandle(PathElement.sequenceElement(i), PathElement.groupElement("name"),
                  PathElement.sequenceElement(p++));
          b = (char) (byte) gpuNameVarHandle.get(GpuInfoArrayMemorySegment, 0L);
          gpuName.append(b);
        }
        results.add(new GPUInfo((int) gpuIdVarHandle.get(GpuInfoArrayMemorySegment, 0L), gpuName.toString().trim(),
                (long) freeMemoryVarHandle.get(GpuInfoArrayMemorySegment, 0L),
                (long) totalMemoryVarHandle.get(GpuInfoArrayMemorySegment, 0L),
                (float) ComputeCapabilityVarHandle.get(GpuInfoArrayMemorySegment, 0L)));
      }
      return results;
    }
  }

  /**
   * A utility method for getting an instance of {@link MemorySegment} for a
   * {@link String}.
   *
   * @param str the string for the expected {@link MemorySegment}
   * @return an instance of {@link MemorySegment}
   */
  public static MemorySegment buildMemorySegment(Arena arena, String str) {
    StringBuilder sb = new StringBuilder(str).append('\0');
    MemoryLayout stringMemoryLayout = MemoryLayout.sequenceLayout(sb.length(), C_CHAR);
    MemorySegment stringMemorySegment = arena.allocate(stringMemoryLayout);

    for (int i = 0; i < sb.length(); i++) {
      VarHandle varHandle = stringMemoryLayout.varHandle(PathElement.sequenceElement(i));
      varHandle.set(stringMemorySegment, 0L, (byte) sb.charAt(i));
    }
    return stringMemorySegment;
  }

  /**
   * A utility method for building a {@link MemorySegment} for a 1D long array.
   *
   * @param data The 1D long array for which the {@link MemorySegment} is needed
   * @return an instance of {@link MemorySegment}
   */
  public static MemorySegment buildMemorySegment(Arena arena, long[] data) {
    int cells = data.length;
    MemoryLayout dataMemoryLayout = MemoryLayout.sequenceLayout(cells, C_LONG);
    MemorySegment dataMemorySegment = arena.allocate(dataMemoryLayout);
    MemorySegment.copy(data, 0, dataMemorySegment, C_LONG, 0, cells);
    return dataMemorySegment;
  }

  /**
   * A utility method for building a {@link MemorySegment} for a 2D float array.
   *
   * @param data The 2D float array for which the {@link MemorySegment} is needed
   * @return an instance of {@link MemorySegment}
   */
  public static MemorySegment buildMemorySegment(Arena arena, float[][] data) {
    long rows = data.length;
    long cols = rows > 0 ? data[0].length : 0;
    MemoryLayout dataMemoryLayout = MemoryLayout.sequenceLayout(rows * cols, C_FLOAT);
    MemorySegment dataMemorySegment = arena.allocate(dataMemoryLayout);
    for (int r = 0; r < rows; r++) {
      MemorySegment.copy(data[r], 0, dataMemorySegment, C_FLOAT, (r * cols * C_FLOAT.byteSize()),
          (int) cols);
    }
    return dataMemorySegment;
  }
}
