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

import static com.nvidia.cuvs.internal.common.LinkerHelper.C_CHAR;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_FLOAT;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_INT;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_LONG;
import static com.nvidia.cuvs.internal.common.LinkerHelper.downcallHandle;
import static com.nvidia.cuvs.internal.panama.PanamaFFMAPI_1.cudaGetDeviceCount;
import static com.nvidia.cuvs.internal.panama.PanamaFFMAPI_1.cudaGetDeviceProperties_v2;
import static com.nvidia.cuvs.internal.panama.PanamaFFMAPI_1.cudaMemGetInfo;
import static com.nvidia.cuvs.internal.panama.PanamaFFMAPI_1.cudaSetDevice;
import static com.nvidia.cuvs.internal.panama.PanamaFFMAPI_1.size_t;
import static java.lang.foreign.ValueLayout.ADDRESS;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemoryLayout.PathElement;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.VarHandle;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;

import com.nvidia.cuvs.GPUInfo;
import com.nvidia.cuvs.internal.panama.DLDataType;
import com.nvidia.cuvs.internal.panama.DLDevice;
import com.nvidia.cuvs.internal.panama.DLManagedTensor;
import com.nvidia.cuvs.internal.panama.DLTensor;
import com.nvidia.cuvs.internal.panama.cudaDeviceProp;

public class Util {

  public static final int CUVS_SUCCESS = 1;
  public static final int CUDA_SUCCESS = 0;

  private static final MethodHandle getLastErrorTextMethodHandle = downcallHandle("cuvsGetLastErrorText",
      FunctionDescriptor.of(ADDRESS));

  private Util() {
  }

  /**
   * Checks the result value of a (CuVS) native method handle call.
   *
   * @param value  the return value
   * @param caller the native method handle that was called
   */
  public static void checkCuVSError(int value, String caller) {
    if (value != CUVS_SUCCESS) {
      String errorMsg = getLastErrorText();
      throw new RuntimeException(caller + " returned " + value + "[" + errorMsg + "]");
    }
  }

  /**
   * Checks the result value of a (CUDA) native method handle call.
   *
   * @param value  the return value
   * @param caller the native method handle that was called
   */
  public static void checkCudaError(int value, String caller) {
    if (value != CUDA_SUCCESS) {
      throw new RuntimeException(caller + " returned " + value);
    }
  }

  static final long MAX_ERROR_TEXT = 1_000_000L;

  static String getLastErrorText() {
    try {
      MemorySegment seg = (MemorySegment) getLastErrorTextMethodHandle.invokeExact();
      if (seg.equals(MemorySegment.NULL)) {
        return "no last error text";
      }
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
    try (var localArena = Arena.ofConfined()) {

      MemorySegment numGpus = localArena.allocate(C_INT);
      int returnValue = cudaGetDeviceCount(numGpus);
      checkCudaError(returnValue, "cudaGetDeviceCount");

      int numGpuCount = numGpus.get(C_INT, 0);
      List<GPUInfo> gpuInfoArr = new ArrayList<GPUInfo>();

      MemorySegment free = localArena.allocate(size_t);
      MemorySegment total = localArena.allocate(size_t);
      MemorySegment deviceProp = cudaDeviceProp.allocate(localArena);

      for (int i = 0; i < numGpuCount; i++) {

        returnValue = cudaSetDevice(i);
        checkCudaError(returnValue, "cudaSetDevice");

        returnValue = cudaGetDeviceProperties_v2(deviceProp, i);
        checkCudaError(returnValue, "cudaGetDeviceProperties_v2");

        returnValue = cudaMemGetInfo(free, total);
        checkCudaError(returnValue, "cudaMemGetInfo");

        float computeCapability = Float
            .parseFloat(cudaDeviceProp.major(deviceProp) + "." + cudaDeviceProp.minor(deviceProp));

        GPUInfo gpuInfo = new GPUInfo(i, cudaDeviceProp.name(deviceProp).getString(0), free.get(C_LONG, 0),
            total.get(C_LONG, 0), computeCapability);

        gpuInfoArr.add(gpuInfo);
      }
      return gpuInfoArr;
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

  public static MemorySegment buildMemorySegment(Arena arena, byte[] data) {
    int cells = data.length;
    MemoryLayout dataMemoryLayout = MemoryLayout.sequenceLayout(cells, C_CHAR);
    MemorySegment dataMemorySegment = arena.allocate(dataMemoryLayout);
    MemorySegment.copy(data, 0, dataMemorySegment, C_CHAR, 0, cells);
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
      MemorySegment.copy(data[r], 0, dataMemorySegment, C_FLOAT, (r * cols * C_FLOAT.byteSize()), (int) cols);
    }
    return dataMemorySegment;
  }

  public static BitSet concatenate(BitSet[] arr, int maxSizeOfEachBitSet) {
    BitSet ret = new BitSet(maxSizeOfEachBitSet * arr.length);
    for (int i = 0; i < arr.length; i++) {
      BitSet b = arr[i];
      if (b == null || b.length() == 0) {
        ret.set(i * maxSizeOfEachBitSet, (i + 1) * maxSizeOfEachBitSet);
      } else {
        for (int j = 0; j < maxSizeOfEachBitSet; j++) {
          ret.set(i * maxSizeOfEachBitSet + j, b.get(j));
        }
      }
    }
    return ret;
  }

  /**
   * @brief Helper function for creating DLManagedTensor instance
   *
   * @param[in] data the data pointer points to the allocated data
   * @param[in] shape the shape of the tensor
   * @param[in] code the type code of base types
   * @param[in] bits the shape of the tensor
   * @param[in] ndim the number of dimensions
   * @return DLManagedTensor
   */
  public static MemorySegment prepareTensor(Arena arena, MemorySegment data, long[] shape, int code, int bits, int ndim,
      int deviceType, int lanes) {

    MemorySegment tensor = DLManagedTensor.allocate(arena);
    MemorySegment dlTensor = DLTensor.allocate(arena);

    DLTensor.data(dlTensor, data);

    MemorySegment dlDevice = DLDevice.allocate(arena);
    DLDevice.device_type(dlDevice, deviceType);
    DLTensor.device(dlTensor, dlDevice);

    DLTensor.ndim(dlTensor, ndim);

    MemorySegment dtype = DLDataType.allocate(arena);
    DLDataType.code(dtype, (byte) code);
    DLDataType.bits(dtype, (byte) bits);
    DLDataType.lanes(dtype, (short) lanes);
    DLTensor.dtype(dlTensor, dtype);

    DLTensor.shape(dlTensor, Util.buildMemorySegment(arena, shape));

    DLTensor.strides(dlTensor, MemorySegment.NULL);

    DLManagedTensor.dl_tensor(tensor, dlTensor);

    return tensor;
  }

}
