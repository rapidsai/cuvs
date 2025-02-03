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

package com.nvidia.cuvs.common;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemoryLayout.PathElement;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.VarHandle;
import java.util.ArrayList;
import java.util.List;

import com.nvidia.cuvs.GPUInfo;
import com.nvidia.cuvs.LibraryException;
import com.nvidia.cuvs.panama.GpuInfo;

public class Util {

  private static Arena arena = null;
  private static Linker linker = null;
  private static SymbolLookup symbolLookup = null;
  private static MemoryLayout intMemoryLayout;
  private static MethodHandle getGpuInfoMethodHandle = null;
  protected static File nativeLibrary;

  static {
    try {
      linker = Linker.nativeLinker();
      arena = Arena.ofShared();
      nativeLibrary = Util.loadLibraryFromJar("/libcuvs_java.so");
      symbolLookup = SymbolLookup.libraryLookup(nativeLibrary.getAbsolutePath(), arena);
      intMemoryLayout = linker.canonicalLayouts().get("int");
      getGpuInfoMethodHandle = linker.downcallHandle(symbolLookup.find("get_gpu_info").get(),
          FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
    } catch (Exception e) {
      throw new LibraryException("LibCuVS Java Library Not Loaded", e);
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
      if (gpuInfo.getComputeCapability() >= minComputeCapability && gpuInfo.getTotalMemory() >= minDeviceMemoryB) {
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
    List<GPUInfo> results = new ArrayList<GPUInfo>();
    MemoryLayout returnValueMemoryLayout = intMemoryLayout;
    MemorySegment returnValueMemorySegment = arena.allocate(returnValueMemoryLayout);
    MemoryLayout numGpuMemoryLayout = intMemoryLayout;
    MemorySegment numGpuMemorySegment = arena.allocate(numGpuMemoryLayout);

    /*
     * Setting a value of 1024 because we cannot predict how much memory to allocate
     * before the function is invoked as cudaGetDeviceCount is inside the
     * get_gpu_info function.
     */
    MemorySegment GpuInfoArrayMemorySegment = GpuInfo.allocateArray(1024, arena);
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

  /**
   * A utility method for getting an instance of {@link MemorySegment} for a
   * {@link String}.
   *
   * @param str the string for the expected {@link MemorySegment}
   * @return an instance of {@link MemorySegment}
   */
  public static MemorySegment buildMemorySegment(Linker linker, Arena arena, String str) {
    MemoryLayout charMemoryLayout = linker.canonicalLayouts().get("char");
    StringBuilder sb = new StringBuilder(str).append('\0');
    MemoryLayout stringMemoryLayout = MemoryLayout.sequenceLayout(sb.length(), charMemoryLayout);
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
  public static MemorySegment buildMemorySegment(Linker linker, Arena arena, long[] data) {
    int cells = data.length;
    MemoryLayout longMemoryLayout = linker.canonicalLayouts().get("long");
    MemoryLayout dataMemoryLayout = MemoryLayout.sequenceLayout(cells, longMemoryLayout);
    MemorySegment dataMemorySegment = arena.allocate(dataMemoryLayout);
    MemorySegment.copy(data, 0, dataMemorySegment, (ValueLayout) longMemoryLayout, 0, cells);
    return dataMemorySegment;
  }

  /**
   * A utility method for building a {@link MemorySegment} for a 2D float array.
   *
   * @param data The 2D float array for which the {@link MemorySegment} is needed
   * @return an instance of {@link MemorySegment}
   */
  public static MemorySegment buildMemorySegment(Linker linker, Arena arena, float[][] data) {
    long rows = data.length;
    long cols = rows > 0 ? data[0].length : 0;
    MemoryLayout floatMemoryLayout = linker.canonicalLayouts().get("float");
    MemoryLayout dataMemoryLayout = MemoryLayout.sequenceLayout(rows * cols, floatMemoryLayout);
    MemorySegment dataMemorySegment = arena.allocate(dataMemoryLayout);
    long floatByteSize = floatMemoryLayout.byteSize();

    for (int r = 0; r < rows; r++) {
      MemorySegment.copy(data[r], 0, dataMemorySegment, (ValueLayout) floatMemoryLayout, (r * cols * floatByteSize),
          (int) cols);
    }

    return dataMemorySegment;
  }

  /**
   * Load the CuVS .so file from environment variable CUVS_JAVA_SO_PATH. If not
   * found there, try to load it from the classpath to a temporary file.
   */
  public static File loadNativeLibrary() throws IOException {
    String libraryPathFromEnvironment = System.getenv("CUVS_JAVA_SO_PATH");
    if (libraryPathFromEnvironment != null) {
      File file = new File(libraryPathFromEnvironment);
      if (!file.exists())
        throw new RuntimeException(
            "Environment variable CUVS_JAVA_SO_PATH points to non-existent file: " + libraryPathFromEnvironment);
      return file;
    }
    return loadLibraryFromJar("/libcuvs_java.so");
  }

  private static File loadLibraryFromJar(String path) throws IOException {
    if (!path.startsWith("/")) {
      throw new IllegalArgumentException("The path has to be absolute (start with '/').");
    }
    // Obtain filename from path
    String[] parts = path.split("/");
    String filename = (parts.length > 1) ? parts[parts.length - 1] : null;

    // Split filename to prefix and suffix (extension)
    String prefix = "";
    String suffix = null;
    if (filename != null) {
      parts = filename.split("\\.", 2);
      prefix = parts[0];
      suffix = (parts.length > 1) ? "." + parts[parts.length - 1] : null;
    }
    // Prepare temporary file
    File temp = File.createTempFile(prefix, suffix);
    InputStream libraryStream = Util.class.getModule().getResourceAsStream(path); // Util.class.getResourceAsStream(path);
    streamCopy(libraryStream, new FileOutputStream(temp));

    return temp;
  }

  private static void streamCopy(InputStream is, OutputStream os) throws LibraryException {
    if (is == null) {
      throw new LibraryException("CuVS Library Not Found in ClassPath");
    }
    byte[] buffer = new byte[1024];
    int readBytes;

    try {
      while ((readBytes = is.read(buffer)) != -1) {
        os.write(buffer, 0, readBytes);
      }
    } catch (IOException e) {
      throw new LibraryException(e);
    } finally {
      // If read/write fails, close streams safely before throwing an exception
      if (os != null)
        try {
          os.close();
        } catch (IOException e) {
          e.printStackTrace();
        }
      if (is != null)
        try {
          is.close();
        } catch (IOException e) {
          e.printStackTrace();
        }
    }
  }
}
