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

import java.io.File;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemoryLayout.PathElement;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SequenceLayout;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.VarHandle;
import java.util.ArrayList;
import java.util.List;

import com.nvidia.cuvs.common.Util;

/**
 * Used for allocating resources for cuVS
 * 
 * @since 25.02
 */
public class CuVSResources implements AutoCloseable {

  public final Arena arena;
  public final Linker linker;
  public final SymbolLookup symbolLookup;
  protected File nativeLibrary;
  private final MethodHandle createResourcesMethodHandle;
  private final MethodHandle destroyResourcesMethodHandle;
  private final MethodHandle getGpuInfoMethodHandle;
  private final MethodHandle getNumGpusMethodHandle;
  private MemorySegment resourcesMemorySegment;
  private MemoryLayout intMemoryLayout;
  private MemoryLayout longMemoryLayout;

  /**
   * Constructor that allocates the resources needed for cuVS
   * 
   * @throws Throwable exception thrown when native function is invoked
   */
  public CuVSResources() throws Throwable {
    linker = Linker.nativeLinker();
    arena = Arena.ofShared();

    nativeLibrary = Util.loadLibraryFromJar("/libcuvs_java.so");
    symbolLookup = SymbolLookup.libraryLookup(nativeLibrary.getAbsolutePath(), arena);
    intMemoryLayout = linker.canonicalLayouts().get("int");
    longMemoryLayout = linker.canonicalLayouts().get("long");

    createResourcesMethodHandle = linker.downcallHandle(symbolLookup.find("create_resources").get(),
        FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS));

    destroyResourcesMethodHandle = linker.downcallHandle(symbolLookup.find("destroy_resources").get(),
        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS));

    getNumGpusMethodHandle = linker.downcallHandle(symbolLookup.find("get_num_gpus").get(),
        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS));

    getGpuInfoMethodHandle = linker.downcallHandle(symbolLookup.find("get_gpu_info").get(), FunctionDescriptor
        .ofVoid(ValueLayout.ADDRESS, intMemoryLayout, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

    getNumGPUs(); // To check if GPUs are found before proceeding.
    createResources();
  }

  /**
   * Gets the number of GPUs on the machine
   * 
   * @return the number of GPUs on the machine
   */
  protected int getNumGPUs() throws Throwable {
    MemoryLayout returnValueMemoryLayout = intMemoryLayout;
    MemorySegment returnValueMemorySegment = arena.allocate(returnValueMemoryLayout);

    MemoryLayout numGPUsMemoryLayout = intMemoryLayout;
    MemorySegment numGPUsMemorySegment = arena.allocate(numGPUsMemoryLayout);

    getNumGpusMethodHandle.invokeExact(returnValueMemorySegment, numGPUsMemorySegment);
    int returnValue = returnValueMemorySegment.get(ValueLayout.JAVA_INT, 0);

    switch (returnValue) {
    case 0: // cudaSuccess
      int result = numGPUsMemorySegment.get(ValueLayout.JAVA_INT, 0);
      if (result == 0)
        throw new GPUException("No GPUs found! The CuVS Java API needs GPU to work.");
      return result;
    case 3: // cudaErrorInitializationError
      throw new GPUException("The API call failed because the CUDA driver and runtime could not be initialized.");
    case 35: // cudaErrorInsufficientDriver
      throw new GPUException("The installed NVIDIA CUDA driver is older than the CUDA runtime library");
    case 100: // cudaErrorNoDevice
      throw new GPUException("No CUDA-capable devices were detected by the installed CUDA driver");
    default:
      throw new GPUException("Returned value: " + returnValue
          + " Please find more details here: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g3f51e3575c2178246db0a94a430e0038");
    }
  }

  /**
   * Gets the GPU information
   * 
   * @return a list of {@link GPUInfo} objects with GPU details
   */
  protected List<GPUInfo> getGPUInfo() throws Throwable {
    int numGPUs = getNumGPUs();
    List<GPUInfo> results = new ArrayList<GPUInfo>();
    MemoryLayout returnValueMemoryLayout = intMemoryLayout;
    MemorySegment returnValueMemorySegment = arena.allocate(returnValueMemoryLayout);

    SequenceLayout gpuIdSequenceLayout = MemoryLayout.sequenceLayout(numGPUs, intMemoryLayout);
    MemorySegment gpuIdMemorySegment = arena.allocate(gpuIdSequenceLayout);

    SequenceLayout freeMemSequenceLayout = MemoryLayout.sequenceLayout(numGPUs, longMemoryLayout);
    MemorySegment freeMemMemorySegment = arena.allocate(freeMemSequenceLayout);

    SequenceLayout totalMemSequenceLayout = MemoryLayout.sequenceLayout(numGPUs, longMemoryLayout);
    MemorySegment totalMemMemorySegment = arena.allocate(totalMemSequenceLayout);

    getGpuInfoMethodHandle.invokeExact(returnValueMemorySegment, numGPUs, gpuIdMemorySegment, freeMemMemorySegment,
        totalMemMemorySegment);

    VarHandle gpuIdVarHandle = gpuIdSequenceLayout.varHandle(PathElement.sequenceElement());
    VarHandle freeMemVarHandle = freeMemSequenceLayout.varHandle(PathElement.sequenceElement());
    VarHandle totalMemVarHandle = totalMemSequenceLayout.varHandle(PathElement.sequenceElement());

    for (int i = 0; i < numGPUs; i++) {
      int gpuId = (int) gpuIdVarHandle.get(gpuIdMemorySegment, 0L, i);
      long freeMemory = (long) freeMemVarHandle.get(freeMemMemorySegment, 0L, i);
      long totalMemory = (long) totalMemVarHandle.get(totalMemMemorySegment, 0L, i);
      results.add(new GPUInfo(gpuId, freeMemory, totalMemory));
    }

    return results;
  }

  /**
   * Creates the resources used internally and returns its reference.
   * 
   * @throws Throwable exception thrown when native function is invoked
   */
  public void createResources() throws Throwable {
    MemoryLayout returnValueMemoryLayout = intMemoryLayout;
    MemorySegment returnValueMemorySegment = arena.allocate(returnValueMemoryLayout);
    resourcesMemorySegment = (MemorySegment) createResourcesMethodHandle.invokeExact(returnValueMemorySegment);
  }

  @Override
  public void close() {
    MemoryLayout returnValueMemoryLayout = intMemoryLayout;
    MemorySegment returnValueMemorySegment = arena.allocate(returnValueMemoryLayout);
    try {
      destroyResourcesMethodHandle.invokeExact(resourcesMemorySegment, returnValueMemorySegment);
    } catch (Throwable e) {
      e.printStackTrace();
    }
    nativeLibrary.delete();
  }

  /**
   * Gets the reference to the cuvsResources MemorySegment.
   * 
   * @return cuvsResources MemorySegment
   */
  protected MemorySegment getMemorySegment() {
    return resourcesMemorySegment;
  }

  /**
   * Returns the loaded libcuvs_java_cagra.so as a {@link SymbolLookup}
   */
  protected SymbolLookup getSymbolLookup() {
    return symbolLookup;
  }

  /**
   * Container for GPU information
   */
  protected class GPUInfo {

    private int gpuId;
    private long freeMemory;
    private long totalMemory;

    public GPUInfo(int gpuId, long freeMemory, long totalMemory) {
      super();
      this.gpuId = gpuId;
      this.freeMemory = freeMemory;
      this.totalMemory = totalMemory;
    }

    public int getGpuId() {
      return gpuId;
    }

    public long getFreeMemory() {
      return freeMemory;
    }

    public long getTotalMemory() {
      return totalMemory;
    }

    @Override
    public String toString() {
      return "GPUInfo [gpuId=" + gpuId + ", freeMemory=" + freeMemory + ", totalMemory=" + totalMemory + "]";
    }
  }
}