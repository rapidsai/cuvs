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

import com.nvidia.cuvs.CuVSResources;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;

import static com.nvidia.cuvs.internal.common.LinkerHelper.C_INT;
import static com.nvidia.cuvs.internal.common.LinkerHelper.downcallHandle;
import static com.nvidia.cuvs.internal.common.Util.checkError;
import static java.lang.foreign.ValueLayout.ADDRESS;

/**
 * Used for allocating resources for cuVS
 *
 * @since 25.02
 */
public class CuVSResourcesImpl implements CuVSResources {

  static final MethodHandle createResourcesMethodHandle = downcallHandle(
      "create_resources", FunctionDescriptor.of(ADDRESS, ADDRESS)
  );

  private static final MethodHandle destroyResourcesMethodHandle = downcallHandle(
       "destroy_resources", FunctionDescriptor.ofVoid(ADDRESS, ADDRESS)
  );

  private final Path tempDirectory;
  private final Arena arena;
  private final MemorySegment resourcesMemorySegment;
  private boolean destroyed;

  /**
   * Constructor that allocates the resources needed for cuVS
   *
   * @throws Throwable exception thrown when native function is invoked
   */
  public CuVSResourcesImpl(Path tempDirectory) throws Throwable {
    this.tempDirectory = tempDirectory;
    try (var localArena = Arena.ofConfined()) {
      MemorySegment returnValue = localArena.allocate(C_INT);
      resourcesMemorySegment = (MemorySegment) createResourcesMethodHandle.invokeExact(returnValue);
      checkError(returnValue.get(C_INT, 0L), "createResourcesMethodHandle");
    }
    arena = Arena.ofShared();
  }

  @Override
  public void close() {
    checkNotDestroyed();
    try (var localArena = Arena.ofConfined()) {
      MemorySegment returnValue = localArena.allocate(C_INT);
      destroyResourcesMethodHandle.invokeExact(resourcesMemorySegment, returnValue);
      checkError(returnValue.get(C_INT, 0L), "destroyResourcesMethodHandle");
    } catch (Throwable e) {
      e.printStackTrace();
    } finally {
      destroyed = true;
    }
    if (!arena.scope().isAlive()) {
      arena.close();
    }
  }

  @Override
  public Path tempDirectory() {
    return tempDirectory;
  }

  private void checkNotDestroyed() {
    if (destroyed) {
      throw new IllegalStateException("destroyed");
    }
  }

  /**
   * Gets the reference to the cuvsResources MemorySegment.
   *
   * @return cuvsResources MemorySegment
   */
  protected MemorySegment getMemorySegment() {
    checkNotDestroyed();
    return resourcesMemorySegment;
  }

  /**
   * The allocation arena used by this resources.
   */
  protected Arena getArena() {
    checkNotDestroyed();
    return arena;
  }

  /**
   * Container for GPU information
   */
  public class GPUInfo {

    private final int gpuId;
    private final long freeMemory;
    private final long totalMemory;
    private final float computeCapability;

    public GPUInfo(int gpuId, long freeMemory, long totalMemory, float computeCapability) {
      super();
      this.gpuId = gpuId;
      this.freeMemory = freeMemory;
      this.totalMemory = totalMemory;
      this.computeCapability = computeCapability;
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

    public float getComputeCapability() {
      return computeCapability;
    }

    @Override
    public String toString() {
      return "GPUInfo [gpuId=" + gpuId + ", freeMemory=" + freeMemory + ", totalMemory=" + totalMemory
          + ", computeCapability=" + computeCapability + "]";
    }

  }
}
