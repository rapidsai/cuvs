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
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsResources_t;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsResourcesCreate;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsResourcesDestroy;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;

import com.nvidia.cuvs.CuVSResources;

/**
 * Used for allocating resources for cuVS
 *
 * @since 25.02
 */
public class CuVSResourcesImpl implements CuVSResources {

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
    arena = Arena.ofShared();
    resourcesMemorySegment = arena.allocate(cuvsResources_t);
    int returnValue = cuvsResourcesCreate(resourcesMemorySegment);
    checkCuVSError(returnValue, "cuvsResourcesCreate");
  }

  @Override
  public void close() {
    checkNotDestroyed();
    int returnValue = cuvsResourcesDestroy(resourcesMemorySegment.get(cuvsResources_t, 0));
    checkCuVSError(returnValue, "cuvsResourcesDestroy");
    destroyed = true;
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
}
