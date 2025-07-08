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
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsResourcesCreate;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsResourcesDestroy;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsResources_t;

import com.nvidia.cuvs.CuVSResources;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Used for allocating resources for cuVS
 *
 * @since 25.02
 */
public class CuVSResourcesImpl implements CuVSResources {

  private final Path tempDirectory;
  private final Arena arena;
  private final long resourceHandle;
  private final ScopedAccess access;

  /**
   * Constructor that allocates the resources needed for cuVS
   *
   * @throws Throwable exception thrown when native function is invoked
   */
  public CuVSResourcesImpl(Path tempDirectory) throws Throwable {
    this.tempDirectory = tempDirectory;
    this.arena = Arena.ofShared();
    try (var localArena = Arena.ofConfined()) {
      var resourcesMemorySegment = localArena.allocate(cuvsResources_t);
      int returnValue = cuvsResourcesCreate(resourcesMemorySegment);
      checkCuVSError(returnValue, "cuvsResourcesCreate");
      this.resourceHandle = resourcesMemorySegment.get(cuvsResources_t, 0);
      this.access = new ScopedAccess() {
        @Override
        public long handle() {
          return resourceHandle;
        }

        @Override
        public void close() {

        }
      };
    }
  }

  @Override
  public ScopedAccess access() {
    return this.access;
  }

  @Override
  public void close() {
    synchronized (this) {
      int returnValue = cuvsResourcesDestroy(resourceHandle);
      checkCuVSError(returnValue, "cuvsResourcesDestroy");
      arena.close();
    }
  }

  @Override
  public Path tempDirectory() {
    return tempDirectory;
  }


  /**
   * The allocation arena used by this resources.
   */
  protected Arena getArena() {
    return arena;
  }
}
