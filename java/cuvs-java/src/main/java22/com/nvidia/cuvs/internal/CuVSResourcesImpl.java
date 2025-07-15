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

/**
 * Used for allocating resources for cuVS
 *
 * @since 25.02
 */
public class CuVSResourcesImpl implements CuVSResources {

  private final Path tempDirectory;
  private final long resourceHandle;
  private boolean destroyed;

  /**
   * Constructor that allocates the resources needed for cuVS
   *
   */
  public CuVSResourcesImpl(Path tempDirectory) {
    this.tempDirectory = tempDirectory;
    try (var localArena = Arena.ofConfined()) {
      var resourcesMemorySegment = localArena.allocate(cuvsResources_t);
      int returnValue = cuvsResourcesCreate(resourcesMemorySegment);
      checkCuVSError(returnValue, "cuvsResourcesCreate");
      resourceHandle = resourcesMemorySegment.get(cuvsResources_t, 0);
    }
  }

  @Override
  public void close() {
    synchronized (this) {
      checkNotDestroyed();
      int returnValue = cuvsResourcesDestroy(resourceHandle);
      checkCuVSError(returnValue, "cuvsResourcesDestroy");
      destroyed = true;
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
   * Gets the opaque CuVSResources handle, to be used whenever we need to pass a cuvsResources_t parameter
   *
   * @return the CuVSResources handle
   */
  long getHandle() {
    checkNotDestroyed();
    return resourceHandle;
  }
}
