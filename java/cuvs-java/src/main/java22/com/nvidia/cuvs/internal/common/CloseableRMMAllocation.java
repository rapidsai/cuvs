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

import static com.nvidia.cuvs.internal.common.LinkerHelper.C_POINTER;
import static com.nvidia.cuvs.internal.common.Util.checkCuVSError;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsRMMAlloc;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsRMMFree;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * A closeable handle for RMM allocations that can be used with try-with-resources.
 * This class manages the lifecycle of a memory allocation made with RMM (RAPIDS Memory Manager).
 * It ensures that the allocated memory is properly released when no longer needed.
 */
public class CloseableRMMAllocation implements CloseableHandle {

  private final long cuvsResourceHandle;
  private long numBytes;
  private MemorySegment pointer;

  private CloseableRMMAllocation(long cuvsResourceHandle, long numBytes, MemorySegment pointer) {
    this.cuvsResourceHandle = cuvsResourceHandle;
    this.numBytes = numBytes;
    this.pointer = pointer;
  }

  /**
   * Copy constructor transfers the ownership of the argument's MemorySegment to this object.
   */
  public CloseableRMMAllocation(CloseableRMMAllocation other) {
    this.cuvsResourceHandle = other.cuvsResourceHandle;
    this.numBytes = other.numBytes;
    this.pointer = other.release();
  }

  public static CloseableRMMAllocation allocateRMMSegment(long cuvsResourceHandle, long numBytes) {
    try (var localArena = Arena.ofConfined()) {
      MemorySegment datasetMemorySegment = localArena.allocate(C_POINTER);
      checkCuVSError(
          cuvsRMMAlloc(cuvsResourceHandle, datasetMemorySegment, numBytes), "cuvsRMMAlloc");
      return new CloseableRMMAllocation(
          cuvsResourceHandle, numBytes, datasetMemorySegment.get(C_POINTER, 0));
    }
  }

  @Override
  public MemorySegment handle() {
    return pointer;
  }

  private MemorySegment release() {
    var oldPointer = pointer;
    pointer = MemorySegment.NULL;
    numBytes = 0;
    return oldPointer;
  }

  private boolean mustClose() {
    return pointer != MemorySegment.NULL;
  }

  @Override
  public void close() {
    if (mustClose()) {
      checkCuVSError(cuvsRMMFree(cuvsResourceHandle, pointer, numBytes), "cuvsRMMFree");
      pointer = MemorySegment.NULL;
    }
  }

  public static CloseableRMMAllocation EMPTY = new CloseableRMMAllocation(0, 0, MemorySegment.NULL);
}
