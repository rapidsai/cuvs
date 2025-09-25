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
import static com.nvidia.cuvs.internal.common.Util.checkCudaError;
import static com.nvidia.cuvs.internal.panama.headers_h.*;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

public class PinnedMemoryBuffer implements AutoCloseable {

  private static final int CHUNK_BYTES =
      8 * 1024 * 1024; // Based on benchmarks, 8MB seems the minimum size to optimize PCIe bandwidth
  private final long hostBufferBytes;

  private MemorySegment hostBuffer = MemorySegment.NULL;

  public PinnedMemoryBuffer(long rows, long columns, ValueLayout valueLayout) {

    long rowBytes = columns * valueLayout.byteSize();
    long matrixBytes = rows * rowBytes;
    if (matrixBytes < CHUNK_BYTES) {
      this.hostBufferBytes = matrixBytes;
    } else if (rowBytes > CHUNK_BYTES) {
      // We need to buffer at least one row at time
      this.hostBufferBytes = rowBytes;
    } else {
      var rowCount = (CHUNK_BYTES / rowBytes);
      this.hostBufferBytes = rowBytes * rowCount;
    }
  }

  private static MemorySegment createPinnedBuffer(long bufferBytes) {
    try (var localArena = Arena.ofConfined()) {
      MemorySegment pointer = localArena.allocate(C_POINTER);
      checkCudaError(cudaMallocHost(pointer, bufferBytes), "cudaMallocHost");
      return pointer.get(C_POINTER, 0);
    }
  }

  private static void destroyPinnedBuffer(MemorySegment bufferSegment) {
    checkCudaError(cudaFreeHost(bufferSegment), "cudaFreeHost");
  }

  public MemorySegment address() {
    if (hostBuffer == MemorySegment.NULL) {
      hostBuffer = createPinnedBuffer(hostBufferBytes);
    }
    return hostBuffer;
  }

  public long size() {
    return hostBufferBytes;
  }

  @Override
  public void close() {
    if (hostBuffer != MemorySegment.NULL) {
      destroyPinnedBuffer(hostBuffer);
      hostBuffer = MemorySegment.NULL;
    }
  }
}
