/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.internal.common;

import static com.nvidia.cuvs.internal.common.LinkerHelper.C_POINTER;
import static com.nvidia.cuvs.internal.common.Util.checkCudaError;
import static com.nvidia.cuvs.internal.panama.headers_h.*;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

public class PinnedMemoryBuffer implements AutoCloseable {

  public static final int CHUNK_BYTES =
      8 * 1024 * 1024; // Based on benchmarks, 8MB seems the minimum size to optimize PCIe bandwidth

  private MemorySegment hostBuffer = MemorySegment.NULL;

  private static MemorySegment createPinnedBuffer() {
    try (var localArena = Arena.ofConfined()) {
      MemorySegment pointer = localArena.allocate(C_POINTER);
      checkCudaError(cudaMallocHost(pointer, PinnedMemoryBuffer.CHUNK_BYTES), "cudaMallocHost");
      return pointer.get(C_POINTER, 0);
    }
  }

  private static void destroyPinnedBuffer(MemorySegment bufferSegment) {
    checkCudaError(cudaFreeHost(bufferSegment), "cudaFreeHost");
  }

  public MemorySegment address() {
    if (hostBuffer == MemorySegment.NULL) {
      hostBuffer = createPinnedBuffer();
    }
    return hostBuffer;
  }

  @Override
  public void close() {
    if (hostBuffer != MemorySegment.NULL) {
      destroyPinnedBuffer(hostBuffer);
      hostBuffer = MemorySegment.NULL;
    }
  }
}
