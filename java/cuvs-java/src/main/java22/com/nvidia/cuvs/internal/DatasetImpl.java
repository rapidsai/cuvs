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

import com.nvidia.cuvs.Dataset;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.concurrent.atomic.AtomicReference;

public class DatasetImpl implements Dataset {
  private final AtomicReference<Arena> arenaReference;
  private final MemorySegment seg;
  private final int size;
  private final int dimensions;

  public DatasetImpl(Arena arena, MemorySegment memorySegment, int size, int dimensions) {
    this.arenaReference = new AtomicReference<>(arena);
    this.seg = memorySegment;
    this.size = size;
    this.dimensions = dimensions;
  }

  @Override
  public void close() {
    var arena = arenaReference.getAndSet(null);
    if (arena != null) {
      arena.close();
    }
  }

  @Override
  public int size() {
    return size;
  }

  @Override
  public int dimensions() {
    return dimensions;
  }

  public MemorySegment asMemorySegment() {
    return seg;
  }

  /**
   * Converts the dataset into a Java heap array of bytes.
   * This method is primarily for 8-bit precision datasets.
   *
   * @return A byte[][] array representing the dataset.
   * @throws IllegalStateException if the dataset is not 8-bit precision.
   */
  public byte[][] toByteArray() {
    if (precision() != 8) {
      throw new IllegalStateException(
          "Dataset is not 8-bit precision. Current precision: " + precision());
    }
    byte[][] result = new byte[size][dimensions];
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < dimensions; j++) {
        result[i][j] = seg.get(java.lang.foreign.ValueLayout.JAVA_BYTE, (long) i * dimensions + j);
      }
    }
    return result;
  }

  @Override
  public int precision() {
    if (size == 0 || dimensions == 0) {
      return 32; // Default for empty datasets
    }

    long totalElements = (long) size * dimensions;
    long segmentBytes = seg.byteSize();

    // Calculate expected bytes for different precisions
    long expectedFloat32Bytes = totalElements * 4; // 4 bytes per float
    long expectedInt8Bytes = totalElements * 1; // 1 byte per int8
    long expectedBinaryBytes = (totalElements + 7) / 8; // Packed bits

    // Match against actual segment size
    if (segmentBytes == expectedFloat32Bytes) {
      return 32;
    } else if (segmentBytes == expectedInt8Bytes) {
      return 8;
    } else if (segmentBytes == expectedBinaryBytes) {
      return 1;
    } else {
      throw new IllegalStateException(
          String.format(
              "Cannot determine precision: segment has %d bytes for %d elements",
              segmentBytes, totalElements));
    }
  }
}
