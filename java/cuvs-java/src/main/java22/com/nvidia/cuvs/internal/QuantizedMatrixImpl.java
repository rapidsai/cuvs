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

import com.nvidia.cuvs.QuantizedMatrix;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Objects;

/**
 * Implementation of QuantizedMatrix that wraps native memory containing quantized data.
 * Supports both binary quantization (1 bit per value) and scalar quantization (8 bits per value).
 */
public final class QuantizedMatrixImpl implements QuantizedMatrix {

  private final MemorySegment nativeSegment;
  private final long rows;
  private final long cols;
  private final int bitsPerValue;
  private final Arena arena;
  private boolean closed = false;

  /**
   * Creates a new QuantizedMatrixImpl.
   *
   * @param nativeSegment the native memory segment containing quantized data
   * @param rows number of rows in the matrix
   * @param cols number of columns in the matrix
   * @param bitsPerValue bits per quantized value (1 for binary, 8 for scalar)
   * @param arena the memory arena managing the native segment
   */
  public QuantizedMatrixImpl(
      MemorySegment nativeSegment, long rows, long cols, int bitsPerValue, Arena arena) {
    this.nativeSegment = Objects.requireNonNull(nativeSegment);
    this.rows = rows;
    this.cols = cols;
    this.bitsPerValue = bitsPerValue;
    this.arena = Objects.requireNonNull(arena);

    if (bitsPerValue != 1 && bitsPerValue != 8) {
      throw new IllegalArgumentException("Only 1-bit and 8-bit quantization are supported");
    }
  }

  @Override
  public int rows() {
    return (int) rows;
  }

  @Override
  public int cols() {
    return (int) cols;
  }

  @Override
  public byte get(int row, int col) {
    ensureOpen();
    validateBounds(row, col);

    if (bitsPerValue == 1) {
      // Binary quantization: extract single bit
      long outputCols = (cols + 7) / 8; // packed bytes per row
      int byteIndex = col / 8;
      int bitIndex = col % 8;
      byte packedByte =
          nativeSegment.getAtIndex(
              java.lang.foreign.ValueLayout.JAVA_BYTE, row * outputCols + byteIndex);
      return (byte) ((packedByte >> bitIndex) & 1);
    } else if (bitsPerValue == 8) {
      // Scalar quantization: extract byte directly
      return nativeSegment.getAtIndex(java.lang.foreign.ValueLayout.JAVA_BYTE, row * cols + col);
    } else {
      throw new IllegalStateException("Unsupported bitsPerValue: " + bitsPerValue);
    }
  }

  @Override
  public byte[][] toArray() {
    ensureOpen();
    byte[][] result = new byte[(int) rows][(int) cols];
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        result[i][j] = get(i, j);
      }
    }
    return result;
  }

  @Override
  public void copyRow(int row, byte[] dest) {
    ensureOpen();
    if (row < 0 || row >= rows) {
      throw new IndexOutOfBoundsException("Row index out of bounds: " + row);
    }
    if (dest.length < cols) {
      throw new IllegalArgumentException(
          "Destination array too small: " + dest.length + " < " + cols);
    }

    for (int j = 0; j < cols; j++) {
      dest[j] = get(row, j);
    }
  }

  private void validateBounds(int row, int col) {
    if (row < 0 || row >= rows) {
      throw new IndexOutOfBoundsException("Row index out of bounds: " + row);
    }
    if (col < 0 || col >= cols) {
      throw new IndexOutOfBoundsException("Column index out of bounds: " + col);
    }
  }

  private void ensureOpen() {
    if (closed) {
      throw new IllegalStateException("QuantizedMatrix has been closed");
    }
  }

  @Override
  public void close() {
    if (!closed) {
      arena.close();
      closed = true;
    }
  }
}
