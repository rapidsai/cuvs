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

/**
 * A wrapper for accessing quantized matrix data.
 * Supports both direct native memory access and deferred heap allocation.
 */
public interface QuantizedMatrix extends AutoCloseable {

  /**
   * Gets the number of rows in the matrix.
   */
  int rows();

  /**
   * Gets the number of columns in the matrix.
   */
  int cols();

  /**
   * Gets the quantized value at (row, col) directly from native memory.
   */
  byte get(int row, int col);

  /**
   * Converts the matrix into a Java heap array.
   */
  byte[][] toArray();

  /**
   * Copies a single row into the provided array.
   *
   * @param row the row index to copy
   * @param dest the destination array (must be at least cols() length)
   */
  void copyRow(int row, byte[] dest);

  /**
   * Frees native memory resources.
   */
  @Override
  void close();
}
