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
 * Represent a contiguous list of elements backed by off-heap memory.
 *
 * @since 25.08
 */
public interface RowView {

  long size();

  /**
   * Returns the integer element at the given position. Asserts that the
   * data type of the dataset on top of which this view is instantiates is
   * {@link CuVSMatrix.DataType#INT}
   *
   * @param index the element index
   */
  int getAsInt(long index);

  /**
   * Returns the integer element at the given position. Asserts that the
   * data type of the dataset on top of which this view is instantiates is
   * {@link CuVSMatrix.DataType#FLOAT}
   *
   * @param index the element index
   */
  float getAsFloat(long index);

  /**
   * Returns the integer element at the given position. Asserts that the
   * data type of the dataset on top of which this view is instantiates is
   * {@link CuVSMatrix.DataType#BYTE}
   *
   * @param index the element index
   */
  byte getAsByte(long index);

  /**
   * Copies the content of this row to an on-heap Java array.
   *
   * @param array the destination array. Must be of length {@link RowView#size()} or bigger.
   */
  void toArray(int[] array);

  /**
   * Copies the content of this row to an on-heap Java array.
   *
   * @param array the destination array. Must be of length {@link RowView#size()} or bigger.
   */
  void toArray(float[] array);

  /**
   * Copies the content of this row to an on-heap Java array.
   *
   * @param array the destination array. Must be of length {@link RowView#size()} or bigger.
   */
  void toArray(byte[] array);
}
