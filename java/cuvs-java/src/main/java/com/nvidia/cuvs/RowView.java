/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
