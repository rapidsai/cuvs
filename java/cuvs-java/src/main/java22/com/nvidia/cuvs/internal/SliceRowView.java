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

import com.nvidia.cuvs.CuVSMatrix;
import com.nvidia.cuvs.RowView;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Locale;

/**
 * A {@link RowView} over a {@link MemorySegment} slice.
 */
class SliceRowView implements RowView {
  private final MemorySegment memorySegment;
  private final long size;
  private final ValueLayout valueLayout;
  private final CuVSMatrix.DataType dataType;
  private final long valueByteSize;

  SliceRowView(
      MemorySegment slice,
      long size,
      ValueLayout valueLayout,
      CuVSMatrix.DataType dataType,
      long valueByteSize) {
    this.memorySegment = slice;
    this.size = size;
    this.valueLayout = valueLayout;
    this.dataType = dataType;
    this.valueByteSize = valueByteSize;
  }

  @Override
  public long size() {
    return size;
  }

  @Override
  public float getAsFloat(long index) {
    assert (index < size)
        : String.format(Locale.ROOT, "Index out of bound ([%d], size [%d])", index, size);
    assert dataType == CuVSMatrix.DataType.FLOAT
        : String.format(
            Locale.ROOT, "Input array is of the wrong type for dataType [%s]", dataType.toString());

    return memorySegment.get((ValueLayout.OfFloat) valueLayout, index * valueByteSize);
  }

  @Override
  public byte getAsByte(long index) {
    assert (index < size)
        : String.format(Locale.ROOT, "Index out of bound ([%d], size [%d])", index, size);
    assert dataType == CuVSMatrix.DataType.BYTE
        : String.format(
            Locale.ROOT, "Input array is of the wrong type for dataType [%s]", dataType.toString());

    return memorySegment.get((ValueLayout.OfByte) valueLayout, index * valueByteSize);
  }

  @Override
  public int getAsInt(long index) {
    assert (index < size)
        : String.format(Locale.ROOT, "Index out of bound ([%d], size [%d])", index, size);
    assert dataType == CuVSMatrix.DataType.INT || dataType == CuVSMatrix.DataType.UINT
        : String.format(
            Locale.ROOT, "Input array is of the wrong type for dataType [%s]", dataType.toString());

    return memorySegment.get((ValueLayout.OfInt) valueLayout, index * valueByteSize);
  }

  @Override
  public void toArray(int[] array) {
    assert (array.length >= size)
        : String.format(
            Locale.ROOT,
            "Input array is not large enough (required: [%d], actual [%d])",
            size,
            array.length);
    assert dataType == CuVSMatrix.DataType.INT || dataType == CuVSMatrix.DataType.UINT
        : String.format(
            Locale.ROOT, "Input array is of the wrong type for dataType [%s]", dataType.toString());
    MemorySegment.copy(memorySegment, valueLayout, 0, array, 0, (int) size);
  }

  @Override
  public void toArray(float[] array) {
    assert (array.length >= size)
        : String.format(
            Locale.ROOT,
            "Input array is not large enough (required: [%d], actual [%d])",
            size,
            array.length);
    assert dataType == CuVSMatrix.DataType.FLOAT
        : String.format(
            Locale.ROOT, "Input array is of the wrong type for dataType [%s]", dataType.toString());
    MemorySegment.copy(memorySegment, valueLayout, 0, array, 0, (int) size);
  }

  @Override
  public void toArray(byte[] array) {
    assert (array.length >= size)
        : String.format(
            Locale.ROOT,
            "Input array is not large enough (required: [%d], actual [%d])",
            size,
            array.length);
    assert dataType == CuVSMatrix.DataType.BYTE
        : String.format(
            Locale.ROOT, "Input array is of the wrong type for dataType [%s]", dataType.toString());
    MemorySegment.copy(memorySegment, valueLayout, 0, array, 0, (int) size);
  }
}
