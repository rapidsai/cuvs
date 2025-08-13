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

import static com.nvidia.cuvs.internal.common.Util.prepareTensor;
import static com.nvidia.cuvs.internal.panama.headers_h.*;

import com.nvidia.cuvs.CuVSHostMatrix;
import com.nvidia.cuvs.RowView;
import java.lang.foreign.*;
import java.lang.invoke.VarHandle;

/**
 * A Dataset implementation backed by host (CPU) memory.
 */
public class CuVSHostMatrixImpl extends CuVSMatrixBaseImpl implements CuVSHostMatrix {
  protected final VarHandle accessor$vh;

  public CuVSHostMatrixImpl(
      MemorySegment memorySegment, long size, long columns, DataType dataType) {
    this(
        memorySegment,
        size,
        columns,
        dataType,
        valueLayoutFromType(dataType),
        MemoryLayout.sequenceLayout(size * columns, valueLayoutFromType(dataType))
            .withByteAlignment(32));
  }

  protected CuVSHostMatrixImpl(
      MemorySegment memorySegment,
      long size,
      long columns,
      DataType dataType,
      ValueLayout valueLayout,
      MemoryLayout sequenceLayout) {
    super(memorySegment, dataType, valueLayout, size, columns);
    this.accessor$vh = sequenceLayout.varHandle(MemoryLayout.PathElement.sequenceElement());
  }

  @Override
  public RowView getRow(long nodeIndex) {
    var valueByteSize = valueLayout.byteSize();
    return new SliceRowView(
        memorySegment.asSlice(nodeIndex * columns * valueByteSize, columns * valueByteSize),
        columns,
        valueLayout,
        dataType,
        valueByteSize);
  }

  @Override
  public void toArray(int[][] array) {
    assert dataType == DataType.INT;
    assert (array.length >= size) : "Input array is not large enough";
    assert (array.length == 0 || array[0].length >= columns) : "Input array is not large enough";
    var valueByteSize = valueLayout.byteSize();
    for (int r = 0; r < size; ++r) {
      MemorySegment.copy(
          memorySegment, valueLayout, r * columns * valueByteSize, array[r], 0, (int) columns);
    }
  }

  @Override
  public void toArray(float[][] array) {
    assert dataType == DataType.FLOAT;
    assert (array.length >= size) : "Input array is not large enough";
    assert (array.length == 0 || array[0].length >= columns) : "Input array is not large enough";
    var valueByteSize = valueLayout.byteSize();
    for (int r = 0; r < size; ++r) {
      MemorySegment.copy(
          memorySegment, valueLayout, r * columns * valueByteSize, array[r], 0, (int) columns);
    }
  }

  @Override
  public void toArray(byte[][] array) {
    assert dataType == DataType.BYTE;
    assert (array.length >= size) : "Input array is not large enough";
    assert (array.length == 0 || array[0].length >= columns) : "Input array is not large enough";
    var valueByteSize = valueLayout.byteSize();
    for (int r = 0; r < size; ++r) {
      MemorySegment.copy(
          memorySegment, valueLayout, r * columns * valueByteSize, array[r], 0, (int) columns);
    }
  }

  @Override
  public void close() {}

  @Override
  public int get(int row, int col) {
    return (int) accessor$vh.get(memorySegment, 0L, (long) row * columns + col);
  }

  @Override
  public MemorySegment toTensor(Arena arena) {
    return prepareTensor(
        arena, memorySegment, new long[] {size, columns}, code(), bits(), kDLCPU());
  }
}
