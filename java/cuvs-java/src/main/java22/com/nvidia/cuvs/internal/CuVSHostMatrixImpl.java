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

import com.nvidia.cuvs.CuVSDeviceMatrix;
import com.nvidia.cuvs.CuVSHostMatrix;
import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.RowView;
import java.lang.foreign.*;
import java.lang.invoke.VarHandle;
import java.util.Locale;

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
  public RowView getRow(long index) {
    assert (index < size)
        : String.format(Locale.ROOT, "Index out of bound ([%d], size [%d])", index, size);

    var valueByteSize = valueLayout.byteSize();
    return new SliceRowView(
        memorySegment.asSlice(index * columns * valueByteSize, columns * valueByteSize),
        columns,
        valueLayout,
        dataType,
        valueByteSize);
  }

  @Override
  public void toArray(int[][] array) {
    assert (array.length >= size)
        : String.format(
            Locale.ROOT,
            "Input array is not large enough (required: [%d], actual [%d])",
            size,
            array.length);
    assert (array.length == 0 || array[0].length >= columns)
        : String.format(
            Locale.ROOT,
            "Input array is not wide enough (required: [%d], actual [%d])",
            columns,
            array[0].length);
    assert dataType == DataType.INT || dataType == DataType.UINT
        : String.format(
            Locale.ROOT, "Input array is of the wrong type for dataType [%s]", dataType.toString());

    var valueByteSize = valueLayout.byteSize();
    for (int r = 0; r < size; ++r) {
      MemorySegment.copy(
          memorySegment, valueLayout, r * columns * valueByteSize, array[r], 0, (int) columns);
    }
  }

  @Override
  public void toArray(float[][] array) {
    assert (array.length >= size)
        : String.format(
            Locale.ROOT,
            "Input array is not large enough (required: [%d], actual [%d])",
            size,
            array.length);
    assert (array.length == 0 || array[0].length >= columns)
        : String.format(
            Locale.ROOT,
            "Input array is not wide enough (required: [%d], actual [%d])",
            columns,
            array[0].length);
    assert dataType == DataType.FLOAT
        : String.format(
            Locale.ROOT, "Input array is of the wrong type for dataType [%s]", dataType.toString());

    var valueByteSize = valueLayout.byteSize();
    for (int r = 0; r < size; ++r) {
      MemorySegment.copy(
          memorySegment, valueLayout, r * columns * valueByteSize, array[r], 0, (int) columns);
    }
  }

  @Override
  public void toArray(byte[][] array) {
    assert (array.length >= size)
        : String.format(
            Locale.ROOT,
            "Input array is not large enough (required: [%d], actual [%d])",
            size,
            array.length);
    assert (array.length == 0 || array[0].length >= columns)
        : String.format(
            Locale.ROOT,
            "Input array is not wide enough (required: [%d], actual [%d])",
            columns,
            array[0].length);
    assert dataType == DataType.BYTE
        : String.format(
            Locale.ROOT, "Input array is of the wrong type for dataType [%s]", dataType.toString());

    var valueByteSize = valueLayout.byteSize();
    for (int r = 0; r < size; ++r) {
      MemorySegment.copy(
          memorySegment, valueLayout, r * columns * valueByteSize, array[r], 0, (int) columns);
    }
  }

  @Override
  public void toHost(CuVSHostMatrix hostMatrix) {
    var targetMatrix = (CuVSHostMatrixImpl) hostMatrix;
    var valueByteSize = valueLayout.byteSize();
    MemorySegment.copy(
        this.memorySegment, 0L, targetMatrix.memorySegment, 0L, size * columns * valueByteSize);
  }

  @Override
  public void toDevice(CuVSDeviceMatrix deviceMatrix, CuVSResources cuVSResources) {
    copyMatrix(this, (CuVSMatrixBaseImpl) deviceMatrix, cuVSResources);
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
