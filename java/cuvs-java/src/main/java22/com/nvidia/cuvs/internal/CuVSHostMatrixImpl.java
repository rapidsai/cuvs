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

  private final int rowStride;
  private final int columnStride;
  private final long rowBytes;
  private final long rowSize;

  public CuVSHostMatrixImpl(
      MemorySegment memorySegment, long size, long columns, DataType dataType) {
    this(
        memorySegment,
        size,
        columns,
        -1,
        -1,
        dataType,
        valueLayoutFromType(dataType),
        sequenceLayoutFromType(size, columns, -1, dataType));
  }

  public CuVSHostMatrixImpl(
      MemorySegment memorySegment,
      long size,
      long columns,
      int rowStride,
      int columnStride,
      DataType dataType) {
    this(
        memorySegment,
        size,
        columns,
        rowStride,
        columnStride,
        dataType,
        valueLayoutFromType(dataType),
        sequenceLayoutFromType(size, columns, rowStride, dataType));
  }

  protected CuVSHostMatrixImpl(
      MemorySegment memorySegment,
      long size,
      long columns,
      int rowStride,
      int columnStride,
      DataType dataType,
      ValueLayout valueLayout,
      MemoryLayout sequenceLayout) {
    super(memorySegment, dataType, valueLayout, size, columns);

    if (rowStride > 0 && rowStride < columns) {
      throw new IllegalArgumentException("Row stride cannot be less than the number of columns");
    }

    this.rowStride = rowStride;
    this.columnStride = columnStride;

    var elementSize = valueLayout.byteSize();
    this.rowSize = rowStride > 0 ? rowStride * elementSize : columns * elementSize;
    this.rowBytes = columns * elementSize;

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

    for (int r = 0; r < size; ++r) {
      MemorySegment.copy(memorySegment, valueLayout, r * rowSize, array[r], 0, (int) columns);
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

    for (int r = 0; r < size; ++r) {
      MemorySegment.copy(memorySegment, valueLayout, r * rowSize, array[r], 0, (int) columns);
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

    for (int r = 0; r < size; ++r) {
      MemorySegment.copy(memorySegment, valueLayout, r * rowSize, array[r], 0, (int) columns);
    }
  }

  @Override
  public void toHost(CuVSHostMatrix hostMatrix) {
    var targetMatrix = (CuVSHostMatrixImpl) hostMatrix;

    if (targetMatrix.columns() != this.columns || targetMatrix.size() != this.size) {
      throw new IllegalArgumentException(
          "Source and target matrices must have the same dimensions");
    }
    if (targetMatrix.dataType() != this.dataType) {
      throw new IllegalArgumentException("Source and target matrices must have the same dataType");
    }

    if (this.rowStride <= 0 && targetMatrix.rowStride <= 0) {
      // copy whole matrix
      MemorySegment.copy(this.memorySegment, 0L, targetMatrix.memorySegment, 0L, size * rowSize);
    } else {
      // copy row-by-row
      for (int r = 0; r < size; ++r) {
        MemorySegment.copy(
            this.memorySegment,
            r * rowSize,
            targetMatrix.memorySegment,
            r * targetMatrix.rowSize,
            rowBytes);
      }
    }
  }

  @Override
  public void toDevice(CuVSDeviceMatrix deviceMatrix, CuVSResources cuVSResources) {
    copyMatrix(this, (CuVSMatrixBaseImpl) deviceMatrix, cuVSResources);
  }

  @Override
  public void close() {}

  @Override
  public int get(int row, int col) {
    var rowPitch = rowStride > 0 ? rowStride : columns;
    return (int) accessor$vh.get(memorySegment, 0L, (long) row * rowPitch + col);
  }

  @Override
  public MemorySegment toTensor(Arena arena) {
    var strides = rowStride >= 0 ? new long[] {rowStride, columnStride} : null;
    return prepareTensor(
        arena, memorySegment, new long[] {size, columns}, strides, code(), bits(), kDLCPU());
  }
}
