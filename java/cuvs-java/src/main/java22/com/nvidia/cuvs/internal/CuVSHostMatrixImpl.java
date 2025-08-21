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

import static com.nvidia.cuvs.internal.common.LinkerHelper.C_CHAR;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_FLOAT;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_INT;
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
  private final ValueLayout valueLayout;
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
    super(memorySegment, dataType, size, columns);
    this.accessor$vh = sequenceLayout.varHandle(MemoryLayout.PathElement.sequenceElement());
    this.valueLayout = valueLayout;
  }

  protected static ValueLayout valueLayoutFromType(DataType dataType) {
    return switch (dataType) {
      case FLOAT -> C_FLOAT;
      case INT, UINT -> C_INT;
      case BYTE -> C_CHAR;
    };
  }

  protected static SequenceLayout sequenceLayoutFromType(
      long size, long columns, DataType dataType) {
    return MemoryLayout.sequenceLayout(size * columns, valueLayoutFromType(dataType))
        .withByteAlignment(32);
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
    assert (dataType == DataType.UINT || dataType == DataType.INT);
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

  public ValueLayout valueLayout() {
    return valueLayout;
  }

  @Override
  public MemorySegment toTensor(Arena arena) {
    return prepareTensor(
        arena, memorySegment, new long[] {size, columns}, code(), bits(), kDLCPU(), 1);
  }

  private static class SliceRowView implements RowView {
    private final MemorySegment memorySegment;
    private final long size;
    private final ValueLayout valueLayout;
    private final DataType dataType;
    private final long valueByteSize;

    SliceRowView(
        MemorySegment slice,
        long size,
        ValueLayout valueLayout,
        DataType dataType,
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
      assert dataType == DataType.FLOAT;
      return memorySegment.get((ValueLayout.OfFloat) valueLayout, index * valueByteSize);
    }

    @Override
    public byte getAsByte(long index) {
      assert dataType == DataType.BYTE;
      return memorySegment.get((ValueLayout.OfByte) valueLayout, index * valueByteSize);
    }

    @Override
    public int getAsInt(long index) {
      assert dataType == DataType.INT;
      return memorySegment.get((ValueLayout.OfInt) valueLayout, index * valueByteSize);
    }

    @Override
    public void toArray(int[] array) {
      assert (array.length >= size) : "Input array is not large enough";
      assert dataType == DataType.INT;
      MemorySegment.copy(memorySegment, valueLayout, 0, array, 0, (int) size);
    }

    @Override
    public void toArray(float[] array) {
      assert (array.length >= size) : "Input array is not large enough";
      assert dataType == DataType.FLOAT;
      MemorySegment.copy(memorySegment, valueLayout, 0, array, 0, (int) size);
    }

    @Override
    public void toArray(byte[] array) {
      assert (array.length >= size) : "Input array is not large enough";
      assert dataType == DataType.BYTE;
      MemorySegment.copy(memorySegment, valueLayout, 0, array, 0, (int) size);
    }
  }
}
