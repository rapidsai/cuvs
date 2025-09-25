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

import static com.nvidia.cuvs.internal.common.Util.*;
import static com.nvidia.cuvs.internal.panama.headers_h.*;

import com.nvidia.cuvs.*;
import com.nvidia.cuvs.internal.common.PinnedMemoryBuffer;
import com.nvidia.cuvs.internal.panama.DLManagedTensor;
import com.nvidia.cuvs.internal.panama.DLTensor;
import java.lang.foreign.*;

public class CuVSDeviceMatrixImpl extends CuVSMatrixBaseImpl implements CuVSDeviceMatrix {

  private long bufferedMatrixRowStart = 0;
  private long bufferedMatrixRowEnd = 0;

  private final CuVSResources resources;

  private final long rowStride;
  private final long columnStride;

  private final PinnedMemoryBuffer hostBuffer;

  protected CuVSDeviceMatrixImpl(
      CuVSResources resources,
      MemorySegment deviceMemorySegment,
      long size,
      long columns,
      DataType dataType,
      ValueLayout valueLayout) {
    this(resources, deviceMemorySegment, size, columns, -1, -1, dataType, valueLayout);
  }

  protected CuVSDeviceMatrixImpl(
      CuVSResources resources,
      MemorySegment deviceMemorySegment,
      long size,
      long columns,
      long rowStride,
      long columnStride,
      DataType dataType,
      ValueLayout valueLayout) {
    super(deviceMemorySegment, dataType, valueLayout, size, columns);
    this.resources = resources;
    this.rowStride = rowStride;
    this.columnStride = columnStride;
    this.hostBuffer = new PinnedMemoryBuffer(size, columns, valueLayout);
  }

  @Override
  public MemorySegment toTensor(Arena arena) {
    var strides = rowStride >= 0 ? new long[] {rowStride, columnStride} : null;
    return prepareTensor(
        arena, memorySegment, new long[] {size, columns}, strides, code(), bits(), kDLCUDA());
  }

  private void populateBuffer(long startRow) {
    try (var localArena = Arena.ofConfined()) {
      long rowBytes = columns * valueLayout.byteSize();
      var endRow = Math.min(startRow + (hostBuffer.size() / rowBytes), size);
      var rowCount = endRow - startRow;

      //      System.out.printf(
      //          Locale.ROOT, "startRow: %d, endRow %d, count: %d\n", startRow, endRow, rowCount);

      MemorySegment sliceManagedTensor = DLManagedTensor.allocate(localArena);
      DLManagedTensor.dl_tensor(sliceManagedTensor, DLTensor.allocate(localArena));

      checkCuVSError(
          cuvsMatrixSliceRows(0, toTensor(localArena), startRow, endRow, sliceManagedTensor),
          "cuvsMatrixSliceRows");
      assert DLTensor.shape(DLManagedTensor.dl_tensor(sliceManagedTensor)).get(C_LONG, 0)
          == rowCount;
      assert DLTensor.shape(DLManagedTensor.dl_tensor(sliceManagedTensor)).getAtIndex(C_LONG, 1)
          == columns;

      MemorySegment bufferTensor =
          prepareTensor(
              localArena,
              hostBuffer.address(),
              new long[] {rowCount, columns},
              code(),
              bits(),
              kDLCPU());

      try (var resourceAccess = resources.access()) {
        checkCuVSError(
            cuvsMatrixCopy(resourceAccess.handle(), sliceManagedTensor, bufferTensor),
            "cuvsMatrixCopy");
        checkCuVSError(cuvsStreamSync(resourceAccess.handle()), "cuvsStreamSync");

        bufferedMatrixRowStart = startRow;
        bufferedMatrixRowEnd = endRow;
      }
    }
  }

  @Override
  public RowView getRow(long row) {
    if (row < bufferedMatrixRowStart || row >= bufferedMatrixRowEnd) {
      populateBuffer(row);
    }
    var valueByteSize = valueLayout.byteSize();
    var startRow = row - bufferedMatrixRowStart;

    return new SliceRowView(
        hostBuffer.address().asSlice(startRow * columns * valueByteSize, columns * valueByteSize),
        columns,
        valueLayout,
        dataType,
        valueByteSize);
  }

  @Override
  public void toArray(int[][] array) {
    assert dataType == DataType.INT || dataType == DataType.UINT;
    assert (array.length >= size) : "Input array is not large enough";
    assert (array.length == 0 || array[0].length >= columns) : "Input array is not large enough";
    try (var localArena = Arena.ofConfined()) {
      var rowBytes = columns * valueLayout.byteSize();
      var tmpRowSegment = localArena.allocate(rowBytes);
      for (int r = 0; r < size; ++r) {
        copyRow(array[r], localArena, r, tmpRowSegment);
      }
    }
  }

  @Override
  public void toArray(float[][] array) {
    assert dataType == DataType.FLOAT;
    assert (array.length >= size) : "Input array is not large enough";
    assert (array.length == 0 || array[0].length >= columns) : "Input array is not large enough";
    try (var localArena = Arena.ofConfined()) {
      var rowBytes = columns * valueLayout.byteSize();
      var tmpRowSegment = localArena.allocate(rowBytes);
      for (int r = 0; r < size; ++r) {
        copyRow(array[r], localArena, r, tmpRowSegment);
      }
    }
  }

  @Override
  public void toArray(byte[][] array) {
    assert dataType == DataType.BYTE;
    assert (array.length >= size) : "Input array is not large enough";
    assert (array.length == 0 || array[0].length >= columns) : "Input array is not large enough";
    try (var localArena = Arena.ofConfined()) {
      var rowSegmentLayout = MemoryLayout.sequenceLayout(columns, valueLayout);
      var tmpRowSegment = localArena.allocate(rowSegmentLayout);
      for (int r = 0; r < size; ++r) {
        copyRow(array[r], localArena, r, tmpRowSegment);
      }
    }
  }

  private void copyRow(Object array, Arena localArena, int r, MemorySegment tmpRowSegment) {
    MemorySegment sliceManagedTensor = DLManagedTensor.allocate(localArena);
    DLManagedTensor.dl_tensor(sliceManagedTensor, DLTensor.allocate(localArena));

    checkCuVSError(
        cuvsMatrixSliceRows(0, toTensor(localArena), r, r + 1, sliceManagedTensor),
        "cuvsMatrixSliceRows");

    MemorySegment bufferTensor =
        prepareTensor(
            localArena, tmpRowSegment, new long[] {1, columns}, code(), bits(), kDLCUDA());
    try (var resourceAccess = resources.access()) {
      checkCuVSError(
          cuvsMatrixCopy(resourceAccess.handle(), sliceManagedTensor, bufferTensor),
          "cuvsMatrixCopy");
      checkCuVSError(cuvsStreamSync(resourceAccess.handle()), "cuvsStreamSync");
    }
    MemorySegment.copy(tmpRowSegment, valueLayout, 0L, array, 0, (int) columns);
  }

  @Override
  public void toHost(CuVSHostMatrix hostMatrix) {
    if (hostMatrix.columns() != columns || hostMatrix.size() != size) {
      throw new IllegalArgumentException("[hostMatrix] must have the same dimensions");
    }
    if (hostMatrix.dataType() != dataType) {
      throw new IllegalArgumentException("[hostMatrix] must have the same dataType");
    }
    try (var localArena = Arena.ofConfined()) {
      var hostMatrixTensor = ((CuVSMatrixInternal) hostMatrix).toTensor(localArena);

      try (var resourceAccess = resources.access()) {
        var cuvsRes = resourceAccess.handle();
        var deviceMatrixTensor = toTensor(localArena);
        checkCuVSError(
            cuvsMatrixCopy(cuvsRes, deviceMatrixTensor, hostMatrixTensor), "cuvsMatrixCopy");
        checkCuVSError(cuvsStreamSync(cuvsRes), "cuvsStreamSync");
      }
    }
  }

  @Override
  public CuVSDeviceMatrix toDevice(CuVSResources resources) {
    return new CuVSDeviceMatrixDelegate(this);
  }

  @Override
  public void toDevice(CuVSDeviceMatrix targetMatrix, CuVSResources cuVSResources) {
    copyMatrix(this, (CuVSMatrixInternal) targetMatrix, cuVSResources);
  }

  @Override
  public void close() {
    hostBuffer.close();
  }

  private static class CuVSDeviceMatrixDelegate implements CuVSDeviceMatrix, CuVSMatrixInternal {
    private final CuVSDeviceMatrixImpl deviceMatrix;

    private CuVSDeviceMatrixDelegate(CuVSDeviceMatrixImpl deviceMatrix) {
      this.deviceMatrix = deviceMatrix;
    }

    @Override
    public long size() {
      return deviceMatrix.size();
    }

    @Override
    public long columns() {
      return deviceMatrix.columns();
    }

    @Override
    public DataType dataType() {
      return deviceMatrix.dataType();
    }

    @Override
    public RowView getRow(long row) {
      return deviceMatrix.getRow(row);
    }

    @Override
    public void toArray(int[][] array) {
      deviceMatrix.toArray(array);
    }

    @Override
    public void toArray(float[][] array) {
      deviceMatrix.toArray(array);
    }

    @Override
    public void toArray(byte[][] array) {
      deviceMatrix.toArray(array);
    }

    @Override
    public void toHost(CuVSHostMatrix hostMatrix) {
      deviceMatrix.toHost(hostMatrix);
    }

    @Override
    public void toDevice(CuVSDeviceMatrix deviceMatrix, CuVSResources cuVSResources) {
      deviceMatrix.toDevice(deviceMatrix, cuVSResources);
    }

    @Override
    public CuVSDeviceMatrix toDevice(CuVSResources cuVSResources) {
      return this;
    }

    @Override
    public MemorySegment memorySegment() {
      return deviceMatrix.memorySegment();
    }

    @Override
    public ValueLayout valueLayout() {
      return deviceMatrix.valueLayout();
    }

    @Override
    public int bits() {
      return deviceMatrix.bits();
    }

    @Override
    public int code() {
      return 0;
    }

    @Override
    public MemorySegment toTensor(Arena arena) {
      return deviceMatrix.toTensor(arena);
    }

    @Override
    public void close() {
      // Do nothing
    }
  }
}
