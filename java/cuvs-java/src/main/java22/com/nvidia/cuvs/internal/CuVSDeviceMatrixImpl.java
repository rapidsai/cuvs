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

import static com.nvidia.cuvs.internal.common.LinkerHelper.C_POINTER;
import static com.nvidia.cuvs.internal.common.Util.*;
import static com.nvidia.cuvs.internal.panama.headers_h.*;

import com.nvidia.cuvs.*;
import com.nvidia.cuvs.internal.panama.DLManagedTensor;
import com.nvidia.cuvs.internal.panama.DLTensor;
import java.lang.foreign.*;

public class CuVSDeviceMatrixImpl extends CuVSMatrixBaseImpl implements CuVSDeviceMatrix {

  private static final int CHUNK_BYTES =
      8 * 1024 * 1024; // Based on benchmarks, 8MB seems the minimum size to optimize PCIe bandwidth
  private final long hostBufferBytes;

  private long bufferedMatrixRowStart = 0;
  private long bufferedMatrixRowEnd = 0;

  private final CuVSResources resources;

  private final long rowStride;
  private final long columnStride;

  private MemorySegment hostBuffer = MemorySegment.NULL;

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

    long rowBytes = columns * valueLayout.byteSize();
    long matrixBytes = size * rowBytes;
    if (matrixBytes < CHUNK_BYTES) {
      this.hostBufferBytes = matrixBytes;
    } else if (rowBytes > CHUNK_BYTES) {
      // We need to buffer at least one row at time
      this.hostBufferBytes = rowBytes;
    } else {
      var rowCount = (CHUNK_BYTES / rowBytes);
      this.hostBufferBytes = rowBytes * rowCount;
    }
  }

  @Override
  public MemorySegment toTensor(Arena arena) {
    var strides = rowStride >= 0 ? new long[] {rowStride, columnStride} : null;
    return prepareTensor(
        arena, memorySegment, new long[] {size, columns}, strides, code(), bits(), kDLCUDA());
  }

  private static MemorySegment createPinnedBuffer(long bufferBytes) {
    try (var localArena = Arena.ofConfined()) {
      MemorySegment pointer = localArena.allocate(C_POINTER);
      checkCudaError(cudaMallocHost(pointer, bufferBytes), "cudaMallocHost");
      return pointer.get(C_POINTER, 0);
    }
  }

  private static void destroyPinnedBuffer(MemorySegment bufferSegment) {
    checkCudaError(cudaFreeHost(bufferSegment), "cudaFreeHost");
  }

  private void populateBuffer(long startRow) {
    if (hostBuffer == MemorySegment.NULL) {
      //      System.out.println("Creating a buffer of size " + hostBufferBytes);
      hostBuffer = createPinnedBuffer(hostBufferBytes);
    }

    try (var localArena = Arena.ofConfined()) {
      long rowBytes = columns * valueLayout.byteSize();
      var endRow = Math.min(startRow + (hostBufferBytes / rowBytes), size);
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
              localArena, hostBuffer, new long[] {rowCount, columns}, code(), bits(), kDLCPU());

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
        hostBuffer.asSlice(startRow * columns * valueByteSize, columns * valueByteSize),
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
      var hostMatrixTensor = ((CuVSHostMatrixImpl) hostMatrix).toTensor(localArena);

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
  public void toDevice(CuVSDeviceMatrix targetMatrix, CuVSResources cuVSResources) {
    copyMatrix(this, (CuVSMatrixBaseImpl) targetMatrix, cuVSResources);
  }

  @Override
  public void close() {
    if (hostBuffer != MemorySegment.NULL) {
      destroyPinnedBuffer(hostBuffer);
      hostBuffer = MemorySegment.NULL;
    }
  }
}
