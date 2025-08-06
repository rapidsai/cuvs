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
import static com.nvidia.cuvs.internal.common.Util.checkCuVSError;
import static com.nvidia.cuvs.internal.common.Util.prepareTensor;
import static com.nvidia.cuvs.internal.panama.headers_h.*;

import com.nvidia.cuvs.*;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

public class CuVSDeviceMatrixImpl extends CuVSMatrixBaseImpl implements CuVSDeviceMatrix {

  private static final int CHUNK_BYTES =
      32 * 1024 * 1024; // 32MB seems like the optimal size to optimize PCIe bandwidth
  private final long hostBufferBytes;

  private long bufferedMatrixRowStart = 0;
  private long bufferedMatrixRowEnd = 0;

  // TODO: can/should we keep resources out this "base" class?
  private final CuVSResources resources;

  protected CuVSDeviceMatrixImpl(
      CuVSResources resources,
      MemorySegment deviceMemorySegment,
      long size,
      long columns,
      DataType dataType) {
    super(deviceMemorySegment, dataType, size, columns);
    this.resources = resources;

    long rowBytes = columns * dataType.bytes();
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
    return prepareTensor(
        arena, memorySegment, new long[] {size, columns}, code(), bits(), kDLCUDA(), 1);
  }

  private MemorySegment hostBuffer = MemorySegment.NULL;

  private void createBuffer() {
    try (var localArena = Arena.ofConfined()) {
      MemorySegment pointer = localArena.allocate(C_POINTER);
      cudaMallocHost(pointer, hostBufferBytes);
      hostBuffer = pointer.get(C_POINTER, 0);
    }
  }

  private void populateBuffer(long startRow) {
    if (hostBuffer == MemorySegment.NULL) {
      createBuffer();
    }
    try (var localArena = Arena.ofConfined()) {
      long rowBytes = columns * dataType.bytes();
      var endRow = Math.min(startRow + (hostBufferBytes / rowBytes), size - 1);
      var rowCount = endRow - startRow + 1;
      MemorySegment sliceTensor =
          prepareTensor(
              localArena,
              MemorySegment.NULL,
              new long[] {rowCount, columns},
              code(),
              bits(),
              kDLCUDA(),
              1);
      cuvsMatrixSliceRows(0, toTensor(localArena), startRow, endRow, sliceTensor);

      MemorySegment bufferTensor =
          prepareTensor(
              localArena, hostBuffer, new long[] {rowCount, columns}, code(), bits(), kDLCUDA(), 1);

      try (var resourceAccess = resources.access()) {
        cuvsMatrixCopy(resourceAccess.handle(), sliceTensor, bufferTensor);

        bufferedMatrixRowStart = startRow;
        bufferedMatrixRowEnd = endRow;
      }
    }
  }

  protected long getMatrixSizeInBytes() {
    return size * columns * dataType.bytes();
  }

  @Override
  public RowView getRow(long row) {
    if (row < bufferedMatrixRowStart || row >= bufferedMatrixRowEnd) {
      populateBuffer(row);
    }

    var valueLayout = valueLayoutFromType(dataType);
    var valueByteSize = valueLayout.byteSize();
    return new SliceRowView(
        memorySegment.asSlice(row * columns * valueByteSize, columns * valueByteSize),
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
    for (int r = 0; r < size; ++r) {
      try (var localArena = Arena.ofConfined()) {
        MemorySegment sliceTensor =
            prepareTensor(
                localArena,
                MemorySegment.NULL,
                new long[] {1, columns},
                code(),
                bits(),
                kDLCUDA(),
                1);
        cuvsMatrixSliceRows(0, toTensor(localArena), r, r + 1, sliceTensor);

        MemorySegment bufferTensor =
            prepareTensor(
                localArena,
                MemorySegment.ofArray(array[r]),
                new long[] {1, columns},
                code(),
                bits(),
                kDLCUDA(),
                1);
        try (var resourceAccess = resources.access()) {
          cuvsMatrixCopy(resourceAccess.handle(), sliceTensor, bufferTensor);
        }
      }
    }
  }

  @Override
  public void toArray(float[][] array) {
    assert dataType == DataType.FLOAT;
    assert (array.length >= size) : "Input array is not large enough";
    assert (array.length == 0 || array[0].length >= columns) : "Input array is not large enough";
    for (int r = 0; r < size; ++r) {
      try (var localArena = Arena.ofConfined()) {
        MemorySegment sliceTensor =
            prepareTensor(
                localArena,
                MemorySegment.NULL,
                new long[] {1, columns},
                code(),
                bits(),
                kDLCUDA(),
                1);
        cuvsMatrixSliceRows(0, toTensor(localArena), r, r + 1, sliceTensor);

        MemorySegment bufferTensor =
            prepareTensor(
                localArena,
                MemorySegment.ofArray(array[r]),
                new long[] {1, columns},
                code(),
                bits(),
                kDLCUDA(),
                1);

        try (var resourceAccess = resources.access()) {
          cuvsMatrixCopy(resourceAccess.handle(), sliceTensor, bufferTensor);
        }
      }
    }
  }

  @Override
  public void toArray(byte[][] array) {
    assert dataType == DataType.BYTE;
    assert (array.length >= size) : "Input array is not large enough";
    assert (array.length == 0 || array[0].length >= columns) : "Input array is not large enough";
    for (int r = 0; r < size; ++r) {
      try (var localArena = Arena.ofConfined()) {
        MemorySegment sliceTensor =
            prepareTensor(
                localArena,
                MemorySegment.NULL,
                new long[] {1, columns},
                code(),
                bits(),
                kDLCUDA(),
                1);
        cuvsMatrixSliceRows(0, toTensor(localArena), r, r + 1, sliceTensor);

        MemorySegment bufferTensor =
            prepareTensor(
                localArena,
                MemorySegment.ofArray(array[r]),
                new long[] {1, columns},
                code(),
                bits(),
                kDLCUDA(),
                1);
        try (var resourceAccess = resources.access()) {
          cuvsMatrixCopy(resourceAccess.handle(), sliceTensor, bufferTensor);
        }
      }
    }
  }

  @Override
  public void close() {
    if (hostBuffer != MemorySegment.NULL) {
      cudaFreeHost(hostBuffer);
      hostBuffer = MemorySegment.NULL;
    }
  }

  CuVSHostMatrix toHost(CuVSResources resources) {
    var graph = new CuVSHostMatrixArenaImpl(size, columns, dataType);
    try (var localArena = Arena.ofConfined()) {
      var graphHostTensor = graph.toTensor(localArena);

      try (var resourceAccess = resources.access()) {
        var cuvsRes = resourceAccess.handle();
        checkCuVSError(cuvsStreamSync(cuvsRes), "cuvsStreamSync");

        var graphDeviceTensor = toTensor(localArena);
        checkCuVSError(
            cuvsMatrixCopy(cuvsRes, graphDeviceTensor, graphHostTensor), "cuvsMatrixCopy");

        checkCuVSError(cuvsStreamSync(cuvsRes), "cuvsStreamSync");
      }
    }
    return graph;
  }
}
