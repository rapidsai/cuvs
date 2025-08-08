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
import java.util.function.Function;

public class CuVSDeviceMatrixImpl extends CuVSMatrixBaseImpl implements CuVSDeviceMatrix {

  private static final int CHUNK_BYTES =
      32 * 1024 * 1024; // 32MB seems like the optimal size to optimize PCIe bandwidth
  private final long hostBufferBytes;

  private final BufferFillFunction bufferFiller;
  private final Function<CuVSResources, CuVSHostMatrix> toHostFunction;
  private final MemorySegment memcopyStream;

  private long bufferedMatrixRowStart = 0;
  private long bufferedMatrixRowEnd = 0;

  // TODO: can/should we keep resources out this "base" class?
  private final CuVSResources resources;

  @FunctionalInterface
  private interface BufferFillFunction {
    void fill(MemorySegment buffer, long startRow, long bufferBytes);
  }

  private MemorySegment hostBuffer = MemorySegment.NULL;

  protected CuVSDeviceMatrixImpl(
      CuVSResources resources,
      MemorySegment deviceMemorySegment,
      long size,
      long columns,
      DataType dataType,
      ValueLayout valueLayout,
      int copyType) {
    super(deviceMemorySegment, dataType, valueLayout, size, columns);
    this.resources = resources;

    var bufferFillerType = copyType & 0xF;
    switch (bufferFillerType) {
      case 1:
        {
          bufferFiller = this::populateBufferWithCuda2D;
          toHostFunction = this::toHostCuda2D;
          this.memcopyStream = MemorySegment.NULL;
        }
        break;

      case 2:
        {
          bufferFiller = this::populateBufferWithCuda2DAsync;
          toHostFunction = this::toHostCuda2DAsync;
          try (var localArena = Arena.ofConfined()) {
            var streamPointer = localArena.allocate(C_POINTER);
            checkCudaError(cudaStreamCreate(streamPointer), "checkCudaError");
            this.memcopyStream = streamPointer.get(C_POINTER, 0);
          }
        }
        break;

      default:
        {
          bufferFiller = this::populateBufferWithCuvs;
          toHostFunction = this::toHostCuvs;
          this.memcopyStream = MemorySegment.NULL;
        }
    }

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
    // TODO: support stride
    return prepareTensor(
        arena, memorySegment, new long[] {size, columns}, code(), bits(), kDLCUDA(), 1);
  }

  @Override
  public CuVSHostMatrix toHost(CuVSResources resources) {
    return toHostFunction.apply(resources);
  }

  public MemorySegment createBuffer(long bufferBytes) {
    try (var localArena = Arena.ofConfined()) {
      MemorySegment pointer = localArena.allocate(C_POINTER);
      checkCudaError(cudaMallocHost(pointer, bufferBytes), "cudaMallocHost");
      return pointer.get(C_POINTER, 0);
    }
  }

  private void populateBuffer(long startRow) {
    if (hostBuffer == MemorySegment.NULL) {
      //      System.out.println("Creating a buffer of size " + hostBufferBytes);
      hostBuffer = createBuffer(hostBufferBytes);
    }

    bufferFiller.fill(hostBuffer, startRow, hostBufferBytes);
  }

  private void populateBufferWithCuvs(MemorySegment buffer, long startRow, long bufferBytes) {
    try (var localArena = Arena.ofConfined()) {
      long rowBytes = columns * valueLayout.byteSize();
      var endRow = Math.min(startRow + (bufferBytes / rowBytes), size);
      var rowCount = endRow - startRow;

      //      System.out.printf(
      //          Locale.ROOT, "startRow: %d, endRow %d, count: %d\n", startRow, endRow, rowCount);

      // TODO: we need stride information too here (optionally)
      MemorySegment sliceTensor =
          prepareTensor(localArena, MemorySegment.NULL, new long[2], code(), bits(), kDLCUDA(), 1);
      checkCuVSError(
          cuvsMatrixSliceRows(0, toTensor(localArena), startRow, endRow, sliceTensor),
          "cuvsMatrixSliceRows");
      assert DLTensor.shape(DLManagedTensor.dl_tensor(sliceTensor)).get(C_LONG, 0) == rowCount;

      MemorySegment bufferTensor =
          prepareTensor(
              localArena, buffer, new long[] {rowCount, columns}, code(), bits(), kDLCUDA(), 1);

      try (var resourceAccess = resources.access()) {
        checkCuVSError(
            cuvsMatrixCopy(resourceAccess.handle(), sliceTensor, bufferTensor), "cuvsMatrixCopy");
        checkCuVSError(cuvsStreamSync(resourceAccess.handle()), "cuvsStreamSync");

        bufferedMatrixRowStart = startRow;
        bufferedMatrixRowEnd = endRow;
      }
    }
  }

  private void populateBufferWithCuda2D(MemorySegment buffer, long startRow, long bufferBytes) {
    long rowBytes = columns * valueLayout.byteSize();
    var endRow = Math.min(startRow + (bufferBytes / rowBytes), size);
    var rowCount = endRow - startRow;

    var src = memorySegment.asSlice(startRow * rowBytes);
    checkCudaError(
        cudaMemcpy2D(buffer, rowBytes, src, rowBytes, rowBytes, rowCount, cudaMemcpyDefault()),
        "cudaMemcpy2D");

    bufferedMatrixRowStart = startRow;
    bufferedMatrixRowEnd = endRow;
  }

  private void populateBufferWithCuda2DAsync(
      MemorySegment buffer, long startRow, long bufferBytes) {
    long rowBytes = columns * valueLayout.byteSize();
    var endRow = Math.min(startRow + (bufferBytes / rowBytes), size);
    var rowCount = endRow - startRow;

    var src = memorySegment.asSlice(startRow * rowBytes);
    checkCudaError(
        cudaMemcpy2DAsync(
            buffer,
            rowBytes,
            src,
            rowBytes,
            rowBytes,
            rowCount,
            cudaMemcpyDefault(),
            memcopyStream),
        "cudaMemcpy2DAsync");
    checkCudaError(cudaStreamSynchronize(memcopyStream), "cudaStreamSynchronize");

    bufferedMatrixRowStart = startRow;
    bufferedMatrixRowEnd = endRow;
  }

  protected long getMatrixSizeInBytes() {
    return size * columns * valueLayout.byteSize();
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
    MemorySegment sliceTensor =
        prepareTensor(
            localArena, MemorySegment.NULL, new long[] {1, columns}, code(), bits(), kDLCUDA(), 1);
    checkCuVSError(
        cuvsMatrixSliceRows(0, toTensor(localArena), r, r + 1, sliceTensor), "cuvsMatrixSliceRows");

    MemorySegment bufferTensor =
        prepareTensor(
            localArena, tmpRowSegment, new long[] {1, columns}, code(), bits(), kDLCUDA(), 1);
    try (var resourceAccess = resources.access()) {
      checkCuVSError(
          cuvsMatrixCopy(resourceAccess.handle(), sliceTensor, bufferTensor), "cuvsMatrixCopy");
      checkCuVSError(cuvsStreamSync(resourceAccess.handle()), "cuvsStreamSync");
    }
    MemorySegment.copy(tmpRowSegment, valueLayout, 0L, array, 0, (int) columns);
  }

  @Override
  public void close() {
    if (hostBuffer != MemorySegment.NULL) {
      checkCudaError(cudaFreeHost(hostBuffer), "cudaFreeHost");
      hostBuffer = MemorySegment.NULL;
    }
    if (memcopyStream != MemorySegment.NULL) {
      checkCudaError(cudaStreamDestroy(memcopyStream), "cudaStreamDestroy");
    }
  }

  CuVSHostMatrix toHostCuvs(CuVSResources resources) {
    var hostMatrix = new CuVSHostMatrixArenaImpl(size, columns, dataType);
    try (var localArena = Arena.ofConfined()) {
      var hostMatrixTensor = hostMatrix.toTensor(localArena);

      try (var resourceAccess = resources.access()) {
        var cuvsRes = resourceAccess.handle();
        checkCuVSError(cuvsStreamSync(cuvsRes), "cuvsStreamSync");

        var deviceMatrixTensor = toTensor(localArena);
        checkCuVSError(
            cuvsMatrixCopy(cuvsRes, deviceMatrixTensor, hostMatrixTensor), "cuvsMatrixCopy");

        checkCuVSError(cuvsStreamSync(cuvsRes), "cuvsStreamSync");
      }
    }
    return hostMatrix;
  }

  CuVSHostMatrix toHostCuda2D(CuVSResources resources) {
    // TODO: stride
    long rowBytes = columns * valueLayout.byteSize();

    var hostMemory = createBuffer(getMatrixSizeInBytes());
    checkCudaError(
        cudaMemcpy2D(
            hostMemory,
            rowBytes,
            memorySegment,
            rowBytes,
            rowBytes,
            size,
            cudaMemcpyDeviceToHost()),
        "cudaMemcpy2D");

    return new CuVSHostMatrixImpl(hostMemory, size, columns, dataType) {
      @Override
      public void close() {
        super.close();
        checkCudaError(cudaFreeHost(hostMemory), "cudaFreeHost");
      }
    };
  }

  CuVSHostMatrix toHostCuda2DAsync(CuVSResources resources) {
    // TODO: stride
    var hostMatrix = new CuVSHostMatrixArenaImpl(size, columns, dataType);

    long rowBytes = columns * valueLayout.byteSize();

    checkCudaError(
        cudaMemcpy2DAsync(
            hostMatrix.memorySegment(),
            rowBytes,
            memorySegment,
            rowBytes,
            rowBytes,
            size,
            cudaMemcpyDeviceToHost(),
            memcopyStream),
        "cudaMemcpy2DAsync");
    checkCudaError(cudaStreamSynchronize(memcopyStream), "cudaStreamSynchronize");
    return hostMatrix;
  }
}
