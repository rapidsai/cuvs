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
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_FLOAT_BYTE_SIZE;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_POINTER;
import static com.nvidia.cuvs.internal.common.Util.CudaMemcpyKind.*;
import static com.nvidia.cuvs.internal.common.Util.allocateRMMSegment;
import static com.nvidia.cuvs.internal.common.Util.checkCuVSError;
import static com.nvidia.cuvs.internal.common.Util.cudaMemcpy;
import static com.nvidia.cuvs.internal.common.Util.prepareTensor;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsRMMFree;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsScalarQuantizerCreate;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsScalarQuantizerDestroy;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsScalarQuantizerInverseTransform;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsScalarQuantizerParamsCreate;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsScalarQuantizerParamsDestroy;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsScalarQuantizerParams_t;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsScalarQuantizerTrain;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsScalarQuantizerTransform;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsScalarQuantizer_t;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsStreamSync;
import static com.nvidia.cuvs.internal.panama.headers_h.kDLCPU;
import static com.nvidia.cuvs.internal.panama.headers_h.kDLCUDA;
import static com.nvidia.cuvs.internal.panama.headers_h.kDLFloat;
import static com.nvidia.cuvs.internal.panama.headers_h.kDLInt;

import com.nvidia.cuvs.CuVSMatrix;
import com.nvidia.cuvs.CuVSMatrix.DataType;
import com.nvidia.cuvs.CuVSMatrix.MemoryKind;
import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.internal.panama.cuvsScalarQuantizerParams;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Objects;

/**
 * Implementation of Scalar8BitQuantizer using Panama Foreign Function Interface.
 */
public class Scalar8BitQuantizerImpl {

  private static final long C_BYTE_SIZE = C_CHAR.byteSize();

  private final CuVSResources resources;
  private final MemorySegment quantizerSegment;
  private boolean destroyed;

  private Scalar8BitQuantizerImpl(CuVSResources resources, MemorySegment quantizerSegment) {
    this.resources = resources;
    this.quantizerSegment = quantizerSegment;
  }

  private void checkNotDestroyed() {
    if (destroyed) {
      throw new IllegalStateException("Scalar8BitQuantizer has been destroyed");
    }
  }

  public void destroy() throws Throwable {
    checkNotDestroyed();
    try {
      int returnValue = cuvsScalarQuantizerDestroy(quantizerSegment);
      checkCuVSError(returnValue, "cuvsScalarQuantizerDestroy");
    } finally {
      destroyed = true;
    }
  }

  public void close() throws Throwable {
    destroy();
  }

  public CuVSMatrix transform(CuVSMatrix dataset) throws Throwable {
    checkNotDestroyed();

    if (dataset.dataType() != DataType.FLOAT) {
      throw new IllegalArgumentException(
          "Scalar8BitQuantizer requires FLOAT input, got " + dataset.dataType());
    }

    try (Arena resultArena = Arena.ofShared();
        var localArena = Arena.ofConfined();
        var resourcesAccessor = resources.access()) {

      long rows = dataset.size();
      long cols = dataset.columns();

      MemorySegment datasetMemSegment = ((CuVSMatrixBaseImpl) dataset).memorySegment();
      long cuvsResourcesPtr = resourcesAccessor.handle();

      // TODO: Currently all datasets are HOST memory kind, preventing testing of GPU quantization
      // flow.
      // Future improvement: Support GPU-native dataset creation in Java API to enable direct GPU
      // quantization testing.
      if (dataset.memoryKind() == MemoryKind.HOST) {
        return performTransformHost(
            cuvsResourcesPtr, quantizerSegment, localArena, rows, cols, datasetMemSegment);
      }

      return performTransformGPU(
          cuvsResourcesPtr,
          quantizerSegment,
          localArena,
          resultArena,
          rows,
          cols,
          datasetMemSegment);
    } catch (Throwable t) {
      throw t;
    }
  }

  /**
   * GPU transform with buffered processing for large datasets.
   */
  private CuVSMatrix performTransformGPU(
      long cuvsResourcesPtr,
      MemorySegment quantizerSeg,
      Arena localArena,
      Arena resultArena,
      long rows,
      long cols,
      MemorySegment datasetMemSegment)
      throws Throwable {

    final long BUFFER_SIZE_BYTES = 1024 * 1024;
    final long FLOAT_SIZE = C_FLOAT_BYTE_SIZE;
    final long FLOATS_PER_BUFFER = BUFFER_SIZE_BYTES / FLOAT_SIZE;
    final long ROWS_PER_BUFFER = Math.max(1, FLOATS_PER_BUFFER / cols);

    CuVSMatrix result = new CuVSHostMatrixArenaImpl(rows, cols, DataType.BYTE);

    for (long startRow = 0; startRow < rows; startRow += ROWS_PER_BUFFER) {
      long endRow = Math.min(startRow + ROWS_PER_BUFFER, rows);
      long chunkRows = endRow - startRow;

      long datasetBytes = FLOAT_SIZE * chunkRows * cols;
      long outputBytes = C_BYTE_SIZE * chunkRows * cols;

      MemorySegment datasetPtr = allocateRMMSegment(cuvsResourcesPtr, datasetBytes);
      MemorySegment outputPtr = allocateRMMSegment(cuvsResourcesPtr, outputBytes);

      try {
        MemorySegment chunkData =
            datasetMemSegment.asSlice(startRow * cols * FLOAT_SIZE, datasetBytes);
        cudaMemcpy(datasetPtr, chunkData, datasetBytes, INFER_DIRECTION);

        long datasetShape[] = {chunkRows, cols};
        MemorySegment datasetTensor =
            prepareTensor(localArena, datasetPtr, datasetShape, kDLFloat(), 32, kDLCUDA(), 1);
        MemorySegment outputTensor =
            prepareTensor(localArena, outputPtr, datasetShape, kDLInt(), 8, kDLCUDA(), 1);

        int returnValue =
            cuvsScalarQuantizerTransform(
                cuvsResourcesPtr, quantizerSeg, datasetTensor, outputTensor);
        checkCuVSError(returnValue, "cuvsScalarQuantizerTransform");

        returnValue = cuvsStreamSync(cuvsResourcesPtr);
        checkCuVSError(returnValue, "cuvsStreamSync");

        MemorySegment resultChunk =
            ((CuVSMatrixBaseImpl) result).memorySegment().asSlice(startRow * cols, outputBytes);
        cudaMemcpy(resultChunk, outputPtr, outputBytes, DEVICE_TO_HOST);

      } finally {
        int returnValue = cuvsRMMFree(cuvsResourcesPtr, datasetPtr, datasetBytes);
        checkCuVSError(returnValue, "cuvsRMMFree");
        returnValue = cuvsRMMFree(cuvsResourcesPtr, outputPtr, outputBytes);
        checkCuVSError(returnValue, "cuvsRMMFree");
      }
    }

    if (result.dataType() != DataType.BYTE) {
      throw new IllegalStateException(
          "Expected BYTE output from scalar quantization, got " + result.dataType());
    }

    return result;
  }

  private MemorySegment buildMemorySegmentFromBytes(Arena arena, byte[][] data) {
    long rows = data.length;
    long cols = rows > 0 ? data[0].length : 0;
    long totalBytes = rows * cols * C_BYTE_SIZE;

    MemorySegment segment = arena.allocate(totalBytes);
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        segment.set(C_CHAR, (i * cols + j) * C_BYTE_SIZE, data[i][j]);
      }
    }
    return segment;
  }

  public static Scalar8BitQuantizerImpl create(CuVSResources resources, CuVSMatrix trainingDataset)
      throws Throwable {
    Objects.requireNonNull(resources);
    Objects.requireNonNull(trainingDataset);

    try (var localArena = Arena.ofConfined();
        var resourcesAccessor = resources.access()) {

      long rows = trainingDataset.size();
      long cols = trainingDataset.columns();
      MemorySegment datasetMemSegment = ((CuVSMatrixBaseImpl) trainingDataset).memorySegment();
      long cuvsResourcesPtr = resourcesAccessor.handle();

      MemorySegment paramsSegment = localArena.allocate(cuvsScalarQuantizerParams_t);
      int returnValue = cuvsScalarQuantizerParamsCreate(paramsSegment);
      checkCuVSError(returnValue, "cuvsScalarQuantizerParamsCreate");

      MemorySegment paramsPtr = paramsSegment.get(C_POINTER, 0);
      cuvsScalarQuantizerParams.quantile(paramsPtr, 0.99f);

      MemorySegment quantizerSegment = localArena.allocate(cuvsScalarQuantizer_t);
      returnValue = cuvsScalarQuantizerCreate(quantizerSegment);
      checkCuVSError(returnValue, "cuvsScalarQuantizerCreate");

      MemorySegment quantizerPtr = quantizerSegment.get(C_POINTER, 0);

      if (trainingDataset.memoryKind() == MemoryKind.HOST) {
        long datasetShape[] = {rows, cols};
        MemorySegment datasetTensor =
            prepareTensor(localArena, datasetMemSegment, datasetShape, kDLFloat(), 32, kDLCPU(), 1);

        returnValue =
            cuvsScalarQuantizerTrain(cuvsResourcesPtr, paramsPtr, datasetTensor, quantizerPtr);
        checkCuVSError(returnValue, "cuvsScalarQuantizerTrain");
      } else {
        performGPUTraining(
            cuvsResourcesPtr, paramsPtr, quantizerPtr, localArena, rows, cols, datasetMemSegment);
      }

      returnValue = cuvsScalarQuantizerParamsDestroy(paramsPtr);
      checkCuVSError(returnValue, "cuvsScalarQuantizerParamsDestroy");

      return new Scalar8BitQuantizerImpl(resources, quantizerPtr);
    }
  }

  public CuVSMatrix inverseTransform(CuVSMatrix quantizedData) throws Throwable {
    checkNotDestroyed();

    if (quantizedData.dataType() != DataType.BYTE) {
      throw new IllegalArgumentException(
          "Inverse transform requires BYTE input, got " + quantizedData.dataType());
    }

    try (Arena resultArena = Arena.ofShared();
        var localArena = Arena.ofConfined();
        var resourcesAccessor = resources.access()) {

      long rows = quantizedData.size();
      long cols = quantizedData.columns();
      MemorySegment quantizedMemSegment = ((CuVSMatrixBaseImpl) quantizedData).memorySegment();
      long cuvsResourcesPtr = resourcesAccessor.handle();

      // TODO: Currently all datasets are HOST memory kind, preventing testing of GPU quantization
      // flow.
      // Future improvement: Support GPU-native dataset creation in Java API to enable direct GPU
      // quantization testing.
      if (quantizedData.memoryKind() == MemoryKind.HOST) {
        return performInverseTransformHost(
            cuvsResourcesPtr, quantizerSegment, localArena, rows, cols, quantizedMemSegment);
      }

      return performInverseTransformGPU(
          cuvsResourcesPtr,
          quantizerSegment,
          localArena,
          resultArena,
          rows,
          cols,
          quantizedMemSegment);
    }
  }

  private CuVSMatrix performInverseTransformGPU(
      long cuvsResourcesPtr,
      MemorySegment quantizerSeg,
      Arena localArena,
      Arena resultArena,
      long rows,
      long cols,
      MemorySegment quantizedMemSegment)
      throws Throwable {

    final long BUFFER_SIZE_BYTES = 1024 * 1024;
    final long BYTE_SIZE = C_BYTE_SIZE;
    final long BYTES_PER_BUFFER = BUFFER_SIZE_BYTES;
    final long ROWS_PER_BUFFER = Math.max(1, BYTES_PER_BUFFER / (cols * BYTE_SIZE));

    CuVSMatrix result = new CuVSHostMatrixArenaImpl(rows, cols, DataType.FLOAT);

    for (long startRow = 0; startRow < rows; startRow += ROWS_PER_BUFFER) {
      long endRow = Math.min(startRow + ROWS_PER_BUFFER, rows);
      long chunkRows = endRow - startRow;

      long quantizedBytes = BYTE_SIZE * chunkRows * cols;
      long outputBytes = C_FLOAT_BYTE_SIZE * chunkRows * cols;

      MemorySegment quantizedPtr = allocateRMMSegment(cuvsResourcesPtr, quantizedBytes);
      MemorySegment outputPtr = allocateRMMSegment(cuvsResourcesPtr, outputBytes);

      try {
        MemorySegment chunkData =
            quantizedMemSegment.asSlice(startRow * cols * BYTE_SIZE, quantizedBytes);
        cudaMemcpy(quantizedPtr, chunkData, quantizedBytes, INFER_DIRECTION);

        long datasetShape[] = {chunkRows, cols};
        MemorySegment quantizedTensor =
            prepareTensor(localArena, quantizedPtr, datasetShape, kDLInt(), 8, kDLCUDA(), 1);
        MemorySegment outputTensor =
            prepareTensor(localArena, outputPtr, datasetShape, kDLFloat(), 32, kDLCUDA(), 1);

        int returnValue =
            cuvsScalarQuantizerInverseTransform(
                cuvsResourcesPtr, quantizerSeg, quantizedTensor, outputTensor);
        checkCuVSError(returnValue, "cuvsScalarQuantizerInverseTransform");

        returnValue = cuvsStreamSync(cuvsResourcesPtr);
        checkCuVSError(returnValue, "cuvsStreamSync");

        MemorySegment resultChunk =
            ((CuVSMatrixBaseImpl) result)
                .memorySegment()
                .asSlice(startRow * cols * C_FLOAT_BYTE_SIZE, outputBytes);
        cudaMemcpy(resultChunk, outputPtr, outputBytes, DEVICE_TO_HOST);

      } finally {
        int returnValue = cuvsRMMFree(cuvsResourcesPtr, quantizedPtr, quantizedBytes);
        checkCuVSError(returnValue, "cuvsRMMFree");
        returnValue = cuvsRMMFree(cuvsResourcesPtr, outputPtr, outputBytes);
        checkCuVSError(returnValue, "cuvsRMMFree");
      }
    }

    return result;
  }

  private static void performGPUTraining(
      long cuvsResourcesPtr,
      MemorySegment paramsPtr,
      MemorySegment quantizerPtr,
      Arena localArena,
      long rows,
      long cols,
      MemorySegment datasetMemSegment)
      throws Throwable {

    final long BUFFER_SIZE_BYTES = 1024 * 1024;
    final long FLOAT_SIZE = C_FLOAT_BYTE_SIZE;
    final long FLOATS_PER_BUFFER = BUFFER_SIZE_BYTES / FLOAT_SIZE;
    final long ROWS_PER_BUFFER = Math.max(1, FLOATS_PER_BUFFER / cols);

    for (long startRow = 0; startRow < rows; startRow += ROWS_PER_BUFFER) {
      long endRow = Math.min(startRow + ROWS_PER_BUFFER, rows);
      long chunkRows = endRow - startRow;

      long datasetBytes = FLOAT_SIZE * chunkRows * cols;

      MemorySegment datasetPtr = allocateRMMSegment(cuvsResourcesPtr, datasetBytes);

      try {
        MemorySegment chunkData =
            datasetMemSegment.asSlice(startRow * cols * FLOAT_SIZE, datasetBytes);
        cudaMemcpy(datasetPtr, chunkData, datasetBytes, INFER_DIRECTION);

        long datasetShape[] = {chunkRows, cols};
        MemorySegment datasetTensor =
            prepareTensor(localArena, datasetPtr, datasetShape, kDLFloat(), 32, kDLCUDA(), 1);

        int returnValue =
            cuvsScalarQuantizerTrain(cuvsResourcesPtr, paramsPtr, datasetTensor, quantizerPtr);
        checkCuVSError(returnValue, "cuvsScalarQuantizerTrain");

        returnValue = cuvsStreamSync(cuvsResourcesPtr);
        checkCuVSError(returnValue, "cuvsStreamSync");

      } finally {
        int returnValue = cuvsRMMFree(cuvsResourcesPtr, datasetPtr, datasetBytes);
        checkCuVSError(returnValue, "cuvsRMMFree");
      }
    }
  }

  private static CuVSMatrix performTransformHost(
      long resPtr,
      MemorySegment quantizerSeg,
      Arena localArena,
      long rows,
      long cols,
      MemorySegment dataSeg)
      throws Throwable {

    var result = new CuVSHostMatrixArenaImpl(rows, cols, DataType.BYTE);

    long[] shape = {rows, cols};
    MemorySegment inTensor = prepareTensor(localArena, dataSeg, shape, kDLFloat(), 32, kDLCPU(), 1);
    MemorySegment outTensor =
        prepareTensor(
            localArena,
            ((CuVSMatrixBaseImpl) result).memorySegment(),
            shape,
            kDLInt(),
            8,
            kDLCPU(),
            1);

    int rv = cuvsScalarQuantizerTransform(resPtr, quantizerSeg, inTensor, outTensor);
    checkCuVSError(rv, "cuvsScalarQuantizerTransform (host)");

    return result;
  }

  private static CuVSMatrix performInverseTransformHost(
      long resPtr,
      MemorySegment quantizerSeg,
      Arena localArena,
      long rows,
      long cols,
      MemorySegment quantizedSeg)
      throws Throwable {

    var result = new CuVSHostMatrixArenaImpl(rows, cols, DataType.FLOAT);

    long[] shape = {rows, cols};
    MemorySegment inTensor =
        prepareTensor(localArena, quantizedSeg, shape, kDLInt(), 8, kDLCPU(), 1);
    MemorySegment outTensor =
        prepareTensor(
            localArena,
            ((CuVSMatrixBaseImpl) result).memorySegment(),
            shape,
            kDLFloat(),
            32,
            kDLCPU(),
            1);

    int rv = cuvsScalarQuantizerInverseTransform(resPtr, quantizerSeg, inTensor, outTensor);
    checkCuVSError(rv, "cuvsScalarQuantizerInverseTransform (host)");

    return result;
  }
}
