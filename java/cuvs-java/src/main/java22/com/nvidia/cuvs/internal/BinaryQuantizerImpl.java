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

import static com.nvidia.cuvs.internal.common.CloseableRMMAllocation.allocateRMMSegment;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_FLOAT_BYTE_SIZE;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_POINTER;
import static com.nvidia.cuvs.internal.common.Util.CudaMemcpyKind.*;
import static com.nvidia.cuvs.internal.common.Util.buildMemorySegment;
import static com.nvidia.cuvs.internal.common.Util.checkCuVSError;
import static com.nvidia.cuvs.internal.common.Util.cudaMemcpy;
import static com.nvidia.cuvs.internal.common.Util.prepareTensor;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsBinaryQuantizerCreate;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsBinaryQuantizerDestroy;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsBinaryQuantizerParamsCreate;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsBinaryQuantizerParamsDestroy;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsBinaryQuantizerParams_t;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsBinaryQuantizerTrain;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsBinaryQuantizerTransformWithParams;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsBinaryQuantizer_t;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsStreamSync;
import static com.nvidia.cuvs.internal.panama.headers_h.kDLCPU;
import static com.nvidia.cuvs.internal.panama.headers_h.kDLCUDA;
import static com.nvidia.cuvs.internal.panama.headers_h.kDLFloat;
import static com.nvidia.cuvs.internal.panama.headers_h.kDLUInt;

import com.nvidia.cuvs.BinaryQuantizer;
import com.nvidia.cuvs.CuVSMatrix;
import com.nvidia.cuvs.CuVSMatrix.DataType;
import com.nvidia.cuvs.CuVSMatrix.MemoryKind;
import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.internal.common.Util;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Implementation of binary quantizer functionality using Panama Foreign Function Interface.
 */
public class BinaryQuantizerImpl {

  public static CuVSMatrix transform(
      CuVSResources cuvsResources, float[][] dataset, BinaryQuantizer.ThresholdType thresholdType)
      throws Throwable {

    try (var localArena = Arena.ofConfined();
        var resourcesAccessor = cuvsResources.access()) {

      long rows = dataset.length;
      long cols = rows > 0 ? dataset[0].length : 0;

      MemorySegment datasetMemSegment = buildMemorySegment(localArena, dataset);
      long cuvsResourcesPtr = resourcesAccessor.handle();

      return performTransformGPU(
          cuvsResourcesPtr,
          localArena,
          rows,
          cols,
          datasetMemSegment,
          HOST_TO_DEVICE,
          thresholdType);
    }
  }

  public static CuVSMatrix transform(
      CuVSResources cuvsResources, CuVSMatrix dataset, BinaryQuantizer.ThresholdType thresholdType)
      throws Throwable {
    if (dataset.dataType() != DataType.FLOAT) {
      throw new IllegalArgumentException(
          "BinaryQuantizer requires FLOAT input, got " + dataset.dataType());
    }

    try (var localArena = Arena.ofConfined();
        var resourcesAccessor = cuvsResources.access()) {

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
            cuvsResourcesPtr, localArena, rows, cols, datasetMemSegment, thresholdType);
      } else {
        CuVSMatrix result =
            performTransformGPU(
                cuvsResourcesPtr,
                localArena,
                rows,
                cols,
                datasetMemSegment,
                INFER_DIRECTION,
                thresholdType);

        if (result.dataType() != DataType.BYTE) {
          throw new IllegalStateException(
              "Expected BYTE output from binary quantization, got " + result.dataType());
        }
        return result;
      }
    }
  }

  private static CuVSMatrix performTransformGPU(
      long cuvsResourcesPtr,
      Arena localArena,
      long rows,
      long cols,
      MemorySegment datasetMemSegment,
      Util.CudaMemcpyKind memcpyDirection,
      BinaryQuantizer.ThresholdType thresholdType)
      throws Throwable {

    final long BUFFER_SIZE_BYTES = 1024 * 1024;
    final long FLOAT_SIZE = C_FLOAT_BYTE_SIZE;
    final long FLOATS_PER_BUFFER = BUFFER_SIZE_BYTES / FLOAT_SIZE;
    final long ROWS_PER_BUFFER = Math.max(1, FLOATS_PER_BUFFER / cols);

    var result = new CuVSHostMatrixArenaImpl(rows, cols, DataType.BYTE);

    long maxDatasetBytes = FLOAT_SIZE * ROWS_PER_BUFFER * cols;
    long maxOutputBytes = ROWS_PER_BUFFER * cols;

    try (var datasetDP = allocateRMMSegment(cuvsResourcesPtr, maxDatasetBytes);
        var outputDP = allocateRMMSegment(cuvsResourcesPtr, maxOutputBytes)) {
      for (long startRow = 0; startRow < rows; startRow += ROWS_PER_BUFFER) {
        long endRow = Math.min(startRow + ROWS_PER_BUFFER, rows);
        long chunkRows = endRow - startRow;
        long actualDatasetBytes = FLOAT_SIZE * chunkRows * cols;
        long actualOutputBytes = chunkRows * cols;

        MemorySegment datasetPtr = datasetDP.handle().asSlice(0, actualDatasetBytes);
        MemorySegment outputPtr = outputDP.handle().asSlice(0, actualOutputBytes);
        MemorySegment chunkData =
            datasetMemSegment.asSlice(startRow * cols * FLOAT_SIZE, actualDatasetBytes);

        cudaMemcpy(datasetPtr, chunkData, actualDatasetBytes, memcpyDirection);

        int returnValue = cuvsStreamSync(cuvsResourcesPtr);
        checkCuVSError(returnValue, "cuvsStreamSync before transform");

        // Create and configure quantizer for this chunk
        MemorySegment paramsSegment = localArena.allocate(cuvsBinaryQuantizerParams_t);
        returnValue = cuvsBinaryQuantizerParamsCreate(paramsSegment);
        checkCuVSError(returnValue, "cuvsBinaryQuantizerParamsCreate");
        MemorySegment paramsPtr = paramsSegment.get(C_POINTER, 0);
        setBinaryQuantizerThreshold(paramsPtr, thresholdType);

        MemorySegment quantizerSegment = localArena.allocate(cuvsBinaryQuantizer_t);
        returnValue = cuvsBinaryQuantizerCreate(quantizerSegment);
        checkCuVSError(returnValue, "cuvsBinaryQuantizerCreate");
        MemorySegment quantizerPtr = quantizerSegment.get(C_POINTER, 0);

        try {
          long datasetShape[] = {chunkRows, cols};
          long outputShape[] = {chunkRows, cols};

          MemorySegment datasetTensor =
              prepareTensor(localArena, datasetPtr, datasetShape, kDLFloat(), 32, kDLCUDA(), 1);
          MemorySegment outputTensor =
              prepareTensor(localArena, outputPtr, outputShape, kDLUInt(), 8, kDLCUDA(), 1);

          returnValue =
              cuvsBinaryQuantizerTrain(cuvsResourcesPtr, paramsPtr, datasetTensor, quantizerPtr);
          checkCuVSError(returnValue, "cuvsBinaryQuantizerTrain (GPU)");

          returnValue =
              cuvsBinaryQuantizerTransformWithParams(
                  cuvsResourcesPtr, quantizerPtr, datasetTensor, outputTensor);
          checkCuVSError(returnValue, "cuvsBinaryQuantizerTransformWithParams (GPU)");

          returnValue = cuvsStreamSync(cuvsResourcesPtr);
          checkCuVSError(returnValue, "cuvsStreamSync after transform");

          MemorySegment resultChunk =
              ((CuVSMatrixBaseImpl) result)
                  .memorySegment()
                  .asSlice(startRow * cols, actualOutputBytes);
          cudaMemcpy(resultChunk, outputPtr, actualOutputBytes, DEVICE_TO_HOST);

        } finally {
          cuvsBinaryQuantizerDestroy(quantizerPtr);
          cuvsBinaryQuantizerParamsDestroy(paramsPtr);
        }
      }
    }

    return result;
  }

  private static CuVSMatrix performTransformHost(
      long cuvsResourcesPtr,
      Arena localArena,
      long rows,
      long cols,
      MemorySegment datasetMemSegment,
      BinaryQuantizer.ThresholdType thresholdType)
      throws Throwable {

    var result = new CuVSHostMatrixArenaImpl(rows, cols, DataType.BYTE);

    MemorySegment paramsSegment = localArena.allocate(cuvsBinaryQuantizerParams_t);
    int returnValue = cuvsBinaryQuantizerParamsCreate(paramsSegment);
    checkCuVSError(returnValue, "cuvsBinaryQuantizerParamsCreate");

    MemorySegment paramsPtr = paramsSegment.get(C_POINTER, 0);
    setBinaryQuantizerThreshold(paramsPtr, thresholdType);

    MemorySegment quantizerSegment = localArena.allocate(cuvsBinaryQuantizer_t);
    returnValue = cuvsBinaryQuantizerCreate(quantizerSegment);
    checkCuVSError(returnValue, "cuvsBinaryQuantizerCreate");

    MemorySegment quantizerPtr = quantizerSegment.get(C_POINTER, 0);

    try {
      long[] datasetShape = {rows, cols};
      MemorySegment datasetTensor =
          prepareTensor(localArena, datasetMemSegment, datasetShape, kDLFloat(), 32, kDLCPU(), 1);

      returnValue =
          cuvsBinaryQuantizerTrain(cuvsResourcesPtr, paramsPtr, datasetTensor, quantizerPtr);
      checkCuVSError(returnValue, "cuvsBinaryQuantizerTrain (host)");

      MemorySegment outputTensor =
          prepareTensor(
              localArena,
              ((CuVSMatrixBaseImpl) result).memorySegment(),
              datasetShape,
              kDLUInt(),
              8,
              kDLCPU(),
              1);

      returnValue =
          cuvsBinaryQuantizerTransformWithParams(
              cuvsResourcesPtr, quantizerPtr, datasetTensor, outputTensor);
      checkCuVSError(returnValue, "cuvsBinaryQuantizerTransformWithParams (host)");

      return result;

    } finally {
      cuvsBinaryQuantizerDestroy(quantizerPtr);
      cuvsBinaryQuantizerParamsDestroy(paramsPtr);
    }
  }

  private static void setBinaryQuantizerThreshold(
      MemorySegment paramsPtr, BinaryQuantizer.ThresholdType thresholdType) {
    paramsPtr.set(ValueLayout.JAVA_INT, 0, thresholdType.getValue());
  }

  /** Creates a new BinaryQuantizer implementation. */
  public static Object create(
      CuVSResources resources,
      CuVSMatrix trainingDataset,
      BinaryQuantizer.ThresholdType thresholdType)
      throws Throwable {
    return new BinaryQuantizerImpl(resources, trainingDataset, thresholdType);
  }

  /** Performs transform using a BinaryQuantizer implementation. */
  public static CuVSMatrix transformWithImpl(Object impl, CuVSMatrix input) throws Throwable {
    if (impl instanceof BinaryQuantizerImpl) {
      return ((BinaryQuantizerImpl) impl).transform(input);
    } else {
      throw new IllegalArgumentException("Invalid implementation object");
    }
  }

  /** Closes BinaryQuantizer implementation. */
  public static void close(Object impl) throws Throwable {
    if (impl instanceof BinaryQuantizerImpl) {
      ((BinaryQuantizerImpl) impl).close();
    }
  }

  private final CuVSResources resources;
  private final CuVSMatrix trainingDataset;
  private final BinaryQuantizer.ThresholdType thresholdType;
  private boolean isClosed = false;

  private BinaryQuantizerImpl(
      CuVSResources resources,
      CuVSMatrix trainingDataset,
      BinaryQuantizer.ThresholdType thresholdType) {
    this.resources = resources;
    this.trainingDataset = trainingDataset;
    this.thresholdType = thresholdType;
  }

  private CuVSMatrix transform(CuVSMatrix input) throws Throwable {
    if (isClosed) {
      throw new IllegalStateException("BinaryQuantizerImpl has been closed");
    }

    if (input.dataType() != DataType.FLOAT) {
      throw new IllegalArgumentException(
          "BinaryQuantizer requires FLOAT input, got " + input.dataType());
    }

    return transform(resources, input, thresholdType);
  }

  private void close() throws Throwable {
    isClosed = true;
  }
}
