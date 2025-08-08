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
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsRMMFree;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsStreamSync;
import static com.nvidia.cuvs.internal.panama.headers_h.kDLCPU;
import static com.nvidia.cuvs.internal.panama.headers_h.kDLCUDA;
import static com.nvidia.cuvs.internal.panama.headers_h.kDLFloat;
import static com.nvidia.cuvs.internal.panama.headers_h.kDLUInt;

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

  private static final long C_BYTE_SIZE = C_CHAR.byteSize();

  public static CuVSMatrix transform(
      CuVSResources cuvsResources, float[][] dataset, int thresholdType) throws Throwable {

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
      CuVSResources cuvsResources, CuVSMatrix dataset, int thresholdType) throws Throwable {
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
      int thresholdType)
      throws Throwable {

    final long BUFFER_SIZE_BYTES = 1024 * 1024;
    final long FLOAT_SIZE = C_FLOAT_BYTE_SIZE;
    final long FLOATS_PER_BUFFER = BUFFER_SIZE_BYTES / FLOAT_SIZE;
    final long ROWS_PER_BUFFER = Math.max(1, FLOATS_PER_BUFFER / cols);

    var result = new CuVSHostMatrixArenaImpl(rows, cols, DataType.BYTE);

    for (long startRow = 0; startRow < rows; startRow += ROWS_PER_BUFFER) {
      long endRow = Math.min(startRow + ROWS_PER_BUFFER, rows);
      long chunkRows = endRow - startRow;

      long datasetBytes = FLOAT_SIZE * chunkRows * cols;
      long outputBytes = chunkRows * cols;

      MemorySegment datasetPtr = allocateRMMSegment(cuvsResourcesPtr, datasetBytes);
      MemorySegment outputPtr = allocateRMMSegment(cuvsResourcesPtr, outputBytes);

      try {
        MemorySegment chunkData =
            datasetMemSegment.asSlice(startRow * cols * FLOAT_SIZE, datasetBytes);
        cudaMemcpy(datasetPtr, chunkData, datasetBytes, memcpyDirection);

        int returnValue = cuvsStreamSync(cuvsResourcesPtr);
        checkCuVSError(returnValue, "cuvsStreamSync before transform");

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
              ((CuVSMatrixBaseImpl) result).memorySegment().asSlice(startRow * cols, outputBytes);
          cudaMemcpy(resultChunk, outputPtr, outputBytes, DEVICE_TO_HOST);

        } finally {
          cuvsBinaryQuantizerDestroy(quantizerPtr);
          cuvsBinaryQuantizerParamsDestroy(paramsPtr);
        }

      } finally {
        int returnValue = cuvsRMMFree(cuvsResourcesPtr, datasetPtr, datasetBytes);
        checkCuVSError(returnValue, "cuvsRMMFree for dataset");

        returnValue = cuvsRMMFree(cuvsResourcesPtr, outputPtr, outputBytes);
        checkCuVSError(returnValue, "cuvsRMMFree for output");
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
      int thresholdType)
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

  private static void setBinaryQuantizerThreshold(MemorySegment paramsPtr, int thresholdType) {
    paramsPtr.set(ValueLayout.JAVA_INT, 0, thresholdType);
  }
}
