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

  /**
   * Constructor for creating a trained scalar quantizer.
   */
  private Scalar8BitQuantizerImpl(CuVSResources resources, MemorySegment quantizerSegment) {
    this.resources = resources;
    this.quantizerSegment = quantizerSegment;
  }

  private void checkNotDestroyed() {
    if (destroyed) {
      throw new IllegalStateException("Scalar8BitQuantizer has been destroyed");
    }
  }

  public int precision() {
    return 8;
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

    // Validate input precision
    if (dataset.precision() != 32) {
      throw new IllegalArgumentException(
          "Scalar8BitQuantizer requires 32-bit float input, got " + dataset.precision() + "-bit");
    }

    try (Arena resultArena = Arena.ofShared();
        var localArena = Arena.ofConfined();
        var resourcesAccessor = resources.access()) {

      long rows = dataset.size();
      long cols = dataset.columns();

      MemorySegment datasetMemSegment = ((CuVSMatrixBaseImpl) dataset).memorySegment();
      long cuvsResourcesPtr = resourcesAccessor.handle();

      if (dataset.memoryKind() == MemoryKind.HOST) {
        // CPU path: no device alloc / cudaMemcpy
        return performTransformHost(
            cuvsResourcesPtr, quantizerSegment, localArena, rows, cols, datasetMemSegment);
      }

      long datasetBytes = C_FLOAT_BYTE_SIZE * rows * cols;
      long outputBytes = C_BYTE_SIZE * rows * cols;

      MemorySegment datasetPtr = allocateRMMSegment(cuvsResourcesPtr, datasetBytes);
      MemorySegment outputPtr = allocateRMMSegment(cuvsResourcesPtr, outputBytes);

      try {
        // Copy input data to device
        cudaMemcpy(datasetPtr, datasetMemSegment, datasetBytes, INFER_DIRECTION);

        // Prepare tensors
        long datasetShape[] = {rows, cols};
        MemorySegment datasetTensor =
            prepareTensor(localArena, datasetPtr, datasetShape, kDLFloat(), 32, kDLCUDA(), 1);
        MemorySegment outputTensor =
            prepareTensor(localArena, outputPtr, datasetShape, kDLInt(), 8, kDLCUDA(), 1);

        // Transform
        int returnValue =
            cuvsScalarQuantizerTransform(
                cuvsResourcesPtr, quantizerSegment, datasetTensor, outputTensor);
        checkCuVSError(returnValue, "cuvsScalarQuantizerTransform");

        returnValue = cuvsStreamSync(cuvsResourcesPtr);
        checkCuVSError(returnValue, "cuvsStreamSync");

        // Copy result back to host
        MemorySegment outputMemSegment = resultArena.allocate(C_CHAR, outputBytes);
        cudaMemcpy(outputMemSegment, outputPtr, outputBytes, DEVICE_TO_HOST);

        // Create matrix using the populated outputMemSegment
        CuVSMatrix result = new CuVSHostMatrixImpl(outputMemSegment, rows, cols, DataType.BYTE);

        // Validate output precision
        if (result.precision() != 8) {
          throw new IllegalStateException(
              "Expected 8-bit output from scalar quantization, got " + result.precision() + "-bit");
        }

        return result;

      } finally {
        // Free device memory
        int returnValue = cuvsRMMFree(cuvsResourcesPtr, datasetPtr, datasetBytes);
        checkCuVSError(returnValue, "cuvsRMMFree");
        returnValue = cuvsRMMFree(cuvsResourcesPtr, outputPtr, outputBytes);
        checkCuVSError(returnValue, "cuvsRMMFree");
      }
    } catch (Throwable t) {
      throw t;
    }
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

  /**
   * Creates a new Scalar8BitQuantizerImpl with training on the provided dataset.
   */
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

      // Create scalar quantizer params
      MemorySegment paramsSegment = localArena.allocate(cuvsScalarQuantizerParams_t);
      int returnValue = cuvsScalarQuantizerParamsCreate(paramsSegment);
      checkCuVSError(returnValue, "cuvsScalarQuantizerParamsCreate");

      MemorySegment paramsPtr = paramsSegment.get(C_POINTER, 0);
      cuvsScalarQuantizerParams.quantile(paramsPtr, 0.99f);

      // Create scalar quantizer
      MemorySegment quantizerSegment = localArena.allocate(cuvsScalarQuantizer_t);
      returnValue = cuvsScalarQuantizerCreate(quantizerSegment);
      checkCuVSError(returnValue, "cuvsScalarQuantizerCreate");

      MemorySegment quantizerPtr = quantizerSegment.get(C_POINTER, 0);

      // Check if we should use CPU or GPU path for training
      if (trainingDataset.memoryKind() == MemoryKind.HOST) {
        // CPU training path
        long datasetShape[] = {rows, cols};
        MemorySegment datasetTensor =
            prepareTensor(localArena, datasetMemSegment, datasetShape, kDLFloat(), 32, kDLCPU(), 1);

        returnValue =
            cuvsScalarQuantizerTrain(cuvsResourcesPtr, paramsPtr, datasetTensor, quantizerPtr);
        checkCuVSError(returnValue, "cuvsScalarQuantizerTrain");
      } else {
        // GPU training path (existing code)
        long datasetBytes = C_FLOAT_BYTE_SIZE * rows * cols;
        MemorySegment datasetPtr = allocateRMMSegment(cuvsResourcesPtr, datasetBytes);
        cudaMemcpy(datasetPtr, datasetMemSegment, datasetBytes, INFER_DIRECTION);

        long datasetShape[] = {rows, cols};
        MemorySegment datasetTensor =
            prepareTensor(localArena, datasetPtr, datasetShape, kDLFloat(), 32, kDLCUDA(), 1);

        returnValue =
            cuvsScalarQuantizerTrain(cuvsResourcesPtr, paramsPtr, datasetTensor, quantizerPtr);
        checkCuVSError(returnValue, "cuvsScalarQuantizerTrain");

        returnValue = cuvsStreamSync(cuvsResourcesPtr);
        checkCuVSError(returnValue, "cuvsStreamSync");

        returnValue = cuvsRMMFree(cuvsResourcesPtr, datasetPtr, datasetBytes);
        checkCuVSError(returnValue, "cuvsRMMFree");
      }

      // Clean up params
      returnValue = cuvsScalarQuantizerParamsDestroy(paramsPtr);
      checkCuVSError(returnValue, "cuvsScalarQuantizerParamsDestroy");

      return new Scalar8BitQuantizerImpl(resources, quantizerPtr);
    }
  }

  public CuVSMatrix inverseTransform(CuVSMatrix quantizedData) throws Throwable {
    checkNotDestroyed();

    if (quantizedData.precision() != 8) {
      throw new IllegalArgumentException(
          "Inverse transform requires 8-bit input, got " + quantizedData.precision() + "-bit");
    }

    try (Arena resultArena = Arena.ofShared();
        var localArena = Arena.ofConfined();
        var resourcesAccessor = resources.access()) {

      long rows = quantizedData.size();
      long cols = quantizedData.columns();
      MemorySegment quantizedMemSegment = ((CuVSMatrixBaseImpl) quantizedData).memorySegment();
      long cuvsResourcesPtr = resourcesAccessor.handle();

      if (quantizedData.memoryKind() == MemoryKind.HOST) {
        // CPU inverse transform path
        return performInverseTransformHost(
            cuvsResourcesPtr, quantizerSegment, localArena, rows, cols, quantizedMemSegment);
      }

      // GPU inverse transform path (existing code)
      long quantizedBytes = C_BYTE_SIZE * rows * cols;
      long outputBytes = C_FLOAT_BYTE_SIZE * rows * cols;

      MemorySegment quantizedPtr = allocateRMMSegment(cuvsResourcesPtr, quantizedBytes);
      MemorySegment outputPtr = allocateRMMSegment(cuvsResourcesPtr, outputBytes);

      try {
        cudaMemcpy(quantizedPtr, quantizedMemSegment, quantizedBytes, INFER_DIRECTION);

        long datasetShape[] = {rows, cols};
        MemorySegment quantizedTensor =
            prepareTensor(localArena, quantizedPtr, datasetShape, kDLInt(), 8, kDLCUDA(), 1);
        MemorySegment outputTensor =
            prepareTensor(localArena, outputPtr, datasetShape, kDLFloat(), 32, kDLCUDA(), 1);

        int returnValue =
            cuvsScalarQuantizerInverseTransform(
                cuvsResourcesPtr, quantizerSegment, quantizedTensor, outputTensor);
        checkCuVSError(returnValue, "cuvsScalarQuantizerInverseTransform");

        returnValue = cuvsStreamSync(cuvsResourcesPtr);
        checkCuVSError(returnValue, "cuvsStreamSync");

        MemorySegment outputMemSegment = resultArena.allocate(C_FLOAT, outputBytes);
        cudaMemcpy(outputMemSegment, outputPtr, outputBytes, DEVICE_TO_HOST);

        return new CuVSHostMatrixImpl(outputMemSegment, rows, cols, DataType.FLOAT);

      } finally {
        int returnValue = cuvsRMMFree(cuvsResourcesPtr, quantizedPtr, quantizedBytes);
        checkCuVSError(returnValue, "cuvsRMMFree");
        returnValue = cuvsRMMFree(cuvsResourcesPtr, outputPtr, outputBytes);
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

    // Create matrix that manages its own arena
    var result = new CuVSHostMatrixArenaImpl(rows, cols, DataType.BYTE);

    long[] shape = {rows, cols};
    MemorySegment inTensor = prepareTensor(localArena, dataSeg, shape, kDLFloat(), 32, kDLCPU(), 1);
    MemorySegment outTensor =
        prepareTensor(localArena, result.memorySegment(), shape, kDLInt(), 8, kDLCPU(), 1);

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

    // Create matrix that manages its own arena
    var result = new CuVSHostMatrixArenaImpl(rows, cols, DataType.FLOAT);

    long[] shape = {rows, cols};
    MemorySegment inTensor =
        prepareTensor(localArena, quantizedSeg, shape, kDLInt(), 8, kDLCPU(), 1);
    MemorySegment outTensor =
        prepareTensor(localArena, result.memorySegment(), shape, kDLFloat(), 32, kDLCPU(), 1);

    int rv = cuvsScalarQuantizerInverseTransform(resPtr, quantizerSeg, inTensor, outTensor);
    checkCuVSError(rv, "cuvsScalarQuantizerInverseTransform (host)");

    return result;
  }
}
