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
import static com.nvidia.cuvs.internal.common.Util.CudaMemcpyKind.*;
import static com.nvidia.cuvs.internal.common.Util.allocateRMMSegment;
import static com.nvidia.cuvs.internal.common.Util.buildMemorySegment;
import static com.nvidia.cuvs.internal.common.Util.checkCuVSError;
import static com.nvidia.cuvs.internal.common.Util.cudaMemcpy;
import static com.nvidia.cuvs.internal.common.Util.prepareTensor;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsBinaryQuantizerTransform;
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

/**
 * Implementation of binary quantizer functionality using Panama Foreign Function Interface.
 */
public class BinaryQuantizerImpl {

  private static final long C_BYTE_SIZE = C_CHAR.byteSize();

  /**
   * Applies binary quantization transform to given dataset.
   *
   * @param cuvsResources CuVS resources
   * @param dataset a two-dimensional float array to transform
   * @return a CuVSMatrix containing the binary quantized data
   */
  public static CuVSMatrix transform(CuVSResources cuvsResources, float[][] dataset)
      throws Throwable {

    try (var localArena = Arena.ofConfined();
        var resourcesAccessor = cuvsResources.access()) {

      long rows = dataset.length;
      long cols = rows > 0 ? dataset[0].length : 0;

      MemorySegment datasetMemSegment = buildMemorySegment(localArena, dataset);
      long cuvsResourcesPtr = resourcesAccessor.handle();

      return performTransform(
          cuvsResourcesPtr, localArena, rows, cols, datasetMemSegment, HOST_TO_DEVICE);
    }
  }

  /**
   * Applies binary quantization transform to given dataset.
   *
   * @param cuvsResources CuVS resources
   * @param dataset a {@link CuVSMatrix} object containing the vectors to transform
   * @return a CuVSMatrix containing the binary quantized data
   */
  public static CuVSMatrix transform(CuVSResources cuvsResources, CuVSMatrix dataset)
      throws Throwable {
    // Validate input precision
    if (dataset.precision() != 32) {
      throw new IllegalArgumentException(
          "BinaryQuantizer requires 32-bit float input, got " + dataset.precision() + "-bit");
    }

    try (var localArena = Arena.ofConfined();
        var resourcesAccessor = cuvsResources.access()) {

      long rows = dataset.size();
      long cols = dataset.columns();

      MemorySegment datasetMemSegment = ((CuVSMatrixBaseImpl) dataset).memorySegment();
      long cuvsResourcesPtr = resourcesAccessor.handle();

      if (dataset.memoryKind() == MemoryKind.HOST) {
        // CPU quantization path
        return performTransformHost(cuvsResourcesPtr, localArena, rows, cols, datasetMemSegment);
      } else {
        CuVSMatrix result =
            performTransform(
                cuvsResourcesPtr, localArena, rows, cols, datasetMemSegment, INFER_DIRECTION);

        // Validate output precision
        if (result.precision() != 8) {
          throw new IllegalStateException(
              "Expected 8-bit output from binary quantization, got " + result.precision() + "-bit");
        }
        return result;
      }
    }
  }

  /**
   * Core transformation logic for GPU path.
   */
  private static CuVSMatrix performTransform(
      long cuvsResourcesPtr,
      Arena localArena,
      long rows,
      long cols,
      MemorySegment datasetMemSegment,
      Util.CudaMemcpyKind memcpyDirection)
      throws Throwable {

    long datasetBytes = C_FLOAT_BYTE_SIZE * rows * cols;
    long outputBytes = rows * cols; // 1 byte per element for uint8_t output

    MemorySegment datasetPtr = allocateRMMSegment(cuvsResourcesPtr, datasetBytes);
    MemorySegment outputPtr = allocateRMMSegment(cuvsResourcesPtr, outputBytes);

    try {
      // Copy input data to device
      cudaMemcpy(datasetPtr, datasetMemSegment, datasetBytes, memcpyDirection);

      // Synchronize before transformation
      int returnValue = cuvsStreamSync(cuvsResourcesPtr);
      checkCuVSError(returnValue, "cuvsStreamSync before transform");

      // Prepare tensors with correct device context and data types
      long datasetShape[] = {rows, cols};
      long outputShape[] = {rows, cols};

      MemorySegment datasetTensor =
          prepareTensor(localArena, datasetPtr, datasetShape, kDLFloat(), 32, kDLCUDA(), 1);
      MemorySegment outputTensor =
          prepareTensor(localArena, outputPtr, outputShape, kDLUInt(), 8, kDLCUDA(), 1);

      // Call native binary quantizer
      returnValue = cuvsBinaryQuantizerTransform(cuvsResourcesPtr, datasetTensor, outputTensor);
      checkCuVSError(returnValue, "cuvsBinaryQuantizerTransform");

      // Synchronize after transformation
      returnValue = cuvsStreamSync(cuvsResourcesPtr);
      checkCuVSError(returnValue, "cuvsStreamSync after transform");

      // Create result matrix that manages its own arena
      var result = new CuVSHostMatrixArenaImpl(rows, cols, DataType.BYTE);

      // Copy result back to host
      cudaMemcpy(result.memorySegment(), outputPtr, outputBytes, DEVICE_TO_HOST);

      return result;

    } finally {
      // Free device memory
      int returnValue = cuvsRMMFree(cuvsResourcesPtr, datasetPtr, datasetBytes);
      checkCuVSError(returnValue, "cuvsRMMFree for dataset");

      returnValue = cuvsRMMFree(cuvsResourcesPtr, outputPtr, outputBytes);
      checkCuVSError(returnValue, "cuvsRMMFree for output");
    }
  }

  /**
   * CPU quantization path - no device memory allocation or transfers
   */
  private static CuVSMatrix performTransformHost(
      long cuvsResourcesPtr,
      Arena localArena,
      long rows,
      long cols,
      MemorySegment datasetMemSegment)
      throws Throwable {

    // Create result matrix that manages its own arena
    var result = new CuVSHostMatrixArenaImpl(rows, cols, DataType.BYTE);

    // Prepare host tensors (device type = CPU)
    long[] datasetShape = {rows, cols};
    MemorySegment datasetTensor =
        prepareTensor(localArena, datasetMemSegment, datasetShape, kDLFloat(), 32, kDLCPU(), 1);
    MemorySegment outputTensor =
        prepareTensor(localArena, result.memorySegment(), datasetShape, kDLUInt(), 8, kDLCPU(), 1);

    // Call native quantizer (will use CPU path automatically)
    int returnValue = cuvsBinaryQuantizerTransform(cuvsResourcesPtr, datasetTensor, outputTensor);
    checkCuVSError(returnValue, "cuvsBinaryQuantizerTransform (host)");

    // Return host-backed matrix
    return result;
  }
}
