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
import static com.nvidia.cuvs.internal.common.Util.buildMemorySegment;
import static com.nvidia.cuvs.internal.common.Util.checkCuVSError;
import static com.nvidia.cuvs.internal.common.Util.cudaMemcpy;
import static com.nvidia.cuvs.internal.common.Util.prepareTensor;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsBinaryQuantizerTransform;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsRMMAlloc;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsRMMFree;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsStreamSync;

import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.Dataset;
import com.nvidia.cuvs.QuantizedMatrix;
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
   * @return a QuantizedMatrix containing the binary quantized data
   */
  public static QuantizedMatrix transform(CuVSResources cuvsResources, float[][] dataset)
      throws Throwable {
    if (!(cuvsResources instanceof CuVSResourcesImpl)) {
      throw new IllegalArgumentException("Unsupported " + cuvsResources);
    }

    CuVSResourcesImpl resources = (CuVSResourcesImpl) cuvsResources;
    Arena resultArena = Arena.ofShared();

    try (var localArena = Arena.ofConfined();
        var resourcesAccessor = resources.access()) {
      long rows = dataset.length;
      long cols = rows > 0 ? dataset[0].length : 0;

      MemorySegment datasetMemSegment = buildMemorySegment(localArena, dataset);
      long cuvsResourcesPtr = resourcesAccessor.handle();
      return performTransform(
          cuvsResourcesPtr, localArena, resultArena, rows, cols, datasetMemSegment, HOST_TO_DEVICE);
    } catch (Throwable t) {
      resultArena.close();
      throw t;
    }
  }

  /**
   * Applies binary quantization transform to given dataset.
   *
   * @param cuvsResources CuVS resources
   * @param dataset a {@link Dataset} object containing the vectors to transform
   * @return a QuantizedMatrix containing the binary quantized data
   */
  public static QuantizedMatrix transform(CuVSResources cuvsResources, Dataset dataset)
      throws Throwable {
    if (!(cuvsResources instanceof CuVSResourcesImpl)) {
      throw new IllegalArgumentException("Unsupported " + cuvsResources);
    }

    CuVSResourcesImpl resources = (CuVSResourcesImpl) cuvsResources;
    Arena resultArena = Arena.ofShared();

    try (var localArena = Arena.ofConfined();
        var resourcesAccessor = resources.access()) {
      long rows = dataset.size();
      long cols = dataset.dimensions();

      MemorySegment datasetMemSegment = ((DatasetImpl) dataset).asMemorySegment();
      long cuvsResourcesPtr = resourcesAccessor.handle();
      return performTransform(
          cuvsResourcesPtr,
          localArena,
          resultArena,
          rows,
          cols,
          datasetMemSegment,
          INFER_DIRECTION);
    } catch (Throwable t) {
      resultArena.close();
      throw t;
    }
  }

  /**
   * Core transformation logic shared by both overloads.
   */
  private static QuantizedMatrix performTransform(
      long cuvsResourcesPtr,
      Arena localArena,
      Arena resultArena,
      long rows,
      long cols,
      MemorySegment datasetMemSegment,
      Util.CudaMemcpyKind memcpyDirection)
      throws Throwable {

    // Allocate device memory for input and output
    MemorySegment datasetD = localArena.allocate(C_POINTER);
    MemorySegment outputD = localArena.allocate(C_POINTER);

    long datasetBytes = C_FLOAT_BYTE_SIZE * rows * cols;
    long outputCols = (cols + 7) / 8; // Ceiling division for bit packing
    long outputBytes = C_BYTE_SIZE * rows * outputCols;

    int returnValue = cuvsRMMAlloc(cuvsResourcesPtr, datasetD, datasetBytes);
    checkCuVSError(returnValue, "cuvsRMMAlloc");

    returnValue = cuvsRMMAlloc(cuvsResourcesPtr, outputD, outputBytes);
    checkCuVSError(returnValue, "cuvsRMMAlloc");

    MemorySegment datasetPtr = datasetD.get(C_POINTER, 0);
    MemorySegment outputPtr = outputD.get(C_POINTER, 0);

    try {
      // Copy input data to device
      cudaMemcpy(datasetPtr, datasetMemSegment, datasetBytes, memcpyDirection);

      // Prepare tensors
      long datasetShape[] = {rows, cols};
      long outputShape[] = {rows, outputCols};
      MemorySegment datasetTensor =
          prepareTensor(localArena, datasetPtr, datasetShape, 2, 32, 2, 2, 1);
      MemorySegment outputTensor = prepareTensor(localArena, outputPtr, outputShape, 1, 8, 2, 2, 1);

      // Transform
      returnValue = cuvsBinaryQuantizerTransform(cuvsResourcesPtr, datasetTensor, outputTensor);
      checkCuVSError(returnValue, "cuvsBinaryQuantizerTransform");

      returnValue = cuvsStreamSync(cuvsResourcesPtr);
      checkCuVSError(returnValue, "cuvsStreamSync");

      // Copy result back to host using the shared result arena
      MemorySegment outputMemSegment = resultArena.allocate(C_CHAR, outputBytes);
      cudaMemcpy(outputMemSegment, outputPtr, outputBytes, DEVICE_TO_HOST);

      // Create QuantizedMatrix with the shared result arena (1 bit per value for binary
      // quantization)
      return new QuantizedMatrixImpl(outputMemSegment, rows, cols, 1, resultArena);

    } finally {
      // Free device memory
      returnValue = cuvsRMMFree(cuvsResourcesPtr, datasetPtr, datasetBytes);
      checkCuVSError(returnValue, "cuvsRMMFree");
      returnValue = cuvsRMMFree(cuvsResourcesPtr, outputPtr, outputBytes);
      checkCuVSError(returnValue, "cuvsRMMFree");
    }
  }
}
