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
import static com.nvidia.cuvs.internal.common.Util.buildMemorySegment;
import static com.nvidia.cuvs.internal.common.Util.checkCuVSError;
import static com.nvidia.cuvs.internal.common.Util.cudaMemcpy;
import static com.nvidia.cuvs.internal.common.Util.prepareTensor;
import static com.nvidia.cuvs.internal.panama.headers_h.cuvsRMMAlloc;
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

import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.Dataset;
import com.nvidia.cuvs.ScalarQuantizer;
import com.nvidia.cuvs.internal.panama.cuvsScalarQuantizerParams;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Objects;

/**
 * Implementation of {@link ScalarQuantizer} using Panama Foreign Function Interface.
 */
public class ScalarQuantizerImpl implements ScalarQuantizer {

  private static final long C_BYTE_SIZE = C_CHAR.byteSize();

  private final CuVSResourcesImpl resources;
  private final MemorySegment quantizerSegment;
  private boolean destroyed;

  /**
   * Constructor for creating a trained scalar quantizer.
   *
   * @param resources the CuVS resources
   * @param quantizerSegment the native quantizer memory segment
   */
  private ScalarQuantizerImpl(CuVSResourcesImpl resources, MemorySegment quantizerSegment) {
    this.resources = resources;
    this.quantizerSegment = quantizerSegment;
  }

  private void checkNotDestroyed() {
    if (destroyed) {
      throw new IllegalStateException("ScalarQuantizer has been destroyed");
    }
  }

  @Override
  public void destroy() throws Throwable {
    checkNotDestroyed();
    try {
      int returnValue = cuvsScalarQuantizerDestroy(quantizerSegment);
      checkCuVSError(returnValue, "cuvsScalarQuantizerDestroy");
    } finally {
      destroyed = true;
    }
  }

  @Override
  public byte[][] transform(float[][] dataset) throws Throwable {
    checkNotDestroyed();
    try (var localArena = Arena.ofConfined()) {
      long rows = dataset.length;
      long cols = rows > 0 ? dataset[0].length : 0;

      Arena arena = resources.getArena();
      MemorySegment datasetMemSegment = buildMemorySegment(localArena, dataset);

      long cuvsResourcesPtr = resources.getHandle();

      // Allocate device memory for input and output
      MemorySegment datasetD = localArena.allocate(C_POINTER);
      MemorySegment outputD = localArena.allocate(C_POINTER);

      long datasetBytes = C_FLOAT_BYTE_SIZE * rows * cols;
      long outputBytes = C_BYTE_SIZE * rows * cols;

      int returnValue = cuvsRMMAlloc(cuvsResourcesPtr, datasetD, datasetBytes);
      checkCuVSError(returnValue, "cuvsRMMAlloc");

      returnValue = cuvsRMMAlloc(cuvsResourcesPtr, outputD, outputBytes);
      checkCuVSError(returnValue, "cuvsRMMAlloc");

      MemorySegment datasetPtr = datasetD.get(C_POINTER, 0);
      MemorySegment outputPtr = outputD.get(C_POINTER, 0);

      // Copy input data to device
      cudaMemcpy(datasetPtr, datasetMemSegment, datasetBytes, HOST_TO_DEVICE);

      // Prepare tensors
      long datasetShape[] = {rows, cols};
      MemorySegment datasetTensor =
          prepareTensor(localArena, datasetPtr, datasetShape, 2, 32, 2, 2, 1);
      MemorySegment outputTensor =
          prepareTensor(localArena, outputPtr, datasetShape, 0, 8, 2, 2, 1);

      // Transform
      returnValue =
          cuvsScalarQuantizerTransform(
              cuvsResourcesPtr, quantizerSegment, datasetTensor, outputTensor);
      checkCuVSError(returnValue, "cuvsScalarQuantizerTransform");

      returnValue = cuvsStreamSync(cuvsResourcesPtr);
      checkCuVSError(returnValue, "cuvsStreamSync");

      // Copy result back to host
      MemorySegment outputMemSegment = localArena.allocate(C_CHAR, outputBytes);
      cudaMemcpy(outputMemSegment, outputPtr, outputBytes, DEVICE_TO_HOST);

      // Free device memory
      returnValue = cuvsRMMFree(cuvsResourcesPtr, datasetPtr, datasetBytes);
      checkCuVSError(returnValue, "cuvsRMMFree");
      returnValue = cuvsRMMFree(cuvsResourcesPtr, outputPtr, outputBytes);
      checkCuVSError(returnValue, "cuvsRMMFree");

      // Convert to byte array
      byte[][] result = new byte[(int) rows][(int) cols];
      for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
          result[i][j] = outputMemSegment.get(C_CHAR, (i * cols + j) * C_BYTE_SIZE);
        }
      }

      return result;
    }
  }

  @Override
  public byte[][] transform(Dataset dataset) throws Throwable {
    checkNotDestroyed();
    try (var localArena = Arena.ofConfined()) {
      long rows = dataset.size();
      long cols = dataset.dimensions();

      Arena arena = resources.getArena();
      MemorySegment datasetMemSegment = ((DatasetImpl) dataset).asMemorySegment();

      long cuvsResourcesPtr = resources.getHandle();

      // Allocate device memory for input and output
      MemorySegment datasetD = localArena.allocate(C_POINTER);
      MemorySegment outputD = localArena.allocate(C_POINTER);

      long datasetBytes = C_FLOAT_BYTE_SIZE * rows * cols;
      long outputBytes = C_BYTE_SIZE * rows * cols;

      int returnValue = cuvsRMMAlloc(cuvsResourcesPtr, datasetD, datasetBytes);
      checkCuVSError(returnValue, "cuvsRMMAlloc");

      returnValue = cuvsRMMAlloc(cuvsResourcesPtr, outputD, outputBytes);
      checkCuVSError(returnValue, "cuvsRMMAlloc");

      MemorySegment datasetPtr = datasetD.get(C_POINTER, 0);
      MemorySegment outputPtr = outputD.get(C_POINTER, 0);

      // Copy input data to device
      cudaMemcpy(datasetPtr, datasetMemSegment, datasetBytes, INFER_DIRECTION);

      // Prepare tensors
      long datasetShape[] = {rows, cols};
      MemorySegment datasetTensor =
          prepareTensor(localArena, datasetPtr, datasetShape, 2, 32, 2, 2, 1);
      MemorySegment outputTensor =
          prepareTensor(localArena, outputPtr, datasetShape, 0, 8, 2, 2, 1);

      // Transform
      returnValue =
          cuvsScalarQuantizerTransform(
              cuvsResourcesPtr, quantizerSegment, datasetTensor, outputTensor);
      checkCuVSError(returnValue, "cuvsScalarQuantizerTransform");

      returnValue = cuvsStreamSync(cuvsResourcesPtr);
      checkCuVSError(returnValue, "cuvsStreamSync");

      // Copy result back to host
      MemorySegment outputMemSegment = localArena.allocate(C_CHAR, outputBytes);
      cudaMemcpy(outputMemSegment, outputPtr, outputBytes, DEVICE_TO_HOST);

      // Free device memory
      returnValue = cuvsRMMFree(cuvsResourcesPtr, datasetPtr, datasetBytes);
      checkCuVSError(returnValue, "cuvsRMMFree");
      returnValue = cuvsRMMFree(cuvsResourcesPtr, outputPtr, outputBytes);
      checkCuVSError(returnValue, "cuvsRMMFree");

      // Convert to byte array
      byte[][] result = new byte[(int) rows][(int) cols];
      for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
          result[i][j] = outputMemSegment.get(C_CHAR, (i * cols + j) * C_BYTE_SIZE);
        }
      }

      return result;
    }
  }

  @Override
  public float[][] inverseTransform(byte[][] quantizedData) throws Throwable {
    checkNotDestroyed();
    try (var localArena = Arena.ofConfined()) {
      long rows = quantizedData.length;
      long cols = rows > 0 ? quantizedData[0].length : 0;

      Arena arena = resources.getArena();
      MemorySegment quantizedMemSegment = buildMemorySegmentFromBytes(localArena, quantizedData);

      long cuvsResourcesPtr = resources.getHandle();

      // Allocate device memory for input and output
      MemorySegment quantizedD = localArena.allocate(C_POINTER);
      MemorySegment outputD = localArena.allocate(C_POINTER);

      long quantizedBytes = C_BYTE_SIZE * rows * cols;
      long outputBytes = C_FLOAT_BYTE_SIZE * rows * cols;

      int returnValue = cuvsRMMAlloc(cuvsResourcesPtr, quantizedD, quantizedBytes);
      checkCuVSError(returnValue, "cuvsRMMAlloc");

      returnValue = cuvsRMMAlloc(cuvsResourcesPtr, outputD, outputBytes);
      checkCuVSError(returnValue, "cuvsRMMAlloc");

      MemorySegment quantizedPtr = quantizedD.get(C_POINTER, 0);
      MemorySegment outputPtr = outputD.get(C_POINTER, 0);

      // Copy input data to device
      cudaMemcpy(quantizedPtr, quantizedMemSegment, quantizedBytes, HOST_TO_DEVICE);

      // Prepare tensors
      long datasetShape[] = {rows, cols};
      MemorySegment quantizedTensor =
          prepareTensor(localArena, quantizedPtr, datasetShape, 0, 8, 2, 2, 1);
      MemorySegment outputTensor =
          prepareTensor(localArena, outputPtr, datasetShape, 2, 32, 2, 2, 1);

      // Inverse transform
      returnValue =
          cuvsScalarQuantizerInverseTransform(
              cuvsResourcesPtr, quantizerSegment, quantizedTensor, outputTensor);
      checkCuVSError(returnValue, "cuvsScalarQuantizerInverseTransform");

      returnValue = cuvsStreamSync(cuvsResourcesPtr);
      checkCuVSError(returnValue, "cuvsStreamSync");

      // Copy result back to host
      MemorySegment outputMemSegment = localArena.allocate(C_FLOAT, outputBytes);
      cudaMemcpy(outputMemSegment, outputPtr, outputBytes, DEVICE_TO_HOST);

      // Free device memory
      returnValue = cuvsRMMFree(cuvsResourcesPtr, quantizedPtr, quantizedBytes);
      checkCuVSError(returnValue, "cuvsRMMFree");
      returnValue = cuvsRMMFree(cuvsResourcesPtr, outputPtr, outputBytes);
      checkCuVSError(returnValue, "cuvsRMMFree");

      // Convert to float array
      float[][] result = new float[(int) rows][(int) cols];
      for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
          result[i][j] = outputMemSegment.get(C_FLOAT, (i * cols + j) * C_FLOAT_BYTE_SIZE);
        }
      }

      return result;
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

  public static ScalarQuantizer.Builder newBuilder(CuVSResources cuvsResources) {
    Objects.requireNonNull(cuvsResources);
    if (!(cuvsResources instanceof CuVSResourcesImpl)) {
      throw new IllegalArgumentException("Unsupported " + cuvsResources);
    }
    return new Builder((CuVSResourcesImpl) cuvsResources);
  }

  /**
   * Builder for creating ScalarQuantizer instances.
   */
  public static class Builder implements ScalarQuantizer.Builder {

    private final CuVSResourcesImpl resources;
    private float quantile = 0.99f;
    private float[][] trainingDataset;
    private Dataset dataset;

    public Builder(CuVSResourcesImpl resources) {
      this.resources = resources;
    }

    @Override
    public Builder withQuantile(float quantile) {
      if (quantile <= 0 || quantile > 1) {
        throw new IllegalArgumentException("Quantile must be in range (0, 1]");
      }
      this.quantile = quantile;
      return this;
    }

    @Override
    public Builder withTrainingDataset(float[][] dataset) {
      this.trainingDataset = dataset;
      return this;
    }

    @Override
    public Builder withTrainingDataset(Dataset dataset) {
      this.dataset = dataset;
      return this;
    }

    @Override
    public ScalarQuantizer build() throws Throwable {
      if (trainingDataset == null && dataset == null) {
        throw new IllegalArgumentException("Training dataset must be provided");
      }

      try (var localArena = Arena.ofConfined()) {
        long rows = dataset != null ? dataset.size() : trainingDataset.length;
        long cols =
            dataset != null ? dataset.dimensions() : (rows > 0 ? trainingDataset[0].length : 0);

        Arena arena = resources.getArena();
        MemorySegment datasetMemSegment =
            dataset != null
                ? ((DatasetImpl) dataset).asMemorySegment()
                : buildMemorySegment(localArena, trainingDataset);

        long cuvsResourcesPtr = resources.getHandle();

        // Create scalar quantizer params
        MemorySegment paramsSegment = arena.allocate(cuvsScalarQuantizerParams_t);
        int returnValue = cuvsScalarQuantizerParamsCreate(paramsSegment);
        checkCuVSError(returnValue, "cuvsScalarQuantizerParamsCreate");

        MemorySegment paramsPtr = paramsSegment.get(C_POINTER, 0);
        cuvsScalarQuantizerParams.quantile(paramsPtr, quantile);

        // Create scalar quantizer
        MemorySegment quantizerSegment = arena.allocate(cuvsScalarQuantizer_t);
        returnValue = cuvsScalarQuantizerCreate(quantizerSegment);
        checkCuVSError(returnValue, "cuvsScalarQuantizerCreate");

        MemorySegment quantizerPtr = quantizerSegment.get(C_POINTER, 0);

        // Allocate device memory for training data
        MemorySegment datasetD = localArena.allocate(C_POINTER);
        long datasetBytes = C_FLOAT_BYTE_SIZE * rows * cols;

        returnValue = cuvsRMMAlloc(cuvsResourcesPtr, datasetD, datasetBytes);
        checkCuVSError(returnValue, "cuvsRMMAlloc");

        MemorySegment datasetPtr = datasetD.get(C_POINTER, 0);
        cudaMemcpy(datasetPtr, datasetMemSegment, datasetBytes, INFER_DIRECTION);

        // Prepare tensor
        long datasetShape[] = {rows, cols};
        MemorySegment datasetTensor =
            prepareTensor(localArena, datasetPtr, datasetShape, 2, 32, 2, 2, 1);

        // Train quantizer
        returnValue =
            cuvsScalarQuantizerTrain(cuvsResourcesPtr, paramsPtr, datasetTensor, quantizerPtr);
        checkCuVSError(returnValue, "cuvsScalarQuantizerTrain");

        returnValue = cuvsStreamSync(cuvsResourcesPtr);
        checkCuVSError(returnValue, "cuvsStreamSync");

        // Free device memory
        returnValue = cuvsRMMFree(cuvsResourcesPtr, datasetPtr, datasetBytes);
        checkCuVSError(returnValue, "cuvsRMMFree");

        // Clean up params
        returnValue = cuvsScalarQuantizerParamsDestroy(paramsPtr);
        checkCuVSError(returnValue, "cuvsScalarQuantizerParamsDestroy");

        return new ScalarQuantizerImpl(resources, quantizerPtr);
      }
    }
  }
}
