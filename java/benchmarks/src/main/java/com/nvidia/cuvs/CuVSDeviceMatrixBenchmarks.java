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

package com.nvidia.cuvs;

import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.util.Random;

@Fork(value = 1, warmups = 0)
@State(Scope.Benchmark)
public class CuVSDeviceMatrixBenchmarks {

  @Param({"2048"})
  private int dims;

  @Param({"16384"})
  private int size;

  private static final Random random = new Random();

  private float[][] data;

  private CuVSResources resources;
  private CuVSDeviceMatrix matrixWithNativeBuffer;
  private CuVSDeviceMatrix matrixWithPinnedBuffer;

  private CuVSHostMatrix hostMatrix;

  private float[][] createRandomData() {
    var array = new float[size][dims];

    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < dims; ++j) {
        array[i][j] = random.nextFloat();
      }
    }
    return array;
  }

  @Setup
  public void initialize() throws Throwable {
    data = createRandomData();
    resources = CuVSResources.create();

    var builder0 = CuVSMatrix.deviceBuilder(resources, size, dims, CuVSMatrix.DataType.FLOAT, 0);
    var builder1 = CuVSMatrix.deviceBuilder(resources, size, dims, CuVSMatrix.DataType.FLOAT, 1);

    for (int i = 0; i < size; ++i) {
      var array = data[i];
      builder0.addVector(array);
      builder1.addVector(array);
    }


    matrixWithNativeBuffer = (CuVSDeviceMatrix) builder0.build();
    matrixWithPinnedBuffer = (CuVSDeviceMatrix) builder1.build();
    hostMatrix = (CuVSHostMatrix) CuVSMatrix.hostBuilder(size, dims, CuVSMatrix.DataType.FLOAT).build();
  }

  @TearDown
  public void cleanUp() {
    if (matrixWithNativeBuffer != null) {
      matrixWithNativeBuffer.close();
    }
    if (matrixWithPinnedBuffer != null) {
      matrixWithPinnedBuffer.close();
    }
    if (hostMatrix != null) {
      hostMatrix.close();
    }
    if (resources != null) {
      resources.close();
    }
  }

  @Benchmark
  public void matrixReadRowsWithPinnedBuffer(Blackhole bh) {
    for (int i = 0; i < size; ++i) {
      bh.consume(matrixWithPinnedBuffer.getRow(i));
    }
  }

  @Benchmark
  public void matrixReadRowsWithNativeBuffer(Blackhole bh) {
    for (int i = 0; i < size; ++i) {
      bh.consume(matrixWithNativeBuffer.getRow(i));
    }
  }

  @Benchmark
  public void matrixCopyToHost() throws Throwable {
    try (var resources = CuVSResources.create()) {
      matrixWithPinnedBuffer.toHost(hostMatrix, resources);
    }
  }

  @Benchmark
  public void matrixDeviceBuilderCudaMemcpyHeap() throws Throwable {
    try (CuVSResources resources = CuVSResources.create()) {
      var builder = CuVSMatrix.deviceBuilder(resources, size, dims, CuVSMatrix.DataType.FLOAT, 0x10);

      for (int i = 0; i < size; ++i) {
        var array = data[i];
        builder.addVector(array);
      }
      var matrix = builder.build();
      matrix.close();
    }
  }

  @Benchmark
  public void matrixDeviceBuilderCudaMemcpyNative() throws Throwable {
    try (CuVSResources resources = CuVSResources.create()) {
      var builder = CuVSMatrix.deviceBuilder(resources, size, dims, CuVSMatrix.DataType.FLOAT, 0);

      for (int i = 0; i < size; ++i) {
        var array = data[i];
        builder.addVector(array);
      }
      var matrix = builder.build();
      matrix.close();
    }
  }

  @Benchmark
  public void matrixDeviceBuilderCudaMemcpyCudaHost() throws Throwable {
    try (CuVSResources resources = CuVSResources.create()) {
      var builder = CuVSMatrix.deviceBuilder(resources, size, dims, CuVSMatrix.DataType.FLOAT, 0x20);

      for (int i = 0; i < size; ++i) {
        var array = data[i];
        builder.addVector(array);
      }
      var matrix = builder.build();
      matrix.close();
    }
  }
}
