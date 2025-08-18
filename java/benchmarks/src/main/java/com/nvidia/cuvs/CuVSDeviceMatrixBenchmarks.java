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
  private CuVSDeviceMatrix deviceMatrix;
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

    var builder0 = CuVSMatrix.deviceBuilder(resources, size, dims, CuVSMatrix.DataType.FLOAT);

    for (int i = 0; i < size; ++i) {
      var array = data[i];
      builder0.addVector(array);
    }

    deviceMatrix = builder0.build();
    hostMatrix = CuVSMatrix.hostBuilder(size, dims, CuVSMatrix.DataType.FLOAT).build();
  }

  @TearDown
  public void cleanUp() {
    if (deviceMatrix != null) {
      deviceMatrix.close();
    }
    if (hostMatrix != null) {
      hostMatrix.close();
    }
    if (resources != null) {
      resources.close();
    }
  }

  @Benchmark
  public void matrixReadRowsFromDevice(Blackhole bh) {
    for (int i = 0; i < size; ++i) {
      bh.consume(deviceMatrix.getRow(i));
    }
  }

  @Benchmark
  public void matrixCopyDeviceToHost() {
    deviceMatrix.toHost(hostMatrix);
  }

  @Benchmark
  public void matrixDeviceBuilder() throws Throwable {
    try (CuVSResources resources = CuVSResources.create()) {
      var builder = CuVSMatrix.deviceBuilder(resources, size, dims, CuVSMatrix.DataType.FLOAT);

      for (int i = 0; i < size; ++i) {
        var array = data[i];
        builder.addVector(array);
      }
      CuVSDeviceMatrix matrix = builder.build();
      matrix.close();
    }
  }
}
