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

/**
 * A Dataset implementation backed by device (GPU) memory.
 */
public interface CuVSDeviceMatrix extends CuVSMatrix {

  /**
   * Returns a new host matrix with data from this device matrix.
   * The returned host matrix will need to be managed by the caller, which will be
   * responsible to call {@link CuVSMatrix#close()} to free its resources when done.
   */
  default CuVSHostMatrix toHost() {
    var hostMatrix = CuVSMatrix.hostBuilder(size(), columns(), dataType()).build();
    toHost(hostMatrix);
    return hostMatrix;
  }

  default CuVSDeviceMatrix toDevice(CuVSResources resources) {
    return new CuVSDeviceMatrixDelegate(this);
  }

  class CuVSDeviceMatrixDelegate implements CuVSDeviceMatrix {

    private final CuVSDeviceMatrix deviceMatrix;

    private CuVSDeviceMatrixDelegate(CuVSDeviceMatrix deviceMatrix) {
      this.deviceMatrix = deviceMatrix;
    }

    @Override
    public long size() {
      return deviceMatrix.size();
    }

    @Override
    public long columns() {
      return deviceMatrix.columns();
    }

    @Override
    public DataType dataType() {
      return deviceMatrix.dataType();
    }

    @Override
    public RowView getRow(long row) {
      return deviceMatrix.getRow(row);
    }

    @Override
    public void toArray(int[][] array) {
      deviceMatrix.toArray(array);
    }

    @Override
    public void toArray(float[][] array) {
      deviceMatrix.toArray(array);
    }

    @Override
    public void toArray(byte[][] array) {
      deviceMatrix.toArray(array);
    }

    @Override
    public MemoryKind memoryKind() {
      return deviceMatrix.memoryKind();
    }

    @Override
    public void toHost(CuVSHostMatrix hostMatrix) {
      deviceMatrix.toHost(hostMatrix);
    }

    @Override
    public void toDevice(CuVSDeviceMatrix deviceMatrix, CuVSResources cuVSResources) {
      this.deviceMatrix.toDevice(deviceMatrix, cuVSResources);
    }

    @Override
    public void close() {
      // Do nothing
    }
  }
}
