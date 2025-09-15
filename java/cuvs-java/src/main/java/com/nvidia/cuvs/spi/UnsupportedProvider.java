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
package com.nvidia.cuvs.spi;

import com.nvidia.cuvs.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;

/**
 * A provider that unconditionally throws UnsupportedOperationException.
 */
final class UnsupportedProvider implements CuVSProvider {

  private final String reasons;

  public UnsupportedProvider(String reasons) {
    this.reasons = reasons;
  }

  @Override
  public CuVSResources newCuVSResources(Path tempDirectory) {
    throw new UnsupportedOperationException(reasons);
  }

  @Override
  public BruteForceIndex.Builder newBruteForceIndexBuilder(CuVSResources cuVSResources) {
    throw new UnsupportedOperationException(reasons);
  }

  @Override
  public CagraIndex.Builder newCagraIndexBuilder(CuVSResources cuVSResources) {
    throw new UnsupportedOperationException(reasons);
  }

  @Override
  public HnswIndex.Builder newHnswIndexBuilder(CuVSResources cuVSResources) {
    throw new UnsupportedOperationException(reasons);
  }

  @Override
  public TieredIndex.Builder newTieredIndexBuilder(CuVSResources cuVSResources) {
    throw new UnsupportedOperationException(reasons);
  }

  @Override
  public CagraIndex mergeCagraIndexes(CagraIndex[] indexes) {
    throw new UnsupportedOperationException(reasons);
  }

  @Override
  public CuVSMatrix.Builder<CuVSHostMatrix> newHostMatrixBuilder(
      long size, long dimensions, CuVSMatrix.DataType dataType) {
    throw new UnsupportedOperationException(reasons);
  }

  @Override
  public CuVSMatrix.Builder<CuVSDeviceMatrix> newDeviceMatrixBuilder(
      CuVSResources cuVSResources, long size, long dimensions, CuVSMatrix.DataType dataType) {
    throw new UnsupportedOperationException(reasons);
  }

  @Override
  public GPUInfoProvider gpuInfoProvider() {
    throw new UnsupportedOperationException(reasons);
  }

  @Override
  public CuVSMatrix.Builder<CuVSDeviceMatrix> newDeviceMatrixBuilder(
      CuVSResources cuVSResources,
      long size,
      long dimensions,
      int rowStride,
      int columnStride,
      CuVSMatrix.DataType dataType) {
    throw new UnsupportedOperationException(reasons);
  }

  @Override
  public MethodHandle newNativeMatrixBuilder() {
    throw new UnsupportedOperationException(reasons);
  }

  @Override
  public CuVSMatrix newMatrixFromArray(float[][] vectors) {
    throw new UnsupportedOperationException(reasons);
  }

  @Override
  public CuVSMatrix newMatrixFromArray(int[][] vectors) {
    throw new UnsupportedOperationException(reasons);
  }

  @Override
  public CuVSMatrix newMatrixFromArray(byte[][] vectors) {
    throw new UnsupportedOperationException(reasons);
  }
}
