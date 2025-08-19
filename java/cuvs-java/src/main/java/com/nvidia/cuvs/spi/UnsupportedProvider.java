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

import com.nvidia.cuvs.BinaryQuantizer;
import com.nvidia.cuvs.BruteForceIndex;
import com.nvidia.cuvs.CagraIndex;
import com.nvidia.cuvs.CuVSMatrix;
import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.HnswIndex;
import com.nvidia.cuvs.TieredIndex;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;

/**
 * A provider that throws UnsupportedOperationException for all operations.
 * Used as a fallback when no proper implementation is available.
 */
final class UnsupportedProvider implements CuVSProvider {

  @Override
  public CuVSResources newCuVSResources(Path tempDirectory) {
    throw new UnsupportedOperationException();
  }

  @Override
  public BruteForceIndex.Builder newBruteForceIndexBuilder(CuVSResources cuVSResources) {
    throw new UnsupportedOperationException();
  }

  @Override
  public CagraIndex.Builder newCagraIndexBuilder(CuVSResources cuVSResources) {
    throw new UnsupportedOperationException();
  }

  @Override
  public HnswIndex.Builder newHnswIndexBuilder(CuVSResources cuVSResources) {
    throw new UnsupportedOperationException();
  }

  @Override
  public TieredIndex.Builder newTieredIndexBuilder(CuVSResources cuVSResources) {
    throw new UnsupportedOperationException();
  }

  @Override
  public CagraIndex mergeCagraIndexes(CagraIndex[] indexes) throws Throwable {
    throw new UnsupportedOperationException();
  }

  @Override
  public CuVSMatrix.Builder newMatrixBuilder(
      int size, int dimensions, CuVSMatrix.DataType dataType) {
    throw new UnsupportedOperationException();
  }

  @Override
  public MethodHandle newNativeMatrixBuilder() {
    throw new UnsupportedOperationException();
  }

  @Override
  public CuVSMatrix newMatrixFromArray(float[][] vectors) {
    throw new UnsupportedOperationException();
  }

  @Override
  public CuVSMatrix newMatrixFromArray(int[][] vectors) {
    throw new UnsupportedOperationException();
  }

  @Override
  public CuVSMatrix newMatrixFromArray(byte[][] vectors) {
    throw new UnsupportedOperationException();
  }

  @Override
  public Object createScalar8BitQuantizerImpl(CuVSResources resources, CuVSMatrix trainingDataset)
      throws Throwable {
    throw new UnsupportedOperationException("CuVS is not supported on this platform");
  }

  @Override
  public CuVSMatrix inverseTransformScalar8Bit(Object impl, CuVSMatrix quantizedData)
      throws Throwable {
    throw new UnsupportedOperationException("CuVS is not supported on this platform");
  }

  @Override
  public Object createBinaryQuantizerImpl(
      CuVSResources resources,
      CuVSMatrix trainingDataset,
      BinaryQuantizer.ThresholdType thresholdType)
      throws Throwable {
    throw new UnsupportedOperationException("CuVS is not supported on this platform");
  }

  @Override
  public CuVSMatrix transformBinaryWithImpl(Object impl, CuVSMatrix input) throws Throwable {
    throw new UnsupportedOperationException("CuVS is not supported on this platform");
  }

  @Override
  public void closeBinaryQuantizer(Object impl) throws Throwable {
    throw new UnsupportedOperationException("CuVS is not supported on this platform");
  }

  @Override
  public CuVSMatrix transformScalar8Bit(Object impl, CuVSMatrix input) throws Throwable {
    throw new UnsupportedOperationException("CuVS is not supported on this platform");
  }

  @Override
  public void closeScalar8BitQuantizer(Object impl) throws Throwable {
    throw new UnsupportedOperationException("CuVS is not supported on this platform");
  }
}
