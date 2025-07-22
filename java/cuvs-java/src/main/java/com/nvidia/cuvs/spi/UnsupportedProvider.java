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

import com.nvidia.cuvs.BruteForceIndex;
import com.nvidia.cuvs.CagraIndex;
import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.Dataset;
import com.nvidia.cuvs.HnswIndex;
import com.nvidia.cuvs.TieredIndex;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;

/**
 * A provider that unconditionally throws UnsupportedOperationException.
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
  public Dataset.Builder newDatasetBuilder(int size, int dimensions) {
    throw new UnsupportedOperationException();
  }

  @Override
  public MethodHandle newNativeDatasetBuilder() {
    throw new UnsupportedOperationException();
  }

  @Override
  public Dataset newArrayDataset(float[][] vectors) {
    throw new UnsupportedOperationException();
  }
}
