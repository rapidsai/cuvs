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
import com.nvidia.cuvs.HnswIndex;
import com.nvidia.cuvs.CagraMergeParams;

import java.nio.file.Path;

/**
 * A provider of low-level cuvs resources and builders.
 */
public interface CuVSProvider {

  Path TMPDIR = Path.of(System.getProperty("java.io.tmpdir"));

  /**
   * The temporary directory to use for intermediate operations.
   * Defaults to {@systemProperty java.io.tmpdir}.
   */
  static Path tempDirectory() {
    return TMPDIR;
  }

  /**
   * The directory where to extract and install the native library.
   * Defaults to {@systemProperty java.io.tmpdir}.
   */
  default Path nativeLibraryPath() {
    return TMPDIR;
  }

  /** Creates a new CuVSResources. */
  CuVSResources newCuVSResources(Path tempDirectory)
      throws Throwable;

  /** Creates a new BruteForceIndex Builder. */
  BruteForceIndex.Builder newBruteForceIndexBuilder(CuVSResources cuVSResources)
      throws UnsupportedOperationException;

  /** Creates a new CagraIndex Builder. */
  CagraIndex.Builder newCagraIndexBuilder(CuVSResources cuVSResources)
      throws UnsupportedOperationException;

  /** Creates a new HnswIndex Builder. */
  HnswIndex.Builder newHnswIndexBuilder(CuVSResources cuVSResources)
      throws UnsupportedOperationException;

  /**
   * Merges multiple CAGRA indexes into a single index.
   *
   * @param indexes Array of CAGRA indexes to merge
   * @return A new merged CAGRA index
   * @throws Throwable if an error occurs during the merge operation
   */
  CagraIndex mergeCagraIndexes(CagraIndex[] indexes) throws Throwable;

  /**
   * Merges multiple CAGRA indexes into a single index with the specified merge parameters.
   * 
   * @param indexes Array of CAGRA indexes to merge
   * @param mergeParams Parameters to control the merge operation, or null to use defaults
   * @return A new merged CAGRA index
   * @throws Throwable if an error occurs during the merge operation
   */
  default CagraIndex mergeCagraIndexes(CagraIndex[] indexes, CagraMergeParams mergeParams) throws Throwable {
    // Default implementation falls back to the method without parameters
    return mergeCagraIndexes(indexes);
  }

  /** Retrieves the system-wide provider. */
  static CuVSProvider provider() {
    return CuVSServiceProvider.Holder.INSTANCE;
  }
}
