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

import com.nvidia.cuvs.spi.CuVSProvider;

import java.nio.file.Path;

/**
 * Used for allocating resources for cuVS
 *
 * @since 25.02
 */
public interface CuVSResources extends AutoCloseable {

  /**
   * Closes this resources and releases any resources associated with it.
   */
  @Override
  void close();


  /**
   * The temporary directory to use for intermediate operations.
   * Defaults to {@systemProperty java.io.tmpdir}.
   */
  Path tempDirectory();

  /**
   * Creates a new resources.
   * Equivalent to
   * <pre>{@code
   *   create(CuVSProvider.tempDirectory())
   * }</pre>
   */
  static CuVSResources create() throws Throwable {
    return create(CuVSProvider.tempDirectory());
  }

  /**
   * Creates a new resources.
   *
   * @param tempDirectory the temporary directory to use for intermediate operations
   * @throws UnsupportedOperationException if the provider does not cuvs
   * @throws LibraryException if the native library cannot be loaded
   */
  static CuVSResources create(Path tempDirectory) throws Throwable {
    return CuVSProvider.provider().newCuVSResources(tempDirectory);
  }
}
