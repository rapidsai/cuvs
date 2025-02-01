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

import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.nio.file.Path;

/**
 * A provider of low-level cuvs resources and builders.
 */
public abstract class CuVSProvider {

  static final Path TMPDIR = Path.of(System.getProperty("java.io.tmpdir"));

  /**
   * The temporary directory to use for intermediate operations.
   * Defaults to {@systemProperty java.io.tmpdir}.
   */
  public static Path tempDirectory() {
    return TMPDIR;
  }

  public abstract CuVSResources newCuVSResources(Path tempDirectory)
      throws Throwable;

  public abstract BruteForceIndex.Builder newBruteForceIndexBuilder(CuVSResources cuVSResources)
      throws UnsupportedOperationException;

  public abstract CagraIndex.Builder newCagraIndexBuilder(CuVSResources cuVSResources)
      throws UnsupportedOperationException;

  public abstract HnswIndex.Builder newHnswIndexBuilder(CuVSResources cuVSResources)
      throws UnsupportedOperationException;

  public static CuVSProvider provider() {
        return CuVSProvider.Holder.INSTANCE;
    }

  private static class Holder {
    static final CuVSProvider INSTANCE = provider();

    static CuVSProvider provider() {
      if (Runtime.version().feature() > 21 && isLinuxAmd64()) {
        try {
          var cls = Class.forName("com.nvidia.cuvs.spi.JDKProvider");
          var ctr = MethodHandles.lookup().findConstructor(cls, MethodType.methodType(void.class));
          return (CuVSProvider) ctr.invoke();
        } catch (Throwable e) {
          throw new AssertionError(e);
        }
      }
      return new UnsupportedProvider();
    }

    /**
     * Returns true iff the architecture is x64 (amd64) and the OS Linux
     * (the * OS we currently support for the native lib).
     */
    static boolean isLinuxAmd64() {
      String name = System.getProperty("os.name");
      return (name.startsWith("Linux")) && System.getProperty("os.arch").equals("amd64");
    }
  }
}
