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

import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.util.ServiceLoader;

/**
 * Service-provider class for {@linkplain CuVSProvider}.
 */
public abstract class CuVSServiceProvider {

  /**
   * Initialize and return an {@link CuVSProvider} provided by this provider.
   * @param builtinProvider the built-in provider.
   * @return the CuVSProvider provided by this provider
   */
  public abstract CuVSProvider get(CuVSProvider builtinProvider);

  static class Holder {
    static final CuVSProvider INSTANCE = loadProvider();

    private static CuVSProvider loadProvider() {
      var builtinProvider = builtinProvider();
      return ServiceLoader.load(CuVSServiceProvider.class)
          .findFirst()
          .map(p -> p.get(builtinProvider))
          .orElse(builtinProvider);
    }

    static CuVSProvider builtinProvider() {
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
