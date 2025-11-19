/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.spi;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.util.ArrayList;
import java.util.List;
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
      var supportedJavaRuntime = Runtime.version().feature() > 21;
      var supportedOs = System.getProperty("os.name").startsWith("Linux");
      var supportedArchitecture = System.getProperty("os.arch").equals("amd64");
      if (supportedJavaRuntime && supportedOs && supportedArchitecture) {
        try {
          var cls = Class.forName("com.nvidia.cuvs.spi.JDKProvider");
          var ctr =
              MethodHandles.lookup()
                  .findStatic(cls, "create", MethodType.methodType(CuVSProvider.class));
          return (CuVSProvider) ctr.invoke();
        } catch (ProviderInitializationException e) {
          return new UnsupportedProvider("Cannot create JDKProvider: " + e.getMessage());
        } catch (Throwable e) {
          throw new AssertionError(e);
        }
      }
      List<String> unsupportedReasons = new ArrayList<>();
      if (!supportedJavaRuntime) {
        unsupportedReasons.add("cuvs-java requires Java Runtime version 22 or greater");
      }
      if (!supportedOs) {
        unsupportedReasons.add("cuvs-java supports only Linux");
      }
      if (!supportedArchitecture) {
        unsupportedReasons.add("cuvs-java supports only x86");
      }

      return new UnsupportedProvider(String.join("; ", unsupportedReasons));
    }
  }
}
