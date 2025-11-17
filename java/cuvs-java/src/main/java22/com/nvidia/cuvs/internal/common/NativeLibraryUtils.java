/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.internal.common;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;

public class NativeLibraryUtils {

  private NativeLibraryUtils() {}

  private static final SymbolLookup LOOKUP =
      SymbolLookup.libraryLookup(System.mapLibraryName("jvm"), Arena.ofAuto())
          .or(SymbolLookup.loaderLookup())
          .or(Linker.nativeLinker().defaultLookup());

  // void * JVM_LoadLibrary(const char *name, jboolean throwException);
  public static MethodHandle JVM_LoadLibrary$mh =
      Linker.nativeLinker()
          .downcallHandle(
              LOOKUP.find("JVM_LoadLibrary").orElseThrow(),
              FunctionDescriptor.of(
                  ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_BOOLEAN));
  // void JVM_UnloadLibrary(void * handle);
  public static MethodHandle JVM_UnloadLibrary$mh =
      Linker.nativeLinker()
          .downcallHandle(
              LOOKUP.find("JVM_UnloadLibrary").orElseThrow(),
              FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));
}
