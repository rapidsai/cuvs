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

package com.nvidia.cuvs.internal.common;

import java.lang.foreign.AddressLayout;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;

/**
 * Utility methods for calling into the native linker.
 */
public class LinkerHelper {

  private static final Linker LINKER = Linker.nativeLinker();
  private static final SymbolLookup SYMBOL_LOOKUP;

  public static final ValueLayout.OfByte C_CHAR = (ValueLayout.OfByte) LINKER.canonicalLayouts().get("char");

  public static final ValueLayout.OfInt C_INT = (ValueLayout.OfInt) LINKER.canonicalLayouts().get("int");

  public static final ValueLayout.OfLong C_LONG = (ValueLayout.OfLong) LINKER.canonicalLayouts().get("long");

  public static final ValueLayout.OfFloat C_FLOAT = (ValueLayout.OfFloat) LINKER.canonicalLayouts().get("float");

  public static final AddressLayout C_POINTER = ((AddressLayout) LINKER.canonicalLayouts().get("void*"))
          .withTargetLayout(MemoryLayout.sequenceLayout(Long.MAX_VALUE, C_CHAR));

  static {
    var nativeLibrary = LoaderUtils.loadNativeLibrary();
    // we use a global arena here, since the symbols obtained
    // from the returned lookup are for the lifetime of the jvm
    SYMBOL_LOOKUP = SymbolLookup.libraryLookup(nativeLibrary.toAbsolutePath(), Arena.global());
  }

  static MemorySegment functionAddress(String function) {
    return SYMBOL_LOOKUP.find(function).orElseThrow(() -> new LinkageError("Native function " + function + " could not be found"));
  }

  public static MethodHandle downcallHandle(String function, FunctionDescriptor functionDescriptor, Linker.Option... options) {
    return LINKER.downcallHandle(functionAddress(function), functionDescriptor, options);
  }

  private LinkerHelper() {}
}
