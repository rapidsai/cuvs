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
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.ValueLayout;

/**
 * Utility methods for calling into the native linker.
 */
public class LinkerHelper {

  private static final Linker LINKER = Linker.nativeLinker();

  public static final ValueLayout.OfByte C_CHAR =
      (ValueLayout.OfByte) LINKER.canonicalLayouts().get("char");

  public static final ValueLayout.OfInt C_INT =
      (ValueLayout.OfInt) LINKER.canonicalLayouts().get("int");

  public static final ValueLayout.OfLong C_LONG =
      (ValueLayout.OfLong) LINKER.canonicalLayouts().get("long");

  public static final ValueLayout.OfFloat C_FLOAT =
      (ValueLayout.OfFloat) LINKER.canonicalLayouts().get("float");

  public static final AddressLayout C_POINTER =
      ((AddressLayout) LINKER.canonicalLayouts().get("void*"))
          .withTargetLayout(MemoryLayout.sequenceLayout(Long.MAX_VALUE, C_CHAR));

  public static final long C_INT_BYTE_SIZE = LinkerHelper.C_INT.byteSize();

  public static final long C_FLOAT_BYTE_SIZE = LinkerHelper.C_FLOAT.byteSize();

  public static final long C_LONG_BYTE_SIZE = LinkerHelper.C_LONG.byteSize();

  private LinkerHelper() {}
}
