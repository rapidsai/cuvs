/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

import java.io.File;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;

import com.nvidia.cuvs.common.Util;

/**
 * Used for allocating resources for cuVS
 * 
 * @since 24.12
 */
public class CuVSResources {

  private final Arena arena;
  private final Linker linker;
  private final SymbolLookup libcuvsNativeLibrary;
  private final MethodHandle createResourceMethodHandle;
  private final MemorySegment memorySegment;

  /**
   * Constructor that allocates the resources needed for cuVS
   * 
   * @throws Throwable exception thrown when native function is invoked
   */
  public CuVSResources() throws Throwable {
    linker = Linker.nativeLinker();
    arena = Arena.ofConfined();

    File nativeLibrary = Util.loadLibraryFromJar("/libcuvs_java.so");
    libcuvsNativeLibrary = SymbolLookup.libraryLookup(nativeLibrary.getAbsolutePath(), arena);

    createResourceMethodHandle = linker.downcallHandle(libcuvsNativeLibrary.find("create_resource").get(),
        FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS));

    MemoryLayout returnValueMemoryLayout = linker.canonicalLayouts().get("int");
    MemorySegment returnValueMemorySegment = arena.allocate(returnValueMemoryLayout);

    memorySegment = (MemorySegment) createResourceMethodHandle.invokeExact(returnValueMemorySegment);
  }

  /**
   * Gets the reference to the cuvsResources MemorySegment.
   * 
   * @return cuvsResources MemorySegment
   */
  protected MemorySegment getMemorySegment() {
    // TODO: Is there a way to not letting a memory segment leak into the public
    // API?
    return memorySegment;
  }

  /**
   * Returns the loaded libcuvs_java.so as a {@link SymbolLookup}
   */
  protected SymbolLookup getLibcuvsNativeLibrary() {
    return libcuvsNativeLibrary;
  }
}