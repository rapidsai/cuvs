/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
 * @since 25.02
 */
public class CuVSResources implements AutoCloseable {

  public final Arena arena;
  public final Linker linker;
  public final SymbolLookup libcuvsNativeLibrary;
  protected File nativeLibrary;

  private final MethodHandle createResourcesMethodHandle;
  private final MethodHandle destroyResourcesMethodHandle;
  private MemorySegment resourcesMemorySegment;

  /**
   * Constructor that allocates the resources needed for cuVS
   * 
   * @throws Throwable exception thrown when native function is invoked
   */
  public CuVSResources() throws Throwable {
    linker = Linker.nativeLinker();
    arena = Arena.ofShared();
    nativeLibrary = Util.loadLibraryFromJar("/libcuvs_java.so");
    libcuvsNativeLibrary = SymbolLookup.libraryLookup(nativeLibrary.getAbsolutePath(), arena);

    createResourcesMethodHandle = linker.downcallHandle(libcuvsNativeLibrary.find("create_resources").get(),
        FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS));

    destroyResourcesMethodHandle = linker.downcallHandle(libcuvsNativeLibrary.find("destroy_resources").get(),
        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS));
    createResources();
  }

  /**
   * Creates the resources used internally and returns its reference.
   * 
   * @throws Throwable exception thrown when native function is invoked
   */
  public void createResources() throws Throwable {
    MemoryLayout returnValueMemoryLayout = linker.canonicalLayouts().get("int");
    MemorySegment returnValueMemorySegment = arena.allocate(returnValueMemoryLayout);
    resourcesMemorySegment = (MemorySegment) createResourcesMethodHandle.invokeExact(returnValueMemorySegment);
  }

  @Override
  public void close() {
    MemoryLayout returnValueMemoryLayout = linker.canonicalLayouts().get("int");
    MemorySegment returnValueMemorySegment = arena.allocate(returnValueMemoryLayout);
    try {
      destroyResourcesMethodHandle.invokeExact(resourcesMemorySegment, returnValueMemorySegment);
    } catch (Throwable e) {
      e.printStackTrace();
    }
    nativeLibrary.delete();
  }

  /**
   * Gets the reference to the cuvsResources MemorySegment.
   * 
   * @return cuvsResources MemorySegment
   */
  protected MemorySegment getMemorySegment() {
    return resourcesMemorySegment;
  }

  /**
   * Returns the loaded libcuvs_java.so as a {@link SymbolLookup}
   */
  protected SymbolLookup getLibcuvsNativeLibrary() {
    return libcuvsNativeLibrary;
  }
}