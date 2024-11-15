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

package com.nvidia.cuvs.cagra;

import java.io.File;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;

public class CuVSResources {

  private Arena arena;
  private Linker linker;
  private MethodHandle createResourcesMemoryHandler;
  private MemorySegment cuvsResourcesMemorySegment;
  private SymbolLookup symbolLookup;

  public CuVSResources() throws Throwable {
    linker = Linker.nativeLinker();
    arena = Arena.ofConfined();

    File workingDirectory = new File(System.getProperty("user.dir"));
    symbolLookup = SymbolLookup.libraryLookup(workingDirectory.getParent() + "/internal/libcuvs_java.so", arena);

    createResourcesMemoryHandler = linker.downcallHandle(symbolLookup.find("create_resource").get(),
        FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS));

    MemoryLayout returnValueMemoryLayout = linker.canonicalLayouts().get("int");
    MemorySegment returnValueMemorySegment = arena.allocate(returnValueMemoryLayout);

    cuvsResourcesMemorySegment = (MemorySegment) createResourcesMemoryHandler.invokeExact(returnValueMemorySegment);
  }

  public MemorySegment getCuvsResourcesMemorySegment() {
    return cuvsResourcesMemorySegment;
  }
}