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
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemoryLayout.PathElement;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SequenceLayout;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.VarHandle;
import java.util.UUID;

/**
 * CagraIndex encapsulates the implementation of crucial methods for interacting
 * with the CAGRA index.
 * <p>
 * CAGRA is a graph-based nearest neighbors algorithm that was built from the
 * ground up for GPU acceleration. CAGRA demonstrates state-of-the art index
 * build and query performance for both small- and large-batch sized search.
 */
public class CagraIndex {

  private final float[][] dataset;
  private final CuVSResources cuvsResources;
  private Arena arena;
  private CagraIndexParams cagraIndexParameters;
  private CagraIndexReference cagraIndesReference;
  private Linker linker;
  private MethodHandle indexMethodHandle;
  private MethodHandle searchMethodHandle;
  private MethodHandle serializeMethodHandle;
  private MethodHandle deserializeMethodHandle;
  private SymbolLookup symbolLookup;

  private CagraIndex(CagraIndexParams indexParameters, float[][] dataset, CuVSResources cuvsResources)
      throws Throwable {
    this.cagraIndexParameters = indexParameters;
    this.dataset = dataset;
    this.init();
    this.cuvsResources = cuvsResources;
    this.cagraIndesReference = build();
  }

  private CagraIndex(InputStream inputStream, CuVSResources cuvsResources) throws Throwable {
    this.cagraIndexParameters = null;
    this.dataset = null;
    this.cuvsResources = cuvsResources;
    this.init();
    this.cagraIndesReference = deserialize(inputStream);
  }

  private void init() throws Throwable {
    linker = Linker.nativeLinker();
    arena = Arena.ofConfined();

    File workingDirectory = new File(System.getProperty("user.dir"));
    symbolLookup = SymbolLookup.libraryLookup(workingDirectory.getParent() + "/internal/libcuvs_java.so", arena);

    indexMethodHandle = linker.downcallHandle(symbolLookup.find("build_index").get(),
        FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, linker.canonicalLayouts().get("long"),
            linker.canonicalLayouts().get("long"), ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

    searchMethodHandle = linker.downcallHandle(symbolLookup.find("search_index").get(),
        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS, linker.canonicalLayouts().get("int"),
            linker.canonicalLayouts().get("long"), linker.canonicalLayouts().get("long"), ValueLayout.ADDRESS,
            ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

    serializeMethodHandle = linker.downcallHandle(symbolLookup.find("serialize_index").get(),
        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

    deserializeMethodHandle = linker.downcallHandle(symbolLookup.find("deserialize_index").get(),
        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

  }

  public MemorySegment getStringSegment(StringBuilder string) {
    string.append('\0');
    MemoryLayout sq = MemoryLayout.sequenceLayout(string.length(), linker.canonicalLayouts().get("char"));
    MemorySegment fln = arena.allocate(sq);

    for (int i = 0; i < string.length(); i++) {
      VarHandle flnVH = sq.varHandle(PathElement.sequenceElement(i));
      flnVH.set(fln, 0L, (byte) string.charAt(i));
    }
    return fln;
  }

  private MemorySegment getMemorySegment(float[][] data) {
    long rows = data.length;
    long cols = data[0].length;

    MemoryLayout dataMemoryLayout = MemoryLayout.sequenceLayout(rows,
        MemoryLayout.sequenceLayout(cols, linker.canonicalLayouts().get("float")));
    MemorySegment dataMemorySegment = arena.allocate(dataMemoryLayout);

    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        VarHandle element = dataMemoryLayout.arrayElementVarHandle(PathElement.sequenceElement(r),
            PathElement.sequenceElement(c));
        element.set(dataMemorySegment, 0, 0, data[r][c]);
      }
    }

    return dataMemorySegment;
  }

  private CagraIndexReference build() throws Throwable {
    long rows = dataset.length;
    long cols = dataset[0].length;
    MemoryLayout returnvalueMemoryLayout = linker.canonicalLayouts().get("int");
    MemorySegment returnvalueMemorySegment = arena.allocate(returnvalueMemoryLayout);

    cagraIndesReference = new CagraIndexReference((MemorySegment) indexMethodHandle.invokeExact(
        getMemorySegment(dataset), rows, cols, cuvsResources.getCuvsResourcesMemorySegment(), returnvalueMemorySegment,
        cagraIndexParameters.getCagraIndexParamsMemorySegment()));

    return cagraIndesReference;
  }

  public SearchResult search(CuVSQuery cuvsQuery) throws Throwable {

    SequenceLayout neighborsSequenceLayout = MemoryLayout.sequenceLayout(50, linker.canonicalLayouts().get("int"));
    SequenceLayout distancesSequenceLayout = MemoryLayout.sequenceLayout(50, linker.canonicalLayouts().get("float"));
    MemorySegment neighborsMemorySegment = arena.allocate(neighborsSequenceLayout);
    MemorySegment distancesMemorySegment = arena.allocate(distancesSequenceLayout);
    MemoryLayout returnValueMemoryLayout = linker.canonicalLayouts().get("int");
    MemorySegment returnValueMemorySegment = arena.allocate(returnValueMemoryLayout);

    searchMethodHandle.invokeExact(cagraIndesReference.getIndexMemorySegment(),
        getMemorySegment(cuvsQuery.getQueryVectors()), cuvsQuery.getTopK(), 4L, 2L,
        cuvsResources.getCuvsResourcesMemorySegment(), neighborsMemorySegment, distancesMemorySegment,
        returnValueMemorySegment, cuvsQuery.getCagraSearchParameters().getCagraSearchParamsMS());

    return new SearchResult(neighborsSequenceLayout, distancesSequenceLayout, neighborsMemorySegment,
        distancesMemorySegment, cuvsQuery.getTopK(), cuvsQuery.getMapping(), cuvsQuery.getQueryVectors().length);
  }

  public void serialize(OutputStream outputStream) throws Throwable {
    MemoryLayout returnValueMemoryLayout = linker.canonicalLayouts().get("int");
    MemorySegment returnValueMemorySegment = arena.allocate(returnValueMemoryLayout);
    String tmpIndexFile = "/tmp/" + UUID.randomUUID().toString() + ".cag";
    serializeMethodHandle.invokeExact(cuvsResources.getCuvsResourcesMemorySegment(),
        cagraIndesReference.getIndexMemorySegment(), returnValueMemorySegment,
        getStringSegment(new StringBuilder(tmpIndexFile)));
    File tempFile = new File(tmpIndexFile);
    FileInputStream fileInputStream = new FileInputStream(tempFile);
    byte[] chunk = new byte[1024];
    int chunkLength = 0;
    while ((chunkLength = fileInputStream.read(chunk)) != -1) {
      outputStream.write(chunk, 0, chunkLength);
    }
    fileInputStream.close();
    tempFile.delete();
  }

  public void serialize(OutputStream outputStream, String tmpFilePath) throws Throwable {
    MemoryLayout returnValueMemoryLayout = linker.canonicalLayouts().get("int");
    MemorySegment returnValueMemorySegment = arena.allocate(returnValueMemoryLayout);
    serializeMethodHandle.invokeExact(cuvsResources.getCuvsResourcesMemorySegment(),
        cagraIndesReference.getIndexMemorySegment(), returnValueMemorySegment,
        getStringSegment(new StringBuilder(tmpFilePath)));
    File tempFile = new File(tmpFilePath);
    FileInputStream fileInputStream = new FileInputStream(tempFile);
    byte[] chunk = new byte[1024];
    int chunkLength = 0;
    while ((chunkLength = fileInputStream.read(chunk)) != -1) {
      outputStream.write(chunk, 0, chunkLength);
    }
    fileInputStream.close();
    tempFile.delete();
  }

  private CagraIndexReference deserialize(InputStream inputStream) throws Throwable {
    MemoryLayout returnValueMemoryLayout = linker.canonicalLayouts().get("int");
    MemorySegment returnValueMemorySegment = arena.allocate(returnValueMemoryLayout);
    String tmpIndexFile = "/tmp/" + UUID.randomUUID().toString() + ".cag";
    CagraIndexReference cagraIndexReference = new CagraIndexReference();

    File tempFile = new File(tmpIndexFile);
    FileOutputStream fileOutputStream = new FileOutputStream(tempFile);
    byte[] chunk = new byte[1024];
    int chunkLength = 0;
    while ((chunkLength = inputStream.read(chunk)) != -1) {
      fileOutputStream.write(chunk, 0, chunkLength);
    }
    deserializeMethodHandle.invokeExact(cuvsResources.getCuvsResourcesMemorySegment(),
        cagraIndexReference.getIndexMemorySegment(), returnValueMemorySegment,
        getStringSegment(new StringBuilder(tmpIndexFile)));

    inputStream.close();
    fileOutputStream.close();
    tempFile.delete();

    return cagraIndexReference;
  }

  public CagraIndexParams getParams() {
    return cagraIndexParameters;
  }

  public CuVSResources getResources() {
    return cuvsResources;
  }

  public static class Builder {

    private CagraIndexParams cagraIndexParams;
    private float[][] dataset;
    private CuVSResources cuvsResources;
    private InputStream inputStream;

    public Builder(CuVSResources cuvsResources) {
      this.cuvsResources = cuvsResources;
    }

    public Builder from(InputStream inputStream) {
      this.inputStream = inputStream;
      return this;
    }

    public Builder withDataset(float[][] dataset) {
      this.dataset = dataset;
      return this;
    }

    public Builder withIndexParams(CagraIndexParams cagraIndexParameters) {
      this.cagraIndexParams = cagraIndexParameters;
      return this;
    }

    public CagraIndex build() throws Throwable {
      if (inputStream != null) {
        return new CagraIndex(inputStream, cuvsResources);
      } else {
        return new CagraIndex(cagraIndexParams, dataset, cuvsResources);
      }
    }
  }

}
