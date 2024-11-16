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
 * build and query performance for both small and large-batch sized search. Know
 * more about this algorithm
 * <a href="https://arxiv.org/abs/2308.15136" target="_blank">here</a>
 * 
 * @since 24.12
 */
public class CagraIndex {

  private final float[][] dataset;
  private final CuVSResources cuvsResources;
  private Arena arena;
  private Linker linker;
  private MethodHandle indexMethodHandle;
  private MethodHandle searchMethodHandle;
  private MethodHandle serializeMethodHandle;
  private MethodHandle deserializeMethodHandle;
  private SymbolLookup symbolLookup;
  private CagraIndexParams cagraIndexParameters;
  private CagraIndexReference cagraIndesReference;

  /**
   * Constructor that initializes CagraIndex with an instance of CagraIndexParams,
   * dataset, and an instance of CuVSResources
   * 
   * @param indexParameters index parameters
   * @param dataset         2D float dataset array
   * @param cuvsResources   cuVS resources instance
   * @throws Throwable exception thrown when native function is invoked
   * @see CagraIndexParams
   * @see CuVSResources
   */
  private CagraIndex(CagraIndexParams indexParameters, float[][] dataset, CuVSResources cuvsResources)
      throws Throwable {
    this.cagraIndexParameters = indexParameters;
    this.dataset = dataset;
    this.initializeMethodHandles();
    this.cuvsResources = cuvsResources;
    this.cagraIndesReference = build();
  }

  /**
   * Constructs an instance of CagraIndex with an instance of InputStream and
   * CuVSResources
   * 
   * @param inputStream   an instance of InputStream (eg. FileInputStream) to read
   *                      a persisted CAGRA index.
   * @param cuvsResources an instance of CuVSResources.
   * @throws Throwable exception thrown when native function is invoked
   */
  private CagraIndex(InputStream inputStream, CuVSResources cuvsResources) throws Throwable {
    this.cagraIndexParameters = null;
    this.dataset = null;
    this.cuvsResources = cuvsResources;
    this.initializeMethodHandles();
    this.cagraIndesReference = deserialize(inputStream);
  }

  /**
   * Initializes the MethodHandles for invoking native methods.
   * 
   * @see MethodHandle
   */
  private void initializeMethodHandles() {
    linker = Linker.nativeLinker();
    arena = Arena.ofConfined();

    File workingDirectory = new File(System.getProperty("user.dir"));
    symbolLookup = SymbolLookup.libraryLookup(workingDirectory.getParent() + "/internal/libcuvs_java.so", arena);

    indexMethodHandle = linker.downcallHandle(symbolLookup.find("build_cagra_index").get(),
        FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, linker.canonicalLayouts().get("long"),
            linker.canonicalLayouts().get("long"), ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

    searchMethodHandle = linker.downcallHandle(symbolLookup.find("search_cagra_index").get(),
        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS, linker.canonicalLayouts().get("int"),
            linker.canonicalLayouts().get("long"), linker.canonicalLayouts().get("long"), ValueLayout.ADDRESS,
            ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

    serializeMethodHandle = linker.downcallHandle(symbolLookup.find("serialize_cagra_index").get(),
        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

    deserializeMethodHandle = linker.downcallHandle(symbolLookup.find("deserialize_cagra_index").get(),
        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

  }

  /**
   * A utility method for getting an instance of MemorySegment for a String.
   * 
   * @param string the string for the expected MemorySegment
   * @return an instance of MemorySegment
   * @see MemoryLayout
   * @see MemorySegment
   * @see StringBuilder
   */
  public MemorySegment getStringMemorySegment(StringBuilder string) {
    string.append('\0');
    MemoryLayout stringMemoryLayout = MemoryLayout.sequenceLayout(string.length(),
        linker.canonicalLayouts().get("char"));
    MemorySegment stringMemorySegment = arena.allocate(stringMemoryLayout);

    for (int i = 0; i < string.length(); i++) {
      VarHandle varHandle = stringMemoryLayout.varHandle(PathElement.sequenceElement(i));
      varHandle.set(stringMemorySegment, 0L, (byte) string.charAt(i));
    }
    return stringMemorySegment;
  }

  /**
   * A utility method for getting an instance of MemorySegment for a 2D float
   * array.
   * 
   * @param data The 2D float array for which the MemorySegment is needed
   * @return an instance of MemorySegment
   * @see MemoryLayout
   * @see MemorySegment
   */
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

  /**
   * Invokes the native build_index function via the Panama API to build the CAGRA
   * index.
   * 
   * @return an instance of CagraIndexReference that holds the pointer to the
   *         index
   * @throws Throwable exception thrown when native function is invoked
   * @see CagraIndexReference
   */
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

  /**
   * Invokes the native search_index via the Panama API for searching a CAGRA
   * index.
   * 
   * @param cuvsQuery an instance of CuVSQuery holding the query and its
   *                  parameters
   * @return an instance of SearchResult containing the results
   * @throws Throwable exception thrown when native function is invoked
   * @see CuVSQuery
   * @see SearchResult
   */
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
        returnValueMemorySegment, cuvsQuery.getCagraSearchParameters().getCagraSearchParamsMemorySegment());

    return new SearchResult(neighborsSequenceLayout, distancesSequenceLayout, neighborsMemorySegment,
        distancesMemorySegment, cuvsQuery.getTopK(), cuvsQuery.getMapping(), cuvsQuery.getQueryVectors().length);
  }

  /**
   * A method to persist a CAGRA index using an instance of OutputStream for
   * writing index bytes.
   * 
   * @param outputStream an instance of OutputStream to write the index bytes into
   * @throws Throwable exception thrown when native function is invoked
   * @see OutputStream
   */
  public void serialize(OutputStream outputStream) throws Throwable {
    MemoryLayout returnValueMemoryLayout = linker.canonicalLayouts().get("int");
    MemorySegment returnValueMemorySegment = arena.allocate(returnValueMemoryLayout);
    String tmpIndexFile = "/tmp/" + UUID.randomUUID().toString() + ".cag";
    serializeMethodHandle.invokeExact(cuvsResources.getCuvsResourcesMemorySegment(),
        cagraIndesReference.getIndexMemorySegment(), returnValueMemorySegment,
        getStringMemorySegment(new StringBuilder(tmpIndexFile)));
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

  /**
   * A method to persist a CAGRA index using an instance of OutputStream and path
   * to the intermediate temporary file.
   * 
   * @param outputStream an instance of OutputStream to write the index bytes to
   * @param tmpFilePath  path to a temporary file where CAGRA index is written
   * @throws Throwable exception thrown when native function is invoked
   * @see OutputStream
   */
  public void serialize(OutputStream outputStream, String tmpFilePath) throws Throwable {
    MemoryLayout returnValueMemoryLayout = linker.canonicalLayouts().get("int");
    MemorySegment returnValueMemorySegment = arena.allocate(returnValueMemoryLayout);
    serializeMethodHandle.invokeExact(cuvsResources.getCuvsResourcesMemorySegment(),
        cagraIndesReference.getIndexMemorySegment(), returnValueMemorySegment,
        getStringMemorySegment(new StringBuilder(tmpFilePath)));
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

  /**
   * Gets an instance of CagraIndexReference by deserializing a CAGRA index using
   * an input stream.
   * 
   * @param inputStream an instance of InputStream (eg. FileInputStream when
   *                    reading an index file).
   * @return an instance of CagraIndexReference.
   * @throws Throwable exception thrown when native function is invoked
   * @see CagraIndexReference
   * @see InputStream
   */
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
        getStringMemorySegment(new StringBuilder(tmpIndexFile)));

    inputStream.close();
    fileOutputStream.close();
    tempFile.delete();

    return cagraIndexReference;
  }

  /**
   * Gets an instance of CagraIndexParams.
   * 
   * @return an instance of CagraIndexParams
   * @see CagraIndexParams
   */
  public CagraIndexParams getCagraIndexParameters() {
    return cagraIndexParameters;
  }

  /**
   * Gets an instance of CuVSResources.
   * 
   * @return an instance of CuVSResources
   * @see CuVSResources
   */
  public CuVSResources getCuVSResources() {
    return cuvsResources;
  }

  /**
   * Builder helps configure and create an instance of CagraIndex.
   */
  public static class Builder {

    private float[][] dataset;
    private CagraIndexParams cagraIndexParams;
    private CuVSResources cuvsResources;
    private InputStream inputStream;

    /**
     * Constructs this Builder with an instance of CuVSResources.
     * 
     * @param cuvsResources an instance of CuVSResources
     * @see CuVSResources
     */
    public Builder(CuVSResources cuvsResources) {
      this.cuvsResources = cuvsResources;
    }

    /**
     * Sets an instance of InputStream typically used when index deserialization is
     * needed.
     * 
     * @param inputStream an instance of InputStream
     * @return an instance of this Builder
     * @see InputStream
     */
    public Builder from(InputStream inputStream) {
      this.inputStream = inputStream;
      return this;
    }

    /**
     * Sets the dataset for building the CAGRA index.
     * 
     * @param dataset a two-dimensional float array
     * @return an instance of this Builder
     */
    public Builder withDataset(float[][] dataset) {
      this.dataset = dataset;
      return this;
    }

    /**
     * Registers an instance of configured CagraIndexParams with this Builder.
     * 
     * @param cagraIndexParameters An instance of CagraIndexParams.
     * @return An instance of this Builder.
     * @see CagraIndexParams
     */
    public Builder withIndexParams(CagraIndexParams cagraIndexParameters) {
      this.cagraIndexParams = cagraIndexParameters;
      return this;
    }

    /**
     * Builds and returns an instance of CagraIndex.
     * 
     * @return an instance of CagraIndex
     * @throws Throwable exception thrown when native function is invoked
     */
    public CagraIndex build() throws Throwable {
      if (inputStream != null) {
        return new CagraIndex(inputStream, cuvsResources);
      } else {
        return new CagraIndex(cagraIndexParams, dataset, cuvsResources);
      }
    }
  }
}
