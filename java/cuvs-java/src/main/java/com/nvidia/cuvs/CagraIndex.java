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
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SequenceLayout;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.util.UUID;

import com.nvidia.cuvs.common.Util;
import com.nvidia.cuvs.panama.cuvsCagraIndex;

/**
 * {@link CagraIndex} encapsulates a CAGRA index, along with methods to interact
 * with it.
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
  private final CuVSResources resources;
  private Arena arena;
  private Linker linker;
  private MethodHandle indexMethodHandle;
  private MethodHandle searchMethodHandle;
  private MethodHandle serializeMethodHandle;
  private MethodHandle deserializeMethodHandle;
  private CagraIndexParams cagraIndexParameters;
  private IndexReference cagraIndexReference;

  /*
   * Constructor for building the index using specified dataset
   */
  private CagraIndex(CagraIndexParams indexParameters, float[][] dataset, CuVSResources resources) throws Throwable {
    this.cagraIndexParameters = indexParameters;
    this.dataset = dataset;
    this.resources = resources;

    initializeMethodHandles();
    this.cagraIndexReference = build();
  }

  /**
   * Constructor for loading the index from an {@link InputStream}
   */
  private CagraIndex(InputStream inputStream, CuVSResources resources) throws Throwable {
    this.cagraIndexParameters = null;
    this.dataset = null;
    this.resources = resources;

    initializeMethodHandles();
    this.cagraIndexReference = deserialize(inputStream);
  }

  /**
   * Initializes the {@link MethodHandles} for invoking native methods.
   * 
   * @throws IOException @{@link IOException} is unable to load the native library
   */
  private void initializeMethodHandles() throws IOException {
    linker = Linker.nativeLinker();
    arena = Arena.ofConfined();

    indexMethodHandle = linker.downcallHandle(resources.getLibcuvsNativeLibrary().find("build_cagra_index").get(),
        FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, linker.canonicalLayouts().get("long"),
            linker.canonicalLayouts().get("long"), ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

    searchMethodHandle = linker.downcallHandle(resources.getLibcuvsNativeLibrary().find("search_cagra_index").get(),
        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS, linker.canonicalLayouts().get("int"),
            linker.canonicalLayouts().get("long"), linker.canonicalLayouts().get("int"), ValueLayout.ADDRESS,
            ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

    serializeMethodHandle = linker.downcallHandle(
        resources.getLibcuvsNativeLibrary().find("serialize_cagra_index").get(),
        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));

    deserializeMethodHandle = linker.downcallHandle(
        resources.getLibcuvsNativeLibrary().find("deserialize_cagra_index").get(),
        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
  }

  /**
   * Invokes the native build_index function via the Panama API to build the
   * {@link CagraIndex}
   * 
   * @return an instance of {@link IndexReference} that holds the pointer to the
   *         index
   */
  private IndexReference build() throws Throwable {
    long rows = dataset.length;
    long cols = dataset[0].length;
    MemoryLayout layout = linker.canonicalLayouts().get("int");
    MemorySegment segment = arena.allocate(layout);

    cagraIndexReference = new IndexReference(
        (MemorySegment) indexMethodHandle.invokeExact(Util.buildMemorySegment(linker, arena, dataset), rows, cols,
            resources.getMemorySegment(), segment, cagraIndexParameters.getMemorySegment()));

    return cagraIndexReference;
  }

  /**
   * Invokes the native search_index via the Panama API for searching a CAGRA
   * index.
   * 
   * @param query an instance of {@link CagraQuery} holding the query vectors and
   *              other parameters
   * @return an instance of {@link CagraSearchResults} containing the results
   */
  public CagraSearchResults search(CagraQuery query) throws Throwable {
    long numQueries = query.getQueryVectors().length;
    long numBlocks = query.getTopK() * numQueries;
    int vectorDimension = numQueries > 0 ? query.getQueryVectors()[0].length : 0;

    SequenceLayout neighborsSequenceLayout = MemoryLayout.sequenceLayout(numBlocks,
        linker.canonicalLayouts().get("int"));
    SequenceLayout distancesSequenceLayout = MemoryLayout.sequenceLayout(numBlocks,
        linker.canonicalLayouts().get("float"));
    MemorySegment neighborsMemorySegment = arena.allocate(neighborsSequenceLayout);
    MemorySegment distancesMemorySegment = arena.allocate(distancesSequenceLayout);
    MemoryLayout returnValueMemoryLayout = linker.canonicalLayouts().get("int");
    MemorySegment returnValueMemorySegment = arena.allocate(returnValueMemoryLayout);

    searchMethodHandle.invokeExact(cagraIndexReference.getMemorySegment(),
        Util.buildMemorySegment(linker, arena, query.getQueryVectors()), query.getTopK(), numQueries, vectorDimension,
        resources.getMemorySegment(), neighborsMemorySegment, distancesMemorySegment, returnValueMemorySegment,
        query.getCagraSearchParameters().getMemorySegment());

    return new CagraSearchResults(neighborsSequenceLayout, distancesSequenceLayout, neighborsMemorySegment,
        distancesMemorySegment, query.getTopK(), query.getMapping(), numQueries);
  }

  /**
   * A method to persist a CAGRA index using an instance of {@link OutputStream}
   * for writing index bytes.
   * 
   * @param outputStream an instance of {@link OutputStream} to write the index
   *                     bytes into
   */
  public void serialize(OutputStream outputStream) throws Throwable {
    serialize(outputStream, File.createTempFile(UUID.randomUUID().toString(), ".cag"));
  }

  /**
   * A method to persist a CAGRA index using an instance of {@link OutputStream}
   * and path to the intermediate temporary file.
   * 
   * @param outputStream an instance of {@link OutputStream} to write the index
   *                     bytes to
   * @param tempFile  an intermediate {@link File} where CAGRA index is written
   *                     temporarily
   */
  public void serialize(OutputStream outputStream, File tempFile) throws Throwable {
    MemoryLayout returnValueMemoryLayout = linker.canonicalLayouts().get("int");
    MemorySegment returnValueMemorySegment = arena.allocate(returnValueMemoryLayout);
    serializeMethodHandle.invokeExact(resources.getMemorySegment(), cagraIndexReference.getMemorySegment(),
        returnValueMemorySegment, Util.buildMemorySegment(linker, arena, tempFile.getAbsolutePath()));
    FileInputStream fileInputStream = new FileInputStream(tempFile);
    byte[] chunk = new byte[1024]; // TODO: Make this configurable
    int chunkLength = 0;
    while ((chunkLength = fileInputStream.read(chunk)) != -1) {
      outputStream.write(chunk, 0, chunkLength);
    }
    fileInputStream.close();
    tempFile.delete();
  }

  /**
   * Gets an instance of {@link IndexReference} by deserializing a CAGRA index
   * using an {@link InputStream}.
   * 
   * @param inputStream an instance of {@link InputStream}
   * @return an instance of {@link IndexReference}.
   */
  private IndexReference deserialize(InputStream inputStream) throws Throwable {
    MemoryLayout returnValueMemoryLayout = linker.canonicalLayouts().get("int");
    MemorySegment returnValueMemorySegment = arena.allocate(returnValueMemoryLayout);
    String tmpIndexFile = "/tmp/" + UUID.randomUUID().toString() + ".cag";
    IndexReference indexReference = new IndexReference();

    File tempFile = new File(tmpIndexFile);
    FileOutputStream fileOutputStream = new FileOutputStream(tempFile);
    byte[] chunk = new byte[1024];
    int chunkLength = 0;
    while ((chunkLength = inputStream.read(chunk)) != -1) {
      fileOutputStream.write(chunk, 0, chunkLength);
    }
    deserializeMethodHandle.invokeExact(resources.getMemorySegment(), indexReference.getMemorySegment(),
        returnValueMemorySegment, Util.buildMemorySegment(linker, arena, tmpIndexFile));

    inputStream.close();
    fileOutputStream.close();
    tempFile.delete();

    return indexReference;
  }

  /**
   * Gets an instance of {@link CagraIndexParams}
   * 
   * @return an instance of {@link CagraIndexParams}
   */
  public CagraIndexParams getCagraIndexParameters() {
    return cagraIndexParameters;
  }

  /**
   * Gets an instance of {@link CuVSResources}
   * 
   * @return an instance of {@link CuVSResources}
   */
  public CuVSResources getCuVSResources() {
    return resources;
  }

  /**
   * Builder helps configure and create an instance of {@link CagraIndex}.
   */
  public static class Builder {

    private float[][] dataset;
    private CagraIndexParams cagraIndexParams;
    private CuVSResources cuvsResources;
    private InputStream inputStream;

    /**
     * Constructs this Builder with an instance of {@link CuVSResources}.
     * 
     * @param cuvsResources an instance of {@link CuVSResources}
     */
    public Builder(CuVSResources cuvsResources) {
      this.cuvsResources = cuvsResources;
    }

    /**
     * Sets an instance of InputStream typically used when index deserialization is
     * needed.
     * 
     * @param inputStream an instance of {@link InputStream}
     * @return an instance of this Builder
     */
    public Builder from(InputStream inputStream) {
      this.inputStream = inputStream;
      return this;
    }

    /**
     * Sets the dataset for building the {@link CagraIndex}.
     * 
     * @param dataset a two-dimensional float array
     * @return an instance of this Builder
     */
    public Builder withDataset(float[][] dataset) {
      this.dataset = dataset;
      return this;
    }

    /**
     * Registers an instance of configured {@link CagraIndexParams} with this
     * Builder.
     * 
     * @param cagraIndexParameters An instance of CagraIndexParams.
     * @return An instance of this Builder.
     */
    public Builder withIndexParams(CagraIndexParams cagraIndexParameters) {
      this.cagraIndexParams = cagraIndexParameters;
      return this;
    }

    /**
     * Builds and returns an instance of CagraIndex.
     * 
     * @return an instance of CagraIndex
     */
    public CagraIndex build() throws Throwable {
      if (inputStream != null) {
        return new CagraIndex(inputStream, cuvsResources);
      } else {
        return new CagraIndex(cagraIndexParams, dataset, cuvsResources);
      }
    }
  }

  /**
   * Holds the memory reference to an index.
   */
  protected static class IndexReference {

    private final MemorySegment memorySegment;

    /**
     * Constructs CagraIndexReference and allocate the MemorySegment.
     */
    protected IndexReference() {
      Arena arena = Arena.ofConfined();
      memorySegment = cuvsCagraIndex.allocate(arena);
    }

    /**
     * Constructs CagraIndexReference with an instance of MemorySegment passed as a
     * parameter.
     * 
     * @param indexMemorySegment the MemorySegment instance to use for containing
     *                           index reference
     */
    protected IndexReference(MemorySegment indexMemorySegment) {
      this.memorySegment = indexMemorySegment;
    }

    /**
     * Gets the instance of index MemorySegment.
     * 
     * @return index MemorySegment
     */
    protected MemorySegment getMemorySegment() {
      return memorySegment;
    }
  }
}
