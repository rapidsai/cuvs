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

package com.nvidia.cuvs.internal;

import static com.nvidia.cuvs.internal.common.LinkerHelper.C_FLOAT;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_INT;
import static com.nvidia.cuvs.internal.common.LinkerHelper.C_LONG;
import static com.nvidia.cuvs.internal.common.LinkerHelper.downcallHandle;
import static com.nvidia.cuvs.internal.common.Util.checkError;
import static java.lang.foreign.ValueLayout.ADDRESS;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SequenceLayout;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.BitSet;
import java.util.Objects;
import java.util.UUID;

import com.nvidia.cuvs.BruteForceIndex;
import com.nvidia.cuvs.BruteForceIndexParams;
import com.nvidia.cuvs.BruteForceQuery;
import com.nvidia.cuvs.CuVSResources;
import com.nvidia.cuvs.Dataset;
import com.nvidia.cuvs.SearchResults;
import com.nvidia.cuvs.internal.common.Util;
import com.nvidia.cuvs.internal.panama.cuvsBruteForceIndex;

/**
 *
 * {@link BruteForceIndex} encapsulates a BRUTEFORCE index, along with methods
 * to interact with it.
 *
 * @since 25.02
 */
public class BruteForceIndexImpl implements BruteForceIndex{

  private static final MethodHandle indexMethodHandle = downcallHandle("build_brute_force_index",
      FunctionDescriptor.of(ADDRESS, ADDRESS, C_LONG, C_LONG, ADDRESS, ADDRESS, C_INT));

  private static final MethodHandle searchMethodHandle = downcallHandle("search_brute_force_index",
      FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, C_INT, C_LONG, C_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, C_LONG));

  private static final MethodHandle destroyIndexMethodHandle = downcallHandle("destroy_brute_force_index",
      FunctionDescriptor.ofVoid(ADDRESS, ADDRESS));

  private static final MethodHandle serializeMethodHandle = downcallHandle("serialize_brute_force_index",
      FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, ADDRESS, ADDRESS));

  private static final MethodHandle deserializeMethodHandle = downcallHandle("deserialize_brute_force_index",
      FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, ADDRESS, ADDRESS));

  private final float[][] vectors;
  private final Dataset dataset;
  private final CuVSResourcesImpl resources;
  private final IndexReference bruteForceIndexReference;
  private final BruteForceIndexParams bruteForceIndexParams;
  private boolean destroyed;

  /**
   * Constructor for building the index using specified dataset
   *
   * @param dataset               the dataset used for creating the BRUTEFORCE
   *                              index
   * @param resources             an instance of {@link CuVSResourcesImpl}
   * @param bruteForceIndexParams an instance of {@link BruteForceIndexParams}
   *                              holding the index parameters
   */
  private BruteForceIndexImpl(float[][] vectors, Dataset dataset, CuVSResourcesImpl resources,
		  BruteForceIndexParams bruteForceIndexParams)
      throws Throwable {
    this.vectors = vectors;
    this.dataset = dataset;
    this.resources = resources;
    this.bruteForceIndexParams = bruteForceIndexParams;
    this.bruteForceIndexReference = build();
  }

  /**
   * Constructor for loading the index from an {@link InputStream}
   *
   * @param inputStream an instance of stream to read the index bytes from
   * @param resources   an instance of {@link CuVSResourcesImpl}
   */
  private BruteForceIndexImpl(InputStream inputStream, CuVSResourcesImpl resources) throws Throwable {
    this.bruteForceIndexParams = null;
    this.vectors = null;
    this.dataset = null;
    this.resources = resources;
    this.bruteForceIndexReference = deserialize(inputStream);
  }

  private void checkNotDestroyed() {
    if (destroyed) {
      throw new IllegalStateException("destroyed");
    }
  }

  /**
   * Invokes the native destroy_brute_force_index function to de-allocate
   * BRUTEFORCE index
   */
  @Override
  public void destroyIndex() throws Throwable {
    checkNotDestroyed();
    try (var localArena = Arena.ofConfined()) {
      MemorySegment returnValue = localArena.allocate(C_INT);
      destroyIndexMethodHandle.invokeExact(bruteForceIndexReference.getMemorySegment(), returnValue);
      checkError(returnValue.get(C_INT, 0L), "destroyIndexMethodHandle");
    } finally {
      destroyed = true;
    }
    if (dataset != null) dataset.close();
  }

  /**
   * Invokes the native build_brute_force_index function via the Panama API to
   * build the {@link BruteForceIndex}
   *
   * @return an instance of {@link IndexReference} that holds the pointer to the
   *         index
   */
  private IndexReference build() throws Throwable {
    long rows = dataset != null? dataset.size(): vectors.length;
    long cols = dataset != null? dataset.dimensions(): (rows > 0 ? vectors[0].length : 0);

    MemorySegment dataSeg = dataset != null? ((DatasetImpl) dataset).seg: 
    	Util.buildMemorySegment(resources.getArena(), vectors);
    try (var localArena = Arena.ofConfined()) {
      MemorySegment returnValue = localArena.allocate(C_INT);
      MemorySegment indexSeg = (MemorySegment) indexMethodHandle.invokeExact(
        dataSeg,
        rows,
        cols,
        resources.getMemorySegment(),
        returnValue,
        bruteForceIndexParams.getNumWriterThreads()
      );
      checkError(returnValue.get(C_INT, 0L), "indexMethodHandle");
      return new IndexReference(indexSeg);
    }
  }

  /**
   * Invokes the native search_brute_force_index via the Panama API for searching
   * a BRUTEFORCE index.
   *
   * @param cuvsQuery an instance of {@link BruteForceQuery} holding the query
   *                  vectors and other parameters
   * @return an instance of {@link BruteForceSearchResults} containing the results
   */
  @Override
  public SearchResults search(BruteForceQuery cuvsQuery) throws Throwable {
    checkNotDestroyed();
    long numQueries = cuvsQuery.getQueryVectors().length;
    long numBlocks = cuvsQuery.getTopK() * numQueries;
    int vectorDimension = numQueries > 0 ? cuvsQuery.getQueryVectors()[0].length : 0;

    SequenceLayout neighborsSequenceLayout = MemoryLayout.sequenceLayout(numBlocks, C_LONG);
    SequenceLayout distancesSequenceLayout = MemoryLayout.sequenceLayout(numBlocks, C_FLOAT);
    MemorySegment neighborsMemorySegment = resources.getArena().allocate(neighborsSequenceLayout);
    MemorySegment distancesMemorySegment = resources.getArena().allocate(distancesSequenceLayout);

    // prepare the prefiltering data
    long prefilterDataLength = 0;
    MemorySegment prefilterDataMemorySegment = MemorySegment.NULL;
    BitSet[] prefilters = cuvsQuery.getPrefilters();
    if (prefilters != null && prefilters.length > 0) {
      BitSet concatenatedFilters = Util.concatenate(prefilters, cuvsQuery.getNumDocs());
      long filters[] = concatenatedFilters.toLongArray();
      prefilterDataMemorySegment = Util.buildMemorySegment(resources.getArena(), filters);
      prefilterDataLength = cuvsQuery.getNumDocs() * prefilters.length;
    }

    MemorySegment querySeg = Util.buildMemorySegment(resources.getArena(), cuvsQuery.getQueryVectors());
    try (var localArena = Arena.ofConfined()) {
      MemorySegment returnValue = localArena.allocate(C_INT);
      searchMethodHandle.invokeExact(
        bruteForceIndexReference.getMemorySegment(),
        querySeg,
        cuvsQuery.getTopK(),
        numQueries,
        vectorDimension,
        resources.getMemorySegment(),
        neighborsMemorySegment,
        distancesMemorySegment,
        returnValue,
        prefilterDataMemorySegment,
        prefilterDataLength
      );
      checkError(returnValue.get(C_INT, 0L), "searchMethodHandle");
    }
    return new BruteForceSearchResults(neighborsSequenceLayout, distancesSequenceLayout, neighborsMemorySegment,
            distancesMemorySegment, cuvsQuery.getTopK(), cuvsQuery.getMapping(), numQueries);
  }

  @Override
  public void serialize(OutputStream outputStream) throws Throwable {
    Path p = Files.createTempFile(resources.tempDirectory(), UUID.randomUUID().toString(), ".bf");
    serialize(outputStream, p);
  }

  @Override
  public void serialize(OutputStream outputStream, Path tempFile) throws Throwable {
    checkNotDestroyed();
    tempFile = tempFile.toAbsolutePath();
    MemorySegment pathSeg = Util.buildMemorySegment(resources.getArena(), tempFile.toString());
    try (var localArena = Arena.ofConfined()) {
      MemorySegment returnValue = localArena.allocate(C_INT);
      serializeMethodHandle.invokeExact(
        resources.getMemorySegment(),
        bruteForceIndexReference.getMemorySegment(),
        returnValue,
        pathSeg
      );
      checkError(returnValue.get(C_INT, 0L), "serializeMethodHandle");

      try (FileInputStream fileInputStream = new FileInputStream(tempFile.toFile())) {
        fileInputStream.transferTo(outputStream);
      } finally {
        Files.deleteIfExists(tempFile);
      }
    }
  }

  /**
   * Gets an instance of {@link IndexReference} by deserializing a BRUTEFORCE
   * index using an {@link InputStream}.
   *
   * @param inputStream an instance of {@link InputStream}
   * @return an instance of {@link IndexReference}.
   */
  private IndexReference deserialize(InputStream inputStream) throws Throwable {
    checkNotDestroyed();
    Path tmpIndexFile = Files.createTempFile(resources.tempDirectory(), UUID.randomUUID().toString(), ".bf");
    tmpIndexFile = tmpIndexFile.toAbsolutePath();
    IndexReference indexReference = new IndexReference(resources);

    try (var in = inputStream;
         FileOutputStream fileOutputStream = new FileOutputStream(tmpIndexFile.toFile())) {
      in.transferTo(fileOutputStream);
      MemorySegment pathSeg = Util.buildMemorySegment(resources.getArena(), tmpIndexFile.toString());
      try (var localArena = Arena.ofConfined()) {
        MemorySegment returnValue = localArena.allocate(C_INT);
        deserializeMethodHandle.invokeExact(
          resources.getMemorySegment(),
          indexReference.getMemorySegment(),
          returnValue,
          pathSeg
        );
        checkError(returnValue.get(C_INT, 0L), "deserializeMethodHandle");
      }
    } finally {
      Files.deleteIfExists(tmpIndexFile);
    }
    return indexReference;
  }

  public static BruteForceIndex.Builder newBuilder(CuVSResources cuvsResources) {
    Objects.requireNonNull(cuvsResources);
    if (!(cuvsResources instanceof CuVSResourcesImpl)) {
      throw new IllegalArgumentException("Unsupported " + cuvsResources);
    }
    return new Builder((CuVSResourcesImpl)cuvsResources);
  }

  /**
   * Builder helps configure and create an instance of {@link BruteForceIndex}.
   */
  public static class Builder implements BruteForceIndex.Builder {

    private float[][] vectors;
    private Dataset dataset;
    private final CuVSResourcesImpl cuvsResources;
    private BruteForceIndexParams bruteForceIndexParams;
    private InputStream inputStream;

    /**
     * Constructs this Builder with an instance of {@link CuVSResourcesImpl}.
     *
     * @param cuvsResources an instance of {@link CuVSResources}
     */
    public Builder(CuVSResourcesImpl cuvsResources) {
      this.cuvsResources = cuvsResources;
    }

    /**
     * Registers an instance of configured {@link BruteForceIndexParams} with this
     * Builder.
     *
     * @param bruteForceIndexParams An instance of BruteForceIndexParams
     * @return An instance of this Builder
     */
    @Override
    public Builder withIndexParams(BruteForceIndexParams bruteForceIndexParams) {
      this.bruteForceIndexParams = bruteForceIndexParams;
      return this;
    }

    /**
     * Sets an instance of InputStream typically used when index deserialization is
     * needed.
     *
     * @param inputStream an instance of {@link InputStream}
     * @return an instance of this Builder
     */
    @Override
    public Builder from(InputStream inputStream) {
      this.inputStream = inputStream;
      return this;
    }

    /**
     * Sets the dataset for building the {@link BruteForceIndex}.
     *
     * @param vectors a two-dimensional float array
     * @return an instance of this Builder
     */
    @Override
    public Builder withDataset(float[][] vectors) {
      this.vectors = vectors;
      return this;
    }

    /**
     * Sets the dataset for building the {@link BruteForceIndex}.
     *
     * @param dataset a {@link Dataset} object containing the vectors
     * @return an instance of this Builder
     */
    @Override
    public Builder withDataset(Dataset dataset) {
      this.dataset = dataset;
      return this;
    }

    /**
     * Builds and returns an instance of {@link BruteForceIndex}.
     *
     * @return an instance of {@link BruteForceIndex}
     */
    @Override
    public BruteForceIndexImpl build() throws Throwable {
      if (inputStream != null) {
        return new BruteForceIndexImpl(inputStream, cuvsResources);
      } else {
        return new BruteForceIndexImpl(vectors, dataset, cuvsResources, bruteForceIndexParams);
      }
    }
  }

  /**
   * Holds the memory reference to a BRUTEFORCE index.
   */
  protected static class IndexReference {

    private final MemorySegment memorySegment;

    /**
     * Constructs CagraIndexReference and allocate the MemorySegment.
     */
    protected IndexReference(CuVSResourcesImpl resources) {
      memorySegment = cuvsBruteForceIndex.allocate(resources.getArena());
    }

    /**
     * Constructs BruteForceIndexReference with an instance of MemorySegment passed
     * as a parameter.
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
