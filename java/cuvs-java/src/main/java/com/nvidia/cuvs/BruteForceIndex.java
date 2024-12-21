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

import java.io.IOException;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SequenceLayout;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;

import com.nvidia.cuvs.common.Util;

/**
 * The BRUTEFORCE method is running the KNN algorithm. It performs an extensive
 * search, and in contrast to ANN methods produces an exact result.
 * 
 * {@link BruteForceIndex} encapsulates a BRUTEFORCE index, along with methods
 * to interact with it.
 * 
 * @since 25.02
 */
public class BruteForceIndex {

  private final float[][] dataset;
  private final long[] prefilterData;
  private final CuVSResources resources;
  private MethodHandle indexMethodHandle;
  private MethodHandle searchMethodHandle;
  private MethodHandle destroyIndexMethodHandle;
  private IndexReference bruteForceIndexReference;
  private BruteForceIndexParams bruteForceIndexParams;
  private MemoryLayout longMemoryLayout;
  private MemoryLayout intMemoryLayout;
  private MemoryLayout floatMemoryLayout;

  /**
   * Constructor for building the index using specified dataset
   * 
   * @param dataset               the dataset used for creating the BRUTEFORCE
   *                              index
   * @param resources             an instance of {@link CuVSResources}
   * @param bruteForceIndexParams an instance of {@link BruteForceIndexParams}
   *                              holding the index parameters
   * @param prefilterData         the prefilter data to use while searching the
   *                              BRUTEFORCE index
   */
  private BruteForceIndex(float[][] dataset, CuVSResources resources, BruteForceIndexParams bruteForceIndexParams,
      long[] prefilterData) throws Throwable {
    this.dataset = dataset;
    this.prefilterData = prefilterData;
    this.resources = resources;
    this.bruteForceIndexParams = bruteForceIndexParams;

    longMemoryLayout = resources.linker.canonicalLayouts().get("long");
    intMemoryLayout = resources.linker.canonicalLayouts().get("int");
    floatMemoryLayout = resources.linker.canonicalLayouts().get("float");

    initializeMethodHandles();
    this.bruteForceIndexReference = build();
  }

  /**
   * Initializes the {@link MethodHandles} for invoking native methods.
   * 
   * @throws IOException @{@link IOException} is unable to load the native library
   */
  private void initializeMethodHandles() throws IOException {
    indexMethodHandle = resources.linker.downcallHandle(
        resources.getSymbolLookup().find("build_brute_force_index").get(),
        FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, longMemoryLayout, longMemoryLayout,
            ValueLayout.ADDRESS, ValueLayout.ADDRESS, intMemoryLayout));

    searchMethodHandle = resources.linker.downcallHandle(
        resources.getSymbolLookup().find("search_brute_force_index").get(),
        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS, intMemoryLayout, longMemoryLayout,
            intMemoryLayout, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
            ValueLayout.ADDRESS, longMemoryLayout));

    destroyIndexMethodHandle = resources.linker.downcallHandle(
        resources.getSymbolLookup().find("destroy_brute_force_index").get(),
        FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS));
  }

  /**
   * Invokes the native destroy_brute_force_index function to de-allocate
   * BRUTEFORCE index
   */
  public void destroyIndex() throws Throwable {
    MemoryLayout returnValueMemoryLayout = intMemoryLayout;
    MemorySegment returnValueMemorySegment = resources.arena.allocate(returnValueMemoryLayout);
    destroyIndexMethodHandle.invokeExact(bruteForceIndexReference.getMemorySegment(), returnValueMemorySegment);
  }

  /**
   * Invokes the native build_brute_force_index function via the Panama API to
   * build the {@link BruteForceIndex}
   * 
   * @return an instance of {@link IndexReference} that holds the pointer to the
   *         index
   */
  private IndexReference build() throws Throwable {
    long rows = dataset.length;
    long cols = rows > 0 ? dataset[0].length : 0;

    MemoryLayout returnValueMemoryLayout = intMemoryLayout;
    MemorySegment returnValueMemorySegment = resources.arena.allocate(returnValueMemoryLayout);

    IndexReference indexReference = new IndexReference((MemorySegment) indexMethodHandle.invokeExact(
        Util.buildMemorySegment(resources.linker, resources.arena, dataset), rows, cols, resources.getMemorySegment(),
        returnValueMemorySegment, bruteForceIndexParams.getNumWriterThreads()));

    return indexReference;
  }

  /**
   * Invokes the native search_brute_force_index via the Panama API for searching
   * a BRUTEFORCE index.
   * 
   * @param cuvsQuery an instance of {@link BruteForceQuery} holding the query
   *                  vectors and other parameters
   * @return an instance of {@link BruteForceSearchResults} containing the results
   */
  public BruteForceSearchResults search(BruteForceQuery cuvsQuery) throws Throwable {
    long numQueries = cuvsQuery.getQueryVectors().length;
    long numBlocks = cuvsQuery.getTopK() * numQueries;
    int vectorDimension = numQueries > 0 ? cuvsQuery.getQueryVectors()[0].length : 0;
    long prefilterDataLength = prefilterData != null ? prefilterData.length : 0;

    SequenceLayout neighborsSequenceLayout = MemoryLayout.sequenceLayout(numBlocks, longMemoryLayout);
    SequenceLayout distancesSequenceLayout = MemoryLayout.sequenceLayout(numBlocks, floatMemoryLayout);
    MemorySegment neighborsMemorySegment = resources.arena.allocate(neighborsSequenceLayout);
    MemorySegment distancesMemorySegment = resources.arena.allocate(distancesSequenceLayout);
    MemoryLayout returnValueMemoryLayout = intMemoryLayout;
    MemorySegment returnValueMemorySegment = resources.arena.allocate(returnValueMemoryLayout);
    MemorySegment prefilterDataMemorySegment = prefilterData != null
        ? Util.buildMemorySegment(resources.linker, resources.arena, prefilterData)
        : MemorySegment.NULL;

    searchMethodHandle.invokeExact(bruteForceIndexReference.getMemorySegment(),
        Util.buildMemorySegment(resources.linker, resources.arena, cuvsQuery.getQueryVectors()), cuvsQuery.getTopK(),
        numQueries, vectorDimension, resources.getMemorySegment(), neighborsMemorySegment, distancesMemorySegment,
        returnValueMemorySegment, prefilterDataMemorySegment, prefilterDataLength);

    return new BruteForceSearchResults(neighborsSequenceLayout, distancesSequenceLayout, neighborsMemorySegment,
        distancesMemorySegment, cuvsQuery.getTopK(), cuvsQuery.getMapping(), numQueries);
  }

  /**
   * Builder helps configure and create an instance of {@link BruteForceIndex}.
   */
  public static class Builder {

    private float[][] dataset;
    private long[] prefilterData;
    private CuVSResources cuvsResources;
    private BruteForceIndexParams bruteForceIndexParams;

    /**
     * Constructs this Builder with an instance of {@link CuVSResources}.
     * 
     * @param cuvsResources an instance of {@link CuVSResources}
     */
    public Builder(CuVSResources cuvsResources) {
      this.cuvsResources = cuvsResources;
    }

    /**
     * Registers an instance of configured {@link BruteForceIndexParams} with this
     * Builder.
     * 
     * @param bruteForceIndexParams An instance of BruteForceIndexParams
     * @return An instance of this Builder
     */
    public Builder withIndexParams(BruteForceIndexParams bruteForceIndexParams) {
      this.bruteForceIndexParams = bruteForceIndexParams;
      return this;
    }

    /**
     * Sets the dataset for building the {@link BruteForceIndex}.
     * 
     * @param dataset a two-dimensional float array
     * @return an instance of this Builder
     */
    public Builder withDataset(float[][] dataset) {
      this.dataset = dataset;
      return this;
    }

    /**
     * Sets the prefilter data for building the {@link BruteForceIndex}.
     * 
     * @param prefilterData a one-dimensional long array
     * @return an instance of this Builder
     */
    public Builder withPrefilterData(long[] prefilterData) {
      this.prefilterData = prefilterData;
      return this;
    }

    /**
     * Builds and returns an instance of {@link BruteForceIndex}.
     * 
     * @return an instance of {@link BruteForceIndex}
     */
    public BruteForceIndex build() throws Throwable {
      return new BruteForceIndex(dataset, cuvsResources, bruteForceIndexParams, prefilterData);
    }
  }

  /**
   * Holds the memory reference to a BRUTEFORCE index.
   */
  protected static class IndexReference {

    private final MemorySegment memorySegment;

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
