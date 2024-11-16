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

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

import com.nvidia.cuvs.panama.cuvsCagraSearchParams;

/**
 * CagraSearchParams encapsulates the logic for configuring and holding search
 * parameters.
 * 
 * @since 24.12
 */
public class CagraSearchParams {

  private int maxQueries;
  private int itopkSize;
  private int maxIterations;
  private int teamSize;
  private int searchWidth;
  private int minIterations;
  private int threadBlockSize;
  private int hashmapMinBitlen;
  private int numRandomSamplings;
  private float hashmapMaxFillRate;
  private long randXorMask;
  private Arena arena;
  private MemorySegment cagraSearchParamsMemorySegment;
  private CuvsCagraSearchAlgo cuvsCagraSearchAlgo;
  private CuvsCagraHashMapMode cuvsCagraHashMapMode;

  /**
   * Enum to denote algorithm used to search CAGRA Index.
   */
  public enum CuvsCagraSearchAlgo {
    /**
     * for large batch sizes
     */
    SINGLE_CTA(0),
    /**
     * for small batch sizes
     */
    MULTI_CTA(1),
    /**
     * MULTI_KERNEL
     */
    MULTI_KERNEL(2),
    /**
     * AUTO
     */
    AUTO(3);

    /**
     * The value for the enum choice.
     */
    public final int value;

    private CuvsCagraSearchAlgo(int value) {
      this.value = value;
    }
  }

  /**
   * Enum to denote Hash Mode used while searching CAGRA index.
   */
  public enum CuvsCagraHashMapMode {
    /**
     * HASH
     */
    HASH(0),
    /**
     * SMALL
     */
    SMALL(1),
    /**
     * AUTO_HASH
     */
    AUTO_HASH(2);

    /**
     * The value for the enum choice.
     */
    public final int value;

    private CuvsCagraHashMapMode(int value) {
      this.value = value;
    }
  }

  /**
   * Constructs an instance of CagraSearchParams with passed search parameters.
   * 
   * @param arena               the Arena instance to use
   * @param maxQueries          the maximum number of queries to search at the
   *                            same time (batch size)
   * @param itopkSize           the number of intermediate search results retained
   *                            during the search
   * @param maxIterations       the upper limit of search iterations
   * @param cuvsCagraSearchAlgo the search implementation is configured
   * @param teamSize            the number of threads used to calculate a single
   *                            distance
   * @param searchWidth         the number of graph nodes to select as the
   *                            starting point for the search in each iteration
   * @param minIterations       the lower limit of search iterations
   * @param threadBlockSize     the thread block size
   * @param hashmapMode         the hashmap type configured
   * @param hashmapMinBitlen    the lower limit of hashmap bit length
   * @param hashmapMaxFillRate  the upper limit of hashmap fill rate
   * @param numRandomSamplings  the number of iterations of initial random seed
   *                            node selection
   * @param randXorMask         the bit mask used for initial random seed node
   *                            selection
   */
  public CagraSearchParams(Arena arena, int maxQueries, int itopkSize, int maxIterations,
      CuvsCagraSearchAlgo cuvsCagraSearchAlgo, int teamSize, int searchWidth, int minIterations, int threadBlockSize,
      CuvsCagraHashMapMode hashmapMode, int hashmapMinBitlen, float hashmapMaxFillRate, int numRandomSamplings,
      long randXorMask) {
    super();
    this.arena = arena;
    this.maxQueries = maxQueries;
    this.itopkSize = itopkSize;
    this.maxIterations = maxIterations;
    this.cuvsCagraSearchAlgo = cuvsCagraSearchAlgo;
    this.teamSize = teamSize;
    this.searchWidth = searchWidth;
    this.minIterations = minIterations;
    this.threadBlockSize = threadBlockSize;
    this.cuvsCagraHashMapMode = hashmapMode;
    this.hashmapMinBitlen = hashmapMinBitlen;
    this.hashmapMaxFillRate = hashmapMaxFillRate;
    this.numRandomSamplings = numRandomSamplings;
    this.randXorMask = randXorMask;
    this.setSearchParametersInTheStubMemorySegment();
  }

  /**
   * Sets the configured search parameters in the MemorySegment.
   */
  public void setSearchParametersInTheStubMemorySegment() {
    cagraSearchParamsMemorySegment = cuvsCagraSearchParams.allocate(arena);
    cuvsCagraSearchParams.max_queries(cagraSearchParamsMemorySegment, maxQueries);
    cuvsCagraSearchParams.itopk_size(cagraSearchParamsMemorySegment, itopkSize);
    cuvsCagraSearchParams.max_iterations(cagraSearchParamsMemorySegment, maxIterations);
    cuvsCagraSearchParams.algo(cagraSearchParamsMemorySegment, cuvsCagraSearchAlgo.value);
    cuvsCagraSearchParams.team_size(cagraSearchParamsMemorySegment, teamSize);
    cuvsCagraSearchParams.search_width(cagraSearchParamsMemorySegment, searchWidth);
    cuvsCagraSearchParams.min_iterations(cagraSearchParamsMemorySegment, minIterations);
    cuvsCagraSearchParams.thread_block_size(cagraSearchParamsMemorySegment, threadBlockSize);
    cuvsCagraSearchParams.hashmap_mode(cagraSearchParamsMemorySegment, cuvsCagraHashMapMode.value);
    cuvsCagraSearchParams.hashmap_min_bitlen(cagraSearchParamsMemorySegment, hashmapMinBitlen);
    cuvsCagraSearchParams.hashmap_max_fill_rate(cagraSearchParamsMemorySegment, hashmapMaxFillRate);
    cuvsCagraSearchParams.num_random_samplings(cagraSearchParamsMemorySegment, numRandomSamplings);
    cuvsCagraSearchParams.rand_xor_mask(cagraSearchParamsMemorySegment, randXorMask);
  }

  /**
   * Gets the maximum number of queries to search at the same time (batch size).
   * 
   * @return the maximum number of queries
   */
  public int getMaxQueries() {
    return maxQueries;
  }

  /**
   * Gets the number of intermediate search results retained during the search.
   * 
   * @return the number of intermediate search results
   */
  public int getItopkSize() {
    return itopkSize;
  }

  /**
   * Gets the upper limit of search iterations.
   * 
   * @return the upper limit value
   */
  public int getMaxIterations() {
    return maxIterations;
  }

  /**
   * Gets the number of threads used to calculate a single distance.
   * 
   * @return the number of threads configured
   */
  public int getTeamSize() {
    return teamSize;
  }

  /**
   * Gets the number of graph nodes to select as the starting point for the search
   * in each iteration.
   * 
   * @return the number of graph nodes
   */
  public int getSearchWidth() {
    return searchWidth;
  }

  /**
   * Gets the lower limit of search iterations.
   * 
   * @return the lower limit value
   */
  public int getMinIterations() {
    return minIterations;
  }

  /**
   * Gets the thread block size.
   * 
   * @return the thread block size
   */
  public int getThreadBlockSize() {
    return threadBlockSize;
  }

  /**
   * Gets the lower limit of hashmap bit length.
   * 
   * @return the lower limit value
   */
  public int getHashmapMinBitlen() {
    return hashmapMinBitlen;
  }

  /**
   * Gets the number of iterations of initial random seed node selection.
   * 
   * @return the number of iterations
   */
  public int getNumRandomSamplings() {
    return numRandomSamplings;
  }

  /**
   * Gets the upper limit of hashmap fill rate.
   * 
   * @return the upper limit of hashmap fill rate
   */
  public float getHashmapMaxFillRate() {
    return hashmapMaxFillRate;
  }

  /**
   * Gets the bit mask used for initial random seed node selection.
   * 
   * @return the bit mask value
   */
  public long getRandXorMask() {
    return randXorMask;
  }

  /**
   * Gets the MemorySegment holding CagraSearchParams.
   * 
   * @return the MemorySegment holding CagraSearchParams
   */
  public MemorySegment getCagraSearchParamsMemorySegment() {
    return cagraSearchParamsMemorySegment;
  }

  /**
   * Gets which search implementation is configured.
   * 
   * @return the search implementation configured
   * @see CuvsCagraSearchAlgo
   */
  public CuvsCagraSearchAlgo getCuvsCagraSearchAlgo() {
    return cuvsCagraSearchAlgo;
  }

  /**
   * Gets the hashmap type configured.
   * 
   * @return the hashmap type configured
   * @see CuvsCagraHashMapMode
   */
  public CuvsCagraHashMapMode getCuvsCagraHashMapMode() {
    return cuvsCagraHashMapMode;
  }

  @Override
  public String toString() {
    return "CagraSearchParams [arena=" + arena + ", maxQueries=" + maxQueries + ", itopkSize=" + itopkSize
        + ", maxIterations=" + maxIterations + ", cuvsCagraSearchAlgo=" + cuvsCagraSearchAlgo + ", teamSize=" + teamSize
        + ", searchWidth=" + searchWidth + ", minIterations=" + minIterations + ", threadBlockSize=" + threadBlockSize
        + ", cuvsCagraHashMapMode=" + cuvsCagraHashMapMode + ", hashmapMinBitlen=" + hashmapMinBitlen
        + ", hashmapMaxFillRate=" + hashmapMaxFillRate + ", numRandomSamplings=" + numRandomSamplings + ", randXorMask="
        + randXorMask + ", cagraSearchParamsMemorySegment=" + cagraSearchParamsMemorySegment + "]";
  }

  /**
   * Builder configures and creates an instance of CagraSearchParams.
   */
  public static class Builder {

    private Arena arena;
    private int maxQueries = 1;
    private int itopkSize = 2;
    private int maxIterations = 3;
    private int teamSize = 4;
    private int searchWidth = 5;
    private int minIterations = 6;
    private int threadBlockSize = 7;
    private int hashmapMinBitlen = 8;
    private int numRandomSamplings = 10;
    private float hashmapMaxFillRate = 9.0f;
    private long randXorMask = 11L;
    private CuvsCagraSearchAlgo cuvsCagraSearchAlgo = CuvsCagraSearchAlgo.MULTI_KERNEL;
    private CuvsCagraHashMapMode cuvsCagraHashMode = CuvsCagraHashMapMode.AUTO_HASH;

    /**
     * Constructs this Builder with an instance of Arena.
     */
    public Builder() {
      this.arena = Arena.ofConfined();
    }

    /**
     * Sets the maximum number of queries to search at the same time (batch size).
     * Auto select when 0.
     * 
     * @param maxQueries the maximum number of queries
     * @return an instance of this Builder
     */
    public Builder withMaxQueries(int maxQueries) {
      this.maxQueries = maxQueries;
      return this;
    }

    /**
     * Sets the number of intermediate search results retained during the search.
     * This is the main knob to adjust trade off between accuracy and search speed.
     * Higher values improve the search accuracy.
     * 
     * @param itopkSize the number of intermediate search results
     * @return an instance of this Builder
     */
    public Builder withItopkSize(int itopkSize) {
      this.itopkSize = itopkSize;
      return this;
    }

    /**
     * Sets the upper limit of search iterations. Auto select when 0.
     * 
     * @param maxIterations the upper limit of search iterations
     * @return an instance of this Builder
     */
    public Builder withMaxIterations(int maxIterations) {
      this.maxIterations = maxIterations;
      return this;
    }

    /**
     * Sets which search implementation to use.
     * 
     * @param cuvsCagraSearchAlgo the search implementation to use
     * @return an instance of this Builder
     * @see CuvsCagraSearchAlgo
     */
    public Builder withAlgo(CuvsCagraSearchAlgo cuvsCagraSearchAlgo) {
      this.cuvsCagraSearchAlgo = cuvsCagraSearchAlgo;
      return this;
    }

    /**
     * Sets the number of threads used to calculate a single distance. 4, 8, 16, or
     * 32.
     * 
     * @param teamSize the number of threads used to calculate a single distance
     * @return an instance of this Builder
     */
    public Builder withTeamSize(int teamSize) {
      this.teamSize = teamSize;
      return this;
    }

    /**
     * Sets the number of graph nodes to select as the starting point for the search
     * in each iteration.
     * 
     * @param searchWidth the number of graph nodes to select
     * @return an instance of this Builder
     */
    public Builder withSearchWidth(int searchWidth) {
      this.searchWidth = searchWidth;
      return this;
    }

    /**
     * Sets the lower limit of search iterations.
     * 
     * @param minIterations the lower limit of search iterations
     * @return an instance of this Builder
     */
    public Builder withMinIterations(int minIterations) {
      this.minIterations = minIterations;
      return this;
    }

    /**
     * Sets the thread block size. 0, 64, 128, 256, 512, 1024. Auto selection when
     * 0.
     * 
     * @param threadBlockSize the thread block size
     * @return an instance of this Builder
     */
    public Builder withThreadBlockSize(int threadBlockSize) {
      this.threadBlockSize = threadBlockSize;
      return this;
    }

    /**
     * Sets the hashmap type. Auto selection when AUTO.
     * 
     * @param cuvsCagraHashMode the hashmap type
     * @return an instance of this Builder
     * @see CuvsCagraHashMapMode
     */
    public Builder withHashmapMode(CuvsCagraHashMapMode cuvsCagraHashMode) {
      this.cuvsCagraHashMode = cuvsCagraHashMode;
      return this;
    }

    /**
     * Sets the lower limit of hashmap bit length. More than 8.
     * 
     * @param hashmapMinBitlen the lower limit of hashmap bit length
     * @return an instance of this Builder
     */
    public Builder withHashmapMinBitlen(int hashmapMinBitlen) {
      this.hashmapMinBitlen = hashmapMinBitlen;
      return this;
    }

    /**
     * Sets the upper limit of hashmap fill rate. More than 0.1, less than 0.9.
     * 
     * @param hashmapMaxFillRate the upper limit of hashmap fill rate
     * @return an instance of this Builder
     */
    public Builder withHashmapMaxFillRate(float hashmapMaxFillRate) {
      this.hashmapMaxFillRate = hashmapMaxFillRate;
      return this;
    }

    /**
     * Sets the number of iterations of initial random seed node selection. 1 or
     * more.
     * 
     * @param numRandomSamplings the number of iterations of initial random seed
     *                           node selection
     * @return an instance of this Builder
     */
    public Builder withNumRandomSamplings(int numRandomSamplings) {
      this.numRandomSamplings = numRandomSamplings;
      return this;
    }

    /**
     * Sets the bit mask used for initial random seed node selection.
     * 
     * @param randXorMask the bit mask used for initial random seed node selection
     * @return an instance of this Builder
     */
    public Builder withRandXorMask(long randXorMask) {
      this.randXorMask = randXorMask;
      return this;
    }

    /**
     * Builds an instance of CagraSearchParams with passed search parameters.
     * 
     * @return an instance of CagraSearchParams
     */
    public CagraSearchParams build() {
      return new CagraSearchParams(arena, maxQueries, itopkSize, maxIterations, cuvsCagraSearchAlgo, teamSize,
          searchWidth, minIterations, threadBlockSize, cuvsCagraHashMode, hashmapMinBitlen, hashmapMaxFillRate,
          numRandomSamplings, randXorMask);
    }
  }
}
