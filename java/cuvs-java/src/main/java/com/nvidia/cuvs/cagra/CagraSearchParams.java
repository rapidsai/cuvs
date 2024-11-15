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

public class CagraSearchParams {

  private Arena arena;
  private int maxQueries;
  private int itopkSize;
  private int maxIterations;
  private CuvsCagraSearchAlgo cuvsCagraSearchAlgo;
  private int teamSize;
  private int searchWidth;
  private int minIterations;
  private int threadBlockSize;
  private CuvsCagraHashMode cuvsCagraHashMode;
  private int hashmapMinBitlen;
  private float hashmapMaxFillRate;
  private int numRandomSamplings;
  private long randXorMask;
  private MemorySegment cagraSearchParamsMemorySegment;

  enum CuvsCagraSearchAlgo {
    SINGLE_CTA(0), MULTI_CTA(1), MULTI_KERNEL(2), AUTO(3);

    public final int label;

    private CuvsCagraSearchAlgo(int label) {
      this.label = label;
    }
  }

  enum CuvsCagraHashMode {
    HASH(0), SMALL(1), AUTO_HASH(2);

    public final int label;

    private CuvsCagraHashMode(int label) {
      this.label = label;
    }
  }

  public CagraSearchParams(Arena arena, int maxQueries, int itopkSize, int maxIterations,
      CuvsCagraSearchAlgo cuvsCagraSearchAlgo, int teamSize, int searchWidth, int minIterations, int threadBlockSize,
      CuvsCagraHashMode hashmapMode, int hashmapMinBitlen, float hashmapMaxFillRate, int numRandomSamplings,
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
    this.cuvsCagraHashMode = hashmapMode;
    this.hashmapMinBitlen = hashmapMinBitlen;
    this.hashmapMaxFillRate = hashmapMaxFillRate;
    this.numRandomSamplings = numRandomSamplings;
    this.randXorMask = randXorMask;
    this.set();
  }

  public void set() {
    cagraSearchParamsMemorySegment = cuvsCagraSearchParams.allocate(arena);
    cuvsCagraSearchParams.max_queries(cagraSearchParamsMemorySegment, maxQueries);
    cuvsCagraSearchParams.itopk_size(cagraSearchParamsMemorySegment, itopkSize);
    cuvsCagraSearchParams.max_iterations(cagraSearchParamsMemorySegment, maxIterations);
    cuvsCagraSearchParams.algo(cagraSearchParamsMemorySegment, cuvsCagraSearchAlgo.label);
    cuvsCagraSearchParams.team_size(cagraSearchParamsMemorySegment, teamSize);
    cuvsCagraSearchParams.search_width(cagraSearchParamsMemorySegment, searchWidth);
    cuvsCagraSearchParams.min_iterations(cagraSearchParamsMemorySegment, minIterations);
    cuvsCagraSearchParams.thread_block_size(cagraSearchParamsMemorySegment, threadBlockSize);
    cuvsCagraSearchParams.hashmap_mode(cagraSearchParamsMemorySegment, cuvsCagraHashMode.label);
    cuvsCagraSearchParams.hashmap_min_bitlen(cagraSearchParamsMemorySegment, hashmapMinBitlen);
    cuvsCagraSearchParams.hashmap_max_fill_rate(cagraSearchParamsMemorySegment, hashmapMaxFillRate);
    cuvsCagraSearchParams.num_random_samplings(cagraSearchParamsMemorySegment, numRandomSamplings);
    cuvsCagraSearchParams.rand_xor_mask(cagraSearchParamsMemorySegment, randXorMask);
  }

  public int getMax_queries() {
    return maxQueries;
  }

  public int getItopk_size() {
    return itopkSize;
  }

  public int getMax_iterations() {
    return maxIterations;
  }

  public CuvsCagraSearchAlgo getAlgo() {
    return cuvsCagraSearchAlgo;
  }

  public int getTeam_size() {
    return teamSize;
  }

  public int getSearch_width() {
    return searchWidth;
  }

  public int getMin_iterations() {
    return minIterations;
  }

  public int getThread_block_size() {
    return threadBlockSize;
  }

  public CuvsCagraHashMode getHashmap_mode() {
    return cuvsCagraHashMode;
  }

  public int getHashmap_min_bitlen() {
    return hashmapMinBitlen;
  }

  public float getHashmap_max_fill_rate() {
    return hashmapMaxFillRate;
  }

  public int getNum_random_samplings() {
    return numRandomSamplings;
  }

  public long getRand_xor_mask() {
    return randXorMask;
  }

  public MemorySegment getCagraSearchParamsMS() {
    return cagraSearchParamsMemorySegment;
  }

  @Override
  public String toString() {
    return "CagraSearchParams [arena=" + arena + ", maxQueries=" + maxQueries + ", itopkSize=" + itopkSize
        + ", maxIterations=" + maxIterations + ", cuvsCagraSearchAlgo=" + cuvsCagraSearchAlgo + ", teamSize=" + teamSize
        + ", searchWidth=" + searchWidth + ", minIterations=" + minIterations + ", threadBlockSize=" + threadBlockSize
        + ", cuvsCagraHashMode=" + cuvsCagraHashMode + ", hashmapMinBitlen=" + hashmapMinBitlen
        + ", hashmapMaxFillRate=" + hashmapMaxFillRate + ", numRandomSamplings=" + numRandomSamplings + ", randXorMask="
        + randXorMask + ", cagraSearchParamsMemorySegment=" + cagraSearchParamsMemorySegment + "]";
  }

  public static class Builder {

    private Arena arena;
    private int maxQueries = 1;
    private int itopkSize = 2;
    private int maxIterations = 3;
    private CuvsCagraSearchAlgo cuvsCagraSearchAlgo = CuvsCagraSearchAlgo.MULTI_KERNEL;
    private int teamSize = 4;
    private int searchWidth = 5;
    private int minIterations = 6;
    private int threadBlockSize = 7;
    private CuvsCagraHashMode cuvsCagraHashMode = CuvsCagraHashMode.AUTO_HASH;
    private int hashmapMinBitlen = 8;
    private float hashmapMaxFillRate = 9.0f;
    private int numRandomSamplings = 10;
    private long randXorMask = 11L;

    public Builder() {
      this.arena = Arena.ofConfined();
    }

    public Builder withMaxQueries(int maxQueries) {
      this.maxQueries = maxQueries;
      return this;
    }

    public Builder withItopkSize(int itopkSize) {
      this.itopkSize = itopkSize;
      return this;
    }

    public Builder withMaxIterations(int maxIterations) {
      this.maxIterations = maxIterations;
      return this;
    }

    public Builder withAlgo(CuvsCagraSearchAlgo cuvsCagraSearchAlgo) {
      this.cuvsCagraSearchAlgo = cuvsCagraSearchAlgo;
      return this;
    }

    public Builder withTeamSize(int teamSize) {
      this.teamSize = teamSize;
      return this;
    }

    public Builder withSearchWidth(int searchWidth) {
      this.searchWidth = searchWidth;
      return this;
    }

    public Builder withMinIterations(int minIterations) {
      this.minIterations = minIterations;
      return this;
    }

    public Builder withThreadBlockSize(int threadBlockSize) {
      this.threadBlockSize = threadBlockSize;
      return this;
    }

    public Builder withHashmapMode(CuvsCagraHashMode cuvsCagraHashMode) {
      this.cuvsCagraHashMode = cuvsCagraHashMode;
      return this;
    }

    public Builder withHashmapMinBitlen(int hashmapMinBitlen) {
      this.hashmapMinBitlen = hashmapMinBitlen;
      return this;
    }

    public Builder withHashmapMaxFillRate(float hashmapMaxFillRate) {
      this.hashmapMaxFillRate = hashmapMaxFillRate;
      return this;
    }

    public Builder withNumRandomSamplings(int numRandomSamplings) {
      this.numRandomSamplings = numRandomSamplings;
      return this;
    }

    public Builder withRandXorMask(long randXorMask) {
      this.randXorMask = randXorMask;
      return this;
    }

    public CagraSearchParams build() throws Throwable {
      return new CagraSearchParams(arena, maxQueries, itopkSize, maxIterations, cuvsCagraSearchAlgo, teamSize,
          searchWidth, minIterations, threadBlockSize, cuvsCagraHashMode, hashmapMinBitlen, hashmapMaxFillRate,
          numRandomSamplings, randXorMask);
    }

  }
}
