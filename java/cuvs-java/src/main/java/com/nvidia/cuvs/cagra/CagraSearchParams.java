package com.nvidia.cuvs.cagra;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

import com.nvidia.cuvs.panama.cuvsCagraSearchParams;

/*
* struct cuvsCagraSearchParams {
*     size_t max_queries;
*     size_t itopk_size;
*     size_t max_iterations;
*     enum cuvsCagraSearchAlgo algo;
*     size_t team_size;
*     size_t search_width;
*     size_t min_iterations;
*     size_t thread_block_size;
*     enum cuvsCagraHashMode hashmap_mode;
*     size_t hashmap_min_bitlen;
*     float hashmap_max_fill_rate;
*     uint32_t num_random_samplings;
*     uint64_t rand_xor_mask;
* }
*/
public class CagraSearchParams {

  private Arena arena;
  private int maxQueries;
  private int itopkSize;
  private int maxIterations;
  private CuvsCagraSearchAlgo algo;
  private int teamSize;
  private int searchWidth;
  private int minIterations;
  private int threadBlockSize;
  private CuvsCagraHashMode hashmapMode;
  private int hashmapMinBitlen;
  private float hashmapMaxFillRate;
  private int numRandomSamplings;
  private long randXorMask;
  private MemorySegment cagraSearchParamsMS;

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

  public CagraSearchParams(Arena arena, int max_queries, int itopk_size, int max_iterations, CuvsCagraSearchAlgo algo,
      int team_size, int search_width, int min_iterations, int thread_block_size, CuvsCagraHashMode hashmap_mode,
      int hashmap_min_bitlen, float hashmap_max_fill_rate, int num_random_samplings, long rand_xor_mask) {
    super();
    this.arena = arena;
    this.maxQueries = max_queries;
    this.itopkSize = itopk_size;
    this.maxIterations = max_iterations;
    this.algo = algo;
    this.teamSize = team_size;
    this.searchWidth = search_width;
    this.minIterations = min_iterations;
    this.threadBlockSize = thread_block_size;
    this.hashmapMode = hashmap_mode;
    this.hashmapMinBitlen = hashmap_min_bitlen;
    this.hashmapMaxFillRate = hashmap_max_fill_rate;
    this.numRandomSamplings = num_random_samplings;
    this.randXorMask = rand_xor_mask;
    this.set();
  }

  public void set() {
    cagraSearchParamsMS = cuvsCagraSearchParams.allocate(arena);
    cuvsCagraSearchParams.max_queries(cagraSearchParamsMS, maxQueries);
    cuvsCagraSearchParams.itopk_size(cagraSearchParamsMS, itopkSize);
    cuvsCagraSearchParams.max_iterations(cagraSearchParamsMS, maxIterations);
    cuvsCagraSearchParams.algo(cagraSearchParamsMS, algo.label);
    cuvsCagraSearchParams.team_size(cagraSearchParamsMS, teamSize);
    cuvsCagraSearchParams.search_width(cagraSearchParamsMS, searchWidth);
    cuvsCagraSearchParams.min_iterations(cagraSearchParamsMS, minIterations);
    cuvsCagraSearchParams.thread_block_size(cagraSearchParamsMS, threadBlockSize);
    cuvsCagraSearchParams.hashmap_mode(cagraSearchParamsMS, hashmapMode.label);
    cuvsCagraSearchParams.hashmap_min_bitlen(cagraSearchParamsMS, hashmapMinBitlen);
    cuvsCagraSearchParams.hashmap_max_fill_rate(cagraSearchParamsMS, hashmapMaxFillRate);
    cuvsCagraSearchParams.num_random_samplings(cagraSearchParamsMS, numRandomSamplings);
    cuvsCagraSearchParams.rand_xor_mask(cagraSearchParamsMS, randXorMask);
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
    return algo;
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
    return hashmapMode;
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
    return cagraSearchParamsMS;
  }

  @Override
  public String toString() {
    return "CagraSearchParams [max_queries=" + maxQueries + ", itopk_size=" + itopkSize + ", max_iterations="
        + maxIterations + ", algo=" + algo + ", team_size=" + teamSize + ", search_width=" + searchWidth
        + ", min_iterations=" + minIterations + ", thread_block_size=" + threadBlockSize + ", hashmap_mode="
        + hashmapMode + ", hashmap_min_bitlen=" + hashmapMinBitlen + ", hashmap_max_fill_rate=" + hashmapMaxFillRate
        + ", num_random_samplings=" + numRandomSamplings + ", rand_xor_mask=" + randXorMask + "]";
  }

  public static class Builder {

    Arena arena;
    int maxQueries = 1;
    int itopkSize = 2;
    int maxIterations = 3;
    CuvsCagraSearchAlgo algo = CuvsCagraSearchAlgo.MULTI_KERNEL;
    int teamSize = 4;
    int searchWidth = 5;
    int minIterations = 6;
    int threadBlockSize = 7;
    CuvsCagraHashMode hashmapMode = CuvsCagraHashMode.AUTO_HASH;
    int hashmapMinBitlen = 8;
    float hashmapMaxFillRate = 9.0f;
    int numRandomSamplings = 10;
    long randXorMask = 11L;

    public Builder() {
      this.arena = Arena.ofConfined();
    }

    public Builder withMaxQueries(int max_queries) {
      this.maxQueries = max_queries;
      return this;
    }

    public Builder withItopkSize(int itopk_size) {
      this.itopkSize = itopk_size;
      return this;
    }

    public Builder withMaxIterations(int max_iterations) {
      this.maxIterations = max_iterations;
      return this;
    }

    public Builder withAlgo(CuvsCagraSearchAlgo algo) {
      this.algo = algo;
      return this;
    }

    public Builder withTeamSize(int team_size) {
      this.teamSize = team_size;
      return this;
    }

    public Builder withSearchWidth(int search_width) {
      this.searchWidth = search_width;
      return this;
    }

    public Builder withMinIterations(int min_iterations) {
      this.minIterations = min_iterations;
      return this;
    }

    public Builder withThreadBlockSize(int thread_block_size) {
      this.threadBlockSize = thread_block_size;
      return this;
    }

    public Builder withHashmapMode(CuvsCagraHashMode hashmap_mode) {
      this.hashmapMode = hashmap_mode;
      return this;
    }

    public Builder withHashmapMinBitlen(int hashmap_min_bitlen) {
      this.hashmapMinBitlen = hashmap_min_bitlen;
      return this;
    }

    public Builder withHashmapMaxFillRate(float hashmap_max_fill_rate) {
      this.hashmapMaxFillRate = hashmap_max_fill_rate;
      return this;
    }

    public Builder withNumRandomSamplings(int num_random_samplings) {
      this.numRandomSamplings = num_random_samplings;
      return this;
    }

    public Builder withRandXorMask(long rand_xor_mask) {
      this.randXorMask = rand_xor_mask;
      return this;
    }

    public CagraSearchParams build() throws Throwable {
      return new CagraSearchParams(arena, maxQueries, itopkSize, maxIterations, algo, teamSize, searchWidth,
          minIterations, threadBlockSize, hashmapMode, hashmapMinBitlen, hashmapMaxFillRate, numRandomSamplings,
          randXorMask);
    }

  }
}
