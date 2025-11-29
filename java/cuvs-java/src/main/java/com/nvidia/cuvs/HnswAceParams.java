/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

/**
 * Parameters for ACE (Augmented Core Extraction) graph build for HNSW.
 * ACE enables building indexes for datasets too large to fit in GPU memory by:
 * 1. Partitioning the dataset in core and augmented partitions using balanced k-means
 * 2. Building sub-indexes for each partition independently
 * 3. Concatenating sub-graphs into a final unified index
 *
 * @since 25.02
 */
public class HnswAceParams {

  private long npartitions;
  private long efConstruction;
  private String buildDir;
  private boolean useDisk;

  private HnswAceParams(long npartitions, long efConstruction, String buildDir, boolean useDisk) {
    this.npartitions = npartitions;
    this.efConstruction = efConstruction;
    this.buildDir = buildDir;
    this.useDisk = useDisk;
  }

  /**
   * Gets the number of partitions for ACE partitioned build.
   *
   * @return the number of partitions
   */
  public long getNpartitions() {
    return npartitions;
  }

  /**
   * Gets the index quality for the ACE build.
   *
   * @return the ef_construction value
   */
  public long getEfConstruction() {
    return efConstruction;
  }

  /**
   * Gets the directory to store ACE build artifacts.
   *
   * @return the build directory path
   */
  public String getBuildDir() {
    return buildDir;
  }

  /**
   * Gets whether disk-based storage is enabled for ACE build.
   *
   * @return true if disk mode is enabled
   */
  public boolean isUseDisk() {
    return useDisk;
  }

  @Override
  public String toString() {
    return "HnswAceParams [npartitions="
        + npartitions
        + ", efConstruction="
        + efConstruction
        + ", buildDir="
        + buildDir
        + ", useDisk="
        + useDisk
        + "]";
  }

  /**
   * Builder configures and creates an instance of {@link HnswAceParams}.
   */
  public static class Builder {

    private long npartitions = 1;
    private long efConstruction = 120;
    private String buildDir = "/tmp/hnsw_ace_build";
    private boolean useDisk = false;

    /**
     * Constructs this Builder.
     */
    public Builder() {}

    /**
     * Sets the number of partitions for ACE partitioned build.
     * Small values might improve recall but potentially degrade performance.
     * 100k - 5M vectors per partition is recommended.
     *
     * @param npartitions the number of partitions
     * @return an instance of Builder
     */
    public Builder withNpartitions(long npartitions) {
      this.npartitions = npartitions;
      return this;
    }

    /**
     * Sets the index quality for the ACE build.
     * Bigger values increase the index quality.
     *
     * @param efConstruction the ef_construction value
     * @return an instance of Builder
     */
    public Builder withEfConstruction(long efConstruction) {
      this.efConstruction = efConstruction;
      return this;
    }

    /**
     * Sets the directory to store ACE build artifacts.
     * Used when useDisk is true or when the graph does not fit in memory.
     *
     * @param buildDir the build directory path
     * @return an instance of Builder
     */
    public Builder withBuildDir(String buildDir) {
      this.buildDir = buildDir;
      return this;
    }

    /**
     * Sets whether to use disk-based storage for ACE build.
     * When true, enables disk-based operations for memory-efficient graph construction.
     *
     * @param useDisk true to enable disk mode
     * @return an instance of Builder
     */
    public Builder withUseDisk(boolean useDisk) {
      this.useDisk = useDisk;
      return this;
    }

    /**
     * Builds an instance of {@link HnswAceParams}.
     *
     * @return an instance of {@link HnswAceParams}
     */
    public HnswAceParams build() {
      return new HnswAceParams(npartitions, efConstruction, buildDir, useDisk);
    }
  }
}
