/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

/**
 * Parameters for ACE (Augmented Core Extraction) graph build algorithm.
 * ACE enables building indices for datasets too large to fit in GPU memory by:
 * 1. Partitioning the dataset in core (closest) and augmented (second-closest)
 * partitions using balanced k-means.
 * 2. Building sub-indices for each partition independently
 * 3. Concatenating sub-graphs into a final unified index
 *
 * @since 25.12
 */
public class CuVSAceParams {

  /** Number of partitions to split the dataset into */
  private final long npartitions;

  /** ef_construction parameter to control index quality in ACE */
  private final long efConstruction;

  /**
   * Directory to store intermediate build files.
   */
  private final String buildDir;

  /** Whether to use disk-based mode for very large datasets */
  private final boolean useDisk;

  private CuVSAceParams(
      long npartitions, long efConstruction, String buildDir, boolean useDisk) {
    this.npartitions = npartitions;
    this.efConstruction = efConstruction;
    this.buildDir = buildDir;
    this.useDisk = useDisk;
  }

  /**
   * Gets the number of partitions.
   *
   * @return the number of partitions
   */
  public long getNpartitions() {
    return npartitions;
  }

  /**
   * Gets the ef_construction parameter.
   *
   * @return the ef_construction parameter
   */
  public long getEfConstruction() {
    return efConstruction;
  }

  /**
   * Gets the build directory path.
   *
   * @return the build directory path
   */
  public String getBuildDir() {
    return buildDir;
  }

  /**
   * Gets whether disk-based mode is enabled.
   *
   * @return true if disk-based mode is enabled
   */
  public boolean isUseDisk() {
    return useDisk;
  }

  @Override
  public String toString() {
    return "CuVSAceParams [npartitions="
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
   * Builder configures and creates an instance of {@link CuVSAceParams}.
   */
  public static class Builder {

    /** Number of partitions to split the dataset into */
    private long npartitions = 1;

    /** ef_construction parameter for HNSW used in ACE */
    private long efConstruction = 120;

    /** Directory to store intermediate build files */
    private String buildDir = "/tmp/ace_build";

    /** Whether to use disk-based mode for very large datasets */
    private boolean useDisk = false;

    public Builder() {}

    /**
     * Sets the number of partitions.
     *
     * @param npartitions the number of partitions
     * @return an instance of Builder
     */
    public Builder withNpartitions(long npartitions) {
      this.npartitions = npartitions;
      return this;
    }

    /**
     * Sets the ef_construction parameter.
     *
     * @param efConstruction the ef_construction parameter
     * @return an instance of Builder
     */
    public Builder withEfConstruction(long efConstruction) {
      this.efConstruction = efConstruction;
      return this;
    }

    /**
     * Sets the build directory path.
     *
     * @param buildDir the build directory path
     * @return an instance of Builder
     */
    public Builder withBuildDir(String buildDir) {
      this.buildDir = buildDir;
      return this;
    }

    /**
     * Sets whether to use disk-based mode.
     *
     * @param useDisk whether to use disk-based mode
     * @return an instance of Builder
     */
    public Builder withUseDisk(boolean useDisk) {
      this.useDisk = useDisk;
      return this;
    }

    /**
     * Builds an instance of {@link CuVSAceParams}.
     *
     * @return an instance of {@link CuVSAceParams}
     */
    public CuVSAceParams build() {
      return new CuVSAceParams(npartitions, efConstruction, buildDir, useDisk);
    }
  }
}
