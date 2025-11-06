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

  /**
   * Number of partitions for ACE (Augmented Core Extraction) partitioned build.
   *
   * Small values might improve recall but potentially degrade performance and increase memory usage.
   * Partitions should not be too small to prevent issues in KNN graph construction. 100k - 5M
   * vectors per partition is recommended depending on the available host and GPU memory. The
   * partition size is on average {@code 2 * (n_rows / npartitions) * dim * sizeof(T)}—the factor 2
   * accounts for core and augmented vectors. Please account for imbalance in the partition sizes
   * (up to 3x in our tests).
   */
  private final long npartitions;

  /**
   * The index quality for the ACE build.
   *
   * Bigger values increase the index quality. At some point, increasing this will no longer improve
   * the quality.
   */
  private final long efConstruction;

  /**
   * Directory to store ACE build artifacts (e.g., KNN graph, optimized graph).
   *
   * Used when {@link #isUseDisk()} is true or when the graph does not fit in host and GPU memory.
   * This should be the fastest disk in the system and hold enough space for twice the dataset, final
   * graph, and label mapping.
   */
  private final String buildDir;

  /**
   * Whether to use disk-based storage for ACE builds.
   *
   * When true, enables disk-based operations for memory-efficient graph construction.
   */
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
   * @see #withNpartitions(long)
   */
  public long getNpartitions() {
    return npartitions;
  }

  /**
   * Gets the {@code ef_construction} parameter.
   *
   * @return the {@code ef_construction} parameter
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
