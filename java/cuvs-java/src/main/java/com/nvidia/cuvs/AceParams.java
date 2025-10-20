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
package com.nvidia.cuvs;

/**
 * Parameters for ACE (Augmented Core Extraction) graph build algorithm.
 *
 * @since 25.12
 */
public class AceParams {

  /**
   * Number of partitions for ACE (Augmented Core Extraction) partitioned build.
   * Small values might improve recall but potentially degrade performance and
   * increase memory usage. Partitions should not be too small to prevent issues
   * in KNN graph construction. 100k - 5M vectors per partition is recommended
   * depending on the available host and GPU memory.
   */
  private final int aceNpartitions;

  /**
   * The index quality for the ACE build.
   * Bigger values increase the index quality. At some point, increasing this will no longer
   * improve the quality.
   */
  private final int aceEfConstruction;

  /**
   * Directory to store ACE build artifacts (e.g., KNN graph, optimized graph).
   * Used when aceNpartitions > 1 or aceUseDisk is true.
   */
  private final String aceBuildDir;

  /**
   * Whether to use disk-based storage for ACE build.
   * When true, enables disk-based operations for memory-efficient graph construction.
   */
  private final boolean aceUseDisk;

  private AceParams(int aceNpartitions, int aceEfConstruction, String aceBuildDir, boolean aceUseDisk) {
    this.aceNpartitions = aceNpartitions;
    this.aceEfConstruction = aceEfConstruction;
    this.aceBuildDir = aceBuildDir;
    this.aceUseDisk = aceUseDisk;
  }

  /**
   * Gets the number of partitions for ACE build.
   *
   * @return the number of partitions
   */
  public int getAceNpartitions() {
    return aceNpartitions;
  }

  /**
   * Gets the ef_construction parameter for ACE build.
   *
   * @return the ef_construction value
   */
  public int getAceEfConstruction() {
    return aceEfConstruction;
  }

  /**
   * Gets the build directory for ACE artifacts.
   *
   * @return the build directory path
   */
  public String getAceBuildDir() {
    return aceBuildDir;
  }

  /**
   * Gets whether disk-based storage is enabled for ACE build.
   *
   * @return true if disk-based storage is enabled
   */
  public boolean isAceUseDisk() {
    return aceUseDisk;
  }

  @Override
  public String toString() {
    return "AceParams [aceNpartitions="
        + aceNpartitions
        + ", aceEfConstruction="
        + aceEfConstruction
        + ", aceBuildDir="
        + aceBuildDir
        + ", aceUseDisk="
        + aceUseDisk
        + "]";
  }

  /**
   * Builder configures and creates an instance of {@link AceParams}.
   */
  public static class Builder {

    private int aceNpartitions = 1;
    private int aceEfConstruction = 120;
    private String aceBuildDir = "/tmp/ace_build";
    private boolean aceUseDisk = false;

    public Builder() {}

    /**
     * Sets the number of partitions for ACE build.
     *
     * @param aceNpartitions the number of partitions
     * @return an instance of Builder
     */
    public Builder withAceNpartitions(int aceNpartitions) {
      this.aceNpartitions = aceNpartitions;
      return this;
    }

    /**
     * Sets the ef_construction parameter for ACE build.
     *
     * @param aceEfConstruction the ef_construction value
     * @return an instance of Builder
     */
    public Builder withAceEfConstruction(int aceEfConstruction) {
      this.aceEfConstruction = aceEfConstruction;
      return this;
    }

    /**
     * Sets the build directory for ACE artifacts.
     *
     * @param aceBuildDir the build directory path
     * @return an instance of Builder
     */
    public Builder withAceBuildDir(String aceBuildDir) {
      this.aceBuildDir = aceBuildDir;
      return this;
    }

    /**
     * Sets whether to use disk-based storage for ACE build.
     *
     * @param aceUseDisk true to enable disk-based storage
     * @return an instance of Builder
     */
    public Builder withAceUseDisk(boolean aceUseDisk) {
      this.aceUseDisk = aceUseDisk;
      return this;
    }

    /**
     * Builds an instance of {@link AceParams}.
     *
     * @return an instance of {@link AceParams}
     */
    public AceParams build() {
      return new AceParams(aceNpartitions, aceEfConstruction, aceBuildDir, aceUseDisk);
    }
  }
}
