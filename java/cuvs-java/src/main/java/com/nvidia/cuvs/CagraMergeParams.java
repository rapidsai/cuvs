/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

public class CagraMergeParams {

  private final CagraIndexParams outputIndexParams;
  private final MergeStrategy strategy;

  /**
   * Constructs a CagraMergeParams with the given output index parameters and merge strategy.
   *
   * @param outputIndexParams Index parameters for the output index
   * @param strategy Merge strategy to use
   */
  private CagraMergeParams(CagraIndexParams outputIndexParams, MergeStrategy strategy) {
    this.outputIndexParams = outputIndexParams;
    this.strategy = strategy;
  }

  /**
   * Gets the index parameters for the output index.
   *
   * @return Index parameters to use for the output index
   */
  public CagraIndexParams getOutputIndexParams() {
    return outputIndexParams;
  }

  /**
   * Gets the merge strategy to use.
   *
   * @return The merge strategy
   */
  public MergeStrategy getStrategy() {
    return strategy;
  }

  /**
   * Strategy to use when merging CAGRA indexes.
   */
  public enum MergeStrategy {
    PHYSICAL(0),

    LOGICAL(1);

    public final int value;

    MergeStrategy(int value) {
      this.value = value;
    }
  }

  /**
   * Builder class for {@link CagraMergeParams}.
   */
  public static class Builder {
    private CagraIndexParams outputIndexParams = new CagraIndexParams.Builder().build();
    private MergeStrategy strategy = MergeStrategy.PHYSICAL; // Default to PHYSICAL

    /**
     * Sets the index parameters for the output index.
     *
     * @param outputIndexParams Index parameters to use for the output index
     * @return This builder
     */
    public Builder withOutputIndexParams(CagraIndexParams outputIndexParams) {
      this.outputIndexParams = outputIndexParams;
      return this;
    }

    /**
     * Sets the merge strategy.
     *
     * @param strategy The merge strategy to use
     * @return This builder
     */
    public Builder withStrategy(MergeStrategy strategy) {
      this.strategy = strategy;
      return this;
    }

    /**
     * Builds the {@link CagraMergeParams} object.
     *
     * @return The built parameters
     */
    public CagraMergeParams build() {
      return new CagraMergeParams(outputIndexParams, strategy);
    }
  }
}
