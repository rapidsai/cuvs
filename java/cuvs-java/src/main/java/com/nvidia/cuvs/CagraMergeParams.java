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
