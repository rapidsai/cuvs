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

public class CuVSIvfPqParams {

  /** CuVS IVF_PQ index parameters */
  private final CuVSIvfPqIndexParams indexParams;

  /** CuVS IVF_PQ search parameters */
  private final CuVSIvfPqSearchParams searchParams;

  /** refinement rate */
  private final float refinementRate;

  private CuVSIvfPqParams(
      CuVSIvfPqIndexParams indexParams, CuVSIvfPqSearchParams searchParams, float refinementRate) {
    super();
    this.indexParams = indexParams;
    this.searchParams = searchParams;
    this.refinementRate = refinementRate;
  }

  /**
   *
   * @return
   */
  public CuVSIvfPqIndexParams getIndexParams() {
    return indexParams;
  }

  /**
   *
   * @return
   */
  public CuVSIvfPqSearchParams getSearchParams() {
    return searchParams;
  }

  /**
   *
   * @return
   */
  public float getRefinementRate() {
    return refinementRate;
  }

  @Override
  public String toString() {
    return "CuVSIvfPqParams [indexParams="
        + indexParams
        + ", searchParams="
        + searchParams
        + ", refinementRate="
        + refinementRate
        + "]";
  }

  /**
   * Builder configures and creates an instance of {@link CuVSIvfPqParams}.
   */
  public static class Builder {

    /** CuVS IVF_PQ index parameters */
    private CuVSIvfPqIndexParams cuVSIvfPqIndexParams = new CuVSIvfPqIndexParams.Builder().build();

    /** CuVS IVF_PQ search parameters */
    private CuVSIvfPqSearchParams cuVSIvfPqSearchParams =
        new CuVSIvfPqSearchParams.Builder().build();

    /** refinement rate */
    private float refinementRate = 2.0f;

    public Builder() {}

    /**
     * Sets the CuVS IVF_PQ index parameters.
     *
     * @param cuVSIvfPqIndexParams the CuVS IVF_PQ index parameters
     * @return an instance of Builder
     */
    public Builder withCuVSIvfPqIndexParams(CuVSIvfPqIndexParams cuVSIvfPqIndexParams) {
      this.cuVSIvfPqIndexParams = cuVSIvfPqIndexParams;
      return this;
    }

    /**
     * Sets the CuVS IVF_PQ search parameters.
     *
     * @param cuVSIvfPqSearchParams the CuVS IVF_PQ search parameters
     * @return an instance of Builder
     */
    public Builder withCuVSIvfPqSearchParams(CuVSIvfPqSearchParams cuVSIvfPqSearchParams) {
      this.cuVSIvfPqSearchParams = cuVSIvfPqSearchParams;
      return this;
    }

    /**
     * Sets the refinement rate, default 2.0.
     *
     * @param refinementRate the refinement rate
     * @return an instance of Builder
     */
    public Builder withRefinementRate(float refinementRate) {
      this.refinementRate = refinementRate;
      return this;
    }

    /**
     * Builds an instance of {@link CuVSIvfPqParams}.
     *
     * @return an instance of {@link CuVSIvfPqParams}
     */
    public CuVSIvfPqParams build() {
      return new CuVSIvfPqParams(cuVSIvfPqIndexParams, cuVSIvfPqSearchParams, refinementRate);
    }
  }
}
