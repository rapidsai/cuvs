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
 * Supplemental parameters to build CAGRA Index.
 *
 * @since 25.02
 */
public class CagraIndexParams {

  private final CagraGraphBuildAlgo cuvsCagraGraphBuildAlgo;
  private final CuvsDistanceType cuvsDistanceType;
  private final int intermediateGraphDegree;
  private final int graphDegree;
  private final int nnDescentNiter;
  private final int numWriterThreads;

  /**
   * Enum that denotes which ANN algorithm is used to build CAGRA graph.
   */
  public enum CagraGraphBuildAlgo {
    /**
     * Select build algorithm automatically
     */
    AUTO_SELECT(0),
    /**
     * Use IVF-PQ to build all-neighbors knn graph
     */
    IVF_PQ(1),
    /**
     * Experimental, use NN-Descent to build all-neighbors knn graph
     */
    NN_DESCENT(2);

    /**
     * The value for the enum choice.
     */
    public final int value;

    private CagraGraphBuildAlgo(int value) {
      this.value = value;
    }
  }

  /**
   * Enum that denotes how to compute distance.
   */
  public enum CuvsDistanceType {

    /**
     * evaluate as dist_ij = sum(x_ik^2) + sum(y_ij)^2 - 2*sum(x_ik * y_jk)
     */
    L2Expanded(0),
    /**
     * same as above, but inside the epilogue, perform square root operation
     */
    L2SqrtExpanded(1),
    /**
     * cosine distance
     */
    CosineExpanded(2),
    /**
     * L1 distance *
     */
    L1(3),
    /**
     * evaluate as dist_ij += (x_ik - y-jk)^2 *
     */
    L2Unexpanded(4),
    /**
     * same as above, but inside the epilogue, perform square root operation
     */
    L2SqrtUnexpanded(5),
    /**
     * basic inner product
     */
    InnerProduct(6),
    /**
     * Chebyshev (Linf) distance
     */
    Linf(7),
    /**
     * Canberra distance
     */
    Canberra(8),
    /**
     * Generalized Minkowski distance
     */
    LpUnexpanded(9),
    /**
     * Correlation distance
     */
    CorrelationExpanded(10),
    /**
     * Jaccard distance
     */
    JaccardExpanded(11),
    /**
     * Hellinger distance
     */
    HellingerExpanded(12),
    /**
     * Haversine distance
     */
    Haversine(13),
    /**
     * Bray-Curtis distance
     */
    BrayCurtis(14),
    /**
     * Jensen-Shannon distance
     */
    JensenShannon(15),
    /**
     * Hamming distance
     */
    HammingUnexpanded(16),
    /**
     * KLDivergence
     */
    KLDivergence(17),
    /**
     * RusselRao
     */
    RusselRaoExpanded(18),
    /**
     * Dice-Sorensen distance
     */
    DiceExpanded(19),
    /**
     * Precomputed (special value)
     */
    Precomputed(100);

    /**
     * The value for the enum choice.
     */
    public final int value;

    private CuvsDistanceType(int value) {
      this.value = value;
    }

  }

  /**
   * Enum that denotes codebook gen options.
   */
  public enum CodebookGen {

    PER_SUBSPACE(0),

    PER_CLUSTER(1);

    /**
     * The value for the enum choice.
     */
    public final int value;

    private CodebookGen(int value) {
      this.value = value;
    }
  }

  /**
   * Enum that denotes cuda datatypes.
   */
  public enum CudaDataType {

    /**
     * real as a half
     */
    CUDA_R_16F(2),

    /**
     * complex as a pair of half numbers
     */
    CUDA_C_16F(6),

    /**
     * real as a float
     */
    CUDA_R_32F(0),

    /**
     * complex as a pair of float numbers
     */
    CUDA_C_32F(4),

    /**
     * real as a double
     */
    CUDA_R_64F(1),

    /**
     * complex as a pair of double numbers
     */
    CUDA_C_64F(5),

    /**
     * real as a signed char
     */
    CUDA_R_8I(3),

    /**
     * complex as a pair of signed char numbers
     */
    CUDA_C_8I(7),

    /**
     * real as a unsigned char
     */
    CUDA_R_8U(8),

    /**
     * complex as a pair of unsigned char numbers
     */
    CUDA_C_8U(9),

    /**
     * real as a signed int
     */
    CUDA_R_32I(10),

    /**
     * complex as a pair of signed int numbers
     */
    CUDA_C_32I(11),

    /**
     * real as a unsigned int
     */
    CUDA_R_32U(12),

    /**
     * complex as a pair of unsigned int numbers
     */
    CUDA_C_32U(13);

    /**
     * The value for the enum choice.
     */
    public final int value;

    private CudaDataType(int value) {
      this.value = value;
    }
  }

  private CagraIndexParams(int intermediateGraphDegree, int graphDegree, CagraGraphBuildAlgo CuvsCagraGraphBuildAlgo,
      int nnDescentNiter, int writerThreads, CuvsDistanceType cuvsDistanceType) {
    this.intermediateGraphDegree = intermediateGraphDegree;
    this.graphDegree = graphDegree;
    this.cuvsCagraGraphBuildAlgo = CuvsCagraGraphBuildAlgo;
    this.nnDescentNiter = nnDescentNiter;
    this.numWriterThreads = writerThreads;
    this.cuvsDistanceType = cuvsDistanceType;
  }

  /**
   * Gets the degree of input graph for pruning.
   *
   * @return the degree of input graph
   */
  public int getIntermediateGraphDegree() {
    return intermediateGraphDegree;
  }

  /**
   * Gets the degree of output graph.
   *
   * @return the degree of output graph
   */
  public int getGraphDegree() {
    return graphDegree;
  }

  /**
   * Gets the {@link CagraGraphBuildAlgo} used to build the index.
   */
  public CagraGraphBuildAlgo getCagraGraphBuildAlgo() {
    return cuvsCagraGraphBuildAlgo;
  }

  /**
   * Gets the number of iterations to run if building with
   * {@link CagraGraphBuildAlgo#NN_DESCENT}
   */
  public int getNNDescentNumIterations() {
    return nnDescentNiter;
  }

  /**
   * Gets the {@link CuvsDistanceType} used to build the index.
   */
  public CuvsDistanceType getCuvsDistanceType() {
    return cuvsDistanceType;
  }

  /**
   * Gets the number of threads used to build the index.
   */
  public int getNumWriterThreads() {
    return numWriterThreads;
  }

  @Override
  public String toString() {
    return "CagraIndexParams [cuvsCagraGraphBuildAlgo=" + cuvsCagraGraphBuildAlgo + ", cuvsDistanceType="
        + cuvsDistanceType + ", intermediateGraphDegree=" + intermediateGraphDegree + ", graphDegree=" + graphDegree
        + ", nnDescentNiter=" + nnDescentNiter + ", numWriterThreads=" + numWriterThreads + "]";
  }

  /**
   * Builder configures and creates an instance of {@link CagraIndexParams}.
   */
  public static class Builder {

    private CagraGraphBuildAlgo cuvsCagraGraphBuildAlgo = CagraGraphBuildAlgo.NN_DESCENT;
    private CuvsDistanceType cuvsDistanceType = CuvsDistanceType.L2Expanded;
    private int intermediateGraphDegree = 128;
    private int graphDegree = 64;
    private int nnDescentNumIterations = 20;
    private int numWriterThreads = 2;

    public Builder() {
    }

    /**
     * Sets the degree of input graph for pruning.
     *
     * @param intermediateGraphDegree degree of input graph for pruning
     * @return an instance of Builder
     */
    public Builder withIntermediateGraphDegree(int intermediateGraphDegree) {
      this.intermediateGraphDegree = intermediateGraphDegree;
      return this;
    }

    /**
     * Sets the degree of output graph.
     *
     * @param graphDegree degree of output graph
     * @return an instance to Builder
     */
    public Builder withGraphDegree(int graphDegree) {
      this.graphDegree = graphDegree;
      return this;
    }

    /**
     * Sets the CuvsCagraGraphBuildAlgo to use.
     *
     * @param cuvsCagraGraphBuildAlgo the CuvsCagraGraphBuildAlgo to use
     * @return an instance of Builder
     */
    public Builder withCagraGraphBuildAlgo(CagraGraphBuildAlgo cuvsCagraGraphBuildAlgo) {
      this.cuvsCagraGraphBuildAlgo = cuvsCagraGraphBuildAlgo;
      return this;
    }

    /**
     * Sets the metric to use.
     *
     * @param cuvsDistanceType the {@link CuvsDistanceType} to use
     * @return an instance of Builder
     */
    public Builder withMetric(CuvsDistanceType cuvsDistanceType) {
      this.cuvsDistanceType = cuvsDistanceType;
      return this;
    }

    /**
     * Sets the Number of Iterations to run if building with
     * {@link CagraGraphBuildAlgo#NN_DESCENT}.
     *
     * @param nnDescentNiter number of Iterations to run if building with
     *                       {@link CagraGraphBuildAlgo#NN_DESCENT}
     * @return an instance of Builder
     */
    public Builder withNNDescentNumIterations(int nnDescentNiter) {
      this.nnDescentNumIterations = nnDescentNiter;
      return this;
    }

    /**
     * Sets the number of writer threads to use for indexing.
     *
     * @param numWriterThreads number of writer threads to use
     * @return an instance of Builder
     */
    public Builder withNumWriterThreads(int numWriterThreads) {
      this.numWriterThreads = numWriterThreads;
      return this;
    }

    /**
     * Builds an instance of {@link CagraIndexParams}.
     *
     * @return an instance of {@link CagraIndexParams}
     */
    public CagraIndexParams build() {
      return new CagraIndexParams(intermediateGraphDegree, graphDegree, cuvsCagraGraphBuildAlgo, nnDescentNumIterations,
          numWriterThreads, cuvsDistanceType);
    }
  }
}
