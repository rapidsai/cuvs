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

import com.nvidia.cuvs.CagraIndexParams.CodebookGen;
import com.nvidia.cuvs.CagraIndexParams.CuvsDistanceType;

public class CuVSIvfPqIndexParams {

  /** Distance type. */
  private final CuvsDistanceType metric;

  /** How PQ codebooks are created. */
  private final CodebookGen codebookKind;

  /** The argument used by some distance metrics. */
  private final float metricArg;

  /** The fraction of data to use during iterative kmeans building. */
  private final double kmeansTrainsetFraction;

  /**
   * The number of inverted lists (clusters)
   *
   * Hint: the number of vectors per cluster (`n_rows/n_lists`) should be
   * approximately 1,000 to 10,000.
   */
  private final int nLists;

  /** The number of iterations searching for kmeans centers (index building). */
  private final int kmeansNIters;

  /**
   * The bit length of the vector element after compression by PQ.
   *
   * Possible values: [4, 5, 6, 7, 8].
   *
   * Hint: the smaller the 'pq_bits', the smaller the index size and the better
   * the search performance, but the lower the recall.
   */
  private final int pqBits;

  /**
   * The dimensionality of the vector after compression by PQ. When zero, an
   * optimal value is selected using a heuristic.
   *
   * NB: `pq_dim * pq_bits` must be a multiple of 8.
   *
   * Hint: a smaller 'pq_dim' results in a smaller index size and better search
   * performance, but lower recall. If 'pq_bits' is 8, 'pq_dim' can be set to any
   * number, but multiple of 8 are desirable for good performance. If 'pq_bits' is
   * not 8, 'pq_dim' should be a multiple of 8. For good performance, it is
   * desirable that 'pq_dim' is a multiple of 32. Ideally, 'pq_dim' should be also
   * a divisor of the dataset dim.
   */
  private final int pqDim;

  /**
   * Whether to add the dataset content to the index, i.e.:
   *
   * - `true` means the index is filled with the dataset vectors and ready to
   * search after calling `build`. - `false` means `build` only trains the
   * underlying model (e.g. quantizer or clustering), but the index is left empty;
   * you'd need to call `extend` on the index afterwards to populate it.
   */
  private final boolean addDataOnBuild;

  /**
   * Apply a random rotation matrix on the input data and queries even if `dim %
   * pq_dim == 0`.
   *
   * Note: if `dim` is not multiple of `pq_dim`, a random rotation is always
   * applied to the input data and queries to transform the working space from
   * `dim` to `rot_dim`, which may be slightly larger than the original space and
   * and is a multiple of `pq_dim` (`rot_dim % pq_dim == 0`). However, this
   * transform is not necessary when `dim` is multiple of `pq_dim` (`dim ==
   * rot_dim`, hence no need in adding "extra" data columns / features).
   *
   * By default, if `dim == rot_dim`, the rotation transform is initialized with
   * the identity matrix. When `force_random_rotation == true`, a random
   * orthogonal transform matrix is generated regardless of the values of `dim`
   * and `pq_dim`.
   */
  private final boolean forceRandomRotation;

  /**
   * By default, the algorithm allocates more space than necessary for individual
   * clusters (`list_data`). This allows to amortize the cost of memory allocation
   * and reduce the number of data copies during repeated calls to `extend`
   * (extending the database).
   *
   * The alternative is the conservative allocation behavior; when enabled, the
   * algorithm always allocates the minimum amount of memory required to store the
   * given number of records. Set this flag to `true` if you prefer to use as
   * little GPU memory for the database as possible.
   */
  private final boolean conservativeMemoryAllocation;

  /**
   * The max number of data points to use per PQ code during PQ codebook training.
   * Using more data points per PQ code may increase the quality of PQ codebook
   * but may also increase the build time. The parameter is applied to both PQ
   * codebook generation methods, i.e., PER_SUBSPACE and PER_CLUSTER. In both
   * cases, we will use `pq_book_size * max_train_points_per_pq_code` training
   * points to train each codebook.
   */
  private final int maxTrainPointsPerPqCode;

  private CuVSIvfPqIndexParams(
      CuvsDistanceType metric,
      CodebookGen codebookKind,
      float metricArg,
      double kmeansTrainsetFraction,
      int nLists,
      int kmeansNIters,
      int pqBits,
      int pqDim,
      boolean addDataOnBuild,
      boolean forceRandomRotation,
      boolean conservativeMemoryAllocation,
      int maxTrainPointsPerPqCode) {
    super();
    this.metric = metric;
    this.codebookKind = codebookKind;
    this.metricArg = metricArg;
    this.kmeansTrainsetFraction = kmeansTrainsetFraction;
    this.nLists = nLists;
    this.kmeansNIters = kmeansNIters;
    this.pqBits = pqBits;
    this.pqDim = pqDim;
    this.addDataOnBuild = addDataOnBuild;
    this.forceRandomRotation = forceRandomRotation;
    this.conservativeMemoryAllocation = conservativeMemoryAllocation;
    this.maxTrainPointsPerPqCode = maxTrainPointsPerPqCode;
  }

  /**
   * Gets the distance type.
   *
   * @return the distance type
   */
  public CuvsDistanceType getMetric() {
    return metric;
  }

  /**
   * Gets how PQ codebooks are created
   *
   * @return how PQ codebooks are created
   */
  public CodebookGen getCodebookKind() {
    return codebookKind;
  }

  /**
   * Gets the argument used by some distance metrics
   *
   * @return the argument used by some distance metrics
   */
  public float getMetricArg() {
    return metricArg;
  }

  /**
   * Gets the fraction of data to use during iterative kmeans building
   *
   * @return the fraction of data to use during iterative kmeans building
   */
  public double getKmeansTrainsetFraction() {
    return kmeansTrainsetFraction;
  }

  /**
   * Gets the number of inverted lists (clusters)
   *
   * @return the number of inverted lists (clusters)
   */
  public int getnLists() {
    return nLists;
  }

  /**
   * Gets the number of iterations searching for kmeans centers
   *
   * @return the number of iterations searching for kmeans centers
   */
  public int getKmeansNIters() {
    return kmeansNIters;
  }

  /**
   * Gets the bit length of the vector element after compression by PQ
   *
   * @return the bit length of the vector element after compression by PQ
   */
  public int getPqBits() {
    return pqBits;
  }

  /**
   * Gets the dimensionality of the vector after compression by PQ
   *
   * @return the dimensionality of the vector after compression by PQ
   */
  public int getPqDim() {
    return pqDim;
  }

  /**
   * Gets whether the dataset content is added to the index
   *
   * @return whether the dataset content is added to the index
   */
  public boolean isAddDataOnBuild() {
    return addDataOnBuild;
  }

  /**
   * Gets the random rotation matrix on the input data and queries
   *
   * @return the random rotation matrix on the input data and queries
   */
  public boolean isForceRandomRotation() {
    return forceRandomRotation;
  }

  /**
   * Gets if conservative allocation behavior is set
   *
   * @return if conservative allocation behavior is set
   */
  public boolean isConservativeMemoryAllocation() {
    return conservativeMemoryAllocation;
  }

  /**
   * Gets whether max number of data points to use per PQ code during PQ codebook
   * training is set
   *
   * @return whether max number of data points to use per PQ code during PQ
   *         codebook training is set
   */
  public int getMaxTrainPointsPerPqCode() {
    return maxTrainPointsPerPqCode;
  }

  @Override
  public String toString() {
    return "CuVSIvfPqIndexParams [metric="
        + metric
        + ", codebookKind="
        + codebookKind
        + ", metricArg="
        + metricArg
        + ", kmeansTrainsetFraction="
        + kmeansTrainsetFraction
        + ", nLists="
        + nLists
        + ", kmeansNIters="
        + kmeansNIters
        + ", pqBits="
        + pqBits
        + ", pqDim="
        + pqDim
        + ", addDataOnBuild="
        + addDataOnBuild
        + ", forceRandomRotation="
        + forceRandomRotation
        + ", conservativeMemoryAllocation="
        + conservativeMemoryAllocation
        + ", maxTrainPointsPerPqCode="
        + maxTrainPointsPerPqCode
        + "]";
  }

  /**
   * Builder configures and creates an instance of {@link CuVSIvfPqIndexParams}.
   */
  public static class Builder {

    /** Distance type. */
    private CuvsDistanceType metric = CuvsDistanceType.L2Expanded;

    /** How PQ codebooks are created. */
    private CodebookGen codebookKind = CodebookGen.PER_SUBSPACE;

    /** The argument used by some distance metrics. */
    private float metricArg = 2.0f;

    /** The fraction of data to use during iterative kmeans building. */
    private double kmeansTrainsetFraction = 0.5;

    /**
     * The number of inverted lists (clusters)
     *
     * Hint: the number of vectors per cluster (`n_rows/n_lists`) should be
     * approximately 1,000 to 10,000.
     */
    private int nLists = 1024;

    /** The number of iterations searching for kmeans centers (index building). */
    private int kmeansNIters = 20;

    /**
     * The bit length of the vector element after compression by PQ.
     *
     * Possible values: [4, 5, 6, 7, 8].
     *
     * Hint: the smaller the 'pq_bits', the smaller the index size and the better
     * the search performance, but the lower the recall.
     */
    private int pqBits = 8;

    /**
     * The dimensionality of the vector after compression by PQ. When zero, an
     * optimal value is selected using a heuristic.
     *
     * NB: `pq_dim * pq_bits` must be a multiple of 8.
     *
     * Hint: a smaller 'pq_dim' results in a smaller index size and better search
     * performance, but lower recall. If 'pq_bits' is 8, 'pq_dim' can be set to any
     * number, but multiple of 8 are desirable for good performance. If 'pq_bits' is
     * not 8, 'pq_dim' should be a multiple of 8. For good performance, it is
     * desirable that 'pq_dim' is a multiple of 32. Ideally, 'pq_dim' should be also
     * a divisor of the dataset dim.
     */
    private int pqDim = 0;

    /**
     * Whether to add the dataset content to the index, i.e.:
     *
     * - `true` means the index is filled with the dataset vectors and ready to
     * search after calling `build`. - `false` means `build` only trains the
     * underlying model (e.g. quantizer or clustering), but the index is left empty;
     * you'd need to call `extend` on the index afterwards to populate it.
     */
    private boolean addDataOnBuild = true;

    /**
     * Apply a random rotation matrix on the input data and queries even if `dim %
     * pq_dim == 0`.
     *
     * Note: if `dim` is not multiple of `pq_dim`, a random rotation is always
     * applied to the input data and queries to transform the working space from
     * `dim` to `rot_dim`, which may be slightly larger than the original space and
     * and is a multiple of `pq_dim` (`rot_dim % pq_dim == 0`). However, this
     * transform is not necessary when `dim` is multiple of `pq_dim` (`dim ==
     * rot_dim`, hence no need in adding "extra" data columns / features).
     *
     * By default, if `dim == rot_dim`, the rotation transform is initialized with
     * the identity matrix. When `force_random_rotation == true`, a random
     * orthogonal transform matrix is generated regardless of the values of `dim`
     * and `pq_dim`.
     */
    private boolean forceRandomRotation = false;

    /**
     * By default, the algorithm allocates more space than necessary for individual
     * clusters (`list_data`). This allows to amortize the cost of memory allocation
     * and reduce the number of data copies during repeated calls to `extend`
     * (extending the database).
     *
     * The alternative is the conservative allocation behavior; when enabled, the
     * algorithm always allocates the minimum amount of memory required to store the
     * given number of records. Set this flag to `true` if you prefer to use as
     * little GPU memory for the database as possible.
     */
    private boolean conservativeMemoryAllocation = false;

    /**
     * The max number of data points to use per PQ code during PQ codebook training.
     * Using more data points per PQ code may increase the quality of PQ codebook
     * but may also increase the build time. The parameter is applied to both PQ
     * codebook generation methods, i.e., PER_SUBSPACE and PER_CLUSTER. In both
     * cases, we will use `pq_book_size * max_train_points_per_pq_code` training
     * points to train each codebook.
     */
    private int maxTrainPointsPerPqCode = 256;

    public Builder() {}

    /**
     * Sets the distance type.
     *
     * @param metric distance type
     * @return an instance of Builder
     */
    public Builder withMetric(CuvsDistanceType metric) {
      this.metric = metric;
      return this;
    }

    /**
     * Sets the argument used by some distance metrics.
     *
     * @param metricArg argument used by some distance metrics
     * @return an instance of Builder
     */
    public Builder withMetricArg(float metricArg) {
      this.metricArg = metricArg;
      return this;
    }

    /**
     * Sets whether to add the dataset content to the index.
     *
     * @param addDataOnBuild whether to add the dataset content to the index
     * @return an instance of Builder
     */
    public Builder withAddDataOnBuild(boolean addDataOnBuild) {
      this.addDataOnBuild = addDataOnBuild;
      return this;
    }

    /**
     * Sets the number of inverted lists (clusters)
     *
     * @param nLists number of inverted lists (clusters)
     * @return an instance of Builder
     */
    public Builder withNLists(int nLists) {
      this.nLists = nLists;
      return this;
    }

    /**
     * Sets the number of iterations searching for kmeans centers
     *
     * @param kmeansNIters number of iterations searching for kmeans centers
     * @return an instance of Builder
     */
    public Builder withKmeansNIters(int kmeansNIters) {
      this.kmeansNIters = kmeansNIters;
      return this;
    }

    /**
     * Sets the fraction of data to use during iterative kmeans building.
     *
     * @param kmeansTrainsetFraction fraction of data to use during iterative kmeans
     *                               building
     * @return an instance of Builder
     */
    public Builder withKmeansTrainsetFraction(double kmeansTrainsetFraction) {
      this.kmeansTrainsetFraction = kmeansTrainsetFraction;
      return this;
    }

    /**
     * Sets the bit length of the vector element after compression by PQ.
     *
     * @param pqBits bit length of the vector element after compression by PQ
     * @return an instance of Builder
     */
    public Builder withPqBits(int pqBits) {
      this.pqBits = pqBits;
      return this;
    }

    /**
     * Sets the dimensionality of the vector after compression by PQ.
     *
     * @param pqDim dimensionality of the vector after compression by PQ
     * @return an instance of Builder
     */
    public Builder withPqDim(int pqDim) {
      this.pqDim = pqDim;
      return this;
    }

    /**
     * Sets how PQ codebooks are created.
     *
     * @param codebookKind how PQ codebooks are created
     * @return an instance of Builder
     */
    public Builder withCodebookKind(CodebookGen codebookKind) {
      this.codebookKind = codebookKind;
      return this;
    }

    /**
     * Sets the random rotation matrix on the input data and queries.
     *
     * @param forceRandomRotation random rotation matrix on the input data and
     *                            queries
     * @return an instance of Builder
     */
    public Builder withForceRandomRotation(boolean forceRandomRotation) {
      this.forceRandomRotation = forceRandomRotation;
      return this;
    }

    /**
     * Sets the conservative allocation behavior
     *
     * @param conservativeMemoryAllocation conservative allocation behavior
     * @return an instance of Builder
     */
    public Builder withConservativeMemoryAllocation(boolean conservativeMemoryAllocation) {
      this.conservativeMemoryAllocation = conservativeMemoryAllocation;
      return this;
    }

    /**
     * Sets the max number of data points to use per PQ code during PQ codebook
     * training
     *
     * @param maxTrainPointsPerPqCode max number of data points to use per PQ code
     *                                during PQ codebook training
     * @return an instance of Builder
     */
    public Builder withMaxTrainPointsPerPqCode(int maxTrainPointsPerPqCode) {
      this.maxTrainPointsPerPqCode = maxTrainPointsPerPqCode;
      return this;
    }

    /**
     * Builds an instance of {@link CuVSIvfPqIndexParams}.
     *
     * @return an instance of {@link CuVSIvfPqIndexParams}
     */
    public CuVSIvfPqIndexParams build() {
      return new CuVSIvfPqIndexParams(
          metric,
          codebookKind,
          metricArg,
          kmeansTrainsetFraction,
          nLists,
          kmeansNIters,
          pqBits,
          pqDim,
          addDataOnBuild,
          forceRandomRotation,
          conservativeMemoryAllocation,
          maxTrainPointsPerPqCode);
    }
  }
}
