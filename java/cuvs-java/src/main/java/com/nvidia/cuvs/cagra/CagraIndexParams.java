/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

package com.nvidia.cuvs.cagra;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

import com.nvidia.cuvs.panama.cuvsCagraIndexParams;

/**
 * Supplemental parameters to build CAGRA Index.
 * 
 * @since 24.12
 */
public class CagraIndexParams {

  private CuvsCagraGraphBuildAlgo cuvsCagraGraphBuildAlgo;
  private MemorySegment cagraIndexParamsMemorySegment;
  private Arena arena;
  private int intermediateGraphDegree;
  private int graphDegree;
  private int nnDescentNiter;

  /**
   * Enum that denotes which ANN algorithm is used to build CAGRA graph.
   */
  public enum CuvsCagraGraphBuildAlgo {
    /**
     * AUTO_SELECT
     */
    AUTO_SELECT(0),
    /**
     * IVF_PQ
     */
    IVF_PQ(1),
    /**
     * NN_DESCENT
     */
    NN_DESCENT(2);

    /**
     * The value for the enum choice.
     */
    public final int label;

    private CuvsCagraGraphBuildAlgo(int label) {
      this.label = label;
    }
  }

  /**
   * Constructor that initializes CagraIndexParams with an instance of Arena,
   * intermediateGraphDegree, graphDegree, CuvsCagraGraphBuildAlgo, and
   * nnDescentNiter.
   * 
   * @param arena                   the Arena instance to use
   * @param intermediateGraphDegree the degree of input graph for pruning
   * @param graphDegree             the degree of output graph
   * @param CuvsCagraGraphBuildAlgo the CuvsCagraGraphBuildAlgo
   * @param nnDescentNiter          the number of Iterations to run if building
   *                                with NN_DESCENT
   */
  public CagraIndexParams(Arena arena, int intermediateGraphDegree, int graphDegree,
      CuvsCagraGraphBuildAlgo CuvsCagraGraphBuildAlgo, int nnDescentNiter) {
    this.arena = arena;
    this.intermediateGraphDegree = intermediateGraphDegree;
    this.graphDegree = graphDegree;
    this.cuvsCagraGraphBuildAlgo = CuvsCagraGraphBuildAlgo;
    this.nnDescentNiter = nnDescentNiter;
    this.setCagraIndexParamsStubValues();
  }

  /**
   * Sets the parameter values in the stub's MemorySegment.
   * 
   * @see MemorySegment
   */
  private void setCagraIndexParamsStubValues() {
    cagraIndexParamsMemorySegment = cuvsCagraIndexParams.allocate(arena);
    cuvsCagraIndexParams.intermediate_graph_degree(cagraIndexParamsMemorySegment, intermediateGraphDegree);
    cuvsCagraIndexParams.graph_degree(cagraIndexParamsMemorySegment, graphDegree);
    cuvsCagraIndexParams.build_algo(cagraIndexParamsMemorySegment, cuvsCagraGraphBuildAlgo.label);
    cuvsCagraIndexParams.nn_descent_niter(cagraIndexParamsMemorySegment, nnDescentNiter);
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
   * Gets the CuvsCagraGraphBuildAlgo.
   * 
   * @return CuvsCagraGraphBuildAlgo selected
   */
  public CuvsCagraGraphBuildAlgo getCuvsCagraGraphBuildAlgo() {
    return cuvsCagraGraphBuildAlgo;
  }

  /**
   * Gets the number of Iterations to run if building with NN_DESCENT.
   * 
   * @return the number of Iterations
   */
  public int getNNDescentNiter() {
    return nnDescentNiter;
  }

  /**
   * Gets the cagraIndexParams MemorySegment.
   * 
   * @return an instance of MemorySegment
   */
  public MemorySegment getCagraIndexParamsMemorySegment() {
    return cagraIndexParamsMemorySegment;
  }

  @Override
  public String toString() {
    return "CagraIndexParams [arena=" + arena + ", intermediateGraphDegree=" + intermediateGraphDegree
        + ", graphDegree=" + graphDegree + ", cuvsCagraGraphBuildAlgo=" + cuvsCagraGraphBuildAlgo + ", nnDescentNiter="
        + nnDescentNiter + ", cagraIndexParamsMemorySegment=" + cagraIndexParamsMemorySegment + "]";
  }

  /**
   * Builder configures and creates an instance of CagraIndexParams.
   */
  public static class Builder {

    private CuvsCagraGraphBuildAlgo cuvsCagraGraphBuildAlgo = CuvsCagraGraphBuildAlgo.NN_DESCENT;
    private Arena arena;
    private int intermediateGraphDegree = 128;
    private int graphDegree = 64;
    private int nnDescentNiter = 20;
    private int writerThreads = 1;

    /**
     * Constructor for builder for initializing and instance of Arena.
     * 
     * @see Arena
     */
    public Builder() {
      this.arena = Arena.ofConfined();
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
    public Builder withCuvsCagraGraphBuildAlgo(CuvsCagraGraphBuildAlgo cuvsCagraGraphBuildAlgo) {
      this.cuvsCagraGraphBuildAlgo = cuvsCagraGraphBuildAlgo;
      return this;
    }

    /**
     * Sets the Number of Iterations to run if building with NN_DESCENT.
     * 
     * @param nnDescentNiter number of Iterations to run if building with NN_DESCENT
     * @return an instance of Builder
     */
    public Builder withNNDescentNiter(int nnDescentNiter) {
      this.nnDescentNiter = nnDescentNiter;
      return this;
    }

    /**
     * Registers the number of writer threads to use for indexing.
     * 
     * @param writerThreads number of writer threads to use
     * @return an instance of Builder
     */
    public Builder withWriterThreads(int writerThreads) {
      this.writerThreads = writerThreads;
      return this;
    }

    /**
     * Builds an instance of CagraIndexParams.
     * 
     * @return an instance of CagraIndexParams
     * @see CagraIndexParams
     */
    public CagraIndexParams build() {
      return new CagraIndexParams(arena, intermediateGraphDegree, graphDegree, cuvsCagraGraphBuildAlgo, nnDescentNiter);
    }
  }
}