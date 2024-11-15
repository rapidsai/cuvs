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

/*
* Supplemental parameters to build CAGRA Index.
*/
public class CagraIndexParams {

  private Arena arena;
  private int intermediateGraphDegree;
  private int graphDegree;
  private CuvsCagraGraphBuildAlgo cuvsCagraGraphBuildAlgo;
  private int nnDescentNiter;
  private MemorySegment cagraIndexParamsMemorySegment;

  public enum CuvsCagraGraphBuildAlgo {
    AUTO_SELECT(0), IVF_PQ(1), NN_DESCENT(2);

    public final int label;

    private CuvsCagraGraphBuildAlgo(int label) {
      this.label = label;
    }
  }

  public CagraIndexParams(Arena arena, int intermediateGraphDegree, int graphDegree,
      CuvsCagraGraphBuildAlgo CuvsCagraGraphBuildAlgo, int nnDescentNiter) {
    this.arena = arena;
    this.intermediateGraphDegree = intermediateGraphDegree;
    this.graphDegree = graphDegree;
    this.cuvsCagraGraphBuildAlgo = CuvsCagraGraphBuildAlgo;
    this.nnDescentNiter = nnDescentNiter;
    this.set();
  }

  private void set() {
    cagraIndexParamsMemorySegment = cuvsCagraIndexParams.allocate(arena);
    cuvsCagraIndexParams.intermediate_graph_degree(cagraIndexParamsMemorySegment, intermediateGraphDegree);
    cuvsCagraIndexParams.graph_degree(cagraIndexParamsMemorySegment, graphDegree);
    cuvsCagraIndexParams.build_algo(cagraIndexParamsMemorySegment, cuvsCagraGraphBuildAlgo.label);
    cuvsCagraIndexParams.nn_descent_niter(cagraIndexParamsMemorySegment, nnDescentNiter);
  }

  public int getIntermediateGraphDegree() {
    return intermediateGraphDegree;
  }

  public int getGraphDegree() {
    return graphDegree;
  }

  public CuvsCagraGraphBuildAlgo getCuvsCagraGraphBuildAlgo() {
    return cuvsCagraGraphBuildAlgo;
  }

  public int getNNDescentNiter() {
    return nnDescentNiter;
  }

  public MemorySegment getCagraIndexParamsMemorySegment() {
    return cagraIndexParamsMemorySegment;
  }

  @Override
  public String toString() {
    return "CagraIndexParams [arena=" + arena + ", intermediateGraphDegree=" + intermediateGraphDegree
        + ", graphDegree=" + graphDegree + ", cuvsCagraGraphBuildAlgo=" + cuvsCagraGraphBuildAlgo + ", nnDescentNiter="
        + nnDescentNiter + ", cagraIndexParamsMemorySegment=" + cagraIndexParamsMemorySegment + "]";
  }

  public static class Builder {

    private Arena arena;
    private int intermediateGraphDegree = 128;
    private int graphDegree = 64;
    private CuvsCagraGraphBuildAlgo cuvsCagraGraphBuildAlgo = CuvsCagraGraphBuildAlgo.NN_DESCENT;
    private int nnDescentNiter = 20;
    private int writerThreads = 1;

    public Builder() {
      this.arena = Arena.ofConfined();
    }

    public Builder withIntermediateGraphDegree(int intermediateGraphDegree) {
      this.intermediateGraphDegree = intermediateGraphDegree;
      return this;
    }

    public Builder withGraphDegree(int graphDegree) {
      this.graphDegree = graphDegree;
      return this;
    }

    public Builder withCuvsCagraGraphBuildAlgo(CuvsCagraGraphBuildAlgo cuvsCagraGraphBuildAlgo) {
      this.cuvsCagraGraphBuildAlgo = cuvsCagraGraphBuildAlgo;
      return this;
    }

    public Builder withNNDescentNiter(int nnDescentNiter) {
      this.nnDescentNiter = nnDescentNiter;
      return this;
    }

    public Builder withWriterThreads(int writerThreads) {
      this.writerThreads = writerThreads;
      return this;
    }

    public CagraIndexParams build() throws Throwable {
      return new CagraIndexParams(arena, intermediateGraphDegree, graphDegree, cuvsCagraGraphBuildAlgo, nnDescentNiter);
    }
  }
}