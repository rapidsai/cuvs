package com.nvidia.cuvs.cagra;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

import com.nvidia.cuvs.panama.cuvsCagraIndexParams;

/*
* struct cuvsCagraIndexParams {
*     size_t intermediate_graph_degree;
*     size_t graph_degree;
*     enum cuvsCagraGraphBuildAlgo build_algo;
*     size_t nn_descent_niter;
* }
*/
public class CagraIndexParams {

  Arena arena;
  int intermediateGraphDegree;
  int graphDegree;
  CuvsCagraGraphBuildAlgo buildAlgo;
  int nnDescentNiter;
  public MemorySegment cagraIndexParamsMS;

  public enum CuvsCagraGraphBuildAlgo {
    AUTO_SELECT(0), IVF_PQ(1), NN_DESCENT(2);

    public final int label;

    private CuvsCagraGraphBuildAlgo(int label) {
      this.label = label;
    }
  }

  public CagraIndexParams(Arena arena, int intermediateGraphDegree, int graphDegree, CuvsCagraGraphBuildAlgo buildAlgo,
      int nnDescentNiter) {
    this.arena = arena;
    this.intermediateGraphDegree = intermediateGraphDegree;
    this.graphDegree = graphDegree;
    this.buildAlgo = buildAlgo;
    this.nnDescentNiter = nnDescentNiter;
    this.set();
  }

  private void set() {
    cagraIndexParamsMS = cuvsCagraIndexParams.allocate(arena);
    cuvsCagraIndexParams.intermediate_graph_degree(cagraIndexParamsMS, intermediateGraphDegree);
    cuvsCagraIndexParams.graph_degree(cagraIndexParamsMS, graphDegree);
    cuvsCagraIndexParams.build_algo(cagraIndexParamsMS, buildAlgo.label);
    cuvsCagraIndexParams.nn_descent_niter(cagraIndexParamsMS, nnDescentNiter);
  }

  public int getIntermediate_graph_degree() {
    return intermediateGraphDegree;
  }

  public int getGraph_degree() {
    return graphDegree;
  }

  public CuvsCagraGraphBuildAlgo getBuild_algo() {
    return buildAlgo;
  }

  public int getNn_descent_niter() {
    return nnDescentNiter;
  }

  @Override
  public String toString() {
    return "CagraIndexParams [intermediate_graph_degree=" + intermediateGraphDegree + ", graph_degree=" + graphDegree
        + ", build_algo=" + buildAlgo + ", nn_descent_niter=" + nnDescentNiter + "]";
  }

  public static class Builder {

    Arena arena;
    int intermediateGraphDegree = 128;
    int graphDegree = 64;
    CuvsCagraGraphBuildAlgo buildAlgo = CuvsCagraGraphBuildAlgo.IVF_PQ;
    int nnDescentNiter = 20;
    int writerThreads = 1;

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

    public Builder withBuildAlgo(CuvsCagraGraphBuildAlgo buildAlgo) {
      this.buildAlgo = buildAlgo;
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
      return new CagraIndexParams(arena, intermediateGraphDegree, graphDegree, buildAlgo, nnDescentNiter);
    }

  }

}
