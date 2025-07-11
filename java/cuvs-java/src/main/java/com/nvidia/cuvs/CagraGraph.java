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

import com.nvidia.cuvs.spi.CuVSProvider;

/**
 * Represent a CAGRA graph backed by off-heap memory.
 *
 * @since 25.08
 */
public interface CagraGraph extends AutoCloseable {

  /**
   * Creates a graph from an on-heap array of neighbours.
   * This method will allocate an additional MemorySegment to hold the graph data.
   */
  static CagraGraph ofArray(int[][] graph) {
    return CuVSProvider.provider().newArrayGraph(graph);
  }

  /**
   * The degree of the graph, aka the length of the neighbour list for each node
   */
  int degree();

  /**
   * The size of the graph, aka the number of nodes
   */
  long size();

  /**
   * Access a single element of the matrix backing the CAGRA graph.
   *
   * @param row the matrix row, i.e. the node index
   * @param col the matrix column, i.e. the i-th neighbour of the {@code row}-th node
   */
  int get(int row, int col);

  /**
   * Get a view (0-copy) of the neighbour data of a node.
   *
   * @param nodeIndex the node for which to return the list of neighbours
   */
  IntList getNeighbours(long nodeIndex);

  @Override
  void close();
}
