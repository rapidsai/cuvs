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

import static com.carrotsearch.randomizedtesting.RandomizedTest.assumeTrue;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import org.junit.Before;
import org.junit.Test;

public class CagraGraphIT extends CuVSTestCase {

  @Before
  public void setup() {
    assumeTrue("not supported on " + System.getProperty("os.name"), isLinuxAmd64());
  }

  // Sample data and query
  private static final int[][] graphData = {
    {1, 2, 3},
    {0, 2, 3},
    {4, 1, 3},
    {3, 0, 2},
    {0, 4, 2}
  };

  @Test
  public void testGraphNeighboursGetAccess() {
    try (var graph = CagraGraph.ofArray(graphData)) {
      for (int n = 0; n < graph.size(); ++n) {
        var neighbours = graph.getNeighbours(n);
        assertEquals(graph.degree(), neighbours.size());
        for (int i = 0; i < graph.degree(); ++i) {
          assertEquals(graphData[n][i], neighbours.get(i));
        }
      }
    }
  }

  @Test
  public void testGraphNeighboursCopyAccess() {
    try (var graph = CagraGraph.ofArray(graphData)) {
      for (int n = 0; n < graph.size(); ++n) {
        var neighbours = graph.getNeighbours(n);
        assertEquals(graph.degree(), neighbours.size());

        int[] copy = new int[(int) neighbours.size()];
        neighbours.toArray(copy);
        assertArrayEquals(graphData[n], copy);
      }
    }
  }

  @Test
  public void testGraphGetAccess() {
    try (var graph = CagraGraph.ofArray(graphData)) {
      for (int n = 0; n < graph.size(); ++n) {
        for (int i = 0; i < graph.degree(); ++i) {
          assertEquals(graphData[n][i], graph.get(n, i));
        }
      }
    }
  }
}
