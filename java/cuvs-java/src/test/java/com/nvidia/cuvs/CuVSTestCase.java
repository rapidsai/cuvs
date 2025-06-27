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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import com.carrotsearch.randomizedtesting.RandomizedContext;
import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class CuVSTestCase {
  protected Random random;
  private static final Logger log = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

  protected void initializeRandom() {
    random = RandomizedContext.current().getRandom();
    log.info("Test seed: " + RandomizedContext.current().getRunnerSeedAsString());
  }

  protected float[][] generateData(Random random, int rows, int cols) {
    float[][] data = new float[rows][cols];
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        data[i][j] = random.nextFloat() * 100;
      }
    }
    return data;
  }

  protected List<List<Integer>> generateExpectedResults(
      int topK, float[][] dataset, float[][] queries, BitSet[] prefilters, Logger log) {
    List<List<Integer>> neighborsResult = new ArrayList<>();
    int dimensions = dataset[0].length;

    for (int q = 0; q < queries.length; q++) {
      float[] query = queries[q];
      Map<Integer, Double> distances = new TreeMap<>();
      for (int j = 0; j < dataset.length; j++) {
        double distance = 0;
        if (prefilters != null && prefilters[q].get(j) == false) {
          distance = Double.POSITIVE_INFINITY;
        } else {
          for (int k = 0; k < dimensions; k++) {
            distance += (query[k] - dataset[j][k]) * (query[k] - dataset[j][k]);
          }
        }
        distances.put(j, Math.sqrt(distance));
      }

      // Sort by distance and select the topK nearest neighbors
      List<Integer> neighbors =
          distances.entrySet().stream()
              .sorted(Map.Entry.comparingByValue())
              .map(Map.Entry::getKey)
              .toList();
      neighborsResult.add(neighbors.subList(0, Math.min(topK * 2 + 10, dataset.length)));
    }

    log.info("Expected results generated successfully.");
    return neighborsResult;
  }

  protected void compareResults(
      SearchResults results,
      List<List<Integer>> expected,
      int topK,
      int datasetSize,
      int numQueries) {

    for (int i = 0; i < numQueries; i++) {
      log.info("Results returned for query " + i + ": " + results.getResults().get(i).keySet());
      log.info(
          "Expected results for query "
              + i
              + ": "
              + expected.get(i).subList(0, Math.min(topK, datasetSize)));
    }

    // actual vs. expected results
    for (int i = 0; i < results.getResults().size(); i++) {
      Map<Integer, Float> result = results.getResults().get(i);

      // Sort result by values (distances) and extract keys
      List<Integer> sortedResultKeys =
          result.entrySet().stream()
              .sorted(Map.Entry.comparingByValue())
              .map(Map.Entry::getKey) // Extract sorted keys
              .toList();

      // just make sure that the first 5 results are in the expected list (which
      // comprises of 2*topK results)
      for (int j = 0; j < Math.min(5, sortedResultKeys.size()); j++) {
        assertTrue(
            "Not found in expected list: " + sortedResultKeys.get(j),
            expected.get(i).contains(sortedResultKeys.get(j)));
      }
    }
  }

  protected static void checkResults(
      List<Map<Integer, Float>> expected, List<Map<Integer, Float>> actual) {
    List<Map<Integer, Float>> sortedExpected = new ArrayList<Map<Integer, Float>>();
    List<Map<Integer, Float>> sortedActual = new ArrayList<Map<Integer, Float>>();
    for (Map<Integer, Float> map : expected) {
      sortedExpected.add(
          new TreeMap<Integer, Float>(map) {
            @Override
            public boolean equals(Object o) {
              Map<Integer, Float> map = (Map<Integer, Float>) o;
              if (this.size() != map.size()) return false;
              for (Integer key : map.keySet()) {
                try {
                  if (Math.abs((float) map.get(key) - ((float) get(key))) < 0.0001f == false) {
                    return false;
                  }
                } catch (Exception ex) {
                  return false;
                }
              }
              return true;
            }
          });
    }
    for (Map<Integer, Float> map : actual) {
      sortedActual.add(new TreeMap<Integer, Float>(map));
    }
    assertEquals(sortedExpected, sortedActual);
  }

  protected static boolean isLinuxAmd64() {
    String name = System.getProperty("os.name");
    return (name.startsWith("Linux")) && System.getProperty("os.arch").equals("amd64");
  }
}
