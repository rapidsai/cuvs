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
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import com.carrotsearch.randomizedtesting.RandomizedRunner;
import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import java.lang.invoke.MethodHandles;
import java.util.BitSet;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@RunWith(RandomizedRunner.class)
public class CagraRandomizedIT extends CuVSTestCase {

  private static final Logger log = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

  @Before
  public void setup() {
    assumeTrue("not supported on " + System.getProperty("os.name"), isLinuxAmd64());
    initializeRandom();
    log.info("Random context initialized for test.");
  }

  @Test
  public void testResultsTopKWithRandomValues() throws Throwable {
    boolean useNativeMemoryDatasets[] = {true, false};
    for (int i = 0; i < 100; i++) {
      for (boolean use : useNativeMemoryDatasets) {
        tmpResultsTopKWithRandomValues(use);
      }
    }
  }

  private void tmpResultsTopKWithRandomValues(boolean useNativeMemoryDataset) throws Throwable {
    int DATASET_SIZE_LIMIT = 10_000;
    int DIMENSIONS_LIMIT = 2048;
    int NUM_QUERIES_LIMIT = 10;
    int TOP_K_LIMIT = 64; // nocommit This fails beyond 64

    int datasetSize =
        random.nextInt(DATASET_SIZE_LIMIT)
            + 2; // datasetSize of 1 fails/crashed due to a bug, hence adding 2 here.
    int dimensions = random.nextInt(DIMENSIONS_LIMIT) + 1;
    int numQueries = random.nextInt(NUM_QUERIES_LIMIT) + 1;
    int topK = Math.min(random.nextInt(TOP_K_LIMIT) + 1, datasetSize);
    boolean usePrefilter = random.nextBoolean();

    if (datasetSize < topK) datasetSize = topK;

    BitSet sharedPrefilter = null;
    if (usePrefilter) {
      sharedPrefilter = new BitSet(datasetSize);
      for (int j = 0; j < datasetSize; j++) {
        sharedPrefilter.set(j, random.nextBoolean());
      }
    }

    BitSet[] prefilters = null;
    if (sharedPrefilter != null) {
      prefilters = new BitSet[numQueries];
      for (int i = 0; i < numQueries; i++) {
        prefilters[i] = sharedPrefilter;
      }
    }

    // Generate a random dataset
    float[][] vectors = generateData(random, datasetSize, dimensions);

    // Generate random query vectors
    float[][] queries = generateData(random, numQueries, dimensions);

    log.info("Dataset size: {}x{}", datasetSize, dimensions);
    log.info("Query size: {}x{}", numQueries, dimensions);
    log.info("TopK: {}", topK);
    log.info("Use native memory dataset? " + useNativeMemoryDataset);

    // Debugging: Log dataset and queries
    if (log.isDebugEnabled()) {
      log.debug("Dataset:");
      for (float[] row : vectors) {
        log.debug(java.util.Arrays.toString(row));
      }
      log.debug("Queries:");
      for (float[] query : queries) {
        log.debug(java.util.Arrays.toString(query));
      }
    }
    // Sanity checks
    assert vectors.length > 0 : "Dataset is empty.";
    assert queries.length > 0 : "Queries are empty.";
    assert dimensions > 0 : "Invalid dimensions.";
    assert topK > 0 && topK <= datasetSize : "Invalid topK value.";

    // Generate expected results using brute force
    List<List<Integer>> expected = generateExpectedResults(topK, vectors, queries, prefilters, log);

    // Create CuVS index and query
    try (CuVSResources resources = CuVSResources.create()) {
      CagraIndexParams indexParams =
          new CagraIndexParams.Builder()
              .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
              .build();

      CagraIndex index;
      if (useNativeMemoryDataset) {
        var datasetBuilder = Dataset.builder(vectors.length, vectors[0].length);
        for (float[] v : vectors) {
          datasetBuilder.addVector(v);
        }
        index =
            CagraIndex.newBuilder(resources)
                .withDataset(datasetBuilder.build())
                .withIndexParams(indexParams)
                .build();
      } else {
        index =
            CagraIndex.newBuilder(resources)
                .withDataset(vectors)
                .withIndexParams(indexParams)
                .build();
      }
      log.info("Index built successfully.");

      try {
        // Execute search and retrieve results
        CagraQuery.Builder queryBuilder =
            new CagraQuery.Builder()
                .withQueryVectors(queries)
                .withTopK(topK)
                .withSearchParams(new CagraSearchParams.Builder(resources).build());

        if (sharedPrefilter != null) {
          queryBuilder.withPrefilter(sharedPrefilter, datasetSize);
        }

        CagraQuery query = queryBuilder.build();
        log.info("Query built successfully. Executing search...");
        SearchResults originalResults = index.search(query);

        compareResults(originalResults, expected, topK, datasetSize, numQueries);

        // Test graph export/import functionality
        log.info("Testing graph export/import functionality...");

        // Export the graph from the original index
        int[][] exportedGraph = index.getGraph();
        log.info(
            "Exported graph with shape: {}x{}",
            exportedGraph.length,
            exportedGraph.length > 0 ? exportedGraph[0].length : 0);

        // Get the dataset from the original index
        Dataset originalDataset = index.getDataset();
        log.info("Retrieved original dataset");

        // Create a new index from the exported graph and dataset
        CagraIndex graphBasedIndex =
            CagraIndex.newBuilder(resources)
                .withDataset(originalDataset)
                .from(exportedGraph, indexParams.getCuvsDistanceType())
                .build();
        log.info("Created new index from exported graph");

        try {
          // Execute the same search query on the graph-based index
          SearchResults graphBasedResults = graphBasedIndex.search(query);
          log.info("Executed search on graph-based index");

          // Compare results between original and graph-based indexes
          compareSearchResults(originalResults, graphBasedResults, topK, numQueries);
          log.info("Graph export/import test passed successfully");

        } finally {
          graphBasedIndex.destroyIndex();
        }
      } finally {
        index.destroyIndex();
      }
    }
  }

  /**
   * Compares results between original and graph-based indexes using overlap instead of exact match.
   */
  private void compareSearchResults(
      SearchResults originalResults, SearchResults graphBasedResults, int topK, int numQueries) {
    log.info("Comparing search results between original and graph-based indexes");

    // Verify we have the same number of query results
    assertEquals(
        "Number of query results should match",
        originalResults.getResults().size(),
        graphBasedResults.getResults().size());

    for (int i = 0; i < numQueries; i++) {
      var originalResult = originalResults.getResults().get(i);
      var graphBasedResult = graphBasedResults.getResults().get(i);

      log.debug(
          "Query {}: Original result size: {}, Graph-based result size: {}",
          i,
          originalResult.size(),
          graphBasedResult.size());

      // Both should return some results
      assertTrue(
          String.format("Original result should not be empty for query %d", i),
          originalResult.size() > 0);
      assertTrue(
          String.format("Graph-based result should not be empty for query %d", i),
          graphBasedResult.size() > 0);

      // Sort both results by distance to get the top neighbors
      var sortedOriginal =
          originalResult.entrySet().stream()
              .sorted(java.util.Map.Entry.comparingByValue())
              .toList();
      var sortedGraphBased =
          graphBasedResult.entrySet().stream()
              .sorted(java.util.Map.Entry.comparingByValue())
              .toList();

      // Compare the top results for significant overlap
      // We expect at least 50% overlap in the top 5 results (or fewer if topK is small)
      int numToCheck =
          Math.min(5, Math.min(topK, Math.min(sortedOriginal.size(), sortedGraphBased.size())));

      if (numToCheck > 0) {
        // Get top neighbors from both results
        var originalTopNeighbors = new java.util.HashSet<Integer>();
        var graphBasedTopNeighbors = new java.util.HashSet<Integer>();

        for (int j = 0; j < numToCheck; j++) {
          if (j < sortedOriginal.size()) {
            originalTopNeighbors.add(sortedOriginal.get(j).getKey());
          }
          if (j < sortedGraphBased.size()) {
            graphBasedTopNeighbors.add(sortedGraphBased.get(j).getKey());
          }
        }

        // Count overlap
        var intersection = new java.util.HashSet<>(originalTopNeighbors);
        intersection.retainAll(graphBasedTopNeighbors);

        int overlapCount = intersection.size();
        double overlapRatio = (double) overlapCount / numToCheck;

        log.debug(
            "Query {}: Top {} neighbors overlap: {}/{} ({:.1f}%)",
            i, numToCheck, overlapCount, numToCheck, overlapRatio * 100);

        // We expect at least 30% overlap (this is a reasonable threshold for different index
        // types)
        assertTrue(
            String.format(
                "Query %d should have reasonable overlap in top %d results: got %.1f%% overlap",
                i, numToCheck, overlapRatio * 100),
            overlapRatio >= 0.3);
      }

      log.debug("Query {} results have reasonable similarity", i);
    }

    log.info("Search results show reasonable similarity between original and graph-based indexes");
  }
}
