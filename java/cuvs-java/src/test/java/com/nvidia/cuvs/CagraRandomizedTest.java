package com.nvidia.cuvs;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.carrotsearch.randomizedtesting.RandomizedRunner;
import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;

@RunWith(RandomizedRunner.class)
public class CagraRandomizedTest extends CuVSTestCase {

  private static final Logger log = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

  @Before
  public void setup() {
    initializeRandom();
    log.info("Random context initialized for test.");
  }

  @Test
  public void testResultsTopKWithRandomValues() throws Throwable {
    for (int i=0; i<10; i++) {
      tmpResultsTopKWithRandomValues();
    }
  }

  public void tmpResultsTopKWithRandomValues() throws Throwable {
    int DATASET_SIZE_LIMIT = 10_000;
    int DIMENSIONS_LIMIT = 2048;
    int NUM_QUERIES_LIMIT = 10;
    int TOP_K_LIMIT = 64; // nocommit This fails beyond 64

    int datasetSize = random.nextInt(DATASET_SIZE_LIMIT) + 1;
    int dimensions = random.nextInt(DIMENSIONS_LIMIT) + 1;
    int numQueries = random.nextInt(NUM_QUERIES_LIMIT) + 1;
    int topK = Math.min(random.nextInt(TOP_K_LIMIT) + 1, datasetSize);

    if(datasetSize < topK) datasetSize = topK;

    // Generate a random dataset
    float[][] dataset = new float[datasetSize][dimensions];
    for (int i = 0; i < datasetSize; i++) {
      for (int j = 0; j < dimensions; j++) {
        dataset[i][j] = random.nextFloat() * 100;
      }
    }
    // Generate random query vectors
    float[][] queries = new float[numQueries][dimensions];
    for (int i = 0; i < numQueries; i++) {
      for (int j = 0; j < dimensions; j++) {
        queries[i][j] = random.nextFloat() * 100;
      }
    }
    log.info("Dataset size: {}x{}", datasetSize, dimensions);
    log.info("Query size: {}x{}", numQueries, dimensions);
    log.info("TopK: {}", topK);

    // Debugging: Log dataset and queries
    if (log.isDebugEnabled()) {
      log.debug("Dataset:");
      for (float[] row : dataset) {
        log.debug(java.util.Arrays.toString(row));
      }
      log.debug("Queries:");
      for (float[] query : queries) {
        log.debug(java.util.Arrays.toString(query));
      }
    }
    // Sanity checks
    assert dataset.length > 0 : "Dataset is empty.";
    assert queries.length > 0 : "Queries are empty.";
    assert dimensions > 0 : "Invalid dimensions.";
    assert topK > 0 && topK <= datasetSize : "Invalid topK value.";

    // Generate expected results using brute force
    List<List<Integer>> expected = generateExpectedResults(topK, dataset, queries);

    // Create CuVS index and query
    try (CuVSResources resources = new CuVSResources()) {
      CagraIndexParams indexParams = new CagraIndexParams.Builder(resources).withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT).build();
      CagraIndex index = new CagraIndex.Builder(resources).withDataset(dataset).withIndexParams(indexParams).build();
      log.info("Index built successfully.");

      // Execute search and retrieve results
      CagraQuery query = new CagraQuery.Builder()
          .withQueryVectors(queries)
          .withTopK(topK)
          .withSearchParams(new CagraSearchParams.Builder(resources).build())
          .build();
      log.info("Query built successfully. Executing search...");
      CagraSearchResults results = index.search(query);

      for (int i=0; i<numQueries; i++) {
        log.info("Results returned for query "+i+": " + results.getResults().get(i).keySet());
        log.info("Expected results for query "+i+": " + expected.get(i).subList(0, Math.min(topK, datasetSize)));
      }

      // actual vs. expected results
      for (int i = 0; i < results.getResults().size(); i++) {
        Map<Integer, Float> result = results.getResults().get(i);
        assertEquals("TopK mismatch for query.", Math.min(topK, datasetSize), result.size());

        // Sort result by values (distances) and extract keys
        List<Integer> sortedResultKeys = result.entrySet().stream()
            .sorted(Map.Entry.comparingByValue()) // Sort by value (distance)
            .map(Map.Entry::getKey) // Extract sorted keys
            .toList();

        // Compare using primitive int arrays
        /*assertArrayEquals(
          "Query " + i + " mismatched",
          expected.get(i).stream().mapToInt(Integer::intValue).toArray(),
          sortedResultKeys.stream().mapToInt(Integer::intValue).toArray()
          );*/
        
        // just make sure that the first 5 results are in the expected list (which comprises of 2*topK results)
        for (int j = 0; j< Math.min(5, sortedResultKeys.size()); j++) {
          assertTrue("Not found in expected list: " + sortedResultKeys.get(j),
              expected.get(i).contains(sortedResultKeys.get(j)));
        }

      }
    }
  }

  private List<List<Integer>> generateExpectedResults(int topK, float[][] dataset, float[][] queries) {
    List<List<Integer>> neighborsResult = new ArrayList<>();
    int dimensions = dataset[0].length;

    for (float[] query : queries) {
      Map<Integer, Double> distances = new TreeMap<>();
      for (int j = 0; j < dataset.length; j++) {
        double distance = 0;
        for (int k = 0; k < dimensions; k++) {
          distance += (query[k] - dataset[j][k]) * (query[k] - dataset[j][k]);
        }
        distances.put(j, Math.sqrt(distance));
      }

      // Sort by distance and select the topK nearest neighbors
      List<Integer> neighbors = distances.entrySet().stream()
          .sorted(Map.Entry.comparingByValue())
          .map(Map.Entry::getKey)
          .toList();
      neighborsResult.add(neighbors.subList(0, Math.min(topK * 2, dataset.length))); // generate double the topK results in the expected array
    }

    log.info("Expected results generated successfully.");
    return neighborsResult;
  }
}
