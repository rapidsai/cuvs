package com.nvidia.cuvs;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.AbstractMap;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.junit.jupiter.api.Test;

import com.nvidia.cuvs.cagra.CagraIndex;
import com.nvidia.cuvs.cagra.CagraIndexParams;
import com.nvidia.cuvs.cagra.CagraSearchParams;
import com.nvidia.cuvs.cagra.CuVSQuery;
import com.nvidia.cuvs.cagra.CuVSResources;
import com.nvidia.cuvs.cagra.SearchResult;

public class CagraBuildAndSearchTest {

  private static final int VECTOR_DIMENSION = 2;
  private static final int DATASET_SIZE = 11;
  private static final int QUERY_SIZE = 2;
  private static final int TOP_K = 4;

  // Helper method to generate random vectors
  private float[][] generateRandomVectors(int size) {
    Random rand = new Random();
    float[][] vectors = new float[size][VECTOR_DIMENSION];
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < VECTOR_DIMENSION; j++) {
        vectors[i][j] = rand.nextFloat();
      }
    }
    return vectors;
  }

  // Helper method to calculate Euclidean distance
  private double calculateDistance(float[] v1, float[] v2) {
    double sum = 0.0;
    for (int i = 0; i < v1.length; i++) {
      sum += Math.pow(v1[i] - v2[i], 2);
    }
    return Math.sqrt(sum);
  }

  // Helper method to calculate top-K distances manually
  private List<Map.Entry<Integer, Double>> getExpectedTopKItems(float[] query, float[][] dataset, int k) {
    PriorityQueue<Map.Entry<Integer, Double>> heap = new PriorityQueue<>(
        Comparator.<Map.Entry<Integer, Double>>comparingDouble(Map.Entry::getValue).reversed());

    for (int i = 0; i < dataset.length; i++) {
      double distance = calculateDistance(query, dataset[i]);
      if (heap.size() < k) {
        heap.offer(new AbstractMap.SimpleEntry<>(i, distance));
      } else if (distance < heap.peek().getValue()) {
        heap.poll();
        heap.offer(new AbstractMap.SimpleEntry<>(i, distance));
      }
    }

    return heap.stream().sorted(Comparator.comparingDouble(Map.Entry::getValue)).collect(Collectors.toList());
  }

  /**
   * A basic test that checks the whole flow - from indexing to search.
   * 
   * @throws Throwable
   */
  @Test
  public void testIndexingAndSearchingFlow() throws Throwable {

    float[][] dataset = generateRandomVectors(DATASET_SIZE);
    float[][] queries = generateRandomVectors(QUERY_SIZE);

    // Create map for dataset IDs
    Map<Integer, Integer> map = IntStream
        .range(0, dataset.length)
        .boxed()
        .collect(Collectors.toMap(i -> i, i -> i));

    CuVSResources res = new CuVSResources();

    // Configure index parameters
    CagraIndexParams cagraIndexParams = new CagraIndexParams
        .Builder()
        .withIntermediateGraphDegree(10)
        .withBuildAlgo(CagraIndexParams.CuvsCagraGraphBuildAlgo.IVF_PQ)
        .build();

    // Create the index with the dataset
    CagraIndex index = new CagraIndex
        .Builder(res)
        .withDataset(dataset)
        .withIndexParams(cagraIndexParams)
        .build();

    // Configure search parameters
    CagraSearchParams cagraSearchParams = new CagraSearchParams
        .Builder()
        .build();

    // Create a query object with the query vectors
    CuVSQuery query = new CuVSQuery
        .Builder()
        .withTopK(TOP_K)
        .withSearchParams(cagraSearchParams)
        .withQueryVectors(queries)
        .withMapping(map)
        .build();

    // Perform the search
    SearchResult rslt = index.search(query);
    List<Map<Integer, Float>> queryResults = rslt.getResults();

    // Validate the results for each query
    for (int queryIndex = 0; queryIndex < QUERY_SIZE; queryIndex++) {

      // Calculate expected top-K items for the current query
      List<Map.Entry<Integer, Double>> expectedTopK = getExpectedTopKItems(queries[queryIndex], dataset, TOP_K);

      // printing expected and actual results
      System.out.println("\nQuery " + queryIndex + ":");
      System.out.println("Expected Top-K Results:");
      expectedTopK.forEach(entry -> System.out.println("ID: " + entry.getKey()));

      System.out.println("Actual Top-K Results:");
      queryResults.get(queryIndex).entrySet().stream().limit(TOP_K)
          .forEach(entry -> System.out.println("ID: " + entry.getKey()));

      // Check each of the topK IDs retrieved for the current query
      List<Integer> returnedIDs = new LinkedList<Integer>(queryResults.get(queryIndex).keySet());
      for (int i = 0; i < TOP_K; i++) {
        assertEquals(expectedTopK.get(i).getKey(), returnedIDs.get(i),
            "Returned ID does not match with expected ID (missing or not in order)");
      }
    }
  }
}