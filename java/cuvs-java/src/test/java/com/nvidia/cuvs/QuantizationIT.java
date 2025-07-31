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
import static org.junit.Assert.*;

import com.carrotsearch.randomizedtesting.RandomizedRunner;
import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.CagraIndexParams.CuvsDistanceType;
import java.lang.invoke.MethodHandles;
import java.util.Map;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@RunWith(RandomizedRunner.class)
public class QuantizationIT extends CuVSTestCase {

  private static final Logger log = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

  @Before
  public void setup() {
    assumeTrue("not supported on " + System.getProperty("os.name"), isLinuxAmd64());
    initializeRandom();
    log.info("Random context initialized for quantization test.");
  }

  private static float[][] createSimpleDataset() {
    return new float[][] {
      {1.0f, 2.0f, 3.0f},
      {4.0f, 5.0f, 6.0f},
      {7.0f, 8.0f, 9.0f},
      {10.0f, 11.0f, 12.0f},
      {13.0f, 14.0f, 15.0f}
    };
  }

  private static float[][] createSimpleQueries() {
    return new float[][] {
      {1.1f, 2.1f, 3.1f},
      {7.1f, 8.1f, 9.1f}
    };
  }

  @Test
  public void testScalarQuantizerBasicFlow() throws Throwable {
    float[][] dataset = createSimpleDataset();
    float[][] queries = createSimpleQueries();

    try (CuVSResources resources = CheckedCuVSResources.create()) {
      // Create float32 dataset
      CuVSMatrix trainingDataset = CuVSMatrix.ofArray(dataset);
      assertEquals(32, trainingDataset.precision());
      assertEquals(dataset.length, trainingDataset.size());
      assertEquals(dataset[0].length, trainingDataset.columns());

      // Create scalar quantizer
      Scalar8BitQuantizer quantizer = new Scalar8BitQuantizer(resources, trainingDataset);
      assertEquals(8, quantizer.precision());
      log.info("Created scalar quantizer with 8-bit precision");

      // Build index with quantized dataset
      CagraIndexParams indexParams =
          new CagraIndexParams.Builder()
              .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
              .withGraphDegree(2)
              .withIntermediateGraphDegree(3)
              .withNumWriterThreads(1)
              .withMetric(CuvsDistanceType.L2Expanded)
              .build();

      CagraIndex index =
          CagraIndex.newBuilder(resources)
              .withDataset(trainingDataset)
              .withQuantizer(quantizer)
              .withIndexParams(indexParams)
              .build();

      log.info("Built index with quantized dataset");

      // Create search parameters
      CagraSearchParams searchParams = new CagraSearchParams.Builder(resources).build();

      // Create quantized query
      CagraQuery query =
          CagraQuery.newBuilder()
              .withQueryVectors(queries)
              .withQuantizer(quantizer)
              .withSearchParams(searchParams)
              .withTopK(3)
              .build();

      assertTrue("Query should have quantized vectors", query.hasQuantizedQueries());
      assertEquals(8, query.getQueryPrecision());
      log.info("Created quantized query");

      // Perform search
      SearchResults results = index.search(query);
      assertNotNull("Search results should not be null", results);

      // Verify results
      assertEquals(
          "Should have results for all queries", queries.length, results.getResults().size());

      for (int i = 0; i < results.getResults().size(); i++) {
        Map<Integer, Float> queryResults = results.getResults().get(i);
        assertFalse("Query " + i + " should return some neighbors", queryResults.isEmpty());

        // Verify all returned IDs are within valid range
        for (Integer id : queryResults.keySet()) {
          assertTrue("Returned ID should be valid", id >= 0 && id < dataset.length);
        }
      }

      log.info(
          "Search completed successfully with {} queries returning neighbors",
          results.getResults().size());

      // Cleanup
      index.destroyIndex();
      quantizer.close();
      trainingDataset.close();
    }
  }

  @Test
  public void testBinaryQuantizerBasicFlow() throws Throwable {
    float[][] dataset = createSimpleDataset();
    float[][] queries = createSimpleQueries();

    try (CuVSResources resources = CheckedCuVSResources.create()) {
      // Create float32 dataset
      CuVSMatrix trainingDataset = CuVSMatrix.ofArray(dataset);
      assertEquals(32, trainingDataset.precision());

      // Create binary quantizer
      BinaryQuantizer quantizer = new BinaryQuantizer(resources);
      assertEquals(8, quantizer.precision());
      log.info("Created binary quantizer");

      // Build index with quantized dataset
      CagraIndexParams indexParams =
          new CagraIndexParams.Builder()
              .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
              .withGraphDegree(2)
              .withIntermediateGraphDegree(3)
              .withNumWriterThreads(1)
              .withMetric(CuvsDistanceType.L2Expanded)
              .build();

      CagraIndex index =
          CagraIndex.newBuilder(resources)
              .withDataset(trainingDataset)
              .withQuantizer(quantizer)
              .withIndexParams(indexParams)
              .build();

      log.info("Built index with binary quantized dataset");

      // Create search parameters
      CagraSearchParams searchParams = new CagraSearchParams.Builder(resources).build();

      // Create quantized query
      CagraQuery query =
          CagraQuery.newBuilder()
              .withQueryVectors(queries)
              .withQuantizer(quantizer)
              .withSearchParams(searchParams)
              .withTopK(3)
              .build();

      assertTrue("Query should have quantized vectors", query.hasQuantizedQueries());
      assertEquals(8, query.getQueryPrecision());

      // Perform search
      SearchResults results = index.search(query);
      assertNotNull("Search results should not be null", results);

      // Verify results
      assertEquals(
          "Should have results for all queries", queries.length, results.getResults().size());

      for (int i = 0; i < results.getResults().size(); i++) {
        Map<Integer, Float> queryResults = results.getResults().get(i);
        assertFalse("Query " + i + " should return some neighbors", queryResults.isEmpty());

        // Verify all returned IDs are within valid range
        for (Integer id : queryResults.keySet()) {
          assertTrue("Returned ID should be valid", id >= 0 && id < dataset.length);
        }
      }

      log.info("Binary quantized search completed successfully");

      // Test that inverse transform is not supported
      CuVSMatrix quantizedQueries = query.getQuantizedQueries();
      assertThrows(
          UnsupportedOperationException.class, () -> quantizer.inverseTransform(quantizedQueries));

      // Cleanup
      index.destroyIndex();
      quantizer.close();
      trainingDataset.close();
      quantizedQueries.close();
    }
  }

  @Test
  public void testScalarQuantizerInverseTransform() throws Throwable {
    float[][] dataset = createSimpleDataset();

    try (CuVSResources resources = CheckedCuVSResources.create()) {
      CuVSMatrix trainingDataset = CuVSMatrix.ofArray(dataset);
      Scalar8BitQuantizer quantizer = new Scalar8BitQuantizer(resources, trainingDataset);

      // Transform and inverse transform
      CuVSMatrix quantized = quantizer.transform(trainingDataset);
      assertEquals(8, quantized.precision());

      CuVSMatrix recovered = quantizer.inverseTransform(quantized);
      assertEquals(32, recovered.precision());
      assertEquals(trainingDataset.size(), recovered.size());
      assertEquals(trainingDataset.columns(), recovered.columns());

      log.info("Inverse transform completed successfully");

      // Cleanup
      quantizer.close();
      trainingDataset.close();
      quantized.close();
      recovered.close();
    }
  }

  @Test
  public void testCPUBinaryQuantization() throws Throwable {
    float[][] dataset = createSimpleDataset();

    try (CuVSResources resources = CheckedCuVSResources.create()) {
      CuVSMatrix inputDataset = CuVSMatrix.ofArray(dataset);
      assertEquals(CuVSMatrix.MemoryKind.HOST, inputDataset.memoryKind());

      BinaryQuantizer quantizer = new BinaryQuantizer(resources);

      // Test binary quantization
      CuVSMatrix quantized = quantizer.transform(inputDataset);
      assertEquals(8, quantized.precision()); // Binary packed into bytes
      assertEquals(CuVSMatrix.MemoryKind.HOST, quantized.memoryKind());

      // Test that inverse transform throws
      assertThrows(
          UnsupportedOperationException.class, () -> quantizer.inverseTransform(quantized));

      quantizer.close();
      inputDataset.close();
      quantized.close();

      log.info("✓ Binary CPU quantization test passed");
    }
  }

  @Test
  public void testRandomizedScalarQuantizedIndexingAndSearch() throws Throwable {
    int maxDatasetSize = 60;
    int maxDimensions = 20;
    int maxQueries = 5;
    int maxTopK = 4;

    for (int iteration = 0; iteration < 5; iteration++) {
      int datasetSize = random.nextInt(maxDatasetSize) + maxTopK;
      int dims = random.nextInt(maxDimensions) + 2;
      int numQueries = random.nextInt(maxQueries) + 1;
      int topK = Math.min(random.nextInt(maxTopK) + 1, datasetSize);

      // Generate dataset and query
      float[][] dataset = new float[datasetSize][dims];
      float[][] queries = new float[numQueries][dims];
      for (int i = 0; i < datasetSize; i++)
        for (int d = 0; d < dims; d++) dataset[i][d] = (float) random.nextGaussian();
      for (int i = 0; i < numQueries; i++)
        for (int d = 0; d < dims; d++) queries[i][d] = (float) random.nextGaussian();

      try (CuVSResources resources = CheckedCuVSResources.create()) {
        CuVSMatrix trainingDataset = CuVSMatrix.ofArray(dataset);
        Scalar8BitQuantizer quantizer = new Scalar8BitQuantizer(resources, trainingDataset);

        CagraIndexParams indexParams =
            new CagraIndexParams.Builder()
                .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
                .withGraphDegree(Math.min(4, datasetSize - 1))
                .withIntermediateGraphDegree(Math.min(5, datasetSize))
                .withNumWriterThreads(1)
                .withMetric(CuvsDistanceType.L2Expanded)
                .build();

        CagraIndex index =
            CagraIndex.newBuilder(resources)
                .withDataset(trainingDataset)
                .withQuantizer(quantizer)
                .withIndexParams(indexParams)
                .build();

        CagraSearchParams searchParams = new CagraSearchParams.Builder(resources).build();
        CagraQuery query =
            CagraQuery.newBuilder()
                .withQueryVectors(queries)
                .withQuantizer(quantizer)
                .withSearchParams(searchParams)
                .withTopK(topK)
                .build();

        SearchResults results = index.search(query);
        assertNotNull("Results must not be null", results);
        assertEquals("One result per query", numQueries, results.getResults().size());

        for (int qi = 0; qi < numQueries; qi++) {
          Map<Integer, Float> result = results.getResults().get(qi);
          // It is possible for result to have < topK entries if datasetSize == topK
          assertFalse("Each query should return non-empty result", result.isEmpty());
          for (Integer id : result.keySet()) {
            assertTrue("ID in result should be in range", id >= 0 && id < datasetSize);
          }
        }

        quantizer.close();
        trainingDataset.close();
        index.destroyIndex();
      }
    }
  }
}
