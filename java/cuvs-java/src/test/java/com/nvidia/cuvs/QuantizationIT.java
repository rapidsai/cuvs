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
import com.nvidia.cuvs.CuVSMatrix.DataType;
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
      CuVSMatrix trainingDataset = CuVSMatrix.ofArray(dataset);
      assertEquals(DataType.FLOAT, trainingDataset.dataType());
      assertEquals(dataset.length, trainingDataset.size());
      assertEquals(dataset[0].length, trainingDataset.columns());

      Scalar8BitQuantizer quantizer = new Scalar8BitQuantizer(resources, trainingDataset);
      assertEquals(DataType.BYTE, quantizer.outputDataType());
      log.info("Created scalar quantizer with BYTE data type");

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

      CagraSearchParams searchParams = new CagraSearchParams.Builder().build();

      CagraQuery query =
          CagraQuery.newBuilder(resources)
              .withQueryVectors(queries)
              .withQuantizer(quantizer)
              .withSearchParams(searchParams)
              .withTopK(3)
              .build();

      assertTrue("Query should have quantized vectors", query.hasQuantizedQueries());
      assertEquals(DataType.BYTE, query.getQueryDataType());
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
      index.close();
      quantizer.close();
      trainingDataset.close();
    }
  }

  @Test
  public void testBinaryQuantizerBasicFlow() throws Throwable {
    float[][] dataset = createSimpleDataset();
    float[][] queries = createSimpleQueries();

    try (CuVSResources resources = CheckedCuVSResources.create()) {
      CuVSMatrix trainingDataset = CuVSMatrix.ofArray(dataset);
      assertEquals(DataType.FLOAT, trainingDataset.dataType());

      BinaryQuantizer quantizer = new BinaryQuantizer(resources, trainingDataset);
      assertEquals(DataType.BYTE, quantizer.outputDataType());
      log.info("Created binary quantizer");

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

      CagraSearchParams searchParams = new CagraSearchParams.Builder().build();

      CagraQuery query =
          CagraQuery.newBuilder(resources)
              .withQueryVectors(queries)
              .withQuantizer(quantizer)
              .withSearchParams(searchParams)
              .withTopK(3)
              .build();

      assertTrue("Query should have quantized vectors", query.hasQuantizedQueries());
      assertEquals(DataType.BYTE, query.getQueryDataType());

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

      CuVSMatrix quantizedQueries = query.getQuantizedQueries();
      try {
        quantizer.inverseTransform(quantizedQueries);
        fail("Expected UnsupportedOperationException to be thrown");
      } catch (UnsupportedOperationException e) {
        assertTrue(
            "Exception message should mention inverse transform",
            e.getMessage().contains("inverse"));
      }

      // Cleanup
      index.close();
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
      assertEquals(DataType.BYTE, quantized.dataType());

      CuVSMatrix recovered = quantizer.inverseTransform(quantized);
      assertEquals(DataType.FLOAT, recovered.dataType());
      assertEquals(trainingDataset.size(), recovered.size());
      assertEquals(trainingDataset.columns(), recovered.columns());

      log.info("Inverse transform completed successfully");

      // Verify quantization worked
      assertEquals(DataType.BYTE, quantized.dataType());
      assertEquals(DataType.FLOAT, recovered.dataType());

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

      BinaryQuantizer quantizer = new BinaryQuantizer(resources, inputDataset);

      CuVSMatrix quantized = quantizer.transform(inputDataset);
      assertEquals(DataType.BYTE, quantized.dataType());
      assertEquals(CuVSMatrix.MemoryKind.HOST, quantized.memoryKind());

      try {
        quantizer.inverseTransform(quantized);
        fail("Expected UnsupportedOperationException to be thrown");
      } catch (UnsupportedOperationException e) {
        assertTrue(
            "Exception message should mention inverse transform",
            e.getMessage().contains("inverse"));
      }

      quantizer.close();
      inputDataset.close();
      quantized.close();

      log.info("Binary CPU quantization test passed");
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

        CagraSearchParams searchParams = new CagraSearchParams.Builder().build();
        CagraQuery query =
            CagraQuery.newBuilder(resources)
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
        index.close();
      }
    }
  }

  @Test
  public void testBinaryQuantizationWithZeroThreshold() throws Throwable {
    float[][] dataset =
        new float[][] {
          {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f}, // crosses zero
          {0.1f, 0.5f, 1.0f, 1.5f, 2.0f}, // all positive
          {-2.0f, -1.5f, -1.0f, -0.5f, 0.0f}, // mostly negative
          {-1.0f, 0.0f, 1.0f, 2.0f, 3.0f} // mixed
        };

    float[][] queries =
        new float[][] {
          {0.5f, 0.5f, 0.5f, 0.5f, 0.5f},
          {-0.5f, -0.5f, -0.5f, -0.5f, -0.5f}
        };

    try (CuVSResources resources = CheckedCuVSResources.create()) {
      CuVSMatrix trainingDataset = CuVSMatrix.ofArray(dataset);

      BinaryQuantizer quantizer =
          new BinaryQuantizer(resources, trainingDataset, BinaryQuantizer.ThresholdType.ZERO);
      assertEquals(DataType.BYTE, quantizer.outputDataType());
      assertEquals(BinaryQuantizer.ThresholdType.ZERO, quantizer.getThresholdType());

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

      log.info("Built index with ZERO threshold binary quantization");

      CagraSearchParams searchParams = new CagraSearchParams.Builder().build();
      CagraQuery query =
          CagraQuery.newBuilder(resources)
              .withQueryVectors(queries)
              .withQuantizer(quantizer)
              .withSearchParams(searchParams)
              .withTopK(2)
              .build();

      assertTrue("Query should have quantized vectors", query.hasQuantizedQueries());
      assertEquals(DataType.BYTE, query.getQueryDataType());

      SearchResults results = index.search(query);
      assertNotNull("Search results should not be null", results);
      assertEquals(
          "Should have results for all queries", queries.length, results.getResults().size());

      CuVSMatrix quantizedDataset = quantizer.transform(trainingDataset);
      byte[][] result = new byte[(int) quantizedDataset.size()][(int) quantizedDataset.columns()];
      quantizedDataset.toArray(result);

      byte firstByte = result[0][0];
      int[] unpackedBits = new int[5];
      for (int i = 0; i < 5; i++) {
        unpackedBits[i] = (firstByte >> i) & 1;
      }

      int[] expectedRow0 = {0, 0, 0, 1, 1};
      assertArrayEquals("ZERO threshold failed for row 0", expectedRow0, unpackedBits);

      index.close();
      quantizer.close();
      trainingDataset.close();
      quantizedDataset.close();

      log.info("ZERO threshold binary quantization test passed");
    }
  }

  @Test
  public void testBinaryQuantizationWithMeanThreshold() throws Throwable {
    float[][] dataset =
        new float[][] {
          {1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, // mean = 3.0
          {0.0f, 1.0f, 2.0f, 3.0f, 4.0f}, // mean = 2.0
          {2.0f, 4.0f, 6.0f, 8.0f, 10.0f}, // mean = 6.0
          {-1.0f, 0.0f, 1.0f, 2.0f, 3.0f} // mean = 1.0
        };

    float[][] queries =
        new float[][] {
          {2.5f, 2.5f, 2.5f, 2.5f, 2.5f},
          {1.0f, 3.0f, 5.0f, 7.0f, 9.0f}
        };

    try (CuVSResources resources = CheckedCuVSResources.create()) {
      CuVSMatrix trainingDataset = CuVSMatrix.ofArray(dataset);

      BinaryQuantizer quantizer =
          new BinaryQuantizer(resources, trainingDataset, BinaryQuantizer.ThresholdType.MEAN);
      assertEquals(BinaryQuantizer.ThresholdType.MEAN, quantizer.getThresholdType());

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

      log.info("Built index with MEAN threshold binary quantization");

      // Perform search
      CagraSearchParams searchParams = new CagraSearchParams.Builder().build();
      CagraQuery query =
          CagraQuery.newBuilder(resources)
              .withQueryVectors(queries)
              .withQuantizer(quantizer)
              .withSearchParams(searchParams)
              .withTopK(2)
              .build();

      SearchResults results = index.search(query);
      assertNotNull("Search results should not be null", results);
      assertEquals(
          "Should have results for all queries", queries.length, results.getResults().size());

      // Calculate overall mean: (3.0 + 2.0 + 6.0 + 1.0) / 4 = 3.0
      // Verify quantization behavior - values > mean should become 1, values <= mean should become
      // 0
      CuVSMatrix quantizedDataset = quantizer.transform(trainingDataset);
      byte[][] result1 = new byte[(int) quantizedDataset.size()][(int) quantizedDataset.columns()];

      // Copy data from CuVSMatrix to array
      quantizedDataset.toArray(result1);

      // Verify that MEAN threshold produces different results than ZERO threshold
      BinaryQuantizer zeroQuantizer =
          new BinaryQuantizer(resources, trainingDataset, BinaryQuantizer.ThresholdType.ZERO);
      CuVSMatrix zeroQuantized = zeroQuantizer.transform(trainingDataset);
      byte[][] zeroData = new byte[(int) zeroQuantized.size()][(int) zeroQuantized.columns()];
      zeroQuantized.toArray(zeroData);

      boolean isDifferent = false;
      for (int i = 0; i < result1.length && !isDifferent; i++) {
        for (int j = 0; j < result1[i].length && !isDifferent; j++) {
          if (result1[i][j] != zeroData[i][j]) {
            isDifferent = true;
          }
        }
      }
      assertTrue("MEAN and ZERO thresholds should produce different results", isDifferent);

      // Cleanup
      index.close();
      quantizer.close();
      zeroQuantizer.close();
      trainingDataset.close();
      quantizedDataset.close();
      zeroQuantized.close();

      log.info("MEAN threshold binary quantization test passed");
    }
  }

  @Test
  public void testBinaryQuantizationWithSamplingMedianThreshold() throws Throwable {
    float[][] dataset = createSimpleDataset();
    float[][] queries = createSimpleQueries();

    try (CuVSResources resources = CheckedCuVSResources.create()) {
      CuVSMatrix trainingDataset = CuVSMatrix.ofArray(dataset);
      assertEquals(DataType.FLOAT, trainingDataset.dataType());

      BinaryQuantizer quantizer =
          new BinaryQuantizer(
              resources, trainingDataset, BinaryQuantizer.ThresholdType.SAMPLING_MEDIAN);
      assertEquals(DataType.BYTE, quantizer.outputDataType());
      log.info("Created binary quantizer with sampling median threshold");

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

      CagraSearchParams searchParams = new CagraSearchParams.Builder().build();

      CagraQuery query =
          CagraQuery.newBuilder(resources)
              .withQueryVectors(queries)
              .withQuantizer(quantizer)
              .withSearchParams(searchParams)
              .withTopK(3)
              .build();

      assertTrue("Query should have quantized vectors", query.hasQuantizedQueries());
      assertEquals(DataType.BYTE, query.getQueryDataType());

      SearchResults results = index.search(query);
      assertNotNull("Search results should not be null", results);

      assertEquals(
          "Should return results for each query", queries.length, results.getResults().size());

      for (int i = 0; i < results.getResults().size(); i++) {
        Map<Integer, Float> result = results.getResults().get(i);
        assertFalse("Each query should return non-empty result", result.isEmpty());
        assertTrue("Should return at most topK results", result.size() <= 3);
      }

      log.info("Binary quantization with sampling median threshold test passed");
    }
  }
}
