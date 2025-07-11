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
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.lang.invoke.MethodHandles;
import java.util.Random;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ScalarQuantizerIT extends CuVSTestCase {

  private static final Logger log = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());
  private CuVSResources cuvsResources;

  @Before
  public void setUp() throws Throwable {
    assumeTrue(isLinuxAmd64());
    random = new Random();
    cuvsResources = CuVSResources.create();
  }

  @After
  public void tearDown() throws Throwable {
    if (cuvsResources != null) {
      cuvsResources.close();
    }
  }

  @Test
  public void testScalarQuantizerBasic() throws Throwable {
    log.info("testScalarQuantizerBasic");

    // Generate test data
    int rows = 1000;
    int cols = 128;
    float[][] trainingData = generateData(random, rows, cols);
    float[][] testData = generateData(random, 10, cols);

    // Create and train scalar quantizer
    ScalarQuantizer quantizer =
        ScalarQuantizer.newBuilder(cuvsResources)
            .withQuantile(0.95f)
            .withTrainingDataset(trainingData)
            .build();

    assertNotNull(quantizer);

    // Test quantization
    byte[][] quantizedData = quantizer.transform(testData);
    assertNotNull(quantizedData);
    assertEquals(testData.length, quantizedData.length);
    assertEquals(testData[0].length, quantizedData[0].length);

    // Test inverse quantization
    float[][] dequantizedData = quantizer.inverseTransform(quantizedData);
    assertNotNull(dequantizedData);
    assertEquals(testData.length, dequantizedData.length);
    assertEquals(testData[0].length, dequantizedData[0].length);

    // The dequantized data should be reasonably close to the original
    // (with some quantization error)
    for (int i = 0; i < testData.length; i++) {
      for (int j = 0; j < testData[i].length; j++) {
        // Allow for some quantization error
        assertTrue(
            "Dequantized data should be reasonably close to original",
            Math.abs(testData[i][j] - dequantizedData[i][j]) < 10.0f);
      }
    }

    quantizer.destroy();
  }

  @Test
  public void testScalarQuantizerWithDataset() throws Throwable {
    log.info("testScalarQuantizerWithDataset");

    // Generate test data
    int rows = 1000;
    int cols = 128;
    float[][] trainingData = generateData(random, rows, cols);

    // Create dataset
    Dataset dataset = Dataset.create(rows, cols);
    for (int i = 0; i < rows; i++) {
      dataset.addVector(trainingData[i]);
    }

    // Create and train scalar quantizer
    ScalarQuantizer quantizer =
        ScalarQuantizer.newBuilder(cuvsResources)
            .withQuantile(0.99f)
            .withTrainingDataset(dataset)
            .build();

    assertNotNull(quantizer);

    // Test quantization with dataset
    byte[][] quantizedData = quantizer.transform(dataset);
    assertNotNull(quantizedData);
    assertEquals(rows, quantizedData.length);
    assertEquals(cols, quantizedData[0].length);

    quantizer.destroy();
    dataset.close();
  }

  @Test
  public void testScalarQuantizerParameterValidation() throws Throwable {
    log.info("testScalarQuantizerParameterValidation");

    int rows = 100;
    int cols = 64;
    float[][] trainingData = generateData(random, rows, cols);

    // Test invalid quantile values
    try {
      ScalarQuantizer.newBuilder(cuvsResources)
          .withQuantile(0.0f)
          .withTrainingDataset(trainingData)
          .build();
      assertTrue("Should have thrown IllegalArgumentException for quantile 0.0", false);
    } catch (IllegalArgumentException e) {
      // Expected
    }

    try {
      ScalarQuantizer.newBuilder(cuvsResources)
          .withQuantile(1.5f)
          .withTrainingDataset(trainingData)
          .build();
      assertTrue("Should have thrown IllegalArgumentException for quantile 1.5", false);
    } catch (IllegalArgumentException e) {
      // Expected
    }

    // Test missing training data
    try {
      ScalarQuantizer.newBuilder(cuvsResources).withQuantile(0.95f).build();
      assertTrue("Should have thrown IllegalArgumentException for missing training data", false);
    } catch (IllegalArgumentException e) {
      // Expected
    }
  }
}
