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

public class BinaryQuantizerIT extends CuVSTestCase {

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
  public void testBinaryQuantizerBasic() throws Throwable {
    log.info("testBinaryQuantizerBasic");

    // Generate test data with both positive and negative values
    int rows = 1000;
    int cols = 128;
    float[][] testData = new float[rows][cols];

    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        // Generate values between -10 and 10
        testData[i][j] = (random.nextFloat() - 0.5f) * 20.0f;
      }
    }

    // Apply binary quantization
    byte[][] quantizedData = BinaryQuantizer.transform(cuvsResources, testData);

    assertNotNull(quantizedData);
    assertEquals(rows, quantizedData.length);
    assertEquals(cols, quantizedData[0].length);

    // Verify binary quantization logic: positive values -> 1, negative/zero -> 0
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        if (testData[i][j] > 0) {
          assertTrue("Positive values should be quantized to 1", quantizedData[i][j] == 1);
        } else {
          assertTrue("Negative/zero values should be quantized to 0", quantizedData[i][j] == 0);
        }
      }
    }
  }

  @Test
  public void testBinaryQuantizerWithDataset() throws Throwable {
    log.info("testBinaryQuantizerWithDataset");

    // Generate test data
    int rows = 1000;
    int cols = 128;
    float[][] testData = new float[rows][cols];

    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        // Generate values between -5 and 5
        testData[i][j] = (random.nextFloat() - 0.5f) * 10.0f;
      }
    }

    // Create dataset
    Dataset dataset = Dataset.create(rows, cols);
    for (int i = 0; i < rows; i++) {
      dataset.addVector(testData[i]);
    }

    // Apply binary quantization
    byte[][] quantizedData = BinaryQuantizer.transform(cuvsResources, dataset);

    assertNotNull(quantizedData);
    assertEquals(rows, quantizedData.length);
    assertEquals(cols, quantizedData[0].length);

    // Verify binary quantization logic
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        if (testData[i][j] > 0) {
          assertTrue("Positive values should be quantized to 1", quantizedData[i][j] == 1);
        } else {
          assertTrue("Negative/zero values should be quantized to 0", quantizedData[i][j] == 0);
        }
      }
    }

    dataset.close();
  }

  @Test
  public void testBinaryQuantizerEdgeCases() throws Throwable {
    log.info("testBinaryQuantizerEdgeCases");

    // Test with specific edge case values
    float[][] testData = {
      {-1.0f, 0.0f, 1.0f, -0.1f, 0.1f},
      {Float.MIN_VALUE, -Float.MIN_VALUE, Float.MAX_VALUE, -Float.MAX_VALUE, 0.0f}
    };

    byte[][] quantizedData = BinaryQuantizer.transform(cuvsResources, testData);

    assertNotNull(quantizedData);
    assertEquals(2, quantizedData.length);
    assertEquals(5, quantizedData[0].length);

    // Verify specific edge cases
    // Row 0: [-1.0, 0.0, 1.0, -0.1, 0.1] -> [0, 0, 1, 0, 1]
    assertEquals(0, quantizedData[0][0]); // -1.0 -> 0
    assertEquals(0, quantizedData[0][1]); // 0.0 -> 0
    assertEquals(1, quantizedData[0][2]); // 1.0 -> 1
    assertEquals(0, quantizedData[0][3]); // -0.1 -> 0
    assertEquals(1, quantizedData[0][4]); // 0.1 -> 1

    // Row 1: [MIN_VALUE, -MIN_VALUE, MAX_VALUE, -MAX_VALUE, 0.0] -> [1, 0, 1, 0, 0]
    assertEquals(1, quantizedData[1][0]); // MIN_VALUE -> 1
    assertEquals(0, quantizedData[1][1]); // -MIN_VALUE -> 0
    assertEquals(1, quantizedData[1][2]); // MAX_VALUE -> 1
    assertEquals(0, quantizedData[1][3]); // -MAX_VALUE -> 0
    assertEquals(0, quantizedData[1][4]); // 0.0 -> 0
  }
}
