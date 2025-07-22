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

    try {
      // Test quantization
      try (QuantizedMatrix quantizedMatrix = quantizer.transform(testData)) {
        assertNotNull(quantizedMatrix);
        assertEquals(testData.length, quantizedMatrix.rows());
        assertEquals(testData[0].length, quantizedMatrix.cols());

        // Convert to byte array for inverse transform
        byte[][] quantizedData = quantizedMatrix.toArray();

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

        // Test inverse quantization with QuantizedMatrix directly
        float[][] dequantizedData2 = quantizer.inverseTransform(quantizedMatrix);
        assertNotNull(dequantizedData2);
        assertEquals(testData.length, dequantizedData2.length);
        assertEquals(testData[0].length, dequantizedData2[0].length);

        // Results should be the same
        for (int i = 0; i < testData.length; i++) {
          for (int j = 0; j < testData[i].length; j++) {
            assertEquals(dequantizedData[i][j], dequantizedData2[i][j], 0.0001f);
          }
        }
      }
    } finally {
      quantizer.destroy();
    }
  }

  @Test
  public void testScalarQuantizerWithDataset() throws Throwable {
    log.info("testScalarQuantizerWithDataset");

    // Generate test data
    int rows = 1000;
    int cols = 128;
    float[][] trainingData = generateData(random, rows, cols);

    // Create dataset
    Dataset dataset = Dataset.ofArray(trainingData);

    try {
      // Create and train scalar quantizer
      ScalarQuantizer quantizer =
          ScalarQuantizer.newBuilder(cuvsResources)
              .withQuantile(0.99f)
              .withTrainingDataset(dataset)
              .build();

      assertNotNull(quantizer);

      try {
        // Test quantization with dataset
        try (QuantizedMatrix quantizedMatrix = quantizer.transform(dataset)) {
          assertNotNull(quantizedMatrix);
          assertEquals(rows, quantizedMatrix.rows());
          assertEquals(cols, quantizedMatrix.cols());

          // Test direct access
          for (int i = 0; i < Math.min(10, rows); i++) {
            for (int j = 0; j < Math.min(10, cols); j++) {
              byte value = quantizedMatrix.get(i, j);
              // Quantized values should be in valid range for int8
              assertTrue("Quantized value should be in valid range", value >= -128 && value <= 127);
            }
          }

          // Test copyRow functionality
          byte[] rowBuffer = new byte[cols];
          quantizedMatrix.copyRow(0, rowBuffer);
          for (int j = 0; j < cols; j++) {
            assertEquals(quantizedMatrix.get(0, j), rowBuffer[j]);
          }
        }
      } finally {
        quantizer.destroy();
      }
    } finally {
      dataset.close();
    }
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

  @Test
  public void testScalarQuantizerResourceManagement() throws Throwable {
    log.info("testScalarQuantizerResourceManagement");

    float[][] trainingData = generateData(random, 100, 64);
    float[][] testData = generateData(random, 10, 64);

    ScalarQuantizer quantizer =
        ScalarQuantizer.newBuilder(cuvsResources)
            .withQuantile(0.95f)
            .withTrainingDataset(trainingData)
            .build();

    try {
      QuantizedMatrix quantizedMatrix = quantizer.transform(testData);

      // Use the matrix
      assertEquals(10, quantizedMatrix.rows());
      assertEquals(64, quantizedMatrix.cols());

      // Close the matrix
      quantizedMatrix.close();

      // Verify that accessing after close throws exception
      try {
        quantizedMatrix.get(0, 0);
        assertTrue("Should throw exception after close", false);
      } catch (IllegalStateException e) {
        assertTrue("Expected exception message", e.getMessage().contains("closed"));
      }
    } finally {
      quantizer.destroy();
    }
  }

  @Test
  public void testScalarQuantizerCopyRow() throws Throwable {
    log.info("testScalarQuantizerCopyRow");

    // Generate test data
    int rows = 10;
    int cols = 8;
    float[][] trainingData = generateData(random, 100, cols);
    float[][] testData = generateData(random, rows, cols);

    ScalarQuantizer quantizer =
        ScalarQuantizer.newBuilder(cuvsResources)
            .withQuantile(0.95f)
            .withTrainingDataset(trainingData)
            .build();

    try {
      try (QuantizedMatrix quantizedMatrix = quantizer.transform(testData)) {
        // Test copyRow functionality
        byte[] rowBuffer = new byte[cols];

        for (int i = 0; i < rows; i++) {
          quantizedMatrix.copyRow(i, rowBuffer);

          // Verify the copied row matches direct access
          for (int j = 0; j < cols; j++) {
            assertEquals(quantizedMatrix.get(i, j), rowBuffer[j]);
          }
        }
      }
    } finally {
      quantizer.destroy();
    }
  }

  @Test
  public void testScalarQuantizerDirectAccess() throws Throwable {
    log.info("testScalarQuantizerDirectAccess");

    // Generate test data
    int rows = 50;
    int cols = 32;
    float[][] trainingData = generateData(random, 200, cols);
    float[][] testData = generateData(random, rows, cols);

    ScalarQuantizer quantizer =
        ScalarQuantizer.newBuilder(cuvsResources)
            .withQuantile(0.90f)
            .withTrainingDataset(trainingData)
            .build();

    try {
      try (QuantizedMatrix quantizedMatrix = quantizer.transform(testData)) {
        assertNotNull(quantizedMatrix);
        assertEquals(rows, quantizedMatrix.rows());
        assertEquals(cols, quantizedMatrix.cols());

        // Test direct access for all elements
        for (int i = 0; i < rows; i++) {
          for (int j = 0; j < cols; j++) {
            byte value = quantizedMatrix.get(i, j);
            // Quantized values should be in valid range for int8
            assertTrue("Quantized value should be in valid range", value >= -128 && value <= 127);
          }
        }

        // Test that toArray() matches direct access
        byte[][] arrayData = quantizedMatrix.toArray();
        assertEquals(rows, arrayData.length);
        assertEquals(cols, arrayData[0].length);

        for (int i = 0; i < rows; i++) {
          for (int j = 0; j < cols; j++) {
            assertEquals(
                "Array data should match direct access",
                quantizedMatrix.get(i, j),
                arrayData[i][j]);
          }
        }
      }
    } finally {
      quantizer.destroy();
    }
  }

  @Test
  public void testScalarQuantizerMatrixConsistency() throws Throwable {
    log.info("testScalarQuantizerMatrixConsistency");

    // Generate test data
    int rows = 20;
    int cols = 16;
    float[][] trainingData = generateData(random, 100, cols);
    float[][] testData = generateData(random, rows, cols);

    ScalarQuantizer quantizer =
        ScalarQuantizer.newBuilder(cuvsResources)
            .withQuantile(0.95f)
            .withTrainingDataset(trainingData)
            .build();

    try {
      try (QuantizedMatrix quantizedMatrix = quantizer.transform(testData)) {
        // Test that multiple calls to get() return the same value
        for (int i = 0; i < Math.min(5, rows); i++) {
          for (int j = 0; j < Math.min(5, cols); j++) {
            byte value1 = quantizedMatrix.get(i, j);
            byte value2 = quantizedMatrix.get(i, j);
            assertEquals("Multiple get() calls should return same value", value1, value2);
          }
        }

        // Test that copyRow() is consistent across multiple calls
        byte[] rowBuffer1 = new byte[cols];
        byte[] rowBuffer2 = new byte[cols];

        for (int i = 0; i < Math.min(3, rows); i++) {
          quantizedMatrix.copyRow(i, rowBuffer1);
          quantizedMatrix.copyRow(i, rowBuffer2);

          for (int j = 0; j < cols; j++) {
            assertEquals(
                "Multiple copyRow() calls should return same data", rowBuffer1[j], rowBuffer2[j]);
          }
        }

        // Test that toArray() is consistent across multiple calls
        byte[][] array1 = quantizedMatrix.toArray();
        byte[][] array2 = quantizedMatrix.toArray();

        for (int i = 0; i < rows; i++) {
          for (int j = 0; j < cols; j++) {
            assertEquals(
                "Multiple toArray() calls should return same data", array1[i][j], array2[i][j]);
          }
        }
      }
    } finally {
      quantizer.destroy();
    }
  }

  @Test
  public void testScalarQuantizerBoundaryConditions() throws Throwable {
    log.info("testScalarQuantizerBoundaryConditions");

    // Test with small matrix
    float[][] smallTrainingData = generateData(random, 10, 4);
    float[][] smallTestData = generateData(random, 2, 4);

    ScalarQuantizer quantizer =
        ScalarQuantizer.newBuilder(cuvsResources)
            .withQuantile(0.95f)
            .withTrainingDataset(smallTrainingData)
            .build();

    try {
      try (QuantizedMatrix quantizedMatrix = quantizer.transform(smallTestData)) {
        assertEquals(2, quantizedMatrix.rows());
        assertEquals(4, quantizedMatrix.cols());

        // Test boundary access
        byte value = quantizedMatrix.get(0, 0);
        assertTrue("First element should be valid", value >= -128 && value <= 127);

        value = quantizedMatrix.get(1, 3);
        assertTrue("Last element should be valid", value >= -128 && value <= 127);

        // Test copyRow with boundary rows
        byte[] rowBuffer = new byte[4];
        quantizedMatrix.copyRow(0, rowBuffer);
        quantizedMatrix.copyRow(1, rowBuffer);

        // Verify no exceptions thrown
        assertTrue("Boundary operations should succeed", true);
      }
    } finally {
      quantizer.destroy();
    }
  }
}
