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

import static com.carrotsearch.randomizedtesting.RandomizedTest.*;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import com.carrotsearch.randomizedtesting.RandomizedRunner;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

@RunWith(RandomizedRunner.class)
public class DatasetIT extends CuVSTestCase {

  @Before
  public void setup() {
    assumeTrue("not supported on " + System.getProperty("os.name"), isLinuxAmd64());
    initializeRandom();
  }

  private static final float DELTA = 1e-9f;

  private static final byte[][] byteData = {
    {1, 2, 3},
    {0, 2, 3},
    {4, 1, 3},
    {3, 0, 2},
    {0, 4, 2}
  };

  private int[][] createIntMatrix() {
    int rows = randomIntBetween(1, 32);
    int cols = randomIntBetween(1, 100);

    int[][] result = new int[rows][cols];

    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        result[r][c] = randomInt();
      }
    }
    return result;
  }

  private float[][] createFloatMatrix() {
    int rows = randomIntBetween(1, 32);
    int cols = randomIntBetween(1, 100);

    float[][] result = new float[rows][cols];

    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        result[r][c] = randomFloat();
      }
    }
    return result;
  }

  @Test
  public void testByteDatasetRowGetAccess() {
    try (var dataset = Dataset.ofArray(byteData)) {
      for (int n = 0; n < dataset.size(); ++n) {
        var row = dataset.getRow(n);
        assertEquals(dataset.columns(), row.size());
        for (int i = 0; i < dataset.columns(); ++i) {
          assertEquals(byteData[n][i], row.getAsByte(i));
        }
      }
    }
  }

  @Test
  public void testByteDatasetRowCopy() {
    try (var dataset = Dataset.ofArray(byteData)) {
      for (int n = 0; n < dataset.size(); ++n) {
        var row = dataset.getRow(n);
        assertEquals(dataset.columns(), row.size());

        var rowCopy = new byte[(int) row.size()];
        row.toArray(rowCopy);
        assertArrayEquals(byteData[n], rowCopy);
      }
    }
  }

  @Test
  public void testByteDatasetCopy() {
    try (var dataset = Dataset.ofArray(byteData)) {
      var dataCopy = new byte[(int) dataset.size()][(int) dataset.columns()];
      dataset.toArray(dataCopy);
      for (int n = 0; n < dataset.size(); ++n) {
        for (int i = 0; i < dataset.columns(); ++i) {
          assertEquals(byteData[n][i], dataCopy[n][i]);
        }
      }
    }
  }

  @Test
  public void testIntDatasetRowGetAccess() {
    var intData = createIntMatrix();
    try (var dataset = Dataset.ofArray(intData)) {
      for (int n = 0; n < dataset.size(); ++n) {
        var row = dataset.getRow(n);
        assertEquals(dataset.columns(), row.size());
        for (int i = 0; i < dataset.columns(); ++i) {
          assertEquals(intData[n][i], row.getAsInt(i));
        }
      }
    }
  }

  @Test
  public void testIntDatasetRowCopy() {
    var intData = createIntMatrix();
    try (var dataset = Dataset.ofArray(intData)) {
      for (int n = 0; n < dataset.size(); ++n) {
        var row = dataset.getRow(n);
        assertEquals(dataset.columns(), row.size());

        var rowCopy = new int[(int) row.size()];
        row.toArray(rowCopy);
        assertArrayEquals(intData[n], rowCopy);
      }
    }
  }

  @Test
  public void testIntDatasetCopy() {
    var intData = createIntMatrix();
    try (var dataset = Dataset.ofArray(intData)) {
      var intDataCopy = new int[(int) dataset.size()][(int) dataset.columns()];
      dataset.toArray(intDataCopy);
      for (int n = 0; n < dataset.size(); ++n) {
        for (int i = 0; i < dataset.columns(); ++i) {
          assertEquals(intData[n][i], intDataCopy[n][i]);
        }
      }
    }
  }

  @Test
  public void testFloatDatasetRowGetAccess() {
    var floatData = createFloatMatrix();
    try (var dataset = Dataset.ofArray(floatData)) {
      for (int n = 0; n < dataset.size(); ++n) {
        var row = dataset.getRow(n);
        assertEquals(dataset.columns(), row.size());
        for (int i = 0; i < dataset.columns(); ++i) {
          assertEquals(floatData[n][i], row.getAsFloat(i), DELTA);
        }
      }
    }
  }

  @Test
  public void testFloatDatasetRowCopy() {
    var floatData = createFloatMatrix();
    try (var dataset = Dataset.ofArray(floatData)) {
      for (int n = 0; n < dataset.size(); ++n) {
        var row = dataset.getRow(n);
        assertEquals(dataset.columns(), row.size());

        var rowCopy = new float[(int) row.size()];
        row.toArray(rowCopy);
        assertArrayEquals(floatData[n], rowCopy, DELTA);
      }
    }
  }

  @Test
  public void testFloatDatasetCopy() {
    var floatData = createFloatMatrix();
    try (var dataset = Dataset.ofArray(floatData)) {
      var dataCopy = new float[(int) dataset.size()][(int) dataset.columns()];
      dataset.toArray(dataCopy);
      for (int n = 0; n < dataset.size(); ++n) {
        for (int i = 0; i < dataset.columns(); ++i) {
          assertEquals(floatData[n][i], dataCopy[n][i], DELTA);
        }
      }
    }
  }

  public void testFloatDatasetBuilder() {
    int rows = randomIntBetween(1, 32);
    int cols = randomIntBetween(1, 100);

    float[][] data = new float[rows][cols];
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        data[c][r] = randomFloat();
      }
    }

    var builder = Dataset.builder(rows, cols, Dataset.DataType.FLOAT);
    for (int r = 0; r < rows; ++r) {
      builder.addVector(data[r]);
    }

    float[][] roundTripData = new float[rows][cols];

    try (var dataset = builder.build()) {
      dataset.toArray(roundTripData);

      for (int n = 0; n < dataset.size(); ++n) {
        for (int i = 0; i < dataset.columns(); ++i) {
          assertEquals(data[n][i], roundTripData[n][i], DELTA);
        }
      }
    }
  }

  public void testIntDatasetBuilder() {
    int rows = randomIntBetween(1, 32);
    int cols = randomIntBetween(1, 100);

    var data = new int[rows][cols];
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        data[c][r] = randomInt();
      }
    }

    var builder = Dataset.builder(rows, cols, Dataset.DataType.INT);
    for (int r = 0; r < rows; ++r) {
      builder.addVector(data[r]);
    }

    var roundTripData = new int[rows][cols];

    try (var dataset = builder.build()) {
      dataset.toArray(roundTripData);

      for (int n = 0; n < dataset.size(); ++n) {
        for (int i = 0; i < dataset.columns(); ++i) {
          assertEquals(data[n][i], roundTripData[n][i]);
        }
      }
    }
  }

  public void testByteDatasetBuilder() {
    int rows = randomIntBetween(1, 32);
    int cols = randomIntBetween(1, 100);

    var data = new byte[rows][cols];
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        data[c][r] = randomByte();
      }
    }

    var builder = Dataset.builder(rows, cols, Dataset.DataType.BYTE);
    for (int r = 0; r < rows; ++r) {
      builder.addVector(data[r]);
    }

    var roundTripData = new byte[rows][cols];

    try (var dataset = builder.build()) {
      dataset.toArray(roundTripData);

      for (int n = 0; n < dataset.size(); ++n) {
        for (int i = 0; i < dataset.columns(); ++i) {
          assertEquals(data[n][i], roundTripData[n][i]);
        }
      }
    }
  }
}
