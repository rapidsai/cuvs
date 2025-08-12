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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@RunWith(RandomizedRunner.class)
public class CuVSMatrixIT extends CuVSTestCase {

  private static final Logger log = LoggerFactory.getLogger(CuVSMatrixIT.class);

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

    return createFloatMatrix(rows, cols);
  }

  private float[][] createFloatMatrix(int rows, int cols) {
    float[][] result = new float[rows][cols];

    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        result[r][c] = randomFloat();
      }
    }
    return result;
  }

  private void testByteDatasetRowGetAccess(CuVSMatrix dataset) {
    for (int n = 0; n < dataset.size(); ++n) {
      var row = dataset.getRow(n);
      assertEquals(dataset.columns(), row.size());
      for (int i = 0; i < dataset.columns(); ++i) {
        assertEquals(byteData[n][i], row.getAsByte(i));
      }
    }
  }

  @Test
  public void testByteHostDatasetRowGetAccess() {
    try (var matrix = CuVSMatrix.ofArray(byteData)) {
      testByteDatasetRowGetAccess(matrix);
    }
  }

  @Test
  public void testByteDeviceDatasetRowGetAccess() throws Throwable {
    try (var resources = CheckedCuVSResources.create()) {
      var builder =
          CuVSMatrix.deviceBuilder(
              resources, byteData.length, byteData[0].length, CuVSMatrix.DataType.BYTE, 0);
      for (int i = 0; i < byteData.length; i++) {
        builder.addVector(byteData[i]);
      }
      try (var matrix = builder.build()) {
        testByteDatasetRowGetAccess(matrix);
      }
    }
  }

  private void testByteDatasetRowCopy(CuVSMatrix dataset) {
    for (int n = 0; n < dataset.size(); ++n) {
      var row = dataset.getRow(n);
      assertEquals(dataset.columns(), row.size());

      var rowCopy = new byte[(int) row.size()];
      row.toArray(rowCopy);
      assertArrayEquals(byteData[n], rowCopy);
    }
  }

  @Test
  public void testByteHostDatasetRowCopy() {
    try (var dataset = CuVSMatrix.ofArray(byteData)) {
      testByteDatasetRowCopy(dataset);
    }
  }

  @Test
  public void testByteDeviceDatasetRowCopy() throws Throwable {
    try (var resources = CheckedCuVSResources.create()) {
      var builder =
          CuVSMatrix.deviceBuilder(
              resources, byteData.length, byteData[0].length, CuVSMatrix.DataType.BYTE, 0);
      for (int i = 0; i < byteData.length; i++) {
        builder.addVector(byteData[i]);
      }
      try (var matrix = builder.build()) {
        testByteDatasetRowCopy(matrix);
      }
    }
  }

  private void testByteDatasetCopy(CuVSMatrix dataset) {
    var dataCopy = new byte[(int) dataset.size()][(int) dataset.columns()];
    dataset.toArray(dataCopy);
    for (int n = 0; n < dataset.size(); ++n) {
      for (int i = 0; i < dataset.columns(); ++i) {
        assertEquals(byteData[n][i], dataCopy[n][i]);
      }
    }
  }

  @Test
  public void testByteHostDatasetCopy() {
    try (var dataset = CuVSMatrix.ofArray(byteData)) {
      testByteDatasetCopy(dataset);
    }
  }

  @Test
  public void testByteDeviceDatasetCopy() throws Throwable {
    try (var resources = CheckedCuVSResources.create()) {
      var builder =
          CuVSMatrix.deviceBuilder(
              resources, byteData.length, byteData[0].length, CuVSMatrix.DataType.BYTE, 0);
      for (int i = 0; i < byteData.length; i++) {
        builder.addVector(byteData[i]);
      }
      try (var matrix = builder.build()) {
        testByteDatasetCopy(matrix);
      }
    }
  }

  @Test
  public void testIntDatasetRowGetAccess() {
    var intData = createIntMatrix();
    try (var dataset = CuVSMatrix.ofArray(intData)) {
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
    try (var dataset = CuVSMatrix.ofArray(intData)) {
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
    try (var dataset = CuVSMatrix.ofArray(intData)) {
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
    try (var dataset = CuVSMatrix.ofArray(floatData)) {
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
    try (var dataset = CuVSMatrix.ofArray(floatData)) {
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
    try (var dataset = CuVSMatrix.ofArray(floatData)) {
      var dataCopy = new float[(int) dataset.size()][(int) dataset.columns()];
      dataset.toArray(dataCopy);
      for (int n = 0; n < dataset.size(); ++n) {
        for (int i = 0; i < dataset.columns(); ++i) {
          assertEquals(floatData[n][i], dataCopy[n][i], DELTA);
        }
      }
    }
  }

  private void testFloatDatasetBuilder(int rows, int cols, CuVSMatrix.Builder builder) {

    float[][] data = new float[rows][cols];
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        data[r][c] = randomFloat();
      }
    }

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

  @Test
  public void testFloatDatasetHostBuilder() {
    int rows = randomIntBetween(1, 32);
    int cols = randomIntBetween(1, 100);
    testFloatDatasetBuilder(
        rows, cols, CuVSMatrix.hostBuilder(rows, cols, CuVSMatrix.DataType.FLOAT));
  }

  @Test
  public void testFloatDatasetDeviceBuilder() throws Throwable {
    int rows = randomIntBetween(1, 1024);
    int cols = randomIntBetween(1, 2048);
    int copyType = randomIntBetween(0, 2);
    try (var resources = CheckedCuVSResources.create()) {
      testFloatDatasetBuilder(
          rows,
          cols,
          CuVSMatrix.deviceBuilder(resources, rows, cols, CuVSMatrix.DataType.FLOAT, copyType));
    }
  }

  private void testIntDatasetBuilder(int rows, int cols, CuVSMatrix.Builder builder) {

    var data = new int[rows][cols];
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        data[r][c] = randomInt();
      }
    }

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

  @Test
  public void testIntDatasetHostBuilder() {
    int rows = randomIntBetween(1, 32);
    int cols = randomIntBetween(1, 100);
    testIntDatasetBuilder(rows, cols, CuVSMatrix.hostBuilder(rows, cols, CuVSMatrix.DataType.INT));
  }

  @Test
  public void testIntDatasetDeviceBuilder() throws Throwable {
    int rows = randomIntBetween(1, 1024);
    int cols = randomIntBetween(1, 2048);
    try (var resources = CheckedCuVSResources.create()) {
      testIntDatasetBuilder(
          rows, cols, CuVSMatrix.deviceBuilder(resources, rows, cols, CuVSMatrix.DataType.INT, 0));
    }
  }

  private void testByteDatasetBuilder(int rows, int cols, CuVSMatrix.Builder builder) {

    var data = new byte[rows][cols];
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        data[r][c] = randomByte();
      }
    }

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

  @Test
  public void testByteDatasetHostBuilder() {
    int rows = randomIntBetween(1, 32);
    int cols = randomIntBetween(1, 100);
    testByteDatasetBuilder(
        rows, cols, CuVSMatrix.hostBuilder(rows, cols, CuVSMatrix.DataType.BYTE));
  }

  @Test
  public void testByteDatasetDeviceBuilder() throws Throwable {
    int rows = randomIntBetween(1, 1024);
    int cols = randomIntBetween(1, 2048);
    try (var resources = CheckedCuVSResources.create()) {
      testByteDatasetBuilder(
          rows, cols, CuVSMatrix.deviceBuilder(resources, rows, cols, CuVSMatrix.DataType.BYTE, 0));
    }
  }

  @Test
  public void testDeviceToHost() throws Throwable {

    final int size = 16 * 1024;
    final int columns = 2048;
    final float[][] data = createFloatMatrix(size, columns);

    try (var resources = CuVSResources.create()) {

      var builder =
          CuVSMatrix.deviceBuilder(resources, size, columns, CuVSMatrix.DataType.FLOAT, 2);
      for (int i = 0; i < size; ++i) {
        var array = data[i];
        builder.addVector(array);
      }

      try (var deviceMatrix = (CuVSDeviceMatrix) builder.build();
          var hostMatrix = deviceMatrix.toHost(resources)) {

        assertEquals(data.length, deviceMatrix.size());
        assertEquals(data[0].length, deviceMatrix.columns());

        assertEquals(deviceMatrix.size(), hostMatrix.size());
        assertEquals(deviceMatrix.columns(), hostMatrix.columns());

        var roundTripData = new float[size][columns];

        hostMatrix.toArray(roundTripData);

        for (int n = 0; n < hostMatrix.size(); ++n) {
          for (int i = 0; i < hostMatrix.columns(); ++i) {
            assertEquals(data[n][i], roundTripData[n][i], 1e-9);
          }
        }
      }
    }
  }
}
