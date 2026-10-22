/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

import static com.carrotsearch.randomizedtesting.RandomizedTest.assumeTrue;
import static org.junit.Assert.assertEquals;

import com.carrotsearch.randomizedtesting.RandomizedRunner;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/** A build must not leave the calling thread's OpenMP thread count changed. */
@RunWith(RandomizedRunner.class)
public class OmpThreadsIT extends CuVSTestCase {

  private MethodHandle ompGetMaxThreads;

  @Before
  public void setup() {
    assumeTrue("not supported on " + System.getProperty("os.name"), isLinuxSupportedArch());
    initializeRandom();
    var symbol =
        SymbolLookup.libraryLookup("libgomp.so.1", Arena.global())
            .find("omp_get_max_threads")
            .orElseThrow();
    ompGetMaxThreads =
        Linker.nativeLinker().downcallHandle(symbol, FunctionDescriptor.of(ValueLayout.JAVA_INT));
  }

  private int ompMaxThreads() throws Throwable {
    return (int) ompGetMaxThreads.invokeExact();
  }

  @Test
  public void testCagraBuildRestoresOmpThreads() throws Throwable {
    var params =
        new CagraIndexParams.Builder()
            .withCagraGraphBuildAlgo(CagraIndexParams.CagraGraphBuildAlgo.NN_DESCENT)
            .withNumWriterThreads(3)
            .build();
    try (var resources = CheckedCuVSResources.create();
        var dataset = CuVSMatrix.ofArray(createFloatMatrix(2000, 32))) {
      int before = ompMaxThreads();
      try (var index =
          CagraIndex.newBuilder(resources).withDataset(dataset).withIndexParams(params).build()) {
        assertEquals(before, ompMaxThreads());
      }
    }
  }

  @Test
  public void testCagraBuildWithoutParamsLeavesOmpThreads() throws Throwable {
    try (var resources = CheckedCuVSResources.create();
        var dataset = CuVSMatrix.ofArray(createFloatMatrix(2000, 32))) {
      int before = ompMaxThreads();
      try (var index = CagraIndex.newBuilder(resources).withDataset(dataset).build()) {
        assertEquals(before, ompMaxThreads());
      }
    }
  }

  @Test
  public void testCagraMergeRestoresOmpThreads() throws Throwable {
    var params =
        new CagraIndexParams.Builder()
            .withCagraGraphBuildAlgo(CagraIndexParams.CagraGraphBuildAlgo.NN_DESCENT)
            .withNumWriterThreads(3)
            .build();
    try (var resources = CheckedCuVSResources.create();
        var dataset1 = CuVSMatrix.ofArray(createFloatMatrix(2000, 32)).toDevice(resources);
        var dataset2 = CuVSMatrix.ofArray(createFloatMatrix(2000, 32)).toDevice(resources);
        var index1 =
            CagraIndex.newBuilder(resources).withDataset(dataset1).withIndexParams(params).build();
        var index2 =
            CagraIndex.newBuilder(resources)
                .withDataset(dataset2)
                .withIndexParams(params)
                .build()) {
      int before = ompMaxThreads();
      try (var merged = CagraIndex.merge(new CagraIndex[] {index1, index2}, params)) {
        assertEquals(before, ompMaxThreads());
      }
    }
  }

  @Test
  public void testBruteForceBuildWithoutParams() throws Throwable {
    try (var resources = CheckedCuVSResources.create();
        var dataset = CuVSMatrix.ofArray(createFloatMatrix(500, 16))) {
      int before = ompMaxThreads();
      try (var index = BruteForceIndex.newBuilder(resources).withDataset(dataset).build()) {
        assertEquals(before, ompMaxThreads());
      }
    }
  }
}
