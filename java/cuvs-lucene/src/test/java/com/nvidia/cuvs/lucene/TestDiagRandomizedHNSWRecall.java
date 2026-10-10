/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.Set;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.analysis.MockAnalyzer;
import org.apache.lucene.tests.analysis.MockTokenizer;
import org.apache.lucene.tests.codecs.asserting.AssertingCodec;
import org.apache.lucene.tests.index.RandomIndexWriter;
import org.apache.lucene.tests.util.English;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.apache.lucene.tests.util.TestUtil;
import org.junit.Test;

/**
 * CI-only diagnostic for TestCuVSRandomizedHNSWVectorSearch. Builds many random indexes drawn from
 * the same distribution as that test, with three vector fields per document holding the same
 * vector: "gpu" (Lucene99AcceleratedHNSWVectorsFormat, the format under test), "cpu" (Lucene's own
 * Lucene99HnswVectorsFormat with default parameters) and "cpu32" (Lucene99HnswVectorsFormat with
 * maxConn=32, beamWidth=32, the HNSW-equivalent parameters the accelerated format is built from).
 * All three fields live in the same segments, so the comparison is on identical data and segment
 * structure. Prints miss statistics (a "miss" is a hit ranked beyond 3*k among the documents that
 * have vectors) and checks hard invariants: valid ids, docs that have vectors, no duplicates,
 * sorted scores, scores matching the exact distance of the returned id, and full reachability.
 */
@SuppressSysoutChecks(bugUrl = "")
public class TestDiagRandomizedHNSWRecall extends LuceneTestCase {

  static final String[] FIELDS = {"gpu", "cpu", "cpu32"};
  static final int[] KS = {1, 2, 3, 5, 8, 13, 21, 34, 55, 64, 89, 128, 200};
  static final int NUM_QUERIES = 10;

  static final class Stats {
    long[] queries = new long[KS.length];
    long[] missQueries = new long[KS.length];
    double[] recallSum = new double[KS.length];
    double[] recallMin = new double[KS.length];
    long invariantViolations = 0;
    long unreachableTrials = 0;
    // Test-equivalent event: query 0, topK uniform in [1, 64] capped at the dataset size.
    long testRuns = 0;
    long testFailures = 0;
    // miss rate (k <= 64) by dimension bucket
    long[] dimQueries = new long[5];
    long[] dimMisses = new long[5];

    Stats() {
      Arrays.fill(recallMin, 1.0);
    }
  }

  static int dimBucket(int dims) {
    if (dims <= 16) return 0;
    if (dims <= 128) return 1;
    if (dims <= 512) return 2;
    if (dims <= 1024) return 3;
    return 4;
  }

  @Test
  public void testSweep() throws Exception {
    assumeTrue("cuVS not supported", isSupported());
    double minutes = Double.parseDouble(System.getProperty("diag.minutes", "5"));
    int maxTrials = Integer.getInteger("diag.trials", 1_000_000);
    long deadline = System.nanoTime() + (long) (minutes * 60e9);

    KnnVectorsFormat gpuFormat = new Lucene99AcceleratedHNSWVectorsFormat();
    KnnVectorsFormat cpuFormat = new Lucene99HnswVectorsFormat();
    KnnVectorsFormat cpu32Format = new Lucene99HnswVectorsFormat(32, 32);
    Codec codec =
        new AssertingCodec() {
          @Override
          public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
            switch (field) {
              case "gpu":
                return gpuFormat;
              case "cpu":
                return cpuFormat;
              default:
                return cpu32Format;
            }
          }
        };

    Stats[] stats = new Stats[FIELDS.length];
    for (int f = 0; f < FIELDS.length; f++) stats[f] = new Stats();
    int printedMisses = 0;
    int trials = 0;

    while (System.nanoTime() < deadline && trials < maxTrials) {
      trials++;
      Random random = random();
      int datasetSize = random.nextInt(1000) + 1;
      int dims = random.nextInt(2048) + 1;
      float[][] dataset = TestUtils.generateDataset(random, datasetSize, dims);
      boolean[] hasVector = new boolean[datasetSize];
      int numVectors = 0;

      try (Directory dir = newDirectory()) {
        RandomIndexWriter writer =
            new RandomIndexWriter(
                random,
                dir,
                newIndexWriterConfig(new MockAnalyzer(random, MockTokenizer.SIMPLE, true))
                    .setMaxBufferedDocs(TestUtil.nextInt(random, 100, 1000))
                    .setCodec(codec)
                    .setMergePolicy(newTieredMergePolicy()));
        for (int i = 0; i < datasetSize; i++) {
          Document doc = new Document();
          doc.add(new StringField("id", String.valueOf(i), Field.Store.YES));
          doc.add(newTextField("field", English.intToEnglish(i), Field.Store.YES));
          boolean skipVector = random.nextInt(10) < 4;
          if (!skipVector || datasetSize < 100) {
            hasVector[i] = true;
            numVectors++;
            for (String field : FIELDS) {
              doc.add(
                  new KnnFloatVectorField(field, dataset[i], VectorSimilarityFunction.EUCLIDEAN));
            }
          }
          writer.addDocument(doc);
        }
        IndexReader reader = writer.getReader();
        writer.close();
        if (numVectors == 0) {
          reader.close();
          continue;
        }
        try {
          IndexSearcher searcher = newSearcher(reader);
          StoredFields storedFields = reader.storedFields();
          List<Integer> segSizes = new ArrayList<>();
          for (LeafReaderContext ctx : reader.leaves()) {
            FloatVectorValues v = ctx.reader().getFloatVectorValues("gpu");
            segSizes.add(v == null ? 0 : v.size());
          }

          float[][] queries = TestUtils.generateQueries(random, dims, NUM_QUERIES);
          int testTopK = Math.min(random.nextInt(64) + 1, datasetSize);

          for (int q = 0; q < NUM_QUERIES; q++) {
            // exact distances and ranks among documents with vectors
            double[] dist = new double[datasetSize];
            Integer[] order = new Integer[numVectors];
            int o = 0;
            for (int j = 0; j < datasetSize; j++) {
              if (!hasVector[j]) continue;
              double d = 0;
              for (int x = 0; x < dims; x++) {
                double diff = queries[q][x] - dataset[j][x];
                d += diff * diff;
              }
              dist[j] = d;
              order[o++] = j;
            }
            Arrays.sort(order, (a, b) -> Double.compare(dist[a], dist[b]));
            int[] rank = new int[datasetSize];
            Arrays.fill(rank, -1);
            for (int r = 0; r < numVectors; r++) rank[order[r]] = r + 1;

            for (int f = 0; f < FIELDS.length; f++) {
              String field = FIELDS[f];
              Stats st = stats[f];
              // Reachability: with k = numVectors every vector must come back.
              if (q == 0) {
                ScoreDoc[] all =
                    searcher.search(
                            new KnnFloatVectorQuery(field, queries[q], numVectors), numVectors)
                        .scoreDocs;
                if (all.length != numVectors) {
                  st.unreachableTrials++;
                  System.out.println(
                      String.format(
                          Locale.ROOT,
                          "DIAGSWEEP UNREACHABLE field=%s trial=%d n=%d nv=%d dims=%d segs=%s"
                              + " returned=%d",
                          field,
                          trials,
                          datasetSize,
                          numVectors,
                          dims,
                          segSizes,
                          all.length));
                }
              }
              List<Integer> ks = new ArrayList<>();
              for (int k : KS) ks.add(k);
              if (q == 0) ks.add(-1); // marker for the test-equivalent topK
              for (int ki = 0; ki < ks.size(); ki++) {
                boolean testEquivalent = ks.get(ki) == -1;
                int k = testEquivalent ? testTopK : ks.get(ki);
                if (!testEquivalent && k > numVectors) continue;
                ScoreDoc[] hits =
                    searcher.search(new KnnFloatVectorQuery(field, queries[q], k), k).scoreDocs;
                // invariants
                String violation = null;
                int expectedCount = Math.min(k, numVectors);
                if (hits.length != expectedCount) {
                  violation = "count " + hits.length + " != " + expectedCount;
                }
                Set<Integer> seen = new HashSet<>();
                int worst = 0;
                int inTopK = 0;
                float prevScore = Float.POSITIVE_INFINITY;
                int[] ids = new int[hits.length];
                for (int h = 0; h < hits.length; h++) {
                  int id = Integer.parseInt(storedFields.document(hits[h].doc).get("id"));
                  ids[h] = id;
                  if (id < 0 || id >= datasetSize || !hasVector[id]) {
                    violation = "invalid id " + id;
                    continue;
                  }
                  if (!seen.add(id)) violation = "duplicate id " + id;
                  float expectedScore =
                      VectorSimilarityFunction.EUCLIDEAN.compare(queries[q], dataset[id]);
                  if (Math.abs(expectedScore - hits[h].score) > 1e-4f * expectedScore + 1e-12f) {
                    violation =
                        "score mismatch id " + id + " " + hits[h].score + " vs " + expectedScore;
                  }
                  if (hits[h].score > prevScore) violation = "unsorted scores at " + h;
                  prevScore = hits[h].score;
                  worst = Math.max(worst, rank[id]);
                  if (rank[id] <= k) inTopK++;
                }
                if (violation != null) {
                  st.invariantViolations++;
                  System.out.println(
                      String.format(
                          Locale.ROOT,
                          "DIAGSWEEP VIOLATION field=%s trial=%d n=%d nv=%d dims=%d segs=%s k=%d:"
                              + " %s",
                          field,
                          trials,
                          datasetSize,
                          numVectors,
                          dims,
                          segSizes,
                          k,
                          violation));
                }
                boolean miss = worst > 3 * k;
                if (testEquivalent) {
                  st.testRuns++;
                  if (miss) st.testFailures++;
                } else {
                  st.queries[ki]++;
                  if (miss) st.missQueries[ki]++;
                  double recall = expectedCount == 0 ? 1.0 : (double) inTopK / expectedCount;
                  st.recallSum[ki] += recall;
                  st.recallMin[ki] = Math.min(st.recallMin[ki], recall);
                  if (k <= 64) {
                    st.dimQueries[dimBucket(dims)]++;
                    if (miss) st.dimMisses[dimBucket(dims)]++;
                  }
                }
                if (miss && (f == 0 || testEquivalent) && printedMisses < 200) {
                  printedMisses++;
                  int kth = order[Math.min(k, numVectors) - 1];
                  int k3 = order[Math.min(3 * k, numVectors) - 1];
                  int[] hitRanks = new int[ids.length];
                  for (int h = 0; h < ids.length; h++) hitRanks[h] = rank[ids[h]];
                  int worstId = -1;
                  for (int id : ids) if (rank[id] == worst) worstId = id;
                  System.out.println(
                      String.format(
                          Locale.ROOT,
                          "DIAGSWEEP MISS field=%s%s trial=%d n=%d nv=%d dims=%d segs=%s k=%d"
                              + " worstId=%d worstRank=%d d(worst)/d(1st)=%.4f"
                              + " d(worst)/d(k-th)=%.4f d(3k-th)/d(1st)=%.4f d(last)/d(1st)=%.4f"
                              + " recall=%d/%d ranks=%s",
                          field,
                          testEquivalent ? "(test)" : "",
                          trials,
                          datasetSize,
                          numVectors,
                          dims,
                          segSizes,
                          k,
                          worstId,
                          worst,
                          Math.sqrt(dist[worstId] / dist[order[0]]),
                          Math.sqrt(dist[worstId] / dist[kth]),
                          Math.sqrt(dist[k3] / dist[order[0]]),
                          Math.sqrt(dist[order[numVectors - 1]] / dist[order[0]]),
                          inTopK,
                          expectedCount,
                          Arrays.toString(hitRanks)));
                }
              }
            }
          }
        } finally {
          reader.close();
        }
      }
      if (trials % 50 == 0) report(stats, trials);
    }
    report(stats, trials);
  }

  static void report(Stats[] stats, int trials) {
    System.out.println("DIAGSWEEP REPORT trials=" + trials);
    for (int f = 0; f < FIELDS.length; f++) {
      Stats st = stats[f];
      StringBuilder sb = new StringBuilder();
      sb.append(
          String.format(
              Locale.ROOT,
              "DIAGSWEEP field=%s violations=%d unreachableTrials=%d testEquivalentFailures=%d/%d;",
              FIELDS[f],
              st.invariantViolations,
              st.unreachableTrials,
              st.testFailures,
              st.testRuns));
      for (int ki = 0; ki < KS.length; ki++) {
        if (st.queries[ki] == 0) continue;
        sb.append(
            String.format(
                Locale.ROOT,
                " k=%d miss=%d/%d recall=%.4f min=%.3f;",
                KS[ki],
                st.missQueries[ki],
                st.queries[ki],
                st.recallSum[ki] / st.queries[ki],
                st.recallMin[ki]));
      }
      sb.append(" dimBuckets(<=16,<=128,<=512,<=1024,<=2048) miss(k<=64)=");
      for (int b = 0; b < 5; b++) {
        sb.append(st.dimMisses[b]).append('/').append(st.dimQueries[b]).append(' ');
      }
      System.out.println(sb);
    }
  }
}
