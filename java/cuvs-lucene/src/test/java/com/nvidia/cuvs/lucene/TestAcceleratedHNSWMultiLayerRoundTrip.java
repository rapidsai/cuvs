/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;
import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

import com.carrotsearch.randomizedtesting.annotations.Name;
import com.carrotsearch.randomizedtesting.annotations.ParametersFactory;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.HnswGraphProvider;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.hnsw.HnswGraph;

/** Exercises the native upper-layer path through serialization and CPU search. */
@SuppressSysoutChecks(bugUrl = "")
public class TestAcceleratedHNSWMultiLayerRoundTrip extends LuceneTestCase {

  private static final String FIELD = "vector";
  private static final int VECTOR_COUNT = 256;
  private static final int GRAPH_DEGREE = 16;

  private final KnnVectorsFormat format;
  private final int dimensions;

  public TestAcceleratedHNSWMultiLayerRoundTrip(
      @Name("knnVectorsFormat") KnnVectorsFormat format, @Name("dimensions") int dimensions) {
    this.format = format;
    this.dimensions = dimensions;
  }

  @ParametersFactory
  public static List<Object[]> parameters() {
    AcceleratedHNSWParams params =
        new AcceleratedHNSWParams.Builder()
            .withStrategy(AcceleratedHNSWParams.Strategy.CUSTOM)
            .withIntermediateGraphDegree(32)
            .withGraphDegree(GRAPH_DEGREE)
            .withHNSWLayer(3)
            .build();
    return Arrays.asList(
        new Object[][] {
          {new Lucene99AcceleratedHNSWVectorsFormat(params), 32},
          {new LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat(params), 129},
          {new LuceneAcceleratedHNSWScalarQuantizedVectorsFormat(params), 32}
        });
  }

  public void testThreeLevelGraphCanBeReopenedAndSearched() throws Exception {
    assumeTrue("cuVS not supported", isSupported());
    float[][] vectors = randomVectors(VECTOR_COUNT, dimensions);

    try (Directory directory = newDirectory()) {
      IndexWriterConfig config =
          newIndexWriterConfig().setCodec(TestUtil.alwaysKnnVectorsFormat(acceleratedOnlyFormat()));
      try (IndexWriter writer = new IndexWriter(directory, config)) {
        for (int id = 0; id < vectors.length; id++) {
          Document document = new Document();
          document.add(new StringField("id", Integer.toString(id), Field.Store.YES));
          document.add(new KnnFloatVectorField(FIELD, vectors[id], EUCLIDEAN));
          writer.addDocument(document);
        }
        writer.forceMerge(1);
      }

      TestUtil.checkIndex(directory);
      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        LeafReader leaf = getOnlyLeafReader(reader);
        HnswGraph graph = graphOf(leaf);
        assertEquals(3, graph.numLevels());

        List<Set<Integer>> nodesByLevel = collectNodes(graph);
        int expectedLevelOneSize = Math.max(2, VECTOR_COUNT / graph.maxConn());
        int expectedLevelTwoSize = Math.max(2, expectedLevelOneSize / graph.maxConn());
        assertEquals(VECTOR_COUNT, nodesByLevel.get(0).size());
        assertEquals(expectedLevelOneSize, nodesByLevel.get(1).size());
        assertEquals(expectedLevelTwoSize, nodesByLevel.get(2).size());
        assertTrue(nodesByLevel.get(0).containsAll(nodesByLevel.get(1)));
        assertTrue(nodesByLevel.get(1).containsAll(nodesByLevel.get(2)));
        assertUpperNeighborsStayOnTheirLevel(graph, nodesByLevel);

        int queryNode = graph.entryNode();
        assertTrue(nodesByLevel.get(2).contains(queryNode));
        IndexSearcher searcher = new IndexSearcher(reader);

        // DIAG: the graph ordinal (== docID in this single dense leaf) and the insertion id differ
        // once a merge policy reorders segments during forceMerge.
        int[] docToId = new int[leaf.maxDoc()];
        int reorderedDocs = 0;
        for (int doc = 0; doc < leaf.maxDoc(); doc++) {
          docToId[doc] = Integer.parseInt(leaf.storedFields().document(doc).get("id"));
          if (docToId[doc] != doc) {
            reorderedDocs++;
          }
        }
        String fmt = format.getClass().getSimpleName();
        String mp = config.getMergePolicy().getClass().getSimpleName();

        // Original (pre-fix) query: vectors[queryNode] interprets the graph ordinal as an id.
        var oldHits = searcher.search(new KnnFloatVectorQuery(FIELD, vectors[queryNode], 10), 10);
        boolean oldFound = false;
        for (var hit : oldHits.scoreDocs) {
          oldFound |= docToId[hit.doc] == queryNode;
        }
        // Fixed query: the entry node's own stored vector, matched by docID.
        float[] entryVector = vectors[docToId[queryNode]];
        var newHits = searcher.search(new KnnFloatVectorQuery(FIELD, entryVector, 10), 10);
        boolean newFound = false;
        for (var hit : newHits.scoreDocs) {
          newFound |= hit.doc == queryNode;
        }
        String oldTarget = diagTarget(leaf, vectors[queryNode], docOfId(docToId, queryNode));
        String newTarget = diagTarget(leaf, entryVector, queryNode);
        System.out.println(
            "PDIAG fmt="
                + fmt
                + " mp="
                + mp
                + " entryOrd="
                + queryNode
                + " entryId="
                + docToId[queryNode]
                + " reorderedDocs="
                + reorderedDocs
                + " oldFound="
                + oldFound
                + " newFound="
                + newFound
                + " old["
                + oldTarget
                + "] new["
                + newTarget
                + "]");
        if (!oldFound || !newFound) {
          System.out.println("PDIAG-HITS fmt=" + fmt + " old=" + describeHits(oldHits, docToId));
          System.out.println("PDIAG-HITS fmt=" + fmt + " new=" + describeHits(newHits, docToId));
        }
        assertEquals(10, newHits.scoreDocs.length);
        assertTrue("the entry-node vector must be returned for its own query", newFound);
      }
    }
  }

  private static int docOfId(int[] docToId, int id) {
    for (int doc = 0; doc < docToId.length; doc++) {
      if (docToId[doc] == id) {
        return doc;
      }
    }
    return -1;
  }

  private static String describeHits(org.apache.lucene.search.TopDocs hits, int[] docToId) {
    StringBuilder sb = new StringBuilder();
    for (var hit : hits.scoreDocs) {
      sb.append(hit.doc).append("/id").append(docToId[hit.doc]).append(':').append(hit.score);
      sb.append(' ');
    }
    return sb.toString();
  }

  /**
   * Scores every doc against {@code query} with the format's own scorer (quantized for BQ/SQ) and
   * with exact float L2, and reports where {@code targetDoc} ranks.
   */
  private static String diagTarget(LeafReader leaf, float[] query, int targetDoc)
      throws IOException {
    var values = leaf.getFloatVectorValues(FIELD);
    var scorer = values.scorer(query);
    var it = scorer.iterator();
    float[] scores = new float[leaf.maxDoc()];
    for (int doc = it.nextDoc(); doc != NO_MORE_DOCS; doc = it.nextDoc()) {
      scores[doc] = scorer.score();
    }
    var exactValues = leaf.getFloatVectorValues(FIELD);
    var exactIt = exactValues.iterator();
    float[] exact = new float[leaf.maxDoc()];
    for (int doc = exactIt.nextDoc(); doc != NO_MORE_DOCS; doc = exactIt.nextDoc()) {
      exact[doc] = EUCLIDEAN.compare(query, exactValues.vectorValue(exactIt.index()));
    }
    float self = scores[targetDoc];
    int ge = 0;
    int gt = 0;
    int exactGe = 0;
    float best = Float.NEGATIVE_INFINITY;
    for (int doc = 0; doc < scores.length; doc++) {
      if (doc != targetDoc && scores[doc] >= self) {
        ge++;
      }
      if (doc != targetDoc && scores[doc] > self) {
        gt++;
      }
      if (doc != targetDoc && exact[doc] >= exact[targetDoc]) {
        exactGe++;
      }
      if (doc != targetDoc) {
        best = Math.max(best, scores[doc]);
      }
    }
    return "doc="
        + targetDoc
        + " selfScore="
        + self
        + " bestOther="
        + best
        + " othersGE="
        + ge
        + " othersGT="
        + gt
        + " exactSelf="
        + exact[targetDoc]
        + " exactOthersGE="
        + exactGe;
  }

  private KnnVectorsFormat acceleratedOnlyFormat() {
    return new KnnVectorsFormat(format.getName()) {
      @Override
      public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
        KnnVectorsWriter writer = format.fieldsWriter(state);
        if (!(writer instanceof Lucene99AcceleratedHNSWVectorsWriter)
            && !(writer instanceof LuceneAcceleratedHNSWBinaryQuantizedVectorsWriter)
            && !(writer instanceof LuceneAcceleratedHNSWScalarQuantizedVectorsWriter)) {
          IOUtils.closeWhileHandlingException(writer);
          throw new AssertionError("CPU fallback must not satisfy native upper-layer tests");
        }
        return writer;
      }

      @Override
      public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
        return format.fieldsReader(state);
      }

      @Override
      public int getMaxDimensions(String fieldName) {
        return format.getMaxDimensions(fieldName);
      }
    };
  }

  private static List<Set<Integer>> collectNodes(HnswGraph graph) throws Exception {
    List<Set<Integer>> nodesByLevel = new ArrayList<>();
    for (int level = 0; level < graph.numLevels(); level++) {
      Set<Integer> nodes = new HashSet<>();
      HnswGraph.NodesIterator iterator = graph.getNodesOnLevel(level);
      while (iterator.hasNext()) {
        assertTrue("duplicate node on level " + level, nodes.add(iterator.nextInt()));
      }
      nodesByLevel.add(nodes);
    }
    return nodesByLevel;
  }

  private static void assertUpperNeighborsStayOnTheirLevel(
      HnswGraph graph, List<Set<Integer>> nodesByLevel) throws Exception {
    for (int level = 1; level < graph.numLevels(); level++) {
      Set<Integer> nodes = nodesByLevel.get(level);
      int arcCount = 0;
      for (int node : nodes) {
        graph.seek(level, node);
        for (int neighbor = graph.nextNeighbor();
            neighbor != NO_MORE_DOCS;
            neighbor = graph.nextNeighbor()) {
          assertTrue("negative upper-layer neighbor", neighbor >= 0);
          assertTrue("upper-layer neighbor outside the index", neighbor < VECTOR_COUNT);
          assertTrue(
              "upper-layer neighbor is not a member of level " + level, nodes.contains(neighbor));
          arcCount++;
        }
      }
      assertTrue("upper layer " + level + " contains no arcs", arcCount > 0);
    }
  }

  private static HnswGraph graphOf(LeafReader leaf) throws Exception {
    KnnVectorsReader reader = ((CodecReader) leaf).getVectorReader();
    if (reader instanceof PerFieldKnnVectorsFormat.FieldsReader fieldsReader) {
      reader = fieldsReader.getFieldReader(FIELD);
    }
    return ((HnswGraphProvider) reader).getGraph(FIELD);
  }

  private static float[][] randomVectors(int count, int dimensions) {
    Random random = new Random(0x2476L);
    float[][] vectors = new float[count][dimensions];
    for (float[] vector : vectors) {
      for (int dimension = 0; dimension < vector.length; dimension++) {
        vector[dimension] = random.nextFloat();
      }
    }
    return vectors;
  }
}
