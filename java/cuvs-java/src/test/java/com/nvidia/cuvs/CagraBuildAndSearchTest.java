package com.nvidia.cuvs;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

import com.nvidia.cuvs.cagra.CagraIndex;
import com.nvidia.cuvs.cagra.CagraIndexParams;
import com.nvidia.cuvs.cagra.CagraSearchParams;
import com.nvidia.cuvs.cagra.CuVSQuery;
import com.nvidia.cuvs.cagra.CuVSResources;
import com.nvidia.cuvs.cagra.SearchResult;

public class CagraBuildAndSearchTest {

  /**
   * A basic test that checks the whole flow - from indexing to search.
   * 
   * @throws Throwable
   */
  @Test
  public void testIndexingAndSearchingFlow() throws Throwable {

    // Sample data and query
    float[][] dataset = {{ 0.74021935f, 0.9209938f }, { 0.03902049f, 0.9689629f }, { 0.92514056f, 0.4463501f }, { 0.6673192f, 0.10993068f }};
    Map<Integer, Integer> map = Map.of(0, 0, 1, 1, 2, 2, 3, 3);
    float[][] queries = {{ 0.48216683f, 0.0428398f }, { 0.5084142f, 0.6545497f }, { 0.51260436f, 0.2643005f }, { 0.05198065f, 0.5789965f }};
    
    // Expected search result
    List<Map<Integer, Float>> expectedQueryResults = Arrays.asList(
        Map.of(3, 0.038782578f, 2, 0.3590463f, 0, 0.83774555f),
        Map.of(0, 0.12472608f, 2, 0.21700792f, 1, 0.31918612f), 
        Map.of(3, 0.047766715f, 2, 0.20332818f, 0, 0.48305473f), 
        Map.of(1, 0.15224178f, 0, 0.59063464f, 3, 0.5986642f));

    // Create resource
    CuVSResources res = new CuVSResources();

    // Configure index parameters
    CagraIndexParams cagraIndexParams = new CagraIndexParams
        .Builder()
        .withBuildAlgo(CagraIndexParams.CuvsCagraGraphBuildAlgo.IVF_PQ)
        .build();

    // Create the index with the dataset
    CagraIndex index = new CagraIndex
        .Builder(res)
        .withDataset(dataset)
        .withIndexParams(cagraIndexParams)
        .build();

    // Configure search parameters
    CagraSearchParams cagraSearchParams = new CagraSearchParams
        .Builder()
        .build();

    // Create a query object with the query vectors
    CuVSQuery query = new CuVSQuery
        .Builder()
        .withTopK(3)
        .withSearchParams(cagraSearchParams)
        .withQueryVectors(queries)
        .withMapping(map)
        .build();

    // Perform the search
    SearchResult searchResults = index.search(query);
    
    // Check results
    assertEquals(expectedQueryResults, searchResults.getResults(), "Results different than expected");
  }
}