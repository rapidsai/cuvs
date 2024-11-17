package com.nvidia.cuvs.common;

import java.util.List;
import java.util.Map;

public interface SearchResults {

	/**
	 * Gets a list results as a map of neighbor IDs to distances.
	 * 
	 * @return a list of results for each query as a map of neighbor IDs to distance
	 */
	public List<Map<Integer, Float>> getResults();
}
