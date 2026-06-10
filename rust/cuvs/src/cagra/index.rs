/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::ffi::CString;
use std::io::{Write, stderr};
use std::path::Path;

use crate::cagra::{IndexParams, SearchParams};
use crate::dlpack::ManagedTensor;
use crate::error::{Error, Result, check_cuvs};
use crate::resources::Resources;

/// CAGRA ANN Index
#[derive(Debug)]
pub struct Index(ffi::cuvsCagraIndex_t);

/// Convert a filesystem path into a `CString` suitable for the cuVS C API,
/// returning `Error::InvalidArgument` instead of panicking for paths that are
/// not valid UTF-8 or that contain an interior NUL byte.
fn path_to_cstring(path: &Path) -> Result<CString> {
    let path_str = path
        .to_str()
        .ok_or_else(|| Error::InvalidArgument(format!("path is not valid UTF-8: {path:?}")))?;
    CString::new(path_str)
        .map_err(|e| Error::InvalidArgument(format!("path contains an interior NUL byte: {e}")))
}

impl Index {
    /// Builds a new Index from the dataset for efficient search.
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `params` - Parameters for building the index
    /// * `dataset` - A row-major matrix on either the host or device to index
    pub fn build<T: Into<ManagedTensor>>(
        res: &Resources,
        params: &IndexParams,
        dataset: T,
    ) -> Result<Index> {
        let dataset: ManagedTensor = dataset.into();
        let index = Index::new()?;
        unsafe {
            check_cuvs(ffi::cuvsCagraBuild(res.0, params.0, dataset.as_ptr(), index.0))?;
        }
        Ok(index)
    }

    /// Creates a new empty index
    pub fn new() -> Result<Index> {
        unsafe {
            let mut index = std::mem::MaybeUninit::<ffi::cuvsCagraIndex_t>::uninit();
            check_cuvs(ffi::cuvsCagraIndexCreate(index.as_mut_ptr()))?;
            Ok(Index(index.assume_init()))
        }
    }

    /// Perform a Approximate Nearest Neighbors search on the Index
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `params` - Parameters to use in searching the index
    /// * `queries` - A matrix in device memory to query for
    /// * `neighbors` - Matrix in device memory that receives the indices of the nearest neighbors
    /// * `distances` - Matrix in device memory that receives the distances of the nearest neighbors
    pub fn search(
        &self,
        res: &Resources,
        params: &SearchParams,
        queries: &ManagedTensor,
        neighbors: &ManagedTensor,
        distances: &ManagedTensor,
    ) -> Result<()> {
        unsafe {
            let prefilter = ffi::cuvsFilter { addr: 0, type_: ffi::cuvsFilterType::NO_FILTER };

            check_cuvs(ffi::cuvsCagraSearch(
                res.0,
                params.0,
                self.0,
                queries.as_ptr(),
                neighbors.as_ptr(),
                distances.as_ptr(),
                prefilter,
            ))
        }
    }

    /// Perform a filtered Approximate Nearest Neighbors search on the Index
    ///
    /// Like [`search`](Self::search), but accepts a bitset filter to exclude
    /// vectors during graph traversal. Filtered vectors are never visited,
    /// giving better recall than post-filtering.
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `params` - Parameters to use in searching the index
    /// * `queries` - A matrix in device memory to query for
    /// * `neighbors` - Matrix in device memory that receives the indices of the nearest neighbors
    /// * `distances` - Matrix in device memory that receives the distances of the nearest neighbors
    /// * `bitset` - A 1-D `uint32` device tensor with `ceil(n_rows / 32)` elements.
    ///   Each bit corresponds to a dataset row: bit 1 = include, bit 0 = exclude.
    pub fn search_with_filter(
        &self,
        res: &Resources,
        params: &SearchParams,
        queries: &ManagedTensor,
        neighbors: &ManagedTensor,
        distances: &ManagedTensor,
        bitset: &ManagedTensor,
    ) -> Result<()> {
        unsafe {
            let prefilter = ffi::cuvsFilter {
                addr: bitset.as_ptr() as usize,
                type_: ffi::cuvsFilterType::BITSET,
            };

            check_cuvs(ffi::cuvsCagraSearch(
                res.0,
                params.0,
                self.0,
                queries.as_ptr(),
                neighbors.as_ptr(),
                distances.as_ptr(),
                prefilter,
            ))
        }
    }

    /// Save the CAGRA index to file.
    ///
    /// Experimental, both the API and the serialization format are subject to change.
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `filename` - The file path for saving the index
    /// * `include_dataset` - Whether to write out the dataset to the file
    pub fn serialize<P: AsRef<Path>>(
        &self,
        res: &Resources,
        filename: P,
        include_dataset: bool,
    ) -> Result<()> {
        let c_filename = path_to_cstring(filename.as_ref())?;
        unsafe {
            check_cuvs(ffi::cuvsCagraSerialize(res.0, c_filename.as_ptr(), self.0, include_dataset))
        }
    }

    /// Save the CAGRA index to file in hnswlib format.
    ///
    /// NOTE: The saved index can only be read by the hnswlib wrapper in cuVS,
    /// as the serialization format is not compatible with the original hnswlib.
    ///
    /// Experimental, both the API and the serialization format are subject to change.
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `filename` - The file path for saving the index
    pub fn serialize_to_hnswlib<P: AsRef<Path>>(&self, res: &Resources, filename: P) -> Result<()> {
        let c_filename = path_to_cstring(filename.as_ref())?;
        unsafe { check_cuvs(ffi::cuvsCagraSerializeToHnswlib(res.0, c_filename.as_ptr(), self.0)) }
    }

    /// Load a CAGRA index from file.
    ///
    /// Experimental, both the API and the serialization format are subject to change.
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `filename` - The path of the file that stores the index
    pub fn deserialize<P: AsRef<Path>>(res: &Resources, filename: P) -> Result<Index> {
        let c_filename = path_to_cstring(filename.as_ref())?;
        let index = Index::new()?;
        unsafe {
            check_cuvs(ffi::cuvsCagraDeserialize(res.0, c_filename.as_ptr(), index.0))?;
        }
        Ok(index)
    }
}

impl Drop for Index {
    fn drop(&mut self) {
        if let Err(e) = check_cuvs(unsafe { ffi::cuvsCagraIndexDestroy(self.0) }) {
            write!(stderr(), "failed to call cagraIndexDestroy {:?}", e)
                .expect("failed to write to stderr");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::s;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;

    const N_DATAPOINTS: usize = 256;
    const N_FEATURES: usize = 16;

    /// Build a small random dataset and a CAGRA index over it.
    fn build_test_index(
        res: &Resources,
        build_params: &IndexParams,
    ) -> (ndarray::Array2<f32>, Index) {
        let dataset =
            ndarray::Array::<f32, _>::random((N_DATAPOINTS, N_FEATURES), Uniform::new(0., 1.0));
        let index = Index::build(res, build_params, &dataset).expect("failed to build cagra index");
        (dataset, index)
    }

    /// Search the first `n_queries` rows of `dataset` against `index` and
    /// assert each query finds itself as the top-1 neighbor. CAGRA search
    /// requires queries and outputs to live in device memory.
    fn search_and_verify_self_neighbors(
        res: &Resources,
        index: &Index,
        dataset: &ndarray::Array2<f32>,
        n_queries: usize,
        k: usize,
    ) {
        let queries = dataset.slice(s![0..n_queries, ..]);
        let queries = ManagedTensor::from(&queries).to_device(res).unwrap();

        let mut neighbors_host = ndarray::Array::<u32, _>::zeros((n_queries, k));
        let neighbors = ManagedTensor::from(&neighbors_host).to_device(res).unwrap();

        let mut distances_host = ndarray::Array::<f32, _>::zeros((n_queries, k));
        let distances = ManagedTensor::from(&distances_host).to_device(res).unwrap();

        let search_params = SearchParams::new().unwrap();
        index.search(res, &search_params, &queries, &neighbors, &distances).expect("search failed");

        distances.to_host(res, &mut distances_host).unwrap();
        neighbors.to_host(res, &mut neighbors_host).unwrap();

        for i in 0..n_queries {
            assert_eq!(
                neighbors_host[[i, 0]],
                i as u32,
                "query {i} should be its own nearest neighbor"
            );
        }
    }

    fn test_cagra(build_params: IndexParams) {
        let res = Resources::new().unwrap();
        let (dataset, index) = build_test_index(&res, &build_params);
        search_and_verify_self_neighbors(&res, &index, &dataset, 4, 10);
    }

    #[test]
    fn test_cagra_index() {
        let build_params = IndexParams::new().unwrap();
        test_cagra(build_params);
    }

    #[test]
    fn test_cagra_compression() {
        use crate::cagra::CompressionParams;
        let build_params =
            IndexParams::new().unwrap().set_compression(CompressionParams::new().unwrap());
        test_cagra(build_params);
    }

    /// Test bitset-filtered search: exclude odd-indexed rows, verify they don't appear.
    #[test]
    fn test_cagra_search_with_filter() {
        let res = Resources::new().unwrap();
        let build_params = IndexParams::new().unwrap();

        let n_datapoints = 256;
        let n_features = 16;
        let dataset =
            ndarray::Array::<f32, _>::random((n_datapoints, n_features), Uniform::new(0., 1.0));

        let index =
            Index::build(&res, &build_params, &dataset).expect("failed to create cagra index");

        // Build a bitset that includes only even-indexed rows
        let n_words = (n_datapoints + 31) / 32;
        let mut bitset_host = ndarray::Array::<u32, _>::zeros(ndarray::Ix1(n_words));
        for i in 0..n_datapoints {
            if i % 2 == 0 {
                bitset_host[i / 32] |= 1u32 << (i % 32);
            }
        }
        let bitset = ManagedTensor::from(&bitset_host).to_device(&res).unwrap();

        // Query with the first 4 even-indexed rows
        let n_queries = 4;
        let queries = dataset.slice(s![0..n_queries * 2;2, ..]); // rows 0, 2, 4, 6
        let queries = ManagedTensor::from(&queries).to_device(&res).unwrap();

        let k = 10;
        let mut neighbors_host = ndarray::Array::<u32, _>::zeros((n_queries, k));
        let neighbors = ManagedTensor::from(&neighbors_host).to_device(&res).unwrap();
        let mut distances_host = ndarray::Array::<f32, _>::zeros((n_queries, k));
        let distances = ManagedTensor::from(&distances_host).to_device(&res).unwrap();

        let search_params = SearchParams::new().unwrap();

        index
            .search_with_filter(&res, &search_params, &queries, &neighbors, &distances, &bitset)
            .unwrap();

        neighbors.to_host(&res, &mut neighbors_host).unwrap();

        // All returned neighbors must be even-indexed (odd rows are filtered out).
        for q in 0..n_queries {
            for n in 0..k {
                let neighbor_id = neighbors_host[[q, n]];
                assert_eq!(
                    neighbor_id % 2,
                    0,
                    "query {q}, neighbor {n}: got odd index {neighbor_id}, expected only even"
                );
            }
        }

        // First query (row 0) should find itself as the nearest neighbor.
        assert_eq!(neighbors_host[[0, 0]], 0);
    }

    /// Test that an index can be searched multiple times without rebuilding.
    /// This validates that `search()` takes `&self` instead of `self`.
    #[test]
    fn test_cagra_multiple_searches() {
        let res = Resources::new().unwrap();
        let build_params = IndexParams::new().unwrap();
        let (dataset, index) = build_test_index(&res, &build_params);

        for _ in 0..3 {
            search_and_verify_self_neighbors(&res, &index, &dataset, 4, 5);
        }
    }

    #[test]
    fn test_cagra_serialize_deserialize() {
        let res = Resources::new().unwrap();
        let build_params = IndexParams::new().unwrap();
        let (dataset, index) = build_test_index(&res, &build_params);

        let filepath = std::env::temp_dir().join("test_cagra_index.bin");
        index.serialize(&res, &filepath, true).expect("failed to serialize cagra index");

        assert!(filepath.exists(), "serialized index file should exist");
        assert!(
            std::fs::metadata(&filepath).unwrap().len() > 0,
            "serialized index file should not be empty"
        );

        let loaded_index =
            Index::deserialize(&res, &filepath).expect("failed to deserialize cagra index");

        // The deserialized index should still find each query as its own
        // nearest neighbor.
        search_and_verify_self_neighbors(&res, &loaded_index, &dataset, 4, 10);

        let _ = std::fs::remove_file(&filepath);
    }

    #[test]
    fn test_cagra_serialize_without_dataset() {
        let res = Resources::new().unwrap();
        let build_params = IndexParams::new().unwrap();
        let (_dataset, index) = build_test_index(&res, &build_params);

        let filepath = std::env::temp_dir().join("test_cagra_index_no_dataset.bin");
        index
            .serialize(&res, &filepath, false)
            .expect("failed to serialize cagra index without dataset");

        assert!(filepath.exists(), "serialized index file should exist");

        let _ = std::fs::remove_file(&filepath);
    }

    #[test]
    fn test_cagra_serialize_to_hnswlib() {
        let res = Resources::new().unwrap();
        let build_params = IndexParams::new().unwrap();
        let (_dataset, index) = build_test_index(&res, &build_params);

        let filepath = std::env::temp_dir().join("test_cagra_index_hnsw.bin");
        index
            .serialize_to_hnswlib(&res, &filepath)
            .expect("failed to serialize cagra index to hnswlib format");

        assert!(filepath.exists(), "serialized hnswlib index file should exist");
        assert!(
            std::fs::metadata(&filepath).unwrap().len() > 0,
            "serialized hnswlib index file should not be empty"
        );

        let _ = std::fs::remove_file(&filepath);
    }

    /// Passing a filename containing an interior NUL byte must surface as an
    /// `InvalidArgument` error rather than panicking inside the serializer.
    #[test]
    fn test_cagra_serialize_rejects_interior_nul() {
        let res = Resources::new().unwrap();
        let build_params = IndexParams::new().unwrap();
        let (_dataset, index) = build_test_index(&res, &build_params);

        // `PathBuf::from` on Unix preserves arbitrary bytes, so we can embed a
        // NUL byte in the path and confirm the helper rejects it.
        let bad_path = std::path::PathBuf::from("/tmp/has\0nul.bin");
        let err = index
            .serialize(&res, &bad_path, true)
            .expect_err("serialize should reject paths with interior NUL");
        assert!(matches!(err, Error::InvalidArgument(_)), "expected InvalidArgument, got {err:?}");
    }
}
