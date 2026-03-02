/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::ffi::CString;
use std::io::{stderr, Write};

use crate::cagra::{IndexParams, SearchParams};
use crate::dlpack::ManagedTensor;
use crate::error::{check_cuvs, Result};
use crate::resources::Resources;

/// CAGRA ANN Index
#[derive(Debug)]
pub struct Index(ffi::cuvsCagraIndex_t);

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
            check_cuvs(ffi::cuvsCagraBuild(
                res.0,
                params.0,
                dataset.as_ptr(),
                index.0,
            ))?;
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
        self,
        res: &Resources,
        params: &SearchParams,
        queries: &ManagedTensor,
        neighbors: &ManagedTensor,
        distances: &ManagedTensor,
    ) -> Result<()> {
        unsafe {
            let prefilter = ffi::cuvsFilter {
                addr: 0,
                type_: ffi::cuvsFilterType::NO_FILTER,
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
    /// * `filename` - The file name for saving the index
    /// * `include_dataset` - Whether to write out the dataset to the file
    pub fn serialize(&self, res: &Resources, filename: &str, include_dataset: bool) -> Result<()> {
        let c_filename = CString::new(filename).expect("filename contains null byte");
        unsafe {
            check_cuvs(ffi::cuvsCagraSerialize(
                res.0,
                c_filename.as_ptr(),
                self.0,
                include_dataset,
            ))
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
    /// * `filename` - The file name for saving the index
    pub fn serialize_to_hnswlib(&self, res: &Resources, filename: &str) -> Result<()> {
        let c_filename = CString::new(filename).expect("filename contains null byte");
        unsafe {
            check_cuvs(ffi::cuvsCagraSerializeToHnswlib(
                res.0,
                c_filename.as_ptr(),
                self.0,
            ))
        }
    }

    /// Load a CAGRA index from file.
    ///
    /// Experimental, both the API and the serialization format are subject to change.
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `filename` - The name of the file that stores the index
    pub fn deserialize(res: &Resources, filename: &str) -> Result<Index> {
        let c_filename = CString::new(filename).expect("filename contains null byte");
        let index = Index::new()?;
        unsafe {
            check_cuvs(ffi::cuvsCagraDeserialize(
                res.0,
                c_filename.as_ptr(),
                index.0,
            ))?;
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
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    fn test_cagra(build_params: IndexParams) {
        let res = Resources::new().unwrap();

        // Create a new random dataset to index
        let n_datapoints = 256;
        let n_features = 16;
        let dataset =
            ndarray::Array::<f32, _>::random((n_datapoints, n_features), Uniform::new(0., 1.0));

        // build the cagra index
        let index =
            Index::build(&res, &build_params, &dataset).expect("failed to create cagra index");

        // use the first 4 points from the dataset as queries : will test that we get them back
        // as their own nearest neighbor
        let n_queries = 4;
        let queries = dataset.slice(s![0..n_queries, ..]);

        let k = 10;

        // CAGRA search API requires queries and outputs to be on device memory
        // copy query data over, and allocate new device memory for the distances/ neighbors
        // outputs
        let queries = ManagedTensor::from(&queries).to_device(&res).unwrap();
        let mut neighbors_host = ndarray::Array::<u32, _>::zeros((n_queries, k));
        let neighbors = ManagedTensor::from(&neighbors_host)
            .to_device(&res)
            .unwrap();

        let mut distances_host = ndarray::Array::<f32, _>::zeros((n_queries, k));
        let distances = ManagedTensor::from(&distances_host)
            .to_device(&res)
            .unwrap();

        let search_params = SearchParams::new().unwrap();

        index
            .search(&res, &search_params, &queries, &neighbors, &distances)
            .unwrap();

        // Copy back to host memory
        distances.to_host(&res, &mut distances_host).unwrap();
        neighbors.to_host(&res, &mut neighbors_host).unwrap();

        // nearest neighbors should be themselves, since queries are from the
        // dataset
        assert_eq!(neighbors_host[[0, 0]], 0);
        assert_eq!(neighbors_host[[1, 0]], 1);
        assert_eq!(neighbors_host[[2, 0]], 2);
        assert_eq!(neighbors_host[[3, 0]], 3);
    }

    #[test]
    fn test_cagra_index() {
        let build_params = IndexParams::new().unwrap();
        test_cagra(build_params);
    }

    #[test]
    fn test_cagra_compression() {
        use crate::cagra::CompressionParams;
        let build_params = IndexParams::new()
            .unwrap()
            .set_compression(CompressionParams::new().unwrap());
        test_cagra(build_params);
    }

    #[test]
    fn test_cagra_serialize_deserialize() {
        let res = Resources::new().unwrap();

        // Create a random dataset
        let n_datapoints = 256;
        let n_features = 16;
        let dataset =
            ndarray::Array::<f32, _>::random((n_datapoints, n_features), Uniform::new(0., 1.0));

        // Build the CAGRA index
        let build_params = IndexParams::new().unwrap();
        let index =
            Index::build(&res, &build_params, &dataset).expect("failed to create cagra index");

        // Serialize to a temp file (including dataset)
        let dir = std::env::temp_dir();
        let filepath = dir.join("test_cagra_index.bin");
        let filepath_str = filepath.to_str().unwrap();
        index
            .serialize(&res, filepath_str, true)
            .expect("failed to serialize cagra index");

        // Verify the file was created
        assert!(filepath.exists(), "serialized index file should exist");
        assert!(
            std::fs::metadata(&filepath).unwrap().len() > 0,
            "serialized index file should not be empty"
        );

        // Deserialize the index
        let loaded_index =
            Index::deserialize(&res, filepath_str).expect("failed to deserialize cagra index");

        // Search the deserialized index with the first 4 points from the dataset
        let n_queries = 4;
        let queries = dataset.slice(s![0..n_queries, ..]);
        let k = 10;

        let queries = ManagedTensor::from(&queries).to_device(&res).unwrap();
        let mut neighbors_host = ndarray::Array::<u32, _>::zeros((n_queries, k));
        let neighbors = ManagedTensor::from(&neighbors_host)
            .to_device(&res)
            .unwrap();

        let mut distances_host = ndarray::Array::<f32, _>::zeros((n_queries, k));
        let distances = ManagedTensor::from(&distances_host)
            .to_device(&res)
            .unwrap();

        let search_params = SearchParams::new().unwrap();

        loaded_index
            .search(&res, &search_params, &queries, &neighbors, &distances)
            .expect("failed to search deserialized index");

        // Copy back to host memory
        distances.to_host(&res, &mut distances_host).unwrap();
        neighbors.to_host(&res, &mut neighbors_host).unwrap();

        // The nearest neighbors should be themselves, since queries are from the dataset
        assert_eq!(neighbors_host[[0, 0]], 0);
        assert_eq!(neighbors_host[[1, 0]], 1);
        assert_eq!(neighbors_host[[2, 0]], 2);
        assert_eq!(neighbors_host[[3, 0]], 3);

        // Clean up
        let _ = std::fs::remove_file(&filepath);
    }

    #[test]
    fn test_cagra_serialize_without_dataset() {
        let res = Resources::new().unwrap();

        // Create a random dataset
        let n_datapoints = 256;
        let n_features = 16;
        let dataset =
            ndarray::Array::<f32, _>::random((n_datapoints, n_features), Uniform::new(0., 1.0));

        // Build the CAGRA index
        let build_params = IndexParams::new().unwrap();
        let index =
            Index::build(&res, &build_params, &dataset).expect("failed to create cagra index");

        // Serialize without dataset (graph only)
        let dir = std::env::temp_dir();
        let filepath = dir.join("test_cagra_index_no_dataset.bin");
        let filepath_str = filepath.to_str().unwrap();
        index
            .serialize(&res, filepath_str, false)
            .expect("failed to serialize cagra index without dataset");

        // Verify the file was created and is smaller than with dataset
        assert!(filepath.exists(), "serialized index file should exist");

        // Clean up
        let _ = std::fs::remove_file(&filepath);
    }

    #[test]
    fn test_cagra_serialize_to_hnswlib() {
        let res = Resources::new().unwrap();

        // Create a random dataset
        let n_datapoints = 256;
        let n_features = 16;
        let dataset =
            ndarray::Array::<f32, _>::random((n_datapoints, n_features), Uniform::new(0., 1.0));

        // Build the CAGRA index
        let build_params = IndexParams::new().unwrap();
        let index =
            Index::build(&res, &build_params, &dataset).expect("failed to create cagra index");

        // Serialize to hnswlib format
        let dir = std::env::temp_dir();
        let filepath = dir.join("test_cagra_index_hnsw.bin");
        let filepath_str = filepath.to_str().unwrap();
        index
            .serialize_to_hnswlib(&res, filepath_str)
            .expect("failed to serialize cagra index to hnswlib format");

        // Verify the file was created
        assert!(
            filepath.exists(),
            "serialized hnswlib index file should exist"
        );
        assert!(
            std::fs::metadata(&filepath).unwrap().len() > 0,
            "serialized hnswlib index file should not be empty"
        );

        // Clean up
        let _ = std::fs::remove_file(&filepath);
    }
}
