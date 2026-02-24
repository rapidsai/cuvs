/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::io::{stderr, Write};

use crate::dlpack::ManagedTensor;
use crate::error::{check_cuvs, Result};
use crate::ivf_flat::{IndexParams, SearchParams};
use crate::resources::Resources;

/// Ivf-Flat ANN Index
#[derive(Debug)]
pub struct Index(ffi::cuvsIvfFlatIndex_t);

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
            check_cuvs(ffi::cuvsIvfFlatBuild(
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
            let mut index = std::mem::MaybeUninit::<ffi::cuvsIvfFlatIndex_t>::uninit();
            check_cuvs(ffi::cuvsIvfFlatIndexCreate(index.as_mut_ptr()))?;
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
            let prefilter = ffi::cuvsFilter {
                addr: 0,
                type_: ffi::cuvsFilterType::NO_FILTER,
            };

            check_cuvs(ffi::cuvsIvfFlatSearch(
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
}

impl Drop for Index {
    fn drop(&mut self) {
        if let Err(e) = check_cuvs(unsafe { ffi::cuvsIvfFlatIndexDestroy(self.0) }) {
            write!(stderr(), "failed to call cuvsIvfFlatIndexDestroy {:?}", e)
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

    #[test]
    fn test_ivf_flat() {
        let build_params = IndexParams::new().unwrap().set_n_lists(64);

        let res = Resources::new().unwrap();

        // Create a new random dataset to index
        let n_datapoints = 1024;
        let n_features = 16;
        let dataset =
            ndarray::Array::<f32, _>::random((n_datapoints, n_features), Uniform::new(0., 1.0));

        let dataset_device = ManagedTensor::from(&dataset).to_device(&res).unwrap();

        // build the ivf-flat index
        let index = Index::build(&res, &build_params, dataset_device)
            .expect("failed to create ivf-flat index");

        // use the first 4 points from the dataset as queries : will test that we get them back
        // as their own nearest neighbor
        let n_queries = 4;
        let queries = dataset.slice(s![0..n_queries, ..]);

        let k = 10;

        // IvfFlat search API requires queries and outputs to be on device memory
        // copy query data over, and allocate new device memory for the distances/ neighbors
        // outputs
        let queries = ManagedTensor::from(&queries).to_device(&res).unwrap();
        let mut neighbors_host = ndarray::Array::<i64, _>::zeros((n_queries, k));
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

    /// Test that an index can be searched multiple times without rebuilding.
    /// This validates that search() takes &self instead of self.
    #[test]
    fn test_ivf_flat_multiple_searches() {
        let build_params = IndexParams::new().unwrap().set_n_lists(64);
        let res = Resources::new().unwrap();

        // Create a random dataset
        let n_datapoints = 1024;
        let n_features = 16;
        let dataset =
            ndarray::Array::<f32, _>::random((n_datapoints, n_features), Uniform::new(0., 1.0));

        let dataset_device = ManagedTensor::from(&dataset).to_device(&res).unwrap();

        // Build the index once
        let index = Index::build(&res, &build_params, dataset_device)
            .expect("failed to create ivf-flat index");

        let search_params = SearchParams::new().unwrap();
        let k = 5;

        // Perform multiple searches on the same index
        for search_iter in 0..3 {
            let n_queries = 4;
            let queries = dataset.slice(s![0..n_queries, ..]);
            let queries = ManagedTensor::from(&queries).to_device(&res).unwrap();

            let mut neighbors_host = ndarray::Array::<i64, _>::zeros((n_queries, k));
            let neighbors = ManagedTensor::from(&neighbors_host)
                .to_device(&res)
                .unwrap();

            let mut distances_host = ndarray::Array::<f32, _>::zeros((n_queries, k));
            let distances = ManagedTensor::from(&distances_host)
                .to_device(&res)
                .unwrap();

            // This should work on every iteration because search() takes &self
            index
                .search(&res, &search_params, &queries, &neighbors, &distances)
                .expect(&format!("search iteration {} failed", search_iter));

            // Copy back to host memory
            distances.to_host(&res, &mut distances_host).unwrap();
            neighbors.to_host(&res, &mut neighbors_host).unwrap();

            // Verify results are consistent
            assert_eq!(
                neighbors_host[[0, 0]],
                0,
                "iteration {}: first query should find itself",
                search_iter
            );
        }
    }
}
