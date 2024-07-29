/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
//! Brute Force KNN

use std::io::{stderr, Write};

use crate::distance_type::DistanceType;
use crate::dlpack::ManagedTensor;
use crate::error::{check_cuvs, Result};
use crate::resources::Resources;

/// Brute Force KNN Index
#[derive(Debug)]
pub struct Index(ffi::cuvsBruteForceIndex_t);

impl Index {
    /// Builds a new Brute Force KNN Index from the dataset for efficient search.
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `metric` - DistanceType to use for building the index
    /// * `metric_arg` - Optional value of `p` for Minkowski distances
    /// * `dataset` - A row-major matrix on either the host or device to index
    pub fn build<T: Into<ManagedTensor>>(
        res: &Resources,
        metric: DistanceType,
        metric_arg: Option<f32>,
        dataset: T,
    ) -> Result<Index> {
        let dataset: ManagedTensor = dataset.into();
        let index = Index::new()?;
        unsafe {
            check_cuvs(ffi::cuvsBruteForceBuild(
                res.0,
                dataset.as_ptr(),
                metric,
                metric_arg.unwrap_or(2.0),
                index.0,
            ))?;
        }
        Ok(index)
    }

    /// Creates a new empty index
    pub fn new() -> Result<Index> {
        unsafe {
            let mut index = std::mem::MaybeUninit::<ffi::cuvsBruteForceIndex_t>::uninit();
            check_cuvs(ffi::cuvsBruteForceIndexCreate(index.as_mut_ptr()))?;
            Ok(Index(index.assume_init()))
        }
    }

    /// Perform a Nearest Neighbors search on the Index
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `queries` - A matrix in device memory to query for
    /// * `neighbors` - Matrix in device memory that receives the indices of the nearest neighbors
    /// * `distances` - Matrix in device memory that receives the distances of the nearest neighbors
    pub fn search(
        self,
        res: &Resources,
        queries: &ManagedTensor,
        neighbors: &ManagedTensor,
        distances: &ManagedTensor,
    ) -> Result<()> {
        unsafe {
            let prefilter = ffi::cuvsFilter {
                addr: 0,
                type_: ffi::cuvsFilterType::NO_FILTER,
            };

            check_cuvs(ffi::cuvsBruteForceSearch(
                res.0,
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
        if let Err(e) = check_cuvs(unsafe { ffi::cuvsBruteForceIndexDestroy(self.0) }) {
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
    use mark_flaky_tests::flaky;

    fn test_bfknn(metric: DistanceType) {
        let res = Resources::new().unwrap();

        // Create a new random dataset to index
        let n_datapoints = 16;
        let n_features = 8;
        let dataset_host =
            ndarray::Array::<f32, _>::random((n_datapoints, n_features), Uniform::new(0., 1.0));

        let dataset = ManagedTensor::from(&dataset_host).to_device(&res).unwrap();

        println!("dataset {:#?}", dataset_host);

        // build the brute force index
        let index =
            Index::build(&res, metric, None, dataset).expect("failed to create brute force index");

        res.sync_stream().unwrap();

        // use the first 4 points from the dataset as queries : will test that we get them back
        // as their own nearest neighbor
        let n_queries = 4;
        let queries = dataset_host.slice(s![0..n_queries, ..]);

        let k = 4;

        println!("queries! {:#?}", queries);
        let queries = ManagedTensor::from(&queries).to_device(&res).unwrap();
        let mut neighbors_host = ndarray::Array::<i64, _>::zeros((n_queries, k));
        let neighbors = ManagedTensor::from(&neighbors_host)
            .to_device(&res)
            .unwrap();

        let mut distances_host = ndarray::Array::<f32, _>::zeros((n_queries, k));
        let distances = ManagedTensor::from(&distances_host)
            .to_device(&res)
            .unwrap();

        index
            .search(&res, &queries, &neighbors, &distances)
            .unwrap();

        // Copy back to host memory
        distances.to_host(&res, &mut distances_host).unwrap();
        neighbors.to_host(&res, &mut neighbors_host).unwrap();
        res.sync_stream().unwrap();

        println!("distances {:#?}", distances_host);
        println!("neighbors {:#?}", neighbors_host);

        // nearest neighbors should be themselves, since queries are from the
        // dataset
        assert_eq!(neighbors_host[[0, 0]], 0);
        assert_eq!(neighbors_host[[1, 0]], 1);
        assert_eq!(neighbors_host[[2, 0]], 2);
        assert_eq!(neighbors_host[[3, 0]], 3);
    }

    /*
        #[test]
        fn test_cosine() {
            test_bfknn(DistanceType::CosineExpanded);
        }
    */

    #[flaky]
    fn test_l2() {
        test_bfknn(DistanceType::L2Expanded);
    }
}
