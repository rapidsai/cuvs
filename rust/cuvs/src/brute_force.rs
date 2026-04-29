/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Brute Force KNN

use std::io::{stderr, Write};
use std::marker::PhantomData;

use crate::distance_type::DistanceType;
use crate::dlpack::ManagedTensor;
use crate::error::{check_cuvs, Result};
use crate::resources::Resources;

/// Brute Force KNN Index.
///
/// The brute force C++ implementation always stores a non-owning view of the
/// dataset, so the index cannot outlive the `ManagedTensor` it was built
/// from. The lifetime parameter `'a` ties the index to that tensor, so the
/// borrow checker catches use-after-free at compile time.
///
/// # Example
///
/// ```no_run
/// # use cuvs::{ManagedTensor, Resources};
/// # use cuvs::brute_force::Index;
/// # use cuvs::distance_type::DistanceType;
/// let res = Resources::new().unwrap();
/// let arr = ndarray::Array::<f32, _>::zeros((64, 8));
/// let tensor = ManagedTensor::from(&arr).to_device(&res).unwrap();
/// let index = Index::build(&res, DistanceType::L2Expanded, None, &tensor).unwrap();
/// // `tensor` must outlive `index`.
/// ```
#[derive(Debug)]
pub struct Index<'a> {
    inner: ffi::cuvsBruteForceIndex_t,
    _dataset: PhantomData<&'a ()>,
}

impl<'a> Index<'a> {
    /// Creates a new empty FFI handle.
    fn create_handle() -> Result<ffi::cuvsBruteForceIndex_t> {
        unsafe {
            let mut index = std::mem::MaybeUninit::<ffi::cuvsBruteForceIndex_t>::uninit();
            check_cuvs(ffi::cuvsBruteForceIndexCreate(index.as_mut_ptr()))?;
            Ok(index.assume_init())
        }
    }

    /// Builds a new Brute Force KNN Index from `dataset`.
    ///
    /// The compiler enforces that `dataset` outlives the returned index,
    /// because the C++ brute force implementation stores a non-owning view
    /// of the input and would otherwise dangle.
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `metric` - DistanceType to use for building the index
    /// * `metric_arg` - Optional value of `p` for Minkowski distances
    /// * `dataset` - A row-major matrix on either the host or device to index
    pub fn build(
        res: &Resources,
        metric: DistanceType,
        metric_arg: Option<f32>,
        dataset: &'a ManagedTensor,
    ) -> Result<Index<'a>> {
        let inner = Self::create_handle()?;
        unsafe {
            check_cuvs(ffi::cuvsBruteForceBuild(
                res.0,
                dataset.as_ptr(),
                metric,
                metric_arg.unwrap_or(2.0),
                inner,
            ))?;
        }
        Ok(Index {
            inner,
            _dataset: PhantomData,
        })
    }

    /// Perform a Nearest Neighbors search on the Index.
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `queries` - A matrix in device memory to query for
    /// * `neighbors` - Matrix in device memory that receives the indices of the nearest neighbors
    /// * `distances` - Matrix in device memory that receives the distances of the nearest neighbors
    pub fn search(
        &self,
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
                self.inner,
                queries.as_ptr(),
                neighbors.as_ptr(),
                distances.as_ptr(),
                prefilter,
            ))
        }
    }
}

impl Drop for Index<'_> {
    fn drop(&mut self) {
        if let Err(e) = check_cuvs(unsafe { ffi::cuvsBruteForceIndexDestroy(self.inner) }) {
            write!(stderr(), "failed to call bruteForceIndexDestroy {:?}", e)
                .expect("failed to write to stderr");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mark_flaky_tests::flaky;
    use ndarray::{s, Array2};
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    /// Build a small random host dataset for the brute-force tests.
    fn make_dataset(n_datapoints: usize, n_features: usize) -> Array2<f32> {
        ndarray::Array::<f32, _>::random((n_datapoints, n_features), Uniform::new(0., 1.0))
    }

    /// Search `index` for the first `n_queries` rows of `dataset_host` and
    /// assert each query is its own nearest neighbor.
    fn search_and_verify_self_neighbors(
        res: &Resources,
        index: &Index<'_>,
        dataset_host: &Array2<f32>,
        n_queries: usize,
        k: usize,
    ) {
        let queries = dataset_host.slice(s![0..n_queries, ..]);
        let queries = ManagedTensor::from(&queries).to_device(res).unwrap();

        let mut neighbors_host = ndarray::Array::<i64, _>::zeros((n_queries, k));
        let neighbors = ManagedTensor::from(&neighbors_host).to_device(res).unwrap();

        let mut distances_host = ndarray::Array::<f32, _>::zeros((n_queries, k));
        let distances = ManagedTensor::from(&distances_host).to_device(res).unwrap();

        index
            .search(res, &queries, &neighbors, &distances)
            .expect("search failed");

        distances.to_host(res, &mut distances_host).unwrap();
        neighbors.to_host(res, &mut neighbors_host).unwrap();
        res.sync_stream().unwrap();

        for i in 0..n_queries {
            assert_eq!(
                neighbors_host[[i, 0]],
                i as i64,
                "query {i} should be its own nearest neighbor"
            );
        }
    }

    fn test_bfknn(metric: DistanceType) {
        let res = Resources::new().unwrap();
        let dataset_host = make_dataset(16, 8);
        let dataset = ManagedTensor::from(&dataset_host).to_device(&res).unwrap();

        let index =
            Index::build(&res, metric, None, &dataset).expect("failed to build brute force index");
        res.sync_stream().unwrap();

        search_and_verify_self_neighbors(&res, &index, &dataset_host, 4, 4);
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

    /// Validates that a borrowed-dataset brute-force index can be searched
    /// multiple times while the dataset is kept alive by the caller.
    #[test]
    fn test_brute_force_multiple_searches() {
        let res = Resources::new().unwrap();
        let dataset_host = make_dataset(64, 8);
        let dataset_device = ManagedTensor::from(&dataset_host).to_device(&res).unwrap();

        let index = Index::build(&res, DistanceType::L2Expanded, None, &dataset_device)
            .expect("failed to build brute force index");
        res.sync_stream().unwrap();

        for _ in 0..3 {
            search_and_verify_self_neighbors(&res, &index, &dataset_host, 4, 4);
        }
        // `dataset_device` is still alive here — the borrow checker enforces it.
    }
}
