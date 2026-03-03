/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::io::{stderr, Write};
use std::marker::PhantomData;

use crate::dlpack::{DatasetOwnership, ManagedTensor};
use crate::error::{check_cuvs, Result};
use crate::ivf_flat::{IndexParams, SearchParams};
use crate::resources::Resources;

/// Ivf-Flat ANN Index
///
/// The lifetime parameter `'a` ensures the dataset outlives the index when built
/// with [`Index::build`]. Use [`Index::build_owned`] for a self-contained index
/// that owns its dataset (e.g., after [`ManagedTensor::to_device`]).
///
/// # Examples
///
/// ## Borrowed dataset (compiler enforces lifetime)
///
/// ```no_run
/// # use cuvs::{ManagedTensor, Resources};
/// # use cuvs::ivf_flat::{Index, IndexParams};
/// let res = Resources::new().unwrap();
/// let arr = ndarray::Array::<f32, _>::zeros((1024, 16));
/// let params = IndexParams::new().unwrap();
/// let tensor = ManagedTensor::from(&arr);
/// let index = Index::build(&res, &params, &tensor).unwrap();
/// // arr and tensor must remain alive while index is in use
/// ```
///
/// ## Owned dataset ('static lifetime)
///
/// ```no_run
/// # use cuvs::{ManagedTensor, Resources};
/// # use cuvs::ivf_flat::{Index, IndexParams};
/// let res = Resources::new().unwrap();
/// let arr = ndarray::Array::<f32, _>::zeros((1024, 16));
/// let params = IndexParams::new().unwrap();
/// let device_tensor = ManagedTensor::from(&arr).to_device(&res).unwrap();
/// let index = Index::build_owned(&res, &params, device_tensor).unwrap();
/// drop(arr); // Fine — index owns the device copy
/// ```
#[derive(Debug)]
pub struct Index<'a> {
    inner: ffi::cuvsIvfFlatIndex_t,
    _data: DatasetOwnership<'a>,
}

impl<'a> Index<'a> {
    /// Creates a new FFI index handle.
    fn create_handle() -> Result<ffi::cuvsIvfFlatIndex_t> {
        unsafe {
            let mut index = std::mem::MaybeUninit::<ffi::cuvsIvfFlatIndex_t>::uninit();
            check_cuvs(ffi::cuvsIvfFlatIndexCreate(index.as_mut_ptr()))?;
            Ok(index.assume_init())
        }
    }

    /// Builds a new IVF-Flat Index from a borrowed dataset.
    ///
    /// The compiler enforces that `dataset` outlives the returned index,
    /// preventing use-after-free when the C library dereferences its
    /// internal view of the data.
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `params` - Parameters for building the index
    /// * `dataset` - A row-major matrix on either the host or device to index
    pub fn build(
        res: &Resources,
        params: &IndexParams,
        dataset: &'a ManagedTensor,
    ) -> Result<Index<'a>> {
        let inner = Self::create_handle()?;
        unsafe {
            check_cuvs(ffi::cuvsIvfFlatBuild(
                res.0,
                params.0,
                dataset.as_ptr(),
                inner,
            ))?;
        }
        Ok(Index {
            inner,
            _data: DatasetOwnership::Borrowed(PhantomData),
        })
    }

    /// Creates a new empty index
    pub fn new() -> Result<Index<'a>> {
        Ok(Index {
            inner: Self::create_handle()?,
            _data: DatasetOwnership::Borrowed(PhantomData),
        })
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
                self.inner,
                queries.as_ptr(),
                neighbors.as_ptr(),
                distances.as_ptr(),
                prefilter,
            ))
        }
    }
}

impl Index<'static> {
    /// Builds a new IVF-Flat Index from an owned dataset.
    ///
    /// The index takes ownership of `dataset`, making it self-contained
    /// with a `'static` lifetime. This is useful when the dataset is a
    /// device copy (from [`ManagedTensor::to_device`]) that should live
    /// as long as the index.
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `params` - Parameters for building the index
    /// * `dataset` - A row-major matrix to index (ownership transferred to the index)
    pub fn build_owned(
        res: &Resources,
        params: &IndexParams,
        dataset: ManagedTensor,
    ) -> Result<Index<'static>> {
        let inner = Self::create_handle()?;
        unsafe {
            check_cuvs(ffi::cuvsIvfFlatBuild(
                res.0,
                params.0,
                dataset.as_ptr(),
                inner,
            ))?;
        }
        Ok(Index {
            inner,
            _data: DatasetOwnership::Owned(dataset),
        })
    }
}

impl Drop for Index<'_> {
    fn drop(&mut self) {
        if let Err(e) = check_cuvs(unsafe { ffi::cuvsIvfFlatIndexDestroy(self.inner) }) {
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

        // build the ivf-flat index (owned — device copy lives in the index)
        let index = Index::build_owned(&res, &build_params, dataset_device)
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

        // Build the index once (owned)
        let index = Index::build_owned(&res, &build_params, dataset_device)
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
                .unwrap_or_else(|e| panic!("search iteration {} failed: {}", search_iter, e));

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

    /// Test that an index built with build (borrowed) ties the dataset lifetime.
    #[test]
    fn test_ivf_flat_borrowed_build() {
        let res = Resources::new().unwrap();
        let build_params = IndexParams::new().unwrap().set_n_lists(64);

        let n_datapoints = 1024;
        let n_features = 16;
        let dataset =
            ndarray::Array::<f32, _>::random((n_datapoints, n_features), Uniform::new(0., 1.0));

        // Create a device tensor and borrow it for the index
        let dataset_device = ManagedTensor::from(&dataset).to_device(&res).unwrap();
        let index = Index::build(&res, &build_params, &dataset_device)
            .expect("failed to create ivf-flat index");

        // Search while the borrowed dataset is still alive
        let n_queries = 4;
        let k = 5;
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

        let search_params = SearchParams::new().unwrap();

        index
            .search(&res, &search_params, &queries, &neighbors, &distances)
            .unwrap();

        distances.to_host(&res, &mut distances_host).unwrap();
        neighbors.to_host(&res, &mut neighbors_host).unwrap();

        assert_eq!(neighbors_host[[0, 0]], 0);
        assert_eq!(neighbors_host[[1, 0]], 1);
        // dataset_device is still alive here — compiler ensures it
    }
}
