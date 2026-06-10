/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::io::{Write, stderr};

use crate::cagra::SearchParams;
use crate::dlpack::ManagedTensor;
use crate::error::{Result, check_cuvs};
use crate::resources::Resources;
use crate::tiered_index::IndexParams;

/// Tiered ANN Index.
///
/// A tiered index couples a brute-force tier that absorbs incremental inserts
/// with an ANN tier (CAGRA by default). Vectors added via [`Index::extend`]
/// land in the brute-force tier and are immediately searchable, even before the
/// ANN tier has been (re)built — this is the defining behavior of the tiered
/// index.
///
/// The C API offers no serialize/deserialize for the tiered index, so this
/// wrapper does not expose persistence (and therefore takes no filesystem
/// paths).
#[derive(Debug)]
pub struct Index(ffi::cuvsTieredIndex_t);

impl Index {
    /// Builds a new tiered Index from the dataset for efficient search.
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `params` - Parameters for building the index (backend + ANN params)
    /// * `dataset` - A row-major matrix on either the host or device to index
    pub fn build<T: Into<ManagedTensor>>(
        res: &Resources,
        params: &IndexParams,
        dataset: T,
    ) -> Result<Index> {
        let dataset: ManagedTensor = dataset.into();
        let index = Index::new()?;
        unsafe {
            check_cuvs(ffi::cuvsTieredIndexBuild(
                res.0,
                params.as_ptr(),
                dataset.as_ptr(),
                index.0,
            ))?;
        }
        Ok(index)
    }

    /// Creates a new empty index handle.
    pub fn new() -> Result<Index> {
        unsafe {
            let mut index = std::mem::MaybeUninit::<ffi::cuvsTieredIndex_t>::uninit();
            check_cuvs(ffi::cuvsTieredIndexCreate(index.as_mut_ptr()))?;
            Ok(Index(index.assume_init()))
        }
    }

    /// Extends the index with new vectors.
    ///
    /// The new vectors are added to the brute-force tier and become immediately
    /// searchable. If `create_ann_index_on_extend` was set and the incremental
    /// tier now exceeds `min_ann_rows`, the ANN tier is (re)built.
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `new_vectors` - A row-major matrix on either the host or device to add
    pub fn extend<T: Into<ManagedTensor>>(&self, res: &Resources, new_vectors: T) -> Result<()> {
        let new_vectors: ManagedTensor = new_vectors.into();
        unsafe { check_cuvs(ffi::cuvsTieredIndexExtend(res.0, new_vectors.as_ptr(), self.0)) }
    }

    /// Performs an Approximate Nearest Neighbors search on the Index.
    ///
    /// `params` must match the ANN backend the index was built with; for the
    /// default CAGRA backend these are [`crate::cagra::SearchParams`].
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `params` - Search parameters for the ANN backend
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
        let no_filter = ffi::cuvsFilter { addr: 0, type_: ffi::cuvsFilterType::NO_FILTER };
        self.search_with_filter(res, params, queries, neighbors, distances, no_filter)
    }

    /// Performs an Approximate Nearest Neighbors search with a prefilter.
    ///
    /// The prefilter is a [`ffi::cuvsFilter`] holding the address of a bitset or
    /// bitmap tensor (and its [`ffi::cuvsFilterType`]) used to exclude vectors
    /// from the result set. Use [`Index::search`] for an unfiltered search.
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `params` - Search parameters for the ANN backend
    /// * `queries` - A matrix in device memory to query for
    /// * `neighbors` - Matrix in device memory that receives the indices of the nearest neighbors
    /// * `distances` - Matrix in device memory that receives the distances of the nearest neighbors
    /// * `prefilter` - A [`ffi::cuvsFilter`] describing the bitset/bitmap to apply
    pub fn search_with_filter(
        &self,
        res: &Resources,
        params: &SearchParams,
        queries: &ManagedTensor,
        neighbors: &ManagedTensor,
        distances: &ManagedTensor,
        prefilter: ffi::cuvsFilter,
    ) -> Result<()> {
        unsafe {
            check_cuvs(ffi::cuvsTieredIndexSearch(
                res.0,
                // The C API takes the backend's search params as an opaque
                // void*; CAGRA is the default backend.
                params.0 as *mut std::os::raw::c_void,
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
        if let Err(e) = check_cuvs(unsafe { ffi::cuvsTieredIndexDestroy(self.0) }) {
            write!(stderr(), "failed to call cuvsTieredIndexDestroy {:?}", e)
                .expect("failed to write to stderr");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tiered_index::AnnAlgo;
    use ndarray::s;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;

    fn default_params() -> IndexParams {
        // Keep min_ann_rows low so the ANN tier (CAGRA) is exercised by the
        // build, while leaving room for extend to land in the brute-force tier.
        IndexParams::new()
            .unwrap()
            .set_algo(AnnAlgo::CUVS_TIERED_INDEX_ALGO_CAGRA)
            .set_min_ann_rows(128)
            .set_create_ann_index_on_extend(true)
    }

    /// (a) Build with an initial dataset and confirm each query finds itself.
    #[test]
    fn test_tiered_build_and_search() {
        let res = Resources::new().unwrap();

        let n_datapoints = 1024;
        let n_features = 16;
        let dataset =
            ndarray::Array::<f32, _>::random((n_datapoints, n_features), Uniform::new(0., 1.0));
        let dataset_device = ManagedTensor::from(&dataset).to_device(&res).unwrap();

        let index = Index::build(&res, &default_params(), dataset_device)
            .expect("failed to build tiered index");

        let n_queries = 4;
        let queries = dataset.slice(s![0..n_queries, ..]);
        let queries = ManagedTensor::from(&queries).to_device(&res).unwrap();

        let k = 10;
        let mut neighbors_host = ndarray::Array::<i64, _>::zeros((n_queries, k));
        let neighbors = ManagedTensor::from(&neighbors_host).to_device(&res).unwrap();
        let distances_host = ndarray::Array::<f32, _>::zeros((n_queries, k));
        let distances = ManagedTensor::from(&distances_host).to_device(&res).unwrap();

        let search_params = SearchParams::new().unwrap();
        index.search(&res, &search_params, &queries, &neighbors, &distances).unwrap();

        neighbors.to_host(&res, &mut neighbors_host).unwrap();
        for i in 0..n_queries {
            assert_eq!(neighbors_host[[i, 0]], i as i64, "query {i} should find itself");
        }
    }

    /// (b) THE KEY TEST: vectors added via extend after build must be
    /// immediately findable — the entire point of the tiered index.
    #[test]
    fn test_tiered_extend_visibility() {
        let res = Resources::new().unwrap();

        let n_features = 16;
        let n_initial = 512;
        let dataset =
            ndarray::Array::<f32, _>::random((n_initial, n_features), Uniform::new(0., 1.0));
        let dataset_device = ManagedTensor::from(&dataset).to_device(&res).unwrap();

        let index = Index::build(&res, &default_params(), dataset_device)
            .expect("failed to build tiered index");

        // New vectors NOT in the original dataset.
        let n_new = 8;
        let new_vectors =
            ndarray::Array::<f32, _>::random((n_new, n_features), Uniform::new(10., 11.0));
        let new_device = ManagedTensor::from(&new_vectors).to_device(&res).unwrap();
        index.extend(&res, new_device).expect("extend failed");

        // Query with the new vectors themselves. Their ids are appended after
        // the initial dataset, so neighbor[i][0] should be n_initial + i.
        let queries = ManagedTensor::from(&new_vectors).to_device(&res).unwrap();
        let k = 5;
        let mut neighbors_host = ndarray::Array::<i64, _>::zeros((n_new, k));
        let neighbors = ManagedTensor::from(&neighbors_host).to_device(&res).unwrap();
        let mut distances_host = ndarray::Array::<f32, _>::zeros((n_new, k));
        let distances = ManagedTensor::from(&distances_host).to_device(&res).unwrap();

        let search_params = SearchParams::new().unwrap();
        index.search(&res, &search_params, &queries, &neighbors, &distances).unwrap();

        neighbors.to_host(&res, &mut neighbors_host).unwrap();
        distances.to_host(&res, &mut distances_host).unwrap();
        for i in 0..n_new {
            assert_eq!(
                neighbors_host[[i, 0]],
                (n_initial + i) as i64,
                "extended vector {i} must be immediately findable as its own nearest neighbor"
            );
            // Self-distance should be ~zero up to float32 rounding.
            assert!(
                distances_host[[i, 0]] < 1e-2,
                "extended vector {i} self-distance {} should be ~0",
                distances_host[[i, 0]]
            );
        }
    }

    /// (c) Repeated extends each remain searchable.
    #[test]
    fn test_tiered_repeated_extends() {
        let res = Resources::new().unwrap();

        let n_features = 16;
        let n_initial = 512;
        let dataset =
            ndarray::Array::<f32, _>::random((n_initial, n_features), Uniform::new(0., 1.0));
        let dataset_device = ManagedTensor::from(&dataset).to_device(&res).unwrap();

        let index = Index::build(&res, &default_params(), dataset_device)
            .expect("failed to build tiered index");

        let search_params = SearchParams::new().unwrap();
        let k = 8;
        let n_batch = 4;
        let mut total = n_initial;

        for round in 0..3 {
            // Use a distinct value range per round so each batch is far from the
            // [0,1] base cluster and from the other rounds.
            let lo = 20.0 + round as f32 * 10.0;
            let new_vectors =
                ndarray::Array::<f32, _>::random((n_batch, n_features), Uniform::new(lo, lo + 1.0));
            let new_device = ManagedTensor::from(&new_vectors).to_device(&res).unwrap();
            index.extend(&res, new_device).expect("extend failed");

            let queries = ManagedTensor::from(&new_vectors).to_device(&res).unwrap();
            let mut neighbors_host = ndarray::Array::<i64, _>::zeros((n_batch, k));
            let neighbors = ManagedTensor::from(&neighbors_host).to_device(&res).unwrap();
            let mut distances_host = ndarray::Array::<f32, _>::zeros((n_batch, k));
            let distances = ManagedTensor::from(&distances_host).to_device(&res).unwrap();

            index.search(&res, &search_params, &queries, &neighbors, &distances).unwrap();
            neighbors.to_host(&res, &mut neighbors_host).unwrap();
            distances.to_host(&res, &mut distances_host).unwrap();
            // Each just-extended vector must be immediately findable: its own id
            // appears in the top-k with a near-zero self-distance. We assert
            // top-k membership rather than exact rank-0 to stay robust against
            // the ANN tier's approximate recall after an on-extend rebuild.
            for i in 0..n_batch {
                let want = (total + i) as i64;
                let pos = (0..k).find(|&j| neighbors_host[[i, j]] == want);
                let pos = pos.unwrap_or_else(|| {
                    panic!(
                        "round {round}: extended vector {i} (id {want}) not found in top-{k}: {:?}",
                        neighbors_host.row(i)
                    )
                });
                // Self-distance is ~0 up to float32 rounding (a handful of
                // ULPs across the feature dimension); real neighbors in other
                // value ranges are orders of magnitude farther.
                assert!(
                    distances_host[[i, pos]] < 1e-2,
                    "round {round}: extended vector {i} self-distance {} should be ~0",
                    distances_host[[i, pos]]
                );
            }
            total += n_batch;
        }
    }

    /// (d) Filtered search: a bitset prefilter that excludes a query's own id
    /// must keep that id out of the result set.
    #[test]
    fn test_tiered_filtered_search() {
        let res = Resources::new().unwrap();

        let n_features = 16;
        let n_datapoints = 1024;
        let dataset =
            ndarray::Array::<f32, _>::random((n_datapoints, n_features), Uniform::new(0., 1.0));
        let dataset_device = ManagedTensor::from(&dataset).to_device(&res).unwrap();

        let index = Index::build(&res, &default_params(), dataset_device)
            .expect("failed to build tiered index");

        // Build a bitset over n_datapoints with all bits set (1 = keep), then
        // clear bit 0 so query 0 cannot return itself. cuvs bitsets are u32
        // words, LSB-first.
        let n_words = n_datapoints.div_ceil(32);
        let mut bitset_host = ndarray::Array::<u32, _>::from_elem(n_words, u32::MAX);
        bitset_host[0] &= !1u32; // clear bit 0 -> exclude id 0
        let bitset = ManagedTensor::from(&bitset_host).to_device(&res).unwrap();
        let prefilter =
            ffi::cuvsFilter { addr: bitset.as_ptr() as usize, type_: ffi::cuvsFilterType::BITSET };

        let n_queries = 1;
        let queries = dataset.slice(s![0..n_queries, ..]);
        let queries = ManagedTensor::from(&queries).to_device(&res).unwrap();
        let k = 5;
        let mut neighbors_host = ndarray::Array::<i64, _>::zeros((n_queries, k));
        let neighbors = ManagedTensor::from(&neighbors_host).to_device(&res).unwrap();
        let distances_host = ndarray::Array::<f32, _>::zeros((n_queries, k));
        let distances = ManagedTensor::from(&distances_host).to_device(&res).unwrap();

        let search_params = SearchParams::new().unwrap();
        index
            .search_with_filter(&res, &search_params, &queries, &neighbors, &distances, prefilter)
            .unwrap();

        neighbors.to_host(&res, &mut neighbors_host).unwrap();
        // id 0 was filtered out, so it must not appear among the neighbors.
        for j in 0..k {
            assert_ne!(neighbors_host[[0, j]], 0, "filtered id 0 must not be returned");
        }
    }
}
