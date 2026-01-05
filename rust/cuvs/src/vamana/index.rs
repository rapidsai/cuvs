/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::ffi::{CStr, CString};
use std::io::{stderr, Write};

use crate::dlpack::ManagedTensor;
use crate::error::{check_cuvs, Result};
use crate::resources::Resources;
use crate::vamana::IndexParams;

/// Vamana ANN Index
#[derive(Debug)]
pub struct Index(ffi::cuvsVamanaIndex_t);

impl Index {
    /// Builds Vamana Index for efficient DiskANN search
    ///
    /// The build uses the Vamana insertion-based algorithm to create the graph. The algorithm
    /// starts with an empty graph and iteratively inserts batches of nodes. Each batch involves
    /// performing a greedy search for each vector to be inserted, and inserting it with edges to
    /// all nodes traversed during the search. Reverse edges are also inserted and robustPrune is applied
    /// to improve graph quality. The index_params struct controls the degree of the final graph.
    ///
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
            check_cuvs(ffi::cuvsVamanaBuild(
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
            let mut index = std::mem::MaybeUninit::<ffi::cuvsVamanaIndex_t>::uninit();
            check_cuvs(ffi::cuvsVamanaIndexCreate(index.as_mut_ptr()))?;
            Ok(Index(index.assume_init()))
        }
    }

    /// Save Vamana index to file
    ///
    /// Matches the file format used by the DiskANN open-source repository, allowing cross-compatibility.
    ///
    /// Serialized Index is to be used by the DiskANN open-source repository for graph search.
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `filename` - The file prefix for where the index is sazved
    /// * `include_dataset` - whether to include the dataset in the serialized index
    pub fn serialize(self, res: &Resources, filename: &str, include_dataset: bool) -> Result<()> {
        let c_filename = CString::new(filename).unwrap();
        unsafe {
            check_cuvs(ffi::cuvsVamanaSerialize(
                res.0,
                c_filename.as_ptr(),
                self.0,
                include_dataset,
            ))
        }
    }
}

impl Drop for Index {
    fn drop(&mut self) {
        if let Err(e) = check_cuvs(unsafe { ffi::cuvsVamanaIndexDestroy(self.0) }) {
            write!(stderr(), "failed to call cuvsVamanaIndexDestroy {:?}", e)
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
    fn test_vamana() {
        let build_params = IndexParams::new().unwrap();

        let res = Resources::new().unwrap();

        // Create a new random dataset to index
        let n_datapoints = 1024;
        let n_features = 16;
        let dataset =
            ndarray::Array::<f32, _>::random((n_datapoints, n_features), Uniform::new(0., 1.0));

        let dataset_device = ManagedTensor::from(&dataset).to_device(&res).unwrap();

        // build the vamana index
        let index = Index::build(&res, &build_params, dataset_device)
            .expect("failed to create vamana index");
    }
}
