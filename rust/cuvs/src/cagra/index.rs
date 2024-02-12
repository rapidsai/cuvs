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

use std::io::{stderr, Write};

use crate::cagra::{IndexParams, SearchParams};
use crate::dlpack::ManagedTensor;
use crate::error::{check_cuvs, Result};
use crate::resources::Resources;

#[derive(Debug)]
pub struct Index {
    index: ffi::cagraIndex_t,
}

impl Index {
    /// Builds a new index
    pub fn build(res: &Resources, params: &IndexParams, dataset: ManagedTensor) -> Result<Index> {
        let index = Index::new()?;
        unsafe {
            check_cuvs(ffi::cagraBuild(
                res.0,
                params.0,
                dataset.as_ptr(),
                index.index,
            ))?;
        }
        Ok(index)
    }

    /// Creates a new empty index
    pub fn new() -> Result<Index> {
        unsafe {
            let mut index = core::mem::MaybeUninit::<ffi::cagraIndex_t>::uninit();
            check_cuvs(ffi::cagraIndexCreate(index.as_mut_ptr()))?;
            Ok(Index {
                index: index.assume_init(),
            })
        }
    }

    pub fn search(
        self,
        res: &Resources,
        params: &SearchParams,
        queries: ManagedTensor,
        neighbors: ManagedTensor,
        distances: ManagedTensor,
    ) -> Result<()> {
        unsafe {
            check_cuvs(ffi::cagraSearch(
                res.0,
                params.0,
                self.index,
                queries.as_ptr(),
                neighbors.as_ptr(),
                distances.as_ptr(),
            ))
        }
    }
}

impl Drop for Index {
    fn drop(&mut self) {
        if let Err(e) = check_cuvs(unsafe { ffi::cagraIndexDestroy(self.index) }) {
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

    #[test]
    fn test_create_empty_index() {
        Index::new().unwrap();
    }

    #[test]
    fn test_index() {
        let res = Resources::new().unwrap();
        let params = IndexParams::new().unwrap();

        let n_features = 16;
        let dataset = ndarray::Array::<f32, _>::random((256, n_features), Uniform::new(0., 1.0));
        let index = Index::build(&res, &params, ManagedTensor::from_ndarray(&dataset))
            .expect("failed to create cagra index");

        // use the first 4 points from the dataset as queries : will test that we get them back
        // as their own nearest neighbor
        let n_queries = 4;
        let queries = dataset.slice(s![0..n_queries, ..]);
        let queries = ManagedTensor::from_ndarray(&queries).to_device().unwrap();

        let k = 10;
        let neighbors =
            ManagedTensor::from_ndarray(&ndarray::Array::<u32, _>::zeros((n_queries, k)))
                .to_device()
                .unwrap();
        let distances =
            ManagedTensor::from_ndarray(&ndarray::Array::<f32, _>::zeros((n_queries, k)))
                .to_device()
                .unwrap();

        let search_params = SearchParams::new().unwrap();

        index
            .search(&res, &search_params, queries, neighbors, distances)
            .unwrap();
    }
}
