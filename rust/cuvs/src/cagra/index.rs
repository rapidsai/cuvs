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

use crate::cagra::IndexParams;
use crate::dlpack::ManagedTensor;
use crate::error::{check_cuvs, Result};
use crate::resources::Resources;

#[derive(Debug)]
pub struct Index {
    index: ffi::cagraIndex_t,
}

impl Index {
    /// Builds a new index
    pub fn build(res: Resources, params: IndexParams, dataset: ManagedTensor) -> Result<Index> {
        let index = Index::new()?;
        unsafe {
            check_cuvs(ffi::cagraBuild(
                res.res,
                params.params,
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

    #[test]
    fn test_create_empty_index() {
        Index::new().unwrap();
    }

    #[test]
    fn test_build() {
        let res = Resources::new().unwrap();
        let params = IndexParams::new().unwrap();

        // TODO: test a more exciting dataset
        let arr = ndarray::Array::<f32, _>::zeros((128, 16));
        let dataset = ManagedTensor::from_ndarray(arr);

        let index = Index::build(res, params, dataset).expect("failed to create cagra index");
    }
}
