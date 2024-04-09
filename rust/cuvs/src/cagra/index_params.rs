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

use crate::error::{check_cuvs, Result};
use std::fmt;
use std::io::{stderr, Write};

pub type BuildAlgo = ffi::cuvsCagraGraphBuildAlgo;

/// Supplemental parameters to build CAGRA Index
pub struct IndexParams(pub ffi::cuvsCagraIndexParams_t);

impl IndexParams {
    /// Returns a new IndexParams
    pub fn new() -> Result<IndexParams> {
        unsafe {
            let mut params = std::mem::MaybeUninit::<ffi::cuvsCagraIndexParams_t>::uninit();
            check_cuvs(ffi::cuvsCagraIndexParamsCreate(params.as_mut_ptr()))?;
            Ok(IndexParams(params.assume_init()))
        }
    }

    /// Degree of input graph for pruning
    pub fn set_intermediate_graph_degree(self, intermediate_graph_degree: usize) -> IndexParams {
        unsafe {
            (*self.0).intermediate_graph_degree = intermediate_graph_degree;
        }
        self
    }

    /// Degree of output graph
    pub fn set_graph_degree(self, graph_degree: usize) -> IndexParams {
        unsafe {
            (*self.0).graph_degree = graph_degree;
        }
        self
    }

    /// ANN algorithm to build knn graph
    pub fn set_build_algo(self, build_algo: BuildAlgo) -> IndexParams {
        unsafe {
            (*self.0).build_algo = build_algo;
        }
        self
    }

    /// Number of iterations to run if building with NN_DESCENT
    pub fn set_nn_descent_niter(self, nn_descent_niter: usize) -> IndexParams {
        unsafe {
            (*self.0).nn_descent_niter = nn_descent_niter;
        }
        self
    }
}

impl fmt::Debug for IndexParams {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // custom debug trait here, default value will show the pointer address
        // for the inner params object which isn't that useful.
        write!(f, "IndexParams({:?})", unsafe { *self.0 })
    }
}

impl Drop for IndexParams {
    fn drop(&mut self) {
        if let Err(e) = check_cuvs(unsafe { ffi::cuvsCagraIndexParamsDestroy(self.0) }) {
            write!(
                stderr(),
                "failed to call cuvsCagraIndexParamsDestroy {:?}",
                e
            )
            .expect("failed to write to stderr");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_params() {
        let params = IndexParams::new()
            .unwrap()
            .set_intermediate_graph_degree(128)
            .set_graph_degree(16)
            .set_build_algo(BuildAlgo::NN_DESCENT)
            .set_nn_descent_niter(10);

        // make sure the setters actually updated internal representation on the c-struct
        unsafe {
            assert_eq!((*params.0).graph_degree, 16);
            assert_eq!((*params.0).intermediate_graph_degree, 128);
            assert_eq!((*params.0).build_algo, BuildAlgo::NN_DESCENT);
            assert_eq!((*params.0).nn_descent_niter, 10);
        }
    }
}
