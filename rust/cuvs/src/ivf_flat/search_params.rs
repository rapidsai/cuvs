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

/// Supplemental parameters to search IvfFlat index
pub struct SearchParams(pub ffi::cuvsIvfFlatSearchParams_t);

impl SearchParams {
    /// Returns a new SearchParams object
    pub fn new() -> Result<SearchParams> {
        unsafe {
            let mut params = std::mem::MaybeUninit::<ffi::cuvsIvfFlatSearchParams_t>::uninit();
            check_cuvs(ffi::cuvsIvfFlatSearchParamsCreate(params.as_mut_ptr()))?;
            Ok(SearchParams(params.assume_init()))
        }
    }

    /// Supplemental parameters to search IVF-Flat index
    pub fn set_n_probes(self, n_probes: u32) -> SearchParams {
        unsafe {
            (*self.0).n_probes = n_probes;
        }
        self
    }
}

impl fmt::Debug for SearchParams {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // custom debug trait here, default value will show the pointer address
        // for the inner params object which isn't that useful.
        write!(f, "SearchParams {{ params: {:?} }}", unsafe { *self.0 })
    }
}

impl Drop for SearchParams {
    fn drop(&mut self) {
        if let Err(e) = check_cuvs(unsafe { ffi::cuvsIvfFlatSearchParamsDestroy(self.0) }) {
            write!(
                stderr(),
                "failed to call cuvsIvfFlatSearchParamsDestroy {:?}",
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
    fn test_search_params() {
        let params = SearchParams::new().unwrap().set_n_probes(128);

        unsafe {
            assert_eq!((*params.0).n_probes, 128);
        }
    }
}
