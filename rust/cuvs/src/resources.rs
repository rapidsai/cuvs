/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::error::{check_cuvs, Result};
use std::io::{stderr, Write};

/// Resources are objects that are shared between function calls,
/// and includes things like CUDA streams, cuBLAS handles and other
/// resources that are expensive to create.
#[derive(Debug)]
pub struct Resources(pub ffi::cuvsResources_t);

impl Resources {
    /// Returns a new Resources object
    pub fn new() -> Result<Resources> {
        let mut res: ffi::cuvsResources_t = 0;
        unsafe {
            check_cuvs(ffi::cuvsResourcesCreate(&mut res))?;
        }
        Ok(Resources(res))
    }

    /// Sets the current cuda stream
    pub fn set_cuda_stream(&self, stream: ffi::cudaStream_t) -> Result<()> {
        unsafe { check_cuvs(ffi::cuvsStreamSet(self.0, stream)) }
    }

    /// Gets the current cuda stream
    pub fn get_cuda_stream(&self) -> Result<ffi::cudaStream_t> {
        unsafe {
            let mut stream = std::mem::MaybeUninit::<ffi::cudaStream_t>::uninit();
            check_cuvs(ffi::cuvsStreamGet(self.0, stream.as_mut_ptr()))?;
            Ok(stream.assume_init())
        }
    }

    /// Syncs the current cuda stream
    pub fn sync_stream(&self) -> Result<()> {
        unsafe { check_cuvs(ffi::cuvsStreamSync(self.0)) }
    }
}

impl Drop for Resources {
    fn drop(&mut self) {
        unsafe {
            if let Err(e) = check_cuvs(ffi::cuvsResourcesDestroy(self.0)) {
                write!(stderr(), "failed to call cuvsResourcesDestroy {:?}", e)
                    .expect("failed to write to stderr");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resources_create() {
        let _ = Resources::new();
    }
}
