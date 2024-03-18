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

use std::fmt;

#[derive(Debug, Clone)]
pub struct CuvsError {
    code: ffi::cuvsError_t,
    text: String,
}

#[derive(Debug, Clone)]
pub enum Error {
    CudaError(ffi::cudaError_t),
    CuvsError(CuvsError),
}

impl std::error::Error for Error {}
impl std::error::Error for CuvsError {}

pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::CudaError(cuda_error) => write!(f, "cudaError={:?}", cuda_error),
            Error::CuvsError(cuvs_error) => write!(f, "cuvsError={:?}", cuvs_error),
        }
    }
}

impl fmt::Display for CuvsError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}:{}", self.code, self.text)
    }
}

/// Simple wrapper to convert a cuvsError_t into a Result
pub fn check_cuvs(err: ffi::cuvsError_t) -> Result<()> {
    match err {
        ffi::cuvsError_t::CUVS_SUCCESS => Ok(()),
        _ => {
            // get a description of the error from cuvs
            let cstr = unsafe {
                let text_ptr = ffi::cuvsGetLastErrorText();
                std::ffi::CStr::from_ptr(text_ptr)
            };
            let text = std::string::String::from_utf8_lossy(cstr.to_bytes()).to_string();

            Err(Error::CuvsError(CuvsError { code: err, text }))
        }
    }
}

pub fn check_cuda(err: ffi::cudaError_t) -> Result<()> {
    match err {
        ffi::cudaError::cudaSuccess => Ok(()),
        _ => Err(Error::CudaError(err)),
    }
}
