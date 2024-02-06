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
pub struct Error {
    err: ffi::cuvsError_t,
}

pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "cuvsError={:?}", self.err)
    }
}

/// Simple wrapper to convert a cuvsError_t into a Result
pub fn check_cuvs(err: ffi::cuvsError_t) -> Result<()> {
    match err {
        ffi::cuvsError_t::CUVS_SUCCESS => Ok(()),
        _ => Err(Error { err }),
    }
}
