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

//! cuVS: Rust bindings for Vector Search on the GPU
//!
//! This crate provides Rust bindings for cuVS, allowing you to run
//! approximate nearest neighbors search on the GPU.
pub mod brute_force;
pub mod cagra;
pub mod distance_type;
mod dlpack;
mod error;
mod resources;

pub use dlpack::ManagedTensor;
pub use error::{Error, Result};
pub use resources::Resources;
