/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! cuVS: Rust bindings for Vector Search on the GPU
//!
//! This crate provides Rust bindings for cuVS, allowing you to run
//! approximate nearest neighbors search on the GPU.
pub mod brute_force;
pub mod cagra;
pub mod cluster;
pub mod distance;
pub mod distance_type;
mod dlpack;
mod error;
pub mod filters;
pub mod ivf_flat;
pub mod ivf_pq;
mod resources;
pub mod vamana;

pub use dlpack::ManagedTensor;
pub use error::{Error, Result};
pub use resources::Resources;
