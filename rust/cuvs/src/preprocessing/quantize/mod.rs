/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Dataset quantizers.
//!
//! Quantizers compress a floating-point dataset into a more compact
//! representation. The [`scalar`] quantizer maps an interval of the input
//! float range onto the full range of an 8-bit integer.
//!
//! The binary and product (PQ) quantizers exposed by the cuVS C API are not
//! yet wrapped in Rust; they are intended to be added in follow-up
//! contributions.
//!
//! Example:
//! ```
//! use cuvs::preprocessing::quantize::scalar::{Quantizer, ScalarQuantizerParams};
//! use cuvs::{ManagedTensor, Resources, Result};
//!
//! use ndarray_rand::rand_distr::Uniform;
//! use ndarray_rand::RandomExt;
//!
//! fn scalar_quantize_example() -> Result<()> {
//!     let res = Resources::new()?;
//!
//!     // Create a new random dataset to quantize
//!     let n_rows = 1024;
//!     let n_cols = 16;
//!     let dataset =
//!         ndarray::Array::<f32, _>::random((n_rows, n_cols), Uniform::new(0., 1.0));
//!     let dataset_device = ManagedTensor::from(&dataset).to_device(&res)?;
//!
//!     // Train a scalar quantizer on the dataset
//!     let params = ScalarQuantizerParams::new()?;
//!     let quantizer = Quantizer::train(&res, &params, &dataset_device)?;
//!
//!     // Quantize the dataset into int8
//!     let mut quantized_host = ndarray::Array::<i8, _>::zeros((n_rows, n_cols));
//!     let quantized = ManagedTensor::from(&quantized_host).to_device(&res)?;
//!     quantizer.transform(&res, &dataset_device, &quantized)?;
//!     quantized.to_host(&res, &mut quantized_host)?;
//!
//!     // Reconstruct an approximation of the original f32 dataset
//!     let mut reconstructed_host = ndarray::Array::<f32, _>::zeros((n_rows, n_cols));
//!     let reconstructed = ManagedTensor::from(&reconstructed_host).to_device(&res)?;
//!     quantizer.inverse_transform(&res, &quantized, &reconstructed)?;
//!     reconstructed.to_host(&res, &mut reconstructed_host)?;
//!
//!     Ok(())
//! }
//! ```

pub mod scalar;
