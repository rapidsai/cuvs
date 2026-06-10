/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Scalar quantizer.
//!
//! The scalar quantizer performs a linear mapping of an interval of the input
//! float range onto the full range of an 8-bit integer. The interval is
//! derived during [`Quantizer::train`] from the dataset, optionally clipping a
//! configurable fraction of outliers (see
//! [`ScalarQuantizerParams::set_quantile`]).

use std::fmt;
use std::io::{Write, stderr};

use crate::dlpack::ManagedTensor;
use crate::error::{Result, check_cuvs};
use crate::resources::Resources;

/// Parameters controlling how a [`Quantizer`] is trained.
pub struct ScalarQuantizerParams(pub ffi::cuvsScalarQuantizerParams_t);

impl ScalarQuantizerParams {
    /// Returns a new `ScalarQuantizerParams` populated with default values.
    pub fn new() -> Result<ScalarQuantizerParams> {
        unsafe {
            let mut params = std::mem::MaybeUninit::<ffi::cuvsScalarQuantizerParams_t>::uninit();
            check_cuvs(ffi::cuvsScalarQuantizerParamsCreate(params.as_mut_ptr()))?;
            Ok(ScalarQuantizerParams(params.assume_init()))
        }
    }

    /// Sets the fraction of the data that is kept once outliers at the top and
    /// bottom of the distribution have been ignored.
    ///
    /// Must be within the range `(0, 1]`. The default is `0.99`.
    pub fn set_quantile(self, quantile: f32) -> ScalarQuantizerParams {
        unsafe {
            (*self.0).quantile = quantile;
        }
        self
    }
}

impl fmt::Debug for ScalarQuantizerParams {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // custom debug impl: the default would just print the raw pointer
        write!(f, "ScalarQuantizerParams({:?})", unsafe { *self.0 })
    }
}

impl Drop for ScalarQuantizerParams {
    fn drop(&mut self) {
        if let Err(e) = check_cuvs(unsafe { ffi::cuvsScalarQuantizerParamsDestroy(self.0) }) {
            write!(stderr(), "failed to call cuvsScalarQuantizerParamsDestroy {:?}", e)
                .expect("failed to write to stderr");
        }
    }
}

/// A trained scalar quantizer.
///
/// Build one with [`Quantizer::train`], then use [`Quantizer::transform`] to
/// quantize a float dataset into int8 and [`Quantizer::inverse_transform`] to
/// reconstruct an approximation of the original float values.
#[derive(Debug)]
pub struct Quantizer(ffi::cuvsScalarQuantizer_t);

impl Quantizer {
    /// Creates a new, untrained quantizer.
    fn new() -> Result<Quantizer> {
        unsafe {
            let mut quantizer = std::mem::MaybeUninit::<ffi::cuvsScalarQuantizer_t>::uninit();
            check_cuvs(ffi::cuvsScalarQuantizerCreate(quantizer.as_mut_ptr()))?;
            Ok(Quantizer(quantizer.assume_init()))
        }
    }

    /// Trains a scalar quantizer on `dataset` for later use in quantizing data.
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `params` - Parameters controlling the quantization (e.g. quantile)
    /// * `dataset` - A row-major `f32`, `f16`, or `f64` matrix on either the host or device
    pub fn train(
        res: &Resources,
        params: &ScalarQuantizerParams,
        dataset: &ManagedTensor,
    ) -> Result<Quantizer> {
        let quantizer = Quantizer::new()?;
        unsafe {
            check_cuvs(ffi::cuvsScalarQuantizerTrain(
                res.0,
                params.0,
                dataset.as_ptr(),
                quantizer.0,
            ))?;
        }
        Ok(quantizer)
    }

    /// Quantizes `dataset` into `out`.
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `dataset` - A row-major `f32`, `f16`, or `f64` matrix to quantize, shape `(m, n)`
    /// * `out` - A row-major `i8` matrix that receives the quantized data, shape `(m, n)`
    ///   — the output dtype must be `i8`: the C API does not validate it and will
    ///   reinterpret the buffer otherwise (unlike `inverse_transform`, whose output
    ///   dtype is validated)
    pub fn transform(
        &self,
        res: &Resources,
        dataset: &ManagedTensor,
        out: &ManagedTensor,
    ) -> Result<()> {
        unsafe {
            check_cuvs(ffi::cuvsScalarQuantizerTransform(
                res.0,
                self.0,
                dataset.as_ptr(),
                out.as_ptr(),
            ))
        }
    }

    /// Reconstructs an approximation of the original float dataset from
    /// previously quantized data.
    ///
    /// Note that scalar quantization is lossy, so the reconstructed values only
    /// approximate the originals.
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `dataset` - A row-major `i8` matrix of quantized data, shape `(m, n)`
    /// * `out` - A row-major `f32` matrix that receives the reconstructed data, shape `(m, n)`
    pub fn inverse_transform(
        &self,
        res: &Resources,
        dataset: &ManagedTensor,
        out: &ManagedTensor,
    ) -> Result<()> {
        unsafe {
            check_cuvs(ffi::cuvsScalarQuantizerInverseTransform(
                res.0,
                self.0,
                dataset.as_ptr(),
                out.as_ptr(),
            ))
        }
    }
}

impl Drop for Quantizer {
    fn drop(&mut self) {
        if let Err(e) = check_cuvs(unsafe { ffi::cuvsScalarQuantizerDestroy(self.0) }) {
            write!(stderr(), "failed to call cuvsScalarQuantizerDestroy {:?}", e)
                .expect("failed to write to stderr");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;

    #[test]
    fn test_scalar_quantizer_params() {
        let params = ScalarQuantizerParams::new().unwrap().set_quantile(0.95);

        // make sure the setter actually updated the internal c-struct
        unsafe {
            assert_eq!((*params.0).quantile, 0.95);
        }
    }

    #[test]
    fn test_scalar_quantizer_roundtrip() {
        let res = Resources::new().unwrap();

        // Create a random dataset to quantize. The data range is [0, 10), so
        // the int8 quantization step is roughly 10 / 256 ~= 0.04.
        let n_rows = 1024;
        let n_cols = 16;
        let data_lo = 0.0f32;
        let data_hi = 10.0f32;
        let dataset =
            ndarray::Array::<f32, _>::random((n_rows, n_cols), Uniform::new(data_lo, data_hi));
        let dataset_device = ManagedTensor::from(&dataset).to_device(&res).unwrap();

        // Train the quantizer (use the full range so we don't clip outliers).
        let params = ScalarQuantizerParams::new().unwrap().set_quantile(1.0);
        let quantizer = Quantizer::train(&res, &params, &dataset_device).unwrap();

        // Quantize the dataset into int8.
        let mut quantized_host = ndarray::Array::<i8, _>::zeros((n_rows, n_cols));
        let quantized = ManagedTensor::from(&quantized_host).to_device(&res).unwrap();
        quantizer.transform(&res, &dataset_device, &quantized).unwrap();
        quantized.to_host(&res, &mut quantized_host).unwrap();

        // The quantized values should span a good chunk of the int8 range,
        // confirming the transform actually did something.
        let q_min = *quantized_host.iter().min().unwrap();
        let q_max = *quantized_host.iter().max().unwrap();
        assert!(
            q_max as i32 - q_min as i32 > 200,
            "quantized values should span most of the int8 range, got [{q_min}, {q_max}]"
        );

        // Reconstruct an approximation of the original f32 values.
        let mut reconstructed_host = ndarray::Array::<f32, _>::zeros((n_rows, n_cols));
        let reconstructed = ManagedTensor::from(&reconstructed_host).to_device(&res).unwrap();
        quantizer.inverse_transform(&res, &quantized, &reconstructed).unwrap();
        reconstructed.to_host(&res, &mut reconstructed_host).unwrap();

        // Compute the max absolute reconstruction error. It should be bounded
        // by a few quantization steps and far below the data range.
        let mut max_abs_err = 0.0f32;
        for (orig, recon) in dataset.iter().zip(reconstructed_host.iter()) {
            let err = (orig - recon).abs();
            if err > max_abs_err {
                max_abs_err = err;
            }
        }

        let data_range = data_hi - data_lo;
        // A loose epsilon: a handful of quantization steps. One step is
        // data_range / 256 ~= 0.04; allow up to ~5 steps of slack.
        let epsilon = data_range / 50.0;
        assert!(
            max_abs_err < epsilon,
            "max abs reconstruction error {max_abs_err} should be below {epsilon}"
        );
        assert!(
            max_abs_err < data_range * 0.05,
            "max abs reconstruction error {max_abs_err} should be far below data range {data_range}"
        );
    }

    #[test]
    fn test_train_unsupported_dtype_errors() {
        let res = Resources::new().unwrap();

        // The C API only supports float (16/32/64-bit) training datasets, and
        // surfaces an integer dataset as an error rather than silently
        // succeeding. (Note: a freshly created, untrained quantizer has
        // min_ == max_ == 0, which produces degenerate output but is *not*
        // reported as an error by the C API, so we exercise the dtype guard
        // instead to cover the error path.)
        let n_rows = 8;
        let n_cols = 4;
        let dataset = ndarray::Array::<i32, _>::zeros((n_rows, n_cols));
        let dataset_device = ManagedTensor::from(&dataset).to_device(&res).unwrap();

        let params = ScalarQuantizerParams::new().unwrap();
        let result = Quantizer::train(&res, &params, &dataset_device);
        assert!(
            result.is_err(),
            "training on an unsupported (integer) dtype should return an error"
        );
    }
}
