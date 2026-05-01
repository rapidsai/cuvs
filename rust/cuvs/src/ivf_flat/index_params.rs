/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::distance_type::DistanceType;
use crate::error::{check_cuvs, Result};
use std::fmt;
use std::io::{stderr, Write};

pub struct IndexParams(pub ffi::cuvsIvfFlatIndexParams_t);

impl IndexParams {
    /// Returns a new IndexParams
    pub fn new() -> Result<IndexParams> {
        unsafe {
            let mut params = std::mem::MaybeUninit::<ffi::cuvsIvfFlatIndexParams_t>::uninit();
            check_cuvs(ffi::cuvsIvfFlatIndexParamsCreate(params.as_mut_ptr()))?;
            Ok(IndexParams(params.assume_init()))
        }
    }

    /// The number of clusters used in the coarse quantizer.
    pub fn set_n_lists(self, n_lists: u32) -> IndexParams {
        unsafe {
            (*self.0).n_lists = n_lists;
        }
        self
    }

    /// DistanceType to use for building the index
    pub fn set_metric(self, metric: DistanceType) -> IndexParams {
        unsafe {
            (*self.0).metric = metric;
        }
        self
    }

    /// The number of iterations searching for kmeans centers during index building.
    pub fn set_metric_arg(self, metric_arg: f32) -> IndexParams {
        unsafe {
            (*self.0).metric_arg = metric_arg;
        }
        self
    }
    /// The number of iterations searching for kmeans centers during index building.
    pub fn set_kmeans_n_iters(self, kmeans_n_iters: u32) -> IndexParams {
        unsafe {
            (*self.0).kmeans_n_iters = kmeans_n_iters;
        }
        self
    }

    /// If kmeans_trainset_fraction is less than 1, then the dataset is
    /// subsampled, and only n_samples * kmeans_trainset_fraction rows
    /// are used for training.
    pub fn set_kmeans_trainset_fraction(self, kmeans_trainset_fraction: f64) -> IndexParams {
        unsafe {
            (*self.0).kmeans_trainset_fraction = kmeans_trainset_fraction;
        }
        self
    }

    /// After training the coarse and fine quantizers, we will populate
    /// the index with the dataset if add_data_on_build == true, otherwise
    /// the index is left empty, and the extend method can be used
    /// to add new vectors to the index.
    pub fn set_add_data_on_build(self, add_data_on_build: bool) -> IndexParams {
        unsafe {
            (*self.0).add_data_on_build = add_data_on_build;
        }
        self
    }
}

impl IndexParams {
    /// Returns a builder for constructing [`IndexParams`] with validated parameters.
    ///
    /// See [`IndexParamsBuilder`] for details.
    pub fn builder() -> IndexParamsBuilder {
        IndexParamsBuilder::default()
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
        if let Err(e) = check_cuvs(unsafe { ffi::cuvsIvfFlatIndexParamsDestroy(self.0) }) {
            write!(
                stderr(),
                "failed to call cuvsIvfFlatIndexParamsDestroy {:?}",
                e
            )
            .expect("failed to write to stderr");
        }
    }
}

/// Builder for IVF-Flat [`IndexParams`] with pre-validated parameters.
///
/// Construct via [`IndexParams::builder()`]. Defaults match the cuVS C API defaults.
pub struct IndexParamsBuilder {
    n_lists: u32,
    metric: Option<DistanceType>,
    metric_arg: f32,
    kmeans_n_iters: u32,
    kmeans_trainset_fraction: f64,
    add_data_on_build: bool,
}

impl Default for IndexParamsBuilder {
    fn default() -> Self {
        Self {
            n_lists: 1024,
            metric: None,
            metric_arg: 2.0,
            kmeans_n_iters: 20,
            kmeans_trainset_fraction: 0.5,
            add_data_on_build: true,
        }
    }
}

impl IndexParamsBuilder {
    /// The number of clusters used in the coarse quantizer.
    ///
    /// Must be > 0.
    pub fn n_lists(mut self, v: u32) -> Self {
        self.n_lists = v;
        self
    }

    /// DistanceType to use for building the index.
    pub fn metric(mut self, v: DistanceType) -> Self {
        self.metric = Some(v);
        self
    }

    /// Metric argument (e.g. p for Minkowski distance).
    pub fn metric_arg(mut self, v: f32) -> Self {
        self.metric_arg = v;
        self
    }

    /// Number of iterations searching for kmeans centers during index building.
    pub fn kmeans_n_iters(mut self, v: u32) -> Self {
        self.kmeans_n_iters = v;
        self
    }

    /// Fraction of dataset used for kmeans training. Must be in (0, 1].
    pub fn kmeans_trainset_fraction(mut self, v: f64) -> Self {
        self.kmeans_trainset_fraction = v;
        self
    }

    /// Populate the index with the dataset during build. When false, use `extend`.
    pub fn add_data_on_build(mut self, v: bool) -> Self {
        self.add_data_on_build = v;
        self
    }

    /// Validate all parameters without allocating any GPU resources.
    pub fn validate(&self) -> crate::error::Result<()> {
        if self.n_lists == 0 {
            return Err(format!("n_lists must be > 0; got {}", self.n_lists).into());
        }
        if self.kmeans_trainset_fraction <= 0.0 || self.kmeans_trainset_fraction > 1.0 {
            return Err(format!(
                "kmeans_trainset_fraction must be in (0, 1]; got {}",
                self.kmeans_trainset_fraction
            )
            .into());
        }
        Ok(())
    }

    /// Validate all parameters and allocate the FFI struct.
    pub fn build(self) -> crate::error::Result<IndexParams> {
        self.validate()?;
        let mut params = IndexParams::new()?
            .set_n_lists(self.n_lists)
            .set_metric_arg(self.metric_arg)
            .set_kmeans_n_iters(self.kmeans_n_iters)
            .set_kmeans_trainset_fraction(self.kmeans_trainset_fraction)
            .set_add_data_on_build(self.add_data_on_build);
        if let Some(metric) = self.metric {
            params = params.set_metric(metric);
        }
        Ok(params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_params() {
        let params = IndexParams::new()
            .unwrap()
            .set_n_lists(128)
            .set_add_data_on_build(false);

        unsafe {
            assert_eq!((*params.0).n_lists, 128);
            assert_eq!((*params.0).add_data_on_build, false);
        }
    }

    #[test]
    fn builder_rejects_zero_n_lists() {
        let err = IndexParams::builder().n_lists(0).validate().unwrap_err();
        assert!(
            err.to_string().contains("n_lists"),
            "error message should name the field: {err}"
        );
    }

    #[test]
    fn builder_accepts_valid_params() {
        assert!(IndexParams::builder()
            .n_lists(256)
            .kmeans_trainset_fraction(0.5)
            .validate()
            .is_ok());
    }
}
