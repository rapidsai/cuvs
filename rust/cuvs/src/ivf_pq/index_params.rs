/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::distance_type::DistanceType;
use crate::error::{check_cuvs, Result};
use std::fmt;
use std::io::{stderr, Write};

pub use ffi::cuvsIvfPqCodebookGen;
pub use ffi::cuvsIvfPqListLayout;

pub struct IndexParams(pub ffi::cuvsIvfPqIndexParams_t);

impl IndexParams {
    /// Returns a new IndexParams
    pub fn new() -> Result<IndexParams> {
        unsafe {
            let mut params = std::mem::MaybeUninit::<ffi::cuvsIvfPqIndexParams_t>::uninit();
            check_cuvs(ffi::cuvsIvfPqIndexParamsCreate(params.as_mut_ptr()))?;
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

    /// The bit length of the vector element after quantization.
    pub fn set_pq_bits(self, pq_bits: u32) -> IndexParams {
        unsafe {
            (*self.0).pq_bits = pq_bits;
        }
        self
    }

    /// The dimensionality of a the vector after product quantization.
    /// When zero, an optimal value is selected using a heuristic. Note
    /// pq_dim * pq_bits must be a multiple of 8. Hint: a smaller 'pq_dim'
    /// results in a smaller index size and better search performance, but
    /// lower recall. If 'pq_bits' is 8, 'pq_dim' can be set to any number,
    /// but multiple of 8 are desirable for good performance. If 'pq_bits'
    /// is not 8, 'pq_dim' should be a multiple of 8. For good performance,
    /// it is desirable that 'pq_dim' is a multiple of 32. Ideally,
    /// 'pq_dim' should be also a divisor of the dataset dim.
    pub fn set_pq_dim(self, pq_dim: u32) -> IndexParams {
        unsafe {
            (*self.0).pq_dim = pq_dim;
        }
        self
    }

    pub fn set_codebook_kind(self, codebook_kind: cuvsIvfPqCodebookGen) -> IndexParams {
        unsafe {
            (*self.0).codebook_kind = codebook_kind;
        }
        self
    }

    /// Memory layout of the IVF-PQ list data.
    /// - FLAT: Codes are stored contiguously, one vector's codes after another.
    /// - INTERLEAVED: Codes are interleaved for optimized search performance.
    ///   This is the default and recommended for search workloads.
    pub fn set_codes_layout(self, codes_layout: cuvsIvfPqListLayout) -> IndexParams {
        unsafe {
            (*self.0).codes_layout = codes_layout;
        }
        self
    }

    /// Apply a random rotation matrix on the input data and queries even
    /// if `dim % pq_dim == 0`. Note: if `dim` is not multiple of `pq_dim`,
    /// a random rotation is always applied to the input data and queries
    /// to transform the working space from `dim` to `rot_dim`, which may
    /// be slightly larger than the original space and and is a multiple
    /// of `pq_dim` (`rot_dim % pq_dim == 0`). However, this transform is
    /// not necessary when `dim` is multiple of `pq_dim` (`dim == rot_dim`,
    /// hence no need in adding "extra" data columns / features). By
    /// default, if `dim == rot_dim`, the rotation transform is
    /// initialized with the identity matrix. When
    /// `force_random_rotation == True`, a random orthogonal transform
    pub fn set_force_random_rotation(self, force_random_rotation: bool) -> IndexParams {
        unsafe {
            (*self.0).force_random_rotation = force_random_rotation;
        }
        self
    }

    /// The max number of data points to use per PQ code during PQ codebook training. Using more data
    /// points per PQ code may increase the quality of PQ codebook but may also increase the build
    /// time. The parameter is applied to both PQ codebook generation methods, i.e., PER_SUBSPACE and
    /// PER_CLUSTER. In both cases, we will use `pq_book_size * max_train_points_per_pq_code` training
    /// points to train each codebook.
    pub fn set_max_train_points_per_pq_code(self, max_pq_points: u32) -> IndexParams {
        unsafe {
            (*self.0).max_train_points_per_pq_code = max_pq_points;
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
        if let Err(e) = check_cuvs(unsafe { ffi::cuvsIvfPqIndexParamsDestroy(self.0) }) {
            write!(
                stderr(),
                "failed to call cuvsIvfPqIndexParamsDestroy {:?}",
                e
            )
            .expect("failed to write to stderr");
        }
    }
}

/// Builder for IVF-PQ [`IndexParams`] with pre-validated parameters.
///
/// Construct via [`IndexParams::builder()`]. Defaults match the cuVS C API defaults.
pub struct IndexParamsBuilder {
    n_lists: u32,
    metric: Option<crate::distance_type::DistanceType>,
    metric_arg: f32,
    kmeans_n_iters: u32,
    kmeans_trainset_fraction: f64,
    pq_bits: u32,
    pq_dim: u32,
    codebook_kind: Option<cuvsIvfPqCodebookGen>,
    codes_layout: Option<cuvsIvfPqListLayout>,
    force_random_rotation: bool,
    max_train_points_per_pq_code: u32,
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
            pq_bits: 8,
            pq_dim: 0,
            codebook_kind: None,
            codes_layout: None,
            force_random_rotation: false,
            max_train_points_per_pq_code: 256,
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
    pub fn metric(mut self, v: crate::distance_type::DistanceType) -> Self {
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

    /// Bit length of the vector element after quantization. Typically 4 or 8.
    pub fn pq_bits(mut self, v: u32) -> Self {
        self.pq_bits = v;
        self
    }

    /// Dimensionality of the vector after product quantization. 0 = auto.
    pub fn pq_dim(mut self, v: u32) -> Self {
        self.pq_dim = v;
        self
    }

    /// Codebook generation method.
    pub fn codebook_kind(mut self, v: cuvsIvfPqCodebookGen) -> Self {
        self.codebook_kind = Some(v);
        self
    }

    /// Memory layout of IVF-PQ list data.
    pub fn codes_layout(mut self, v: cuvsIvfPqListLayout) -> Self {
        self.codes_layout = Some(v);
        self
    }

    /// Apply a random rotation matrix on input data and queries.
    pub fn force_random_rotation(mut self, v: bool) -> Self {
        self.force_random_rotation = v;
        self
    }

    /// Max number of data points per PQ code during codebook training.
    pub fn max_train_points_per_pq_code(mut self, v: u32) -> Self {
        self.max_train_points_per_pq_code = v;
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
            .set_pq_bits(self.pq_bits)
            .set_pq_dim(self.pq_dim)
            .set_force_random_rotation(self.force_random_rotation)
            .set_max_train_points_per_pq_code(self.max_train_points_per_pq_code)
            .set_add_data_on_build(self.add_data_on_build);
        if let Some(metric) = self.metric {
            params = params.set_metric(metric);
        }
        if let Some(kind) = self.codebook_kind {
            params = params.set_codebook_kind(kind);
        }
        if let Some(layout) = self.codes_layout {
            params = params.set_codes_layout(layout);
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

    #[test]
    fn existing_setter_api_unchanged() {
        let params = IndexParams::new()
            .unwrap()
            .set_n_lists(128)
            .set_add_data_on_build(false);
        unsafe {
            assert_eq!((*params.0).n_lists, 128);
            assert_eq!((*params.0).add_data_on_build, false);
        }
    }
}
