/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::distance_type::DistanceType;
use crate::error::{check_cuvs, Result};
use std::fmt;
use std::io::{stderr, Write};

pub struct IndexParams(pub ffi::cuvsVamanaIndexParams_t);

impl IndexParams {
    /// Returns a new IndexParams
    pub fn new() -> Result<IndexParams> {
        unsafe {
            let mut params = std::mem::MaybeUninit::<ffi::cuvsVamanaIndexParams_t>::uninit();
            check_cuvs(ffi::cuvsVamanaIndexParamsCreate(params.as_mut_ptr()))?;
            Ok(IndexParams(params.assume_init()))
        }
    }

    /// DistanceType to use for building the index
    pub fn set_metric(self, metric: DistanceType) -> IndexParams {
        unsafe {
            (*self.0).metric = metric;
        }
        self
    }

    /// Maximum degree of output graph corresponds to the R parameter in the original Vamana
    /// literature.
    pub fn set_graph_degree(self, graph_degree: u32) -> IndexParams {
        unsafe {
            (*self.0).graph_degree = graph_degree;
        }
        self
    }

    /// Maximum number of visited nodes per search corresponds to the L parameter in the Vamana
    /// literature
    pub fn set_visited_size(self, visited_size: u32) -> IndexParams {
        unsafe {
            (*self.0).visited_size = visited_size;
        }
        self
    }

    /// Number of Vamana vector insertion iterations (each iteration inserts all vectors).
    pub fn set_vamana_iters(self, vamana_iters: f32) -> IndexParams {
        unsafe {
            (*self.0).vamana_iters = vamana_iters;
        }
        self
    }

    /// Alpha for pruning parameter
    pub fn set_alpha(self, alpha: f32) -> IndexParams {
        unsafe {
            (*self.0).alpha = alpha;
        }
        self
    }

    /// Maximum fraction of dataset inserted per batch.
    /// Larger max batch decreases graph quality, but improves speed
    pub fn set_max_fraction(self, max_fraction: f32) -> IndexParams {
        unsafe {
            (*self.0).max_fraction = max_fraction;
        }
        self
    }

    /// Base of growth rate of batch sizes
    pub fn set_batch_base(self, batch_base: f32) -> IndexParams {
        unsafe {
            (*self.0).batch_base = batch_base;
        }
        self
    }

    /// Size of candidate queue structure - should be (2^x)-1
    pub fn set_queue_size(self, queue_size: u32) -> IndexParams {
        unsafe {
            (*self.0).queue_size = queue_size;
        }
        self
    }

    /// Max batchsize of reverse edge processing (reduces memory footprint)
    pub fn set_reverse_batchsize(self, reverse_batchsize: u32) -> IndexParams {
        unsafe {
            (*self.0).reverse_batchsize = reverse_batchsize;
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
        if let Err(e) = check_cuvs(unsafe { ffi::cuvsVamanaIndexParamsDestroy(self.0) }) {
            write!(
                stderr(),
                "failed to call cuvsVamanaIndexParamsDestroy {:?}",
                e
            )
            .expect("failed to write to stderr");
        }
    }
}

/// Builder for Vamana [`IndexParams`] with pre-validated parameters.
///
/// Construct via [`IndexParams::builder()`]. Defaults match the cuVS C API defaults.
pub struct IndexParamsBuilder {
    graph_degree: u32,
    visited_size: u32,
    vamana_iters: f32,
    alpha: f32,
    max_fraction: f32,
    batch_base: f32,
    queue_size: u32,
    reverse_batchsize: u32,
    metric: Option<DistanceType>,
}

impl Default for IndexParamsBuilder {
    fn default() -> Self {
        Self {
            graph_degree: 64,
            visited_size: 75,
            vamana_iters: 2.0,
            alpha: 1.2,
            max_fraction: 0.06,
            batch_base: 2.0,
            queue_size: 255,
            reverse_batchsize: 1_000_000,
            metric: None,
        }
    }
}

impl IndexParamsBuilder {
    /// Maximum degree of the output graph (R parameter in Vamana literature).
    ///
    /// Must be > 0.
    pub fn graph_degree(mut self, v: u32) -> Self {
        self.graph_degree = v;
        self
    }

    /// Maximum number of visited nodes per search (L parameter in Vamana literature).
    ///
    /// Must be >= `graph_degree`.
    pub fn visited_size(mut self, v: u32) -> Self {
        self.visited_size = v;
        self
    }

    /// Number of Vamana vector insertion iterations.
    pub fn vamana_iters(mut self, v: f32) -> Self {
        self.vamana_iters = v;
        self
    }

    /// Alpha for pruning parameter.
    ///
    /// Must be > 0.
    pub fn alpha(mut self, v: f32) -> Self {
        self.alpha = v;
        self
    }

    /// Maximum fraction of dataset inserted per batch.
    pub fn max_fraction(mut self, v: f32) -> Self {
        self.max_fraction = v;
        self
    }

    /// Base of growth rate of batch sizes.
    pub fn batch_base(mut self, v: f32) -> Self {
        self.batch_base = v;
        self
    }

    /// Size of candidate queue structure.
    pub fn queue_size(mut self, v: u32) -> Self {
        self.queue_size = v;
        self
    }

    /// Max batchsize of reverse edge processing.
    pub fn reverse_batchsize(mut self, v: u32) -> Self {
        self.reverse_batchsize = v;
        self
    }

    /// DistanceType to use for building the index.
    pub fn metric(mut self, v: DistanceType) -> Self {
        self.metric = Some(v);
        self
    }

    /// Validate all parameters without allocating any GPU resources.
    pub fn validate(&self) -> crate::error::Result<()> {
        if self.graph_degree == 0 {
            return Err(format!("graph_degree must be > 0; got {}", self.graph_degree).into());
        }
        if self.visited_size < self.graph_degree {
            return Err(format!(
                "visited_size ({}) must be >= graph_degree ({})",
                self.visited_size, self.graph_degree
            )
            .into());
        }
        if self.alpha <= 0.0 {
            return Err(format!("alpha must be > 0; got {}", self.alpha).into());
        }
        Ok(())
    }

    /// Validate all parameters and allocate the FFI struct.
    pub fn build(self) -> crate::error::Result<IndexParams> {
        self.validate()?;
        let mut params = IndexParams::new()?
            .set_graph_degree(self.graph_degree)
            .set_visited_size(self.visited_size)
            .set_vamana_iters(self.vamana_iters)
            .set_alpha(self.alpha)
            .set_max_fraction(self.max_fraction)
            .set_batch_base(self.batch_base)
            .set_queue_size(self.queue_size)
            .set_reverse_batchsize(self.reverse_batchsize);
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
            .set_alpha(1.0)
            .set_visited_size(128);

        unsafe {
            assert_eq!((*params.0).alpha, 1.0);
            assert_eq!((*params.0).visited_size, 128);
        }
    }

    #[test]
    fn builder_rejects_zero_graph_degree() {
        let err = IndexParams::builder()
            .graph_degree(0)
            .validate()
            .unwrap_err();
        assert!(
            err.to_string().contains("graph_degree"),
            "error message should name the field: {err}"
        );
    }

    #[test]
    fn builder_rejects_visited_size_less_than_graph_degree() {
        let err = IndexParams::builder()
            .graph_degree(64)
            .visited_size(32)
            .validate()
            .unwrap_err();
        assert!(
            err.to_string().contains("visited_size"),
            "error message should name the field: {err}"
        );
    }

    #[test]
    fn builder_accepts_valid_params() {
        assert!(IndexParams::builder()
            .graph_degree(32)
            .visited_size(75)
            .alpha(1.2)
            .validate()
            .is_ok());
    }
}
