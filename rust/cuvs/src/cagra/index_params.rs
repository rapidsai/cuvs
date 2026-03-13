/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::error::{check_cuvs, Result};
use std::fmt;
use std::io::{stderr, Write};

pub type BuildAlgo = ffi::cuvsCagraGraphBuildAlgo;

/// Supplemental parameters to build CAGRA Index
pub struct CompressionParams(pub ffi::cuvsCagraCompressionParams_t);

impl CompressionParams {
    /// Returns a new CompressionParams
    pub fn new() -> Result<CompressionParams> {
        unsafe {
            let mut params = std::mem::MaybeUninit::<ffi::cuvsCagraCompressionParams_t>::uninit();
            check_cuvs(ffi::cuvsCagraCompressionParamsCreate(params.as_mut_ptr()))?;
            Ok(CompressionParams(params.assume_init()))
        }
    }

    /// The bit length of the vector element after compression by PQ.
    pub fn set_pq_bits(self, pq_bits: u32) -> CompressionParams {
        unsafe {
            (*self.0).pq_bits = pq_bits;
        }
        self
    }

    /// The dimensionality of the vector after compression by PQ. When zero,
    /// an optimal value is selected using a heuristic.
    pub fn set_pq_dim(self, pq_dim: u32) -> CompressionParams {
        unsafe {
            (*self.0).pq_dim = pq_dim;
        }
        self
    }

    /// Vector Quantization (VQ) codebook size - number of "coarse cluster
    /// centers". When zero, an optimal value is selected using a heuristic.
    pub fn set_vq_n_centers(self, vq_n_centers: u32) -> CompressionParams {
        unsafe {
            (*self.0).vq_n_centers = vq_n_centers;
        }
        self
    }

    /// The number of iterations searching for kmeans centers (both VQ & PQ
    /// phases).
    pub fn set_kmeans_n_iters(self, kmeans_n_iters: u32) -> CompressionParams {
        unsafe {
            (*self.0).kmeans_n_iters = kmeans_n_iters;
        }
        self
    }

    /// The fraction of data to use during iterative kmeans building (VQ
    /// phase). When zero, an optimal value is selected using a heuristic.
    pub fn set_vq_kmeans_trainset_fraction(
        self,
        vq_kmeans_trainset_fraction: f64,
    ) -> CompressionParams {
        unsafe {
            (*self.0).vq_kmeans_trainset_fraction = vq_kmeans_trainset_fraction;
        }
        self
    }

    /// The fraction of data to use during iterative kmeans building (PQ
    /// phase). When zero, an optimal value is selected using a heuristic.
    pub fn set_pq_kmeans_trainset_fraction(
        self,
        pq_kmeans_trainset_fraction: f64,
    ) -> CompressionParams {
        unsafe {
            (*self.0).pq_kmeans_trainset_fraction = pq_kmeans_trainset_fraction;
        }
        self
    }
}

pub struct IndexParams(pub ffi::cuvsCagraIndexParams_t, Option<CompressionParams>);

impl IndexParams {
    /// Returns a new IndexParams
    pub fn new() -> Result<IndexParams> {
        unsafe {
            let mut params = std::mem::MaybeUninit::<ffi::cuvsCagraIndexParams_t>::uninit();
            check_cuvs(ffi::cuvsCagraIndexParamsCreate(params.as_mut_ptr()))?;
            Ok(IndexParams(params.assume_init(), None))
        }
    }

    /// Degree of input graph for pruning
    pub fn set_intermediate_graph_degree(self, intermediate_graph_degree: usize) -> IndexParams {
        unsafe {
            (*self.0).intermediate_graph_degree = intermediate_graph_degree;
        }
        self
    }

    /// Degree of output graph
    pub fn set_graph_degree(self, graph_degree: usize) -> IndexParams {
        unsafe {
            (*self.0).graph_degree = graph_degree;
        }
        self
    }

    /// ANN algorithm to build knn graph
    pub fn set_build_algo(self, build_algo: BuildAlgo) -> IndexParams {
        unsafe {
            (*self.0).build_algo = build_algo;
        }
        self
    }

    /// Number of iterations to run if building with NN_DESCENT
    pub fn set_nn_descent_niter(self, nn_descent_niter: usize) -> IndexParams {
        unsafe {
            (*self.0).nn_descent_niter = nn_descent_niter;
        }
        self
    }

    pub fn set_compression(mut self, compression: CompressionParams) -> IndexParams {
        unsafe {
            (*self.0).compression = compression.0;
        }
        // Note: we're moving the ownership of compression here to avoid having it cleaned up
        // and leaving a dangling pointer
        self.1 = Some(compression);
        self
    }
}

impl IndexParams {
    /// Returns a builder for constructing [`IndexParams`] with validated parameters.
    ///
    /// Unlike the `IndexParams::new()?.set_*()` setter chain, [`IndexParamsBuilder::build`]
    /// validates all parameters in Rust before any FFI allocation. Invalid values produce a
    /// clear error message naming the offending field and its valid range, before any GPU
    /// work begins.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use cuvs::cagra::IndexParams;
    ///
    /// let params = IndexParams::builder()
    ///     .graph_degree(32)
    ///     .intermediate_graph_degree(64)
    ///     .nn_descent_niter(20)
    ///     .build()
    ///     .unwrap();
    /// ```
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

impl fmt::Debug for CompressionParams {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "CompressionParams({:?})", unsafe { *self.0 })
    }
}

impl Drop for IndexParams {
    fn drop(&mut self) {
        if let Err(e) = check_cuvs(unsafe { ffi::cuvsCagraIndexParamsDestroy(self.0) }) {
            write!(
                stderr(),
                "failed to call cuvsCagraIndexParamsDestroy {:?}",
                e
            )
            .expect("failed to write to stderr");
        }
    }
}

impl Drop for CompressionParams {
    fn drop(&mut self) {
        if let Err(e) = check_cuvs(unsafe { ffi::cuvsCagraCompressionParamsDestroy(self.0) }) {
            write!(
                stderr(),
                "failed to call cuvsCagraCompressionParamsDestroy {:?}",
                e
            )
            .expect("failed to write to stderr");
        }
    }
}

/// Builder for [`IndexParams`] with pre-validated parameters.
///
/// Construct via [`IndexParams::builder()`]. Call [`IndexParamsBuilder::build`] to
/// validate all parameters and allocate the FFI struct in one step.
///
/// Defaults match the cuVS C API defaults: `graph_degree=64`,
/// `intermediate_graph_degree=128`, `nn_descent_niter=20`.
pub struct IndexParamsBuilder {
    graph_degree: usize,
    intermediate_graph_degree: usize,
    nn_descent_niter: usize,
    build_algo: Option<BuildAlgo>,
    compression: Option<CompressionParams>,
}

impl Default for IndexParamsBuilder {
    fn default() -> Self {
        Self {
            graph_degree: 64,
            intermediate_graph_degree: 128,
            nn_descent_niter: 20,
            build_algo: None,
            compression: None,
        }
    }
}

impl IndexParamsBuilder {
    /// Degree of output graph.
    ///
    /// Must be > 0. Values that are multiples of 32 are preferred for warp alignment.
    pub fn graph_degree(mut self, v: usize) -> Self {
        self.graph_degree = v;
        self
    }

    /// Degree of input graph for pruning.
    ///
    /// Must be >= `graph_degree`.
    pub fn intermediate_graph_degree(mut self, v: usize) -> Self {
        self.intermediate_graph_degree = v;
        self
    }

    /// Number of iterations to run if building with NN_DESCENT.
    ///
    /// Must be > 0.
    pub fn nn_descent_niter(mut self, v: usize) -> Self {
        self.nn_descent_niter = v;
        self
    }

    /// ANN algorithm to build knn graph.
    pub fn build_algo(mut self, v: BuildAlgo) -> Self {
        self.build_algo = Some(v);
        self
    }

    /// Vector compression parameters.
    pub fn compression(mut self, v: CompressionParams) -> Self {
        self.compression = Some(v);
        self
    }

    /// Validate all parameters without allocating any GPU resources.
    ///
    /// Returns `Ok(())` if all parameters are valid, or `Err` with a message naming
    /// the offending field and its valid range.
    pub fn validate(&self) -> crate::error::Result<()> {
        if self.graph_degree == 0 {
            return Err(format!("graph_degree must be > 0; got {}", self.graph_degree).into());
        }
        if self.intermediate_graph_degree < self.graph_degree {
            return Err(format!(
                "intermediate_graph_degree ({}) must be >= graph_degree ({})",
                self.intermediate_graph_degree, self.graph_degree
            )
            .into());
        }
        if self.nn_descent_niter == 0 {
            return Err(format!(
                "nn_descent_niter must be > 0; got {}",
                self.nn_descent_niter
            )
            .into());
        }
        Ok(())
    }

    /// Validate all parameters and allocate the FFI struct.
    ///
    /// Returns `Err` with a message naming the offending field and its valid range
    /// before any GPU work begins.
    pub fn build(self) -> crate::error::Result<IndexParams> {
        self.validate()?;
        let mut params = IndexParams::new()?
            .set_graph_degree(self.graph_degree)
            .set_intermediate_graph_degree(self.intermediate_graph_degree)
            .set_nn_descent_niter(self.nn_descent_niter);
        if let Some(algo) = self.build_algo {
            params = params.set_build_algo(algo);
        }
        if let Some(compression) = self.compression {
            params = params.set_compression(compression);
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
            .set_intermediate_graph_degree(128)
            .set_graph_degree(16)
            .set_build_algo(BuildAlgo::NN_DESCENT)
            .set_nn_descent_niter(10)
            .set_compression(
                CompressionParams::new()
                    .unwrap()
                    .set_pq_bits(4)
                    .set_pq_dim(8),
            );

        // make sure the setters actually updated internal representation on the c-struct
        unsafe {
            assert_eq!((*params.0).graph_degree, 16);
            assert_eq!((*params.0).intermediate_graph_degree, 128);
            assert_eq!((*params.0).build_algo, BuildAlgo::NN_DESCENT);
            assert_eq!((*params.0).nn_descent_niter, 10);
            assert_eq!((*(*params.0).compression).pq_dim, 8);
            assert_eq!((*(*params.0).compression).pq_bits, 4);
        }
    }

    // --- IndexParamsBuilder tests ---

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
    fn builder_rejects_invalid_intermediate_degree() {
        let err = IndexParams::builder()
            .graph_degree(32)
            .intermediate_graph_degree(16)
            .validate()
            .unwrap_err();
        assert!(
            err.to_string().contains("intermediate_graph_degree"),
            "error message should name the field: {err}"
        );
    }

    #[test]
    fn builder_rejects_zero_niter() {
        let err = IndexParams::builder()
            .nn_descent_niter(0)
            .validate()
            .unwrap_err();
        assert!(
            err.to_string().contains("nn_descent_niter"),
            "error message should name the field: {err}"
        );
    }

    #[test]
    fn builder_accepts_valid_params() {
        assert!(IndexParams::builder()
            .graph_degree(32)
            .intermediate_graph_degree(64)
            .nn_descent_niter(20)
            .validate()
            .is_ok());
    }

    #[test]
    fn builder_round_trips_to_ffi() {
        // Built params must produce the same FFI struct values as the manual setter chain.
        let via_builder = IndexParams::builder()
            .graph_degree(32)
            .intermediate_graph_degree(64)
            .nn_descent_niter(20)
            .build()
            .unwrap();
        let via_setters = IndexParams::new()
            .unwrap()
            .set_graph_degree(32)
            .set_intermediate_graph_degree(64)
            .set_nn_descent_niter(20);
        unsafe {
            assert_eq!((*via_builder.0).graph_degree, (*via_setters.0).graph_degree);
            assert_eq!(
                (*via_builder.0).intermediate_graph_degree,
                (*via_setters.0).intermediate_graph_degree
            );
            assert_eq!(
                (*via_builder.0).nn_descent_niter,
                (*via_setters.0).nn_descent_niter
            );
        }
    }

    #[test]
    fn existing_setter_api_unchanged() {
        // Ensure the original API still compiles and sets values correctly.
        let params = IndexParams::new()
            .unwrap()
            .set_graph_degree(32)
            .set_intermediate_graph_degree(64)
            .set_nn_descent_niter(20);
        unsafe {
            assert_eq!((*params.0).graph_degree, 32);
            assert_eq!((*params.0).intermediate_graph_degree, 64);
            assert_eq!((*params.0).nn_descent_niter, 20);
        }
    }
}
