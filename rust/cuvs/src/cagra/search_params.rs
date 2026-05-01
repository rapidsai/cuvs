/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::error::{check_cuvs, Result};
use std::fmt;
use std::io::{stderr, Write};

pub type SearchAlgo = ffi::cuvsCagraSearchAlgo;
pub type HashMode = ffi::cuvsCagraHashMode;

/// Supplemental parameters to search CAGRA index
pub struct SearchParams(pub ffi::cuvsCagraSearchParams_t);

impl SearchParams {
    /// Returns a new SearchParams object
    pub fn new() -> Result<SearchParams> {
        unsafe {
            let mut params = std::mem::MaybeUninit::<ffi::cuvsCagraSearchParams_t>::uninit();
            check_cuvs(ffi::cuvsCagraSearchParamsCreate(params.as_mut_ptr()))?;
            Ok(SearchParams(params.assume_init()))
        }
    }

    /// Maximum number of queries to search at the same time (batch size). Auto select when 0
    pub fn set_max_queries(self, max_queries: usize) -> SearchParams {
        unsafe {
            (*self.0).max_queries = max_queries;
        }
        self
    }

    /// Number of intermediate search results retained during the search.
    /// This is the main knob to adjust trade off between accuracy and search speed.
    /// Higher values improve the search accuracy
    pub fn set_itopk_size(self, itopk_size: usize) -> SearchParams {
        unsafe {
            (*self.0).itopk_size = itopk_size;
        }
        self
    }

    /// Upper limit of search iterations. Auto select when 0.
    pub fn set_max_iterations(self, max_iterations: usize) -> SearchParams {
        unsafe {
            (*self.0).max_iterations = max_iterations;
        }
        self
    }

    /// Which search implementation to use.
    pub fn set_algo(self, algo: SearchAlgo) -> SearchParams {
        unsafe {
            (*self.0).algo = algo;
        }
        self
    }

    /// Number of threads used to calculate a single distance. 4, 8, 16, or 32.
    pub fn set_team_size(self, team_size: usize) -> SearchParams {
        unsafe {
            (*self.0).team_size = team_size;
        }
        self
    }

    /// Lower limit of search iterations.
    pub fn set_min_iterations(self, min_iterations: usize) -> SearchParams {
        unsafe {
            (*self.0).min_iterations = min_iterations;
        }
        self
    }

    /// Thread block size. 0, 64, 128, 256, 512, 1024. Auto selection when 0.
    pub fn set_thread_block_size(self, thread_block_size: usize) -> SearchParams {
        unsafe {
            (*self.0).thread_block_size = thread_block_size;
        }
        self
    }

    /// Hashmap type. Auto selection when AUTO.
    pub fn set_hashmap_mode(self, hashmap_mode: HashMode) -> SearchParams {
        unsafe {
            (*self.0).hashmap_mode = hashmap_mode;
        }
        self
    }

    /// Lower limit of hashmap bit length. More than 8.
    pub fn set_hashmap_min_bitlen(self, hashmap_min_bitlen: usize) -> SearchParams {
        unsafe {
            (*self.0).hashmap_min_bitlen = hashmap_min_bitlen;
        }
        self
    }

    /// Upper limit of hashmap fill rate. More than 0.1, less than 0.9.
    pub fn set_hashmap_max_fill_rate(self, hashmap_max_fill_rate: f32) -> SearchParams {
        unsafe {
            (*self.0).hashmap_max_fill_rate = hashmap_max_fill_rate;
        }
        self
    }

    /// Number of iterations of initial random seed node selection. 1 or more.
    pub fn set_num_random_samplings(self, num_random_samplings: u32) -> SearchParams {
        unsafe {
            (*self.0).num_random_samplings = num_random_samplings;
        }
        self
    }

    /// Bit mask used for initial random seed node selection.
    pub fn set_rand_xor_mask(self, rand_xor_mask: u64) -> SearchParams {
        unsafe {
            (*self.0).rand_xor_mask = rand_xor_mask;
        }
        self
    }
}

impl SearchParams {
    /// Returns a builder for constructing [`SearchParams`] with validated parameters.
    ///
    /// See [`SearchParamsBuilder`] for details.
    pub fn builder() -> SearchParamsBuilder {
        SearchParamsBuilder::default()
    }
}

impl fmt::Debug for SearchParams {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // custom debug trait here, default value will show the pointer address
        // for the inner params object which isn't that useful.
        write!(f, "SearchParams {{ params: {:?} }}", unsafe { *self.0 })
    }
}

impl Drop for SearchParams {
    fn drop(&mut self) {
        if let Err(e) = check_cuvs(unsafe { ffi::cuvsCagraSearchParamsDestroy(self.0) }) {
            write!(
                stderr(),
                "failed to call cuvsCagraSearchParamsDestroy {:?}",
                e
            )
            .expect("failed to write to stderr");
        }
    }
}

/// Builder for [`SearchParams`] with pre-validated parameters.
///
/// Construct via [`SearchParams::builder()`]. Call [`SearchParamsBuilder::build`] to
/// validate all parameters and allocate the FFI struct in one step.
pub struct SearchParamsBuilder {
    itopk_size: usize,
    max_queries: usize,
    max_iterations: usize,
    min_iterations: usize,
    team_size: usize,
    thread_block_size: usize,
    hashmap_max_fill_rate: f32,
    hashmap_min_bitlen: usize,
    num_random_samplings: u32,
    rand_xor_mask: u64,
    algo: Option<SearchAlgo>,
    hashmap_mode: Option<HashMode>,
}

impl Default for SearchParamsBuilder {
    fn default() -> Self {
        Self {
            itopk_size: 64,
            max_queries: 0,
            max_iterations: 0,
            min_iterations: 0,
            team_size: 0,
            thread_block_size: 0,
            hashmap_max_fill_rate: 0.5,
            hashmap_min_bitlen: 0,
            num_random_samplings: 1,
            rand_xor_mask: 0x128394,
            algo: None,
            hashmap_mode: None,
        }
    }
}

impl SearchParamsBuilder {
    /// Number of intermediate search results retained during the search.
    ///
    /// Must be a power of 2 (or 0 to use the cuVS default).
    pub fn itopk_size(mut self, v: usize) -> Self {
        self.itopk_size = v;
        self
    }

    /// Maximum number of queries to search at the same time. 0 = auto.
    pub fn max_queries(mut self, v: usize) -> Self {
        self.max_queries = v;
        self
    }

    /// Upper limit of search iterations. 0 = auto.
    pub fn max_iterations(mut self, v: usize) -> Self {
        self.max_iterations = v;
        self
    }

    /// Lower limit of search iterations.
    pub fn min_iterations(mut self, v: usize) -> Self {
        self.min_iterations = v;
        self
    }

    /// Number of threads used to calculate a single distance.
    ///
    /// Must be 0 (auto), 4, 8, 16, or 32.
    pub fn team_size(mut self, v: usize) -> Self {
        self.team_size = v;
        self
    }

    /// Thread block size. 0 (auto), 64, 128, 256, 512, or 1024.
    pub fn thread_block_size(mut self, v: usize) -> Self {
        self.thread_block_size = v;
        self
    }

    /// Upper limit of hashmap fill rate.
    ///
    /// Must be in the exclusive range (0.1, 0.9).
    pub fn hashmap_max_fill_rate(mut self, v: f32) -> Self {
        self.hashmap_max_fill_rate = v;
        self
    }

    /// Lower limit of hashmap bit length.
    pub fn hashmap_min_bitlen(mut self, v: usize) -> Self {
        self.hashmap_min_bitlen = v;
        self
    }

    /// Number of iterations of initial random seed node selection.
    pub fn num_random_samplings(mut self, v: u32) -> Self {
        self.num_random_samplings = v;
        self
    }

    /// Bit mask used for initial random seed node selection.
    pub fn rand_xor_mask(mut self, v: u64) -> Self {
        self.rand_xor_mask = v;
        self
    }

    /// Which search implementation to use.
    pub fn algo(mut self, v: SearchAlgo) -> Self {
        self.algo = Some(v);
        self
    }

    /// Hashmap type.
    pub fn hashmap_mode(mut self, v: HashMode) -> Self {
        self.hashmap_mode = Some(v);
        self
    }

    /// Validate all parameters without allocating any GPU resources.
    ///
    /// Returns `Ok(())` if all parameters are valid, or `Err` with a message naming
    /// the offending field and its valid range.
    pub fn validate(&self) -> crate::error::Result<()> {
        if self.itopk_size != 0 && !self.itopk_size.is_power_of_two() {
            return Err(format!(
                "itopk_size must be a power of 2 or 0 (auto); got {}",
                self.itopk_size
            )
            .into());
        }
        const VALID_TEAM_SIZES: &[usize] = &[0, 4, 8, 16, 32];
        if !VALID_TEAM_SIZES.contains(&self.team_size) {
            return Err(format!(
                "team_size must be one of {{0, 4, 8, 16, 32}}; got {}",
                self.team_size
            )
            .into());
        }
        if self.hashmap_max_fill_rate <= 0.1 || self.hashmap_max_fill_rate >= 0.9 {
            return Err(format!(
                "hashmap_max_fill_rate must be in (0.1, 0.9); got {}",
                self.hashmap_max_fill_rate
            )
            .into());
        }
        Ok(())
    }

    /// Validate all parameters and allocate the FFI struct.
    ///
    /// Returns `Err` with a message naming the offending field and its valid range
    /// before any GPU work begins.
    pub fn build(self) -> crate::error::Result<SearchParams> {
        self.validate()?;
        let mut params = SearchParams::new()?
            .set_itopk_size(self.itopk_size)
            .set_max_queries(self.max_queries)
            .set_max_iterations(self.max_iterations)
            .set_min_iterations(self.min_iterations)
            .set_team_size(self.team_size)
            .set_thread_block_size(self.thread_block_size)
            .set_hashmap_max_fill_rate(self.hashmap_max_fill_rate)
            .set_hashmap_min_bitlen(self.hashmap_min_bitlen)
            .set_num_random_samplings(self.num_random_samplings)
            .set_rand_xor_mask(self.rand_xor_mask);
        if let Some(algo) = self.algo {
            params = params.set_algo(algo);
        }
        if let Some(mode) = self.hashmap_mode {
            params = params.set_hashmap_mode(mode);
        }
        Ok(params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_params() {
        let params = SearchParams::new().unwrap().set_itopk_size(128);

        unsafe {
            assert_eq!((*params.0).itopk_size, 128);
        }
    }

    // --- SearchParamsBuilder tests ---

    #[test]
    fn builder_rejects_non_power_of_two_itopk() {
        let err = SearchParams::builder()
            .itopk_size(100)
            .validate()
            .unwrap_err();
        assert!(
            err.to_string().contains("itopk_size"),
            "error message should name the field: {err}"
        );
    }

    #[test]
    fn builder_rejects_invalid_team_size() {
        let err = SearchParams::builder().team_size(7).validate().unwrap_err();
        assert!(
            err.to_string().contains("team_size"),
            "error message should name the field: {err}"
        );
    }

    #[test]
    fn builder_rejects_fill_rate_too_high() {
        let err = SearchParams::builder()
            .hashmap_max_fill_rate(0.95)
            .validate()
            .unwrap_err();
        assert!(
            err.to_string().contains("hashmap_max_fill_rate"),
            "error message should name the field: {err}"
        );
    }

    #[test]
    fn builder_rejects_fill_rate_too_low() {
        let err = SearchParams::builder()
            .hashmap_max_fill_rate(0.05)
            .validate()
            .unwrap_err();
        assert!(
            err.to_string().contains("hashmap_max_fill_rate"),
            "error message should name the field: {err}"
        );
    }

    #[test]
    fn builder_accepts_valid_params() {
        assert!(SearchParams::builder()
            .itopk_size(64)
            .team_size(8)
            .hashmap_max_fill_rate(0.5)
            .validate()
            .is_ok());
    }

    #[test]
    fn builder_accepts_zero_itopk_as_auto() {
        // itopk_size=0 means "auto select" in cuVS — should be valid
        assert!(SearchParams::builder().itopk_size(0).validate().is_ok());
    }
}
