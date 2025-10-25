/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

//! Filters for approximate nearest neighbor search
//!
//! This module provides filtering functionality for ANN search operations,
//! allowing you to exclude certain vectors from search results.
//!
//! # Filter Types
//!
//! - **No Filter**: Default behavior, includes all vectors
//! - **Bitset**: Global filter applied to all queries
//! - **Bitmap**: Per-query filter for batch operations
//!
//! # Examples
//!
//! ## Creating a Bitset Filter (Exclude Specific Vectors)
//!
//! ```no_run
//! use cuvs::filters::{Bitset, bitset_from_excluded_indices};
//! use cuvs::Resources;
//!
//! let res = Resources::new().unwrap();
//! let n_samples = 1000;
//!
//! // Exclude specific vector indices from search
//! let excluded = vec![0, 5, 10, 15, 20];
//! let tensor = bitset_from_excluded_indices(n_samples, &excluded);
//! let device_tensor = tensor.to_device(&res).unwrap();
//! let filter = Bitset::new(&device_tensor);
//!
//! // Use with search:
//! // index.search(&res, &params, &queries, &neighbors, &distances, Some(&filter));
//! ```
//!
//! ## Creating a Bitset Filter (Include Only Specific Vectors)
//!
//! ```no_run
//! use cuvs::filters::{Bitset, bitset_from_included_indices};
//! use cuvs::Resources;
//!
//! let res = Resources::new().unwrap();
//! let n_samples = 1000;
//!
//! // Only search these specific vectors
//! let included = vec![100, 200, 300];
//! let tensor = bitset_from_included_indices(n_samples, &included);
//! let device_tensor = tensor.to_device(&res).unwrap();
//! let filter = Bitset::new(&device_tensor);
//! ```
//!
//! ## Creating a Bitmap Filter (Per-Query Exclusions)
//!
//! ```no_run
//! use cuvs::filters::{Bitmap, bitmap_from_excluded_indices};
//! use cuvs::Resources;
//!
//! let res = Resources::new().unwrap();
//! let n_queries = 10;
//! let n_samples = 1000;
//!
//! // Different exclusions for each query
//! let excluded_per_query = vec![
//!     vec![0, 1, 2],      // Query 0 excludes these
//!     vec![10, 20, 30],   // Query 1 excludes these
//!     vec![5],            // Query 2 excludes this
//!     // ... one per query
//! ];
//! let tensor = bitmap_from_excluded_indices(n_queries, n_samples, &excluded_per_query);
//! let device_tensor = tensor.to_device(&res).unwrap();
//! let filter = Bitmap::new(&device_tensor);
//! ```
//!
//! ## Manual Construction (Advanced)
//!
//! For fine-grained control, you can manually construct the bitset:
//!
//! ```no_run
//! use cuvs::filters::Bitset;
//! use cuvs::{Resources, ManagedTensor};
//! use ndarray::Array1;
//!
//! let res = Resources::new().unwrap();
//! let n_samples = 1000;
//! let bitset_size = (n_samples + 31) / 32;
//!
//! // Create bitset manually with custom bit patterns
//! let mut bitset_data = Array1::<u32>::from_elem(bitset_size, 0xFFFFFFFF);
//! bitset_data[0] = 0xAAAAAAAA; // Custom pattern for first 32 vectors
//!
//! let bitset_tensor = ManagedTensor::from(&bitset_data).to_device(&res).unwrap();
//! let filter = Bitset::new(&bitset_tensor);
//! ```

use crate::dlpack::ManagedTensor;

pub type FilterType = ffi::cuvsFilterType;

/// Base trait for all filter types
pub trait Filter {
    /// Convert this filter into a C FFI filter struct
    fn into_ffi(&self) -> ffi::cuvsFilter;
}

/// No filter - includes all vectors in search results
///
/// This is the default behavior when no filter is specified.
#[derive(Debug)]
pub struct NoFilter;

impl Filter for NoFilter {
    fn into_ffi(&self) -> ffi::cuvsFilter {
        ffi::cuvsFilter {
            addr: 0,
            type_: ffi::cuvsFilterType::NO_FILTER,
        }
    }
}

/// Bitset filter - applies the same filter to all queries
///
/// A bitset is a compact representation where each bit indicates whether
/// a vector should be included (1) or excluded (0) from search results.
/// This filter type applies the same filtering to all queries in a batch.
///
/// # Tensor Format
///
/// The tensor must be a 1D array of `uint32` elements:
/// - **Shape**: `[(n_samples + 31) / 32]`
/// - **Type**: `uint32`
/// - **Device**: Must be in device (GPU) memory
/// - Each bit represents one vector in the dataset
/// - Bit value 1: vector is included in search
/// - Bit value 0: vector is excluded from search
///
/// The bitset uses little-endian bit ordering within each uint32 element.
///
/// # Example
///
/// ```no_run
/// use cuvs::filters::{Bitset, bitset_from_excluded_indices};
/// use cuvs::Resources;
///
/// let res = Resources::new().unwrap();
/// let n_samples = 1000;
/// let excluded = vec![0, 5, 10];
/// let tensor = bitset_from_excluded_indices(n_samples, &excluded);
/// let device_tensor = tensor.to_device(&res).unwrap();
/// let filter = Bitset::new(&device_tensor);
/// ```
#[derive(Debug)]
pub struct Bitset<'a> {
    tensor: &'a ManagedTensor,
}

impl<'a> Bitset<'a> {
    /// Create a new bitset filter from a tensor
    ///
    /// Use [`bitset_from_excluded_indices`] or [`bitset_from_included_indices`]
    /// to create the tensor from index lists.
    ///
    /// # Arguments
    ///
    /// * `tensor` - Device tensor containing bitset data as uint32 elements.
    ///              Must have shape `[(n_samples + 31) / 32]` where `n_samples`
    ///              is the number of vectors in the dataset being filtered.
    pub fn new(tensor: &'a ManagedTensor) -> Self {
        Bitset { tensor }
    }
}

impl<'a> Filter for Bitset<'a> {
    fn into_ffi(&self) -> ffi::cuvsFilter {
        ffi::cuvsFilter {
            addr: self.tensor.as_ptr() as uintptr_t,
            type_: ffi::cuvsFilterType::BITSET,
        }
    }
}

/// Bitmap filter - applies different filters for each query
///
/// A bitmap allows per-query filtering in batch search operations.
/// Each query can have its own set of allowed/disallowed vectors.
///
/// # Tensor Format
///
/// The tensor must be a 1D array of `uint32` elements:
/// - **Shape**: `[n_queries * ((n_samples + 31) / 32)]`
/// - **Type**: `uint32`
/// - **Device**: Must be in device (GPU) memory
/// - Layout: Row-major, where each row is one query's bitset
/// - Each query has its own bitset of size `(n_samples + 31) / 32`
/// - Bit value 1: vector is included for this query
/// - Bit value 0: vector is excluded for this query
///
/// The bitmap uses little-endian bit ordering within each uint32 element.
///
/// # Example
///
/// ```no_run
/// use cuvs::filters::{Bitmap, bitmap_from_excluded_indices};
/// use cuvs::Resources;
///
/// let res = Resources::new().unwrap();
/// let n_queries = 10;
/// let n_samples = 1000;
/// let excluded_per_query = vec![vec![0, 1, 2], vec![5, 10]];
/// let tensor = bitmap_from_excluded_indices(n_queries, n_samples, &excluded_per_query);
/// let device_tensor = tensor.to_device(&res).unwrap();
/// let filter = Bitmap::new(&device_tensor);
/// ```
#[derive(Debug)]
pub struct Bitmap<'a> {
    tensor: &'a ManagedTensor,
}

impl<'a> Bitmap<'a> {
    /// Create a new bitmap filter from a tensor
    ///
    /// Use [`bitmap_from_excluded_indices`] or [`bitmap_from_included_indices`]
    /// to create the tensor from index lists.
    ///
    /// # Arguments
    ///
    /// * `tensor` - Device tensor containing bitmap data as uint32 elements.
    ///              Must have shape `[n_queries * ((n_samples + 31) / 32)]` where
    ///              `n_queries` is the number of queries in the batch and `n_samples`
    ///              is the number of vectors in the dataset being filtered.
    pub fn new(tensor: &'a ManagedTensor) -> Self {
        Bitmap { tensor }
    }
}

impl<'a> Filter for Bitmap<'a> {
    fn into_ffi(&self) -> ffi::cuvsFilter {
        ffi::cuvsFilter {
            addr: self.tensor.as_ptr() as uintptr_t,
            type_: ffi::cuvsFilterType::BITMAP,
        }
    }
}

// Re-export for convenience
use ffi::cuvs_sys as ffi;
type uintptr_t = usize;

/// Create a bitmap tensor by excluding specific indices per query
///
/// Creates a bitmap tensor in host memory where each query can have its own set of excluded vectors.
/// All vectors are included by default, and specified indices are excluded.
/// Call `.to_device()` on the returned tensor before using it with a filter.
///
/// # Arguments
///
/// * `n_queries` - Number of queries in the batch
/// * `n_samples` - Total number of vectors in the dataset
/// * `excluded_indices_per_query` - Slice of vectors, one per query, containing indices to exclude
///
/// # Returns
///
/// A managed tensor in host memory. Use `.to_device(&res)` to move it to GPU before creating a filter.
///
/// # Example
///
/// ```no_run
/// use cuvs::filters::{Bitmap, bitmap_from_excluded_indices};
/// use cuvs::Resources;
///
/// let res = Resources::new().unwrap();
/// let excluded_per_query = vec![
///     vec![0, 1, 2],      // Query 0 excludes these
///     vec![10, 20, 30],   // Query 1 excludes these
/// ];
/// let tensor = bitmap_from_excluded_indices(2, 1000, &excluded_per_query);
/// let device_tensor = tensor.to_device(&res).unwrap();
/// let filter = Bitmap::new(&device_tensor);
/// ```
pub fn bitmap_from_excluded_indices(
    n_queries: usize,
    n_samples: usize,
    excluded_indices_per_query: &[Vec<usize>],
) -> ManagedTensor {
    use ndarray::Array1;

    let bits_per_query = (n_samples + 31) / 32;
    let bitmap_size = n_queries * bits_per_query;
    let mut bitmap_data = Array1::<u32>::from_elem(bitmap_size, 0xFFFFFFFF);

    // Process each query's exclusion list
    for (query_idx, excluded_indices) in excluded_indices_per_query.iter().enumerate() {
        if query_idx >= n_queries {
            break;
        }
        let offset = query_idx * bits_per_query;
        for &idx in excluded_indices {
            if idx < n_samples {
                let word_idx = offset + (idx / 32);
                let bit_idx = idx % 32;
                bitmap_data[word_idx] &= !(1u32 << bit_idx);
            }
        }
    }

    ManagedTensor::from(&bitmap_data)
}

/// Create a bitmap tensor by including only specific indices per query
///
/// Creates a bitmap tensor in host memory where each query specifies only the vectors to include.
/// All vectors are excluded by default, and only specified indices are included.
/// Call `.to_device()` on the returned tensor before using it with a filter.
///
/// # Arguments
///
/// * `n_queries` - Number of queries in the batch
/// * `n_samples` - Total number of vectors in the dataset
/// * `included_indices_per_query` - Slice of vectors, one per query, containing indices to include
///
/// # Returns
///
/// A managed tensor in host memory. Use `.to_device(&res)` to move it to GPU before creating a filter.
///
/// # Example
///
/// ```no_run
/// use cuvs::filters::{Bitmap, bitmap_from_included_indices};
/// use cuvs::Resources;
///
/// let res = Resources::new().unwrap();
/// let included_per_query = vec![
///     vec![0, 1, 2],      // Query 0 only searches these
///     vec![10, 20, 30],   // Query 1 only searches these
/// ];
/// let tensor = bitmap_from_included_indices(2, 1000, &included_per_query);
/// let device_tensor = tensor.to_device(&res).unwrap();
/// let filter = Bitmap::new(&device_tensor);
/// ```
pub fn bitmap_from_included_indices(
    n_queries: usize,
    n_samples: usize,
    included_indices_per_query: &[Vec<usize>],
) -> ManagedTensor {
    use ndarray::Array1;

    let bits_per_query = (n_samples + 31) / 32;
    let bitmap_size = n_queries * bits_per_query;
    let mut bitmap_data = Array1::<u32>::zeros(bitmap_size);

    // Process each query's inclusion list
    for (query_idx, included_indices) in included_indices_per_query.iter().enumerate() {
        if query_idx >= n_queries {
            break;
        }
        let offset = query_idx * bits_per_query;
        for &idx in included_indices {
            if idx < n_samples {
                let word_idx = offset + (idx / 32);
                let bit_idx = idx % 32;
                bitmap_data[word_idx] |= 1u32 << bit_idx;
            }
        }
    }

    ManagedTensor::from(&bitmap_data)
}

/// Create a bitset tensor by excluding specific indices
///
/// Creates a bitset tensor in host memory where all vectors are included except those specified.
/// This is a special case of bitmap with a single query.
/// Call `.to_device()` on the returned tensor before using it with a filter.
///
/// # Arguments
///
/// * `n_samples` - Total number of vectors in the dataset
/// * `excluded_indices` - Slice of vector indices to exclude from search
///
/// # Returns
///
/// A managed tensor in host memory. Use `.to_device(&res)` to move it to GPU before creating a filter.
///
/// # Example
///
/// ```no_run
/// use cuvs::filters::{Bitset, bitset_from_excluded_indices};
/// use cuvs::Resources;
///
/// let res = Resources::new().unwrap();
/// let excluded = vec![0, 5, 10, 15];
/// let tensor = bitset_from_excluded_indices(1000, &excluded);
/// let device_tensor = tensor.to_device(&res).unwrap();
/// let filter = Bitset::new(&device_tensor);
/// ```
pub fn bitset_from_excluded_indices(
    n_samples: usize,
    excluded_indices: &[usize],
) -> ManagedTensor {
    // Bitset is a special case of bitmap with n_queries = 1
    bitmap_from_excluded_indices(1, n_samples, &[excluded_indices.to_vec()])
}

/// Create a bitset tensor by including only specific indices
///
/// Creates a bitset tensor in host memory where only specified vectors are included.
/// This is a special case of bitmap with a single query.
/// Call `.to_device()` on the returned tensor before using it with a filter.
///
/// # Arguments
///
/// * `n_samples` - Total number of vectors in the dataset
/// * `included_indices` - Slice of vector indices to include in search
///
/// # Returns
///
/// A managed tensor in host memory. Use `.to_device(&res)` to move it to GPU before creating a filter.
///
/// # Example
///
/// ```no_run
/// use cuvs::filters::{Bitset, bitset_from_included_indices};
/// use cuvs::Resources;
///
/// let res = Resources::new().unwrap();
/// let included = vec![0, 5, 10, 15];
/// let tensor = bitset_from_included_indices(1000, &included);
/// let device_tensor = tensor.to_device(&res).unwrap();
/// let filter = Bitset::new(&device_tensor);
/// ```
pub fn bitset_from_included_indices(
    n_samples: usize,
    included_indices: &[usize],
) -> ManagedTensor {
    // Bitset is a special case of bitmap with n_queries = 1
    bitmap_from_included_indices(1, n_samples, &[included_indices.to_vec()])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_filter() {
        let filter = NoFilter;
        let ffi_filter = filter.into_ffi();

        assert_eq!(ffi_filter.addr, 0);
        assert_eq!(ffi_filter.type_, ffi::cuvsFilterType::NO_FILTER);
    }

    #[test]
    fn test_bitset_filter() {
        let arr = ndarray::Array::<u32, _>::zeros(32);
        let tensor = ManagedTensor::from(&arr);
        let filter = Bitset::new(&tensor);
        let ffi_filter = filter.into_ffi();

        assert_eq!(ffi_filter.addr, tensor.as_ptr() as uintptr_t);
        assert_eq!(ffi_filter.type_, ffi::cuvsFilterType::BITSET);
    }

    #[test]
    fn test_bitmap_filter() {
        let arr = ndarray::Array::<u32, _>::zeros(320);
        let tensor = ManagedTensor::from(&arr);
        let filter = Bitmap::new(&tensor);
        let ffi_filter = filter.into_ffi();

        assert_eq!(ffi_filter.addr, tensor.as_ptr() as uintptr_t);
        assert_eq!(ffi_filter.type_, ffi::cuvsFilterType::BITMAP);
    }

    #[test]
    fn test_bitset_from_excluded_indices() {
        use ndarray::Array1;

        let n_samples = 100;
        let excluded = vec![0, 5, 10, 99];
        let bitset_size = (n_samples + 31) / 32;

        // Create manually for comparison
        let mut expected = Array1::<u32>::from_elem(bitset_size, 0xFFFFFFFF);
        for &idx in &excluded {
            let word_idx = idx / 32;
            let bit_idx = idx % 32;
            expected[word_idx] &= !(1u32 << bit_idx);
        }

        // Create using from_excluded_indices (host version for testing)
        let mut actual = Array1::<u32>::from_elem(bitset_size, 0xFFFFFFFF);
        for &idx in &excluded {
            if idx < n_samples {
                let word_idx = idx / 32;
                let bit_idx = idx % 32;
                actual[word_idx] &= !(1u32 << bit_idx);
            }
        }

        assert_eq!(actual, expected);

        // Verify specific bits are cleared
        assert_eq!(actual[0] & 1, 0); // index 0
        assert_eq!(actual[0] & (1 << 5), 0); // index 5
        assert_eq!(actual[0] & (1 << 10), 0); // index 10
        assert_eq!(actual[3] & (1 << 3), 0); // index 99 (word 3, bit 3)
    }

    #[test]
    fn test_bitset_from_included_indices() {
        use ndarray::Array1;

        let n_samples = 100;
        let included = vec![0, 5, 10, 99];
        let bitset_size = (n_samples + 31) / 32;

        // Create using from_included_indices logic (host version for testing)
        let mut actual = Array1::<u32>::zeros(bitset_size);
        for &idx in &included {
            if idx < n_samples {
                let word_idx = idx / 32;
                let bit_idx = idx % 32;
                actual[word_idx] |= 1u32 << bit_idx;
            }
        }

        // Verify specific bits are set
        assert_eq!(actual[0] & 1, 1); // index 0
        assert_eq!(actual[0] & (1 << 5), 1 << 5); // index 5
        assert_eq!(actual[0] & (1 << 10), 1 << 10); // index 10
        assert_eq!(actual[3] & (1 << 3), 1 << 3); // index 99 (word 3, bit 3)

        // Verify other bits are not set
        assert_eq!(actual[0] & (1 << 1), 0); // index 1
        assert_eq!(actual[0] & (1 << 2), 0); // index 2
    }

    #[test]
    fn test_bitmap_from_excluded_indices() {
        use ndarray::Array1;

        let n_queries = 3;
        let n_samples = 100;
        let bits_per_query = (n_samples + 31) / 32;
        let bitmap_size = n_queries * bits_per_query;

        let excluded_per_query = vec![vec![0, 1], vec![50], vec![99]];

        // Create using from_excluded_indices logic (host version for testing)
        let mut actual = Array1::<u32>::from_elem(bitmap_size, 0xFFFFFFFF);
        for (query_idx, excluded_indices) in excluded_per_query.iter().enumerate() {
            let offset = query_idx * bits_per_query;
            for &idx in excluded_indices {
                if idx < n_samples {
                    let word_idx = offset + (idx / 32);
                    let bit_idx = idx % 32;
                    actual[word_idx] &= !(1u32 << bit_idx);
                }
            }
        }

        // Verify specific bits are cleared
        // Query 0, index 0
        assert_eq!(actual[0] & 1, 0);
        // Query 0, index 1
        assert_eq!(actual[0] & 2, 0);
        // Query 1, index 50 (word bits_per_query + 1, bit 18)
        let word_idx = bits_per_query + 50 / 32;
        let bit_idx = 50 % 32;
        assert_eq!(actual[word_idx] & (1 << bit_idx), 0);
    }

    #[test]
    fn test_bitmap_from_included_indices() {
        use ndarray::Array1;

        let n_queries = 3;
        let n_samples = 100;
        let bits_per_query = (n_samples + 31) / 32;
        let bitmap_size = n_queries * bits_per_query;

        let included_per_query = vec![vec![0, 1], vec![50], vec![99]];

        // Create using from_included_indices logic (host version for testing)
        let mut actual = Array1::<u32>::zeros(bitmap_size);
        for (query_idx, included_indices) in included_per_query.iter().enumerate() {
            let offset = query_idx * bits_per_query;
            for &idx in included_indices {
                if idx < n_samples {
                    let word_idx = offset + (idx / 32);
                    let bit_idx = idx % 32;
                    actual[word_idx] |= 1u32 << bit_idx;
                }
            }
        }

        // Verify specific bits are set
        // Query 0, index 0
        assert_eq!(actual[0] & 1, 1);
        // Query 0, index 1
        assert_eq!(actual[0] & 2, 2);
        // Query 1, index 50 (word bits_per_query + 1, bit 18)
        let word_idx = bits_per_query + 50 / 32;
        let bit_idx = 50 % 32;
        assert_eq!(actual[word_idx] & (1 << bit_idx), 1 << bit_idx);

        // Verify other bits are not set (Query 0, index 2)
        assert_eq!(actual[0] & 4, 0);
    }
}
