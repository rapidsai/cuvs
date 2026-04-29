/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::ffi::CString;
use std::io::{stderr, Write};
use std::marker::PhantomData;
use std::path::Path;

use crate::cagra::{IndexParams, SearchParams};
use crate::dlpack::ManagedTensor;
use crate::error::{check_cuvs, Error, Result};
use crate::resources::Resources;

/// CAGRA ANN Index.
///
/// CAGRA's behavior is runtime-dependent: the underlying C++ helper
/// `make_aligned_dataset()` stores a non-owning view of the dataset when the
/// input is device-accessible, row-major and 16-byte aligned, and copies
/// otherwise. Because the non-owning path can't be ruled out at compile time,
/// the returned index carries the lifetime of the `ManagedTensor` used to
/// build it, so the borrow checker prevents use-after-free regardless of which
/// path the C++ side takes.
///
/// Indices produced by [`Index::deserialize`] own their data and carry a
/// `'static` lifetime, since the serialized file is fully read into memory
/// managed by the C++ library.
///
/// # Example
///
/// ```no_run
/// # use cuvs::{ManagedTensor, Resources};
/// # use cuvs::cagra::{Index, IndexParams};
/// let res = Resources::new().unwrap();
/// let arr = ndarray::Array::<f32, _>::zeros((256, 16));
/// let params = IndexParams::new().unwrap();
/// let tensor = ManagedTensor::from(&arr);
/// let index = Index::build(&res, &params, &tensor).unwrap();
/// // `arr` and `tensor` must outlive `index`.
/// ```
#[derive(Debug)]
pub struct Index<'a> {
    inner: ffi::cuvsCagraIndex_t,
    _dataset: PhantomData<&'a ()>,
}

/// Convert a filesystem path into a `CString` suitable for the cuVS C API,
/// returning `Error::InvalidArgument` instead of panicking for paths that are
/// not valid UTF-8 or that contain an interior NUL byte.
fn path_to_cstring(path: &Path) -> Result<CString> {
    let path_str = path
        .to_str()
        .ok_or_else(|| Error::InvalidArgument(format!("path is not valid UTF-8: {path:?}")))?;
    CString::new(path_str)
        .map_err(|e| Error::InvalidArgument(format!("path contains an interior NUL byte: {e}")))
}

impl<'a> Index<'a> {
    /// Creates a new empty FFI index handle.
    fn create_handle() -> Result<ffi::cuvsCagraIndex_t> {
        unsafe {
            let mut index = std::mem::MaybeUninit::<ffi::cuvsCagraIndex_t>::uninit();
            check_cuvs(ffi::cuvsCagraIndexCreate(index.as_mut_ptr()))?;
            Ok(index.assume_init())
        }
    }

    /// Builds a new CAGRA Index from `dataset`.
    ///
    /// The compiler enforces that `dataset` outlives the returned index, so
    /// the C++ library's internal view (when it takes the non-owning path)
    /// can never dangle.
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `params` - Parameters for building the index
    /// * `dataset` - A row-major matrix on either the host or device to index
    pub fn build(
        res: &Resources,
        params: &IndexParams,
        dataset: &'a ManagedTensor,
    ) -> Result<Index<'a>> {
        let inner = Self::create_handle()?;
        unsafe {
            check_cuvs(ffi::cuvsCagraBuild(
                res.0,
                params.0,
                dataset.as_ptr(),
                inner,
            ))?;
        }
        Ok(Index {
            inner,
            _dataset: PhantomData,
        })
    }

    /// Perform an Approximate Nearest Neighbors search on the Index.
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `params` - Parameters to use in searching the index
    /// * `queries` - A matrix in device memory to query for
    /// * `neighbors` - Matrix in device memory that receives the indices of the nearest neighbors
    /// * `distances` - Matrix in device memory that receives the distances of the nearest neighbors
    pub fn search(
        &self,
        res: &Resources,
        params: &SearchParams,
        queries: &ManagedTensor,
        neighbors: &ManagedTensor,
        distances: &ManagedTensor,
    ) -> Result<()> {
        unsafe {
            let prefilter = ffi::cuvsFilter {
                addr: 0,
                type_: ffi::cuvsFilterType::NO_FILTER,
            };

            check_cuvs(ffi::cuvsCagraSearch(
                res.0,
                params.0,
                self.inner,
                queries.as_ptr(),
                neighbors.as_ptr(),
                distances.as_ptr(),
                prefilter,
            ))
        }
    }

    /// Save the CAGRA index to file.
    ///
    /// Experimental, both the API and the serialization format are subject to change.
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `filename` - The file path for saving the index
    /// * `include_dataset` - Whether to write out the dataset to the file
    pub fn serialize<P: AsRef<Path>>(
        &self,
        res: &Resources,
        filename: P,
        include_dataset: bool,
    ) -> Result<()> {
        let c_filename = path_to_cstring(filename.as_ref())?;
        unsafe {
            check_cuvs(ffi::cuvsCagraSerialize(
                res.0,
                c_filename.as_ptr(),
                self.inner,
                include_dataset,
            ))
        }
    }

    /// Save the CAGRA index to file in hnswlib format.
    ///
    /// NOTE: The saved index can only be read by the hnswlib wrapper in cuVS,
    /// as the serialization format is not compatible with the original hnswlib.
    ///
    /// Experimental, both the API and the serialization format are subject to change.
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `filename` - The file path for saving the index
    pub fn serialize_to_hnswlib<P: AsRef<Path>>(&self, res: &Resources, filename: P) -> Result<()> {
        let c_filename = path_to_cstring(filename.as_ref())?;
        unsafe {
            check_cuvs(ffi::cuvsCagraSerializeToHnswlib(
                res.0,
                c_filename.as_ptr(),
                self.inner,
            ))
        }
    }
}

impl Index<'static> {
    /// Load a CAGRA index from file.
    ///
    /// The returned index owns the data read from disk, so it has a `'static`
    /// lifetime and isn't tied to any input `ManagedTensor`.
    ///
    /// Experimental, both the API and the serialization format are subject to change.
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `filename` - The path of the file that stores the index
    pub fn deserialize<P: AsRef<Path>>(res: &Resources, filename: P) -> Result<Index<'static>> {
        let c_filename = path_to_cstring(filename.as_ref())?;
        let inner = Self::create_handle()?;
        unsafe {
            check_cuvs(ffi::cuvsCagraDeserialize(res.0, c_filename.as_ptr(), inner))?;
        }
        Ok(Index {
            inner,
            _dataset: PhantomData,
        })
    }
}

impl Drop for Index<'_> {
    fn drop(&mut self) {
        if let Err(e) = check_cuvs(unsafe { ffi::cuvsCagraIndexDestroy(self.inner) }) {
            write!(stderr(), "failed to call cagraIndexDestroy {:?}", e)
                .expect("failed to write to stderr");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{s, Array2};
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    const N_DATAPOINTS: usize = 256;
    const N_FEATURES: usize = 16;

    /// Build a random host dataset for the CAGRA tests.
    fn make_dataset() -> Array2<f32> {
        ndarray::Array::<f32, _>::random((N_DATAPOINTS, N_FEATURES), Uniform::new(0., 1.0))
    }

    /// Run a self-neighbor search against `index` using the first `n_queries`
    /// rows of `dataset` and assert each query finds itself as its top-1
    /// neighbor. CAGRA search requires queries and outputs on device memory.
    fn search_and_verify_self_neighbors(
        res: &Resources,
        index: &Index<'_>,
        dataset: &Array2<f32>,
        n_queries: usize,
        k: usize,
    ) {
        let queries = dataset.slice(s![0..n_queries, ..]);
        let queries = ManagedTensor::from(&queries).to_device(res).unwrap();

        let mut neighbors_host = ndarray::Array::<u32, _>::zeros((n_queries, k));
        let neighbors = ManagedTensor::from(&neighbors_host).to_device(res).unwrap();

        let mut distances_host = ndarray::Array::<f32, _>::zeros((n_queries, k));
        let distances = ManagedTensor::from(&distances_host).to_device(res).unwrap();

        let search_params = SearchParams::new().unwrap();
        index
            .search(res, &search_params, &queries, &neighbors, &distances)
            .expect("search failed");

        distances.to_host(res, &mut distances_host).unwrap();
        neighbors.to_host(res, &mut neighbors_host).unwrap();

        for i in 0..n_queries {
            assert_eq!(
                neighbors_host[[i, 0]],
                i as u32,
                "query {i} should be its own nearest neighbor"
            );
        }
    }

    fn test_cagra(build_params: IndexParams) {
        let res = Resources::new().unwrap();
        let dataset = make_dataset();
        let tensor = ManagedTensor::from(&dataset);
        let index =
            Index::build(&res, &build_params, &tensor).expect("failed to build cagra index");
        search_and_verify_self_neighbors(&res, &index, &dataset, 4, 10);
    }

    #[test]
    fn test_cagra_index() {
        let build_params = IndexParams::new().unwrap();
        test_cagra(build_params);
    }

    #[test]
    fn test_cagra_compression() {
        use crate::cagra::CompressionParams;
        let build_params = IndexParams::new()
            .unwrap()
            .set_compression(CompressionParams::new().unwrap());
        test_cagra(build_params);
    }

    /// Validates that an index built against a borrowed `ManagedTensor` can be
    /// searched multiple times (covers the `search(&self, …)` contract).
    #[test]
    fn test_cagra_multiple_searches() {
        let res = Resources::new().unwrap();
        let build_params = IndexParams::new().unwrap();
        let dataset = make_dataset();
        let tensor = ManagedTensor::from(&dataset);
        let index =
            Index::build(&res, &build_params, &tensor).expect("failed to build cagra index");

        for _ in 0..3 {
            search_and_verify_self_neighbors(&res, &index, &dataset, 4, 5);
        }
    }

    #[test]
    fn test_cagra_serialize_deserialize() {
        let res = Resources::new().unwrap();
        let build_params = IndexParams::new().unwrap();
        let dataset = make_dataset();
        let tensor = ManagedTensor::from(&dataset);
        let index =
            Index::build(&res, &build_params, &tensor).expect("failed to build cagra index");

        let filepath = std::env::temp_dir().join("test_cagra_index.bin");
        index
            .serialize(&res, &filepath, true)
            .expect("failed to serialize cagra index");

        assert!(filepath.exists(), "serialized index file should exist");
        assert!(
            std::fs::metadata(&filepath).unwrap().len() > 0,
            "serialized index file should not be empty"
        );

        // `deserialize` returns `Index<'static>` — the file owns its data, so
        // it can safely outlive any input `ManagedTensor`.
        let loaded_index: Index<'static> =
            Index::deserialize(&res, &filepath).expect("failed to deserialize cagra index");
        search_and_verify_self_neighbors(&res, &loaded_index, &dataset, 4, 10);

        let _ = std::fs::remove_file(&filepath);
    }

    #[test]
    fn test_cagra_serialize_without_dataset() {
        let res = Resources::new().unwrap();
        let build_params = IndexParams::new().unwrap();
        let dataset = make_dataset();
        let tensor = ManagedTensor::from(&dataset);
        let index =
            Index::build(&res, &build_params, &tensor).expect("failed to build cagra index");

        let filepath = std::env::temp_dir().join("test_cagra_index_no_dataset.bin");
        index
            .serialize(&res, &filepath, false)
            .expect("failed to serialize cagra index without dataset");

        assert!(filepath.exists(), "serialized index file should exist");

        let _ = std::fs::remove_file(&filepath);
    }

    #[test]
    fn test_cagra_serialize_to_hnswlib() {
        let res = Resources::new().unwrap();
        let build_params = IndexParams::new().unwrap();
        let dataset = make_dataset();
        let tensor = ManagedTensor::from(&dataset);
        let index =
            Index::build(&res, &build_params, &tensor).expect("failed to build cagra index");

        let filepath = std::env::temp_dir().join("test_cagra_index_hnsw.bin");
        index
            .serialize_to_hnswlib(&res, &filepath)
            .expect("failed to serialize cagra index to hnswlib format");

        assert!(
            filepath.exists(),
            "serialized hnswlib index file should exist"
        );
        assert!(
            std::fs::metadata(&filepath).unwrap().len() > 0,
            "serialized hnswlib index file should not be empty"
        );

        let _ = std::fs::remove_file(&filepath);
    }

    /// Passing a filename containing an interior NUL byte must surface as an
    /// `InvalidArgument` error rather than panicking inside the serializer.
    #[test]
    fn test_cagra_serialize_rejects_interior_nul() {
        let res = Resources::new().unwrap();
        let build_params = IndexParams::new().unwrap();
        let dataset = make_dataset();
        let tensor = ManagedTensor::from(&dataset);
        let index =
            Index::build(&res, &build_params, &tensor).expect("failed to build cagra index");

        // `PathBuf::from` on Unix preserves arbitrary bytes, so we can embed a
        // NUL byte in the path and confirm the helper rejects it.
        let bad_path = std::path::PathBuf::from("/tmp/has\0nul.bin");
        let err = index
            .serialize(&res, &bad_path, true)
            .expect_err("serialize should reject paths with interior NUL");
        assert!(
            matches!(err, Error::InvalidArgument(_)),
            "expected InvalidArgument, got {err:?}"
        );
    }
}
