/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! The tiered index couples a brute-force tier that absorbs incremental inserts
//! with an ANN tier (CAGRA by default; IVF-Flat and IVF-PQ are also available).
//!
//! Vectors added with [`Index::extend`] land in the brute-force tier and are
//! immediately searchable — even before the ANN tier has been (re)built. Once
//! the incremental tier exceeds `min_ann_rows`, the ANN tier is built (or
//! rebuilt on extend when `create_ann_index_on_extend` is set).
//!
//! The C API does not provide serialize/deserialize for the tiered index, so
//! this module does not expose persistence.
//!
//! Example:
//! ```
//!
//! use cuvs::tiered_index::{AnnAlgo, Index, IndexParams};
//! use cuvs::cagra::SearchParams;
//! use cuvs::{ManagedTensor, Resources, Result};
//!
//! use ndarray::s;
//! use ndarray_rand::rand_distr::Uniform;
//! use ndarray_rand::RandomExt;
//!
//! fn tiered_index_example() -> Result<()> {
//!     let res = Resources::new()?;
//!
//!     // Create a new random dataset to index
//!     let n_datapoints = 1024;
//!     let n_features = 16;
//!     let dataset =
//!         ndarray::Array::<f32, _>::random((n_datapoints, n_features), Uniform::new(0., 1.0));
//!
//!     // Build the tiered index, backed by CAGRA for its ANN tier
//!     let build_params = IndexParams::new()?
//!         .set_algo(AnnAlgo::CUVS_TIERED_INDEX_ALGO_CAGRA)
//!         .set_min_ann_rows(128)
//!         .set_create_ann_index_on_extend(true);
//!     let dataset_device = ManagedTensor::from(&dataset).to_device(&res)?;
//!     let index = Index::build(&res, &build_params, dataset_device)?;
//!
//!     // Add new vectors after build: they are immediately searchable.
//!     let new_vectors =
//!         ndarray::Array::<f32, _>::random((8, n_features), Uniform::new(0., 1.0));
//!     let new_device = ManagedTensor::from(&new_vectors).to_device(&res)?;
//!     index.extend(&res, new_device)?;
//!
//!     // Search using the first 4 points from the dataset as queries
//!     let n_queries = 4;
//!     let queries = dataset.slice(s![0..n_queries, ..]);
//!     let queries = ManagedTensor::from(&queries).to_device(&res)?;
//!
//!     let k = 10;
//!     let mut neighbors_host = ndarray::Array::<i64, _>::zeros((n_queries, k));
//!     let neighbors = ManagedTensor::from(&neighbors_host).to_device(&res)?;
//!     let mut distances_host = ndarray::Array::<f32, _>::zeros((n_queries, k));
//!     let distances = ManagedTensor::from(&distances_host).to_device(&res)?;
//!
//!     // CAGRA-backed tiered index searches with CAGRA SearchParams
//!     let search_params = SearchParams::new()?;
//!     index.search(&res, &search_params, &queries, &neighbors, &distances)?;
//!
//!     neighbors.to_host(&res, &mut neighbors_host)?;
//!     distances.to_host(&res, &mut distances_host)?;
//!     println!("Neighbors {:?}", neighbors_host);
//!     println!("Distances {:?}", distances_host);
//!     Ok(())
//! }
//! ```

mod index;
mod index_params;

pub use index::Index;
pub use index_params::{AnnAlgo, IndexParams};
