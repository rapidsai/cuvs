/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! The IVF-SQ method is an ANN algorithm. Like IVF-Flat it uses an inverted file
//! index (IVF), but stores the vectors using scalar quantization (SQ), trading a
//! small amount of accuracy for a substantial reduction in memory footprint.
//! This algorithm provides simple knobs to reduce the overall search space and
//! to trade-off accuracy for speed.
//!
//! Example:
//! ```
//!
//! use cuvs::ivf_sq::{Index, IndexParams, SearchParams};
//! use cuvs::{ManagedTensor, Resources, Result};
//!
//! use ndarray::s;
//! use ndarray_rand::rand_distr::Uniform;
//! use ndarray_rand::RandomExt;
//!
//! fn ivf_sq_example() -> Result<()> {
//!     let res = Resources::new()?;
//!
//!     // Create a new random dataset to index
//!     let n_datapoints = 65536;
//!     let n_features = 512;
//!     let dataset =
//!         ndarray::Array::<f32, _>::random((n_datapoints, n_features), Uniform::new(0., 1.0));
//!
//!     // build the ivf-sq index
//!     let build_params = IndexParams::new()?;
//!     let index = Index::build(&res, &build_params, &dataset)?;
//!     println!(
//!         "Indexed {}x{} datapoints into ivf-sq index",
//!         n_datapoints, n_features
//!     );
//!
//!     // use the first 4 points from the dataset as queries : will test that we get them back
//!     // as their own nearest neighbor
//!     let n_queries = 4;
//!     let queries = dataset.slice(s![0..n_queries, ..]);
//!
//!     let k = 10;
//!
//!     // Ivf-SQ search API requires queries and outputs to be on device memory
//!     // copy query data over, and allocate new device memory for the distances/ neighbors
//!     // outputs
//!     let queries = ManagedTensor::from(&queries).to_device(&res)?;
//!     let mut neighbors_host = ndarray::Array::<i64, _>::zeros((n_queries, k));
//!     let neighbors = ManagedTensor::from(&neighbors_host).to_device(&res)?;
//!
//!     let mut distances_host = ndarray::Array::<f32, _>::zeros((n_queries, k));
//!     let distances = ManagedTensor::from(&distances_host).to_device(&res)?;
//!
//!     let search_params = SearchParams::new()?;
//!
//!     index.search(&res, &search_params, &queries, &neighbors, &distances)?;
//!
//!     // Copy back to host memory
//!     distances.to_host(&res, &mut distances_host)?;
//!     neighbors.to_host(&res, &mut neighbors_host)?;
//!
//!     // nearest neighbors should be themselves, since queries are from the
//!     // dataset
//!     println!("Neighbors {:?}", neighbors_host);
//!     println!("Distances {:?}", distances_host);
//!     Ok(())
//! }
//! ```

mod index;
mod index_params;
mod search_params;

pub use index::Index;
pub use index_params::IndexParams;
pub use search_params::SearchParams;
