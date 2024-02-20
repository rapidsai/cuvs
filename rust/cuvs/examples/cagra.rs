/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

use cuvs::cagra::{IndexParams, SearchParams, Index};
use cuvs::{ManagedTensor, Result, Resources};

use ndarray::s;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

/// Example showing how to index and search data with CAGRA
fn cagra_example() -> Result<()> {
    let res = Resources::new()?;

    // Create a new random dataset to index
    let n_datapoints = 65536;
    let n_features = 512;
    let dataset =
        ndarray::Array::<f32, _>::random((n_datapoints, n_features), Uniform::new(0., 1.0));

    // build the cagra index
    let build_params = IndexParams::new()?;
    let index = Index::build(&res, &build_params, &ManagedTensor::from_ndarray(&dataset))?;
    println!("Indexed {}x{} datapoints into cagra index", n_datapoints, n_features);


    // use the first 4 points from the dataset as queries : will test that we get them back
    // as their own nearest neighbor
    let n_queries = 4;
    let queries = dataset.slice(s![0..n_queries, ..]);

    let k = 10;

    // CAGRA search API requires queries and outputs to be on device memory
    // copy query data over, and allocate new device memory for the distances/ neighbors
    // outputs
    let queries = ManagedTensor::from_ndarray(&queries).to_device()?;
    let mut neighbors_host = ndarray::Array::<u32, _>::zeros((n_queries, k));
    let neighbors = ManagedTensor::from_ndarray(&neighbors_host).to_device()?;

    let mut distances_host = ndarray::Array::<f32, _>::zeros((n_queries, k));
    let distances = ManagedTensor::from_ndarray(&distances_host).to_device()?;

    let search_params = SearchParams::new()?;

    index.search(&res, &search_params, &queries, &neighbors, &distances)?;

    // Copy back to host memory
    distances.to_host(&mut distances_host)?;
    neighbors.to_host(&mut neighbors_host)?;

    // nearest neighbors should be themselves, since queries are from the
    // dataset
    println!("Neighbors {:?}", neighbors_host);
    println!("Distances {:?}", distances_host);
    Ok(())
}

fn main() {
    if let Err(e) = cagra_example() {
        println!("Failed to run CAGRA: {:?}", e);
    }
}
