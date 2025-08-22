/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

//! Kmeans clustering API's
//!
//! Example:
//! ```
//!
//! use cuvs::cluster::kmeans;
//! use cuvs::{ManagedTensor, Resources, Result};
//!
//! use ndarray_rand::rand_distr::Uniform;
//! use ndarray_rand::RandomExt;
//!
//! fn kmeans_example() -> Result<()> {
//!     let res = Resources::new()?;
//!
//!     // Create a new random dataset to index
//!     let n_datapoints = 65536;
//!     let n_features = 512;
//!     let n_clusters = 8;
//!     let dataset =
//!         ndarray::Array::<f32, _>::random((n_datapoints, n_features), Uniform::new(0., 1.0));
//!     let dataset = ManagedTensor::from(&dataset).to_device(&res)?;
//!
//!     let centroids_host = ndarray::Array::<f32, _>::zeros((n_clusters, n_features));
//!     let mut centroids = ManagedTensor::from(&centroids_host).to_device(&res)?;
//!
//!     // find the centroids with the kmeans index
//!     let kmeans_params = kmeans::Params::new()?.set_n_clusters(n_clusters as i32);
//!     let (inertia, n_iter) = kmeans::fit(&res, &kmeans_params, &dataset, &None, &mut centroids)?;
//!     Ok(())
//! }
//! ```

mod params;

pub use params::Params;

use crate::dlpack::ManagedTensor;
use crate::error::{check_cuvs, Result};
use crate::resources::Resources;

/// Find clusters with the k-means algorithm
///
/// # Arguments
///
/// * `res` - Resources to use
/// * `params` - Parameters to use to fit KMeans model
/// * `x` - A matrix in device memory - shape (m, k)
/// * `sample_weight` - Optional device matrix shape (n_clusters, 1)
/// * `centroids` - Output device matrix, that has the centroids for each cluster
///   shape (n_clusters, k)
pub fn fit(
    res: &Resources,
    params: &Params,
    x: &ManagedTensor,
    sample_weight: &Option<ManagedTensor>,
    centroids: &mut ManagedTensor,
) -> Result<(f64, i32)> {
    let mut inertia: f64 = 0.0;
    let mut niter: i32 = 0;

    unsafe {
        let sample_weight_dlpack = match sample_weight {
            Some(tensor) => tensor.as_ptr(),
            None => std::ptr::null_mut(),
        };
        check_cuvs(ffi::cuvsKMeansFit(
            res.0,
            params.0,
            x.as_ptr(),
            sample_weight_dlpack,
            centroids.as_ptr(),
            &mut inertia as *mut f64,
            &mut niter as *mut i32,
        ))?;
    }
    Ok((inertia, niter))
}

/// Predict clusters with the k-means algorithm
///
/// # Arguments
///
/// * `res` - Resources to use
/// * `params` - Parameters to use to fit KMeans model
/// * `x` - Input matrix in device memory - shape (m, k)
/// * `sample_weight` - Optional device matrix shape (n_clusters, 1)
/// * `centroids` - Centroids calculated by fit in device memory, shape (n_clusters, k)
/// * `labels` - preallocated CUDA array interface matrix shape (m, 1) to hold the output labels
/// * `normalize_weight` - whether or not to normalize the weights
pub fn predict(
    res: &Resources,
    params: &Params,
    x: &ManagedTensor,
    sample_weight: &Option<ManagedTensor>,
    centroids: &ManagedTensor,
    labels: &mut ManagedTensor,
    normalize_weight: bool,
) -> Result<f64> {
    let mut inertia: f64 = 0.0;

    unsafe {
        let sample_weight_dlpack = match sample_weight {
            Some(tensor) => tensor.as_ptr(),
            None => std::ptr::null_mut(),
        };
        check_cuvs(ffi::cuvsKMeansPredict(
            res.0,
            params.0,
            x.as_ptr(),
            sample_weight_dlpack,
            centroids.as_ptr(),
            labels.as_ptr(),
            normalize_weight,
            &mut inertia as *mut f64,
        ))?;
    }
    Ok(inertia)
}

/// Compute cluster cost given an input matrix and existing centroids
/// # Arguments
///
/// * `res` - Resources to use
/// * `x` - Input matrix in device memory - shape (m, k)
/// * `centroids` - Centroids calculated by fit in device memory, shape (n_clusters, k)
pub fn cluster_cost(res: &Resources, x: &ManagedTensor, centroids: &ManagedTensor) -> Result<f64> {
    let mut inertia: f64 = 0.0;

    unsafe {
        check_cuvs(ffi::cuvsKMeansClusterCost(
            res.0,
            x.as_ptr(),
            centroids.as_ptr(),
            &mut inertia as *mut f64,
        ))?;
    }
    Ok(inertia)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    #[test]
    fn test_kmeans() {
        let res = Resources::new().unwrap();

        let n_clusters = 4;

        // Create a new random dataset to index
        let n_datapoints = 256;
        let n_features = 16;
        let dataset =
            ndarray::Array::<f32, _>::random((n_datapoints, n_features), Uniform::new(0., 1.0));
        let dataset = ManagedTensor::from(&dataset).to_device(&res).unwrap();

        let centroids_host = ndarray::Array::<f32, _>::zeros((n_clusters, n_features));
        let mut centroids = ManagedTensor::from(&centroids_host)
            .to_device(&res)
            .unwrap();

        let params = Params::new().unwrap().set_n_clusters(n_clusters as i32);

        // compute the inertia, before fitting centroids
        let original_inertia = cluster_cost(&res, &dataset, &centroids).unwrap();

        // fit the centroids, make sure that inertia has gone down
        let (inertia, n_iter) = fit(&res, &params, &dataset, &None, &mut centroids).unwrap();

        assert!(inertia < original_inertia);
        assert!(n_iter >= 1);

        let mut labels_host = ndarray::Array::<i32, _>::zeros((n_clusters,));
        let mut labels = ManagedTensor::from(&labels_host).to_device(&res).unwrap();

        // make sure the prediction for each centroid is the centroid itself
        predict(
            &res,
            &params,
            &centroids,
            &None,
            &centroids,
            &mut labels,
            false,
        )
        .unwrap();

        labels.to_host(&res, &mut labels_host).unwrap();
        assert_eq!(labels_host[[0,]], 0);
        assert_eq!(labels_host[[1,]], 1);
        assert_eq!(labels_host[[2,]], 2);
        assert_eq!(labels_host[[3,]], 3);
    }
}
