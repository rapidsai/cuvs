/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! K-means clustering.
//!
//! [`fit`] computes cluster centroids for a dataset, [`predict`] assigns points
//! to clusters, and [`cluster_cost`] reports the inertia. All inputs and outputs
//! reside in device memory and are passed through the
//! [`IntoDlTensor`](crate::IntoDlTensor) /
//! [`IntoDlTensorMut`](crate::IntoDlTensorMut) traits; see the
//! [`dlpack`](crate::dlpack) module for the tensor model.

mod params;

pub use params::Params;

use crate::dlpack::{IntoDlTensor, IntoDlTensorMut};
use crate::error::{Result, check_cuvs};
use crate::resources::Resources;

/// Fits k-means centroids to `x`, returning `(inertia, n_iterations)`.
///
/// `x` (shape `m × k`) is the input matrix and `centroids` (shape
/// `n_clusters × k`) receives the fitted centroids; `sample_weight` is an
/// optional per-sample weight. All reside in device memory and implement
/// [`IntoDlTensor`](crate::IntoDlTensor) /
/// [`IntoDlTensorMut`](crate::IntoDlTensorMut).
pub fn fit<'a>(
    res: &Resources,
    params: &Params,
    x: impl IntoDlTensor<'a>,
    sample_weight: Option<impl IntoDlTensor<'a>>,
    centroids: impl IntoDlTensorMut<'a>,
) -> Result<(f64, i32)> {
    let x = x.into_dl_tensor()?;
    let sample_weight = sample_weight.map(|w| w.into_dl_tensor()).transpose()?;
    let centroids = centroids.into_dl_tensor_mut()?;
    let mut inertia: f64 = 0.0;
    let mut niter: i32 = 0;
    let mut sample_weight_c = sample_weight.as_ref().map(|w| w.to_c());
    let sample_weight_ptr =
        sample_weight_c.as_mut().map(|w| w.as_mut_ptr()).unwrap_or(std::ptr::null_mut());

    unsafe {
        check_cuvs(ffi::cuvsKMeansFit(
            res.0,
            params.0,
            x.to_c().as_mut_ptr(),
            sample_weight_ptr,
            centroids.to_c().as_mut_ptr(),
            &mut inertia as *mut f64,
            &mut niter as *mut i32,
        ))?;
    }
    Ok((inertia, niter))
}

/// Assigns each row of `x` to its nearest centroid, writing cluster labels into
/// `labels` and returning the inertia.
///
/// `x` (shape `m × k`), `centroids` (shape `n_clusters × k`), the optional
/// `sample_weight`, and `labels` (shape `m × 1`) reside in device memory and
/// implement [`IntoDlTensor`](crate::IntoDlTensor) /
/// [`IntoDlTensorMut`](crate::IntoDlTensorMut). `normalize_weight` selects
/// whether the sample weights are normalized.
pub fn predict<'a>(
    res: &Resources,
    params: &Params,
    x: impl IntoDlTensor<'a>,
    sample_weight: Option<impl IntoDlTensor<'a>>,
    centroids: impl IntoDlTensor<'a>,
    labels: impl IntoDlTensorMut<'a>,
    normalize_weight: bool,
) -> Result<f64> {
    let x = x.into_dl_tensor()?;
    let sample_weight = sample_weight.map(|w| w.into_dl_tensor()).transpose()?;
    let centroids = centroids.into_dl_tensor()?;
    let labels = labels.into_dl_tensor_mut()?;
    let mut inertia: f64 = 0.0;
    let mut sample_weight_c = sample_weight.as_ref().map(|w| w.to_c());
    let sample_weight_ptr =
        sample_weight_c.as_mut().map(|w| w.as_mut_ptr()).unwrap_or(std::ptr::null_mut());

    unsafe {
        check_cuvs(ffi::cuvsKMeansPredict(
            res.0,
            params.0,
            x.to_c().as_mut_ptr(),
            sample_weight_ptr,
            centroids.to_c().as_mut_ptr(),
            labels.to_c().as_mut_ptr(),
            normalize_weight,
            &mut inertia as *mut f64,
        ))?;
    }
    Ok(inertia)
}

/// Computes the k-means cost (inertia) of `x` against existing `centroids`.
///
/// `x` (shape `m × k`) and `centroids` (shape `n_clusters × k`) reside in device
/// memory and implement [`IntoDlTensor`](crate::IntoDlTensor).
pub fn cluster_cost<'a>(
    res: &Resources,
    x: impl IntoDlTensor<'a>,
    centroids: impl IntoDlTensor<'a>,
) -> Result<f64> {
    let x = x.into_dl_tensor()?;
    let centroids = centroids.into_dl_tensor()?;
    let mut inertia: f64 = 0.0;

    unsafe {
        check_cuvs(ffi::cuvsKMeansClusterCost(
            res.0,
            x.to_c().as_mut_ptr(),
            centroids.to_c().as_mut_ptr(),
            &mut inertia as *mut f64,
        ))?;
    }
    Ok(inertia)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::DeviceTensor;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;

    #[test]
    fn test_kmeans() {
        let res = Resources::new().unwrap();

        let n_clusters = 4;

        // Create a new random dataset to index
        let n_datapoints = 256;
        let n_features = 16;
        let dataset_host = ndarray::Array::<f32, _>::random(
            (n_datapoints, n_features),
            Uniform::new(0., 1.0).unwrap(),
        );
        let dataset = DeviceTensor::from_host(&res, &dataset_host).unwrap();

        let centroids_host = ndarray::Array::<f32, _>::zeros((n_clusters, n_features));
        let mut centroids = DeviceTensor::from_host(&res, &centroids_host).unwrap();

        let params = Params::new().unwrap().set_n_clusters(n_clusters as i32);

        // compute the inertia, before fitting centroids
        let original_inertia = cluster_cost(&res, &dataset, &centroids).unwrap();

        // fit the centroids, make sure that inertia has gone down
        let (inertia, n_iter) =
            fit(&res, &params, &dataset, None::<&DeviceTensor<f32>>, &mut centroids).unwrap();

        assert!(inertia < original_inertia);
        assert!(n_iter >= 1);

        let mut labels_host = ndarray::Array::<i32, _>::zeros((n_clusters,));
        let mut labels = DeviceTensor::<i32>::zeros(&res, &[n_clusters]).unwrap();

        // make sure the prediction for each centroid is the centroid itself
        predict(
            &res,
            &params,
            &centroids,
            None::<&DeviceTensor<f32>>,
            &centroids,
            &mut labels,
            false,
        )
        .unwrap();

        labels.copy_to_host(&res, &mut labels_host).unwrap();
        assert_eq!(labels_host[[0,]], 0);
        assert_eq!(labels_host[[1,]], 1);
        assert_eq!(labels_host[[2,]], 2);
        assert_eq!(labels_host[[3,]], 3);
    }
}
