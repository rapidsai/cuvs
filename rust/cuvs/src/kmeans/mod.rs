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
//! K-Means Clustering

use std::io::{stderr, Write};

use crate::distance_type::DistanceType;
use crate::dlpack::ManagedTensor;
use crate::error::{check_cuvs, Result};
use crate::resources::Resources;

/// K-Means initialization methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InitMethod {
    /// Sample the centroids using the kmeans++ strategy
    KMeansPlusPlus,
    /// Sample the centroids uniformly at random
    Random,
    /// User provides the array of initial centroids
    Array,
}

impl From<InitMethod> for ffi::cuvsKMeansInitMethod {
    fn from(method: InitMethod) -> Self {
        match method {
            InitMethod::KMeansPlusPlus => ffi::cuvsKMeansInitMethod::KMeansPlusPlus,
            InitMethod::Random => ffi::cuvsKMeansInitMethod::Random,
            InitMethod::Array => ffi::cuvsKMeansInitMethod::Array,
        }
    }
}

/// K-Means clustering parameters
#[derive(Debug, Clone)]
pub struct Params {
    /// Distance metric to use for clustering
    pub metric: DistanceType,
    /// Number of clusters to form
    pub n_clusters: i32,
    /// Initialization method
    pub init: InitMethod,
    /// Maximum number of iterations
    pub max_iter: i32,
    /// Relative tolerance for convergence
    pub tol: f64,
    /// Number of times k-means will be run with different seeds
    pub n_init: i32,
    /// Oversampling factor for k-means|| algorithm
    pub oversampling_factor: f64,
    /// Batch size for samples in distance computation
    pub batch_samples: i32,
    /// Batch size for centroids in distance computation
    pub batch_centroids: i32,
    /// Whether to check inertia for convergence
    pub inertia_check: bool,
    /// Whether to use hierarchical (balanced) kmeans
    pub hierarchical: bool,
    /// Number of training iterations for hierarchical k-means
    pub hierarchical_n_iters: i32,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            metric: DistanceType::L2Expanded,
            n_clusters: 8,
            init: InitMethod::KMeansPlusPlus,
            max_iter: 300,
            tol: 1e-4,
            n_init: 1,
            oversampling_factor: 2.0,
            batch_samples: 0,
            batch_centroids: 0,
            inertia_check: true,
            hierarchical: false,
            hierarchical_n_iters: 0,
        }
    }
}

impl Params {
    /// Create a new Params instance with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of clusters
    pub fn n_clusters(mut self, n_clusters: i32) -> Self {
        self.n_clusters = n_clusters;
        self
    }

    /// Set the initialization method
    pub fn init(mut self, init: InitMethod) -> Self {
        self.init = init;
        self
    }

    /// Set the distance metric
    pub fn metric(mut self, metric: DistanceType) -> Self {
        self.metric = metric;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: i32) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the tolerance for convergence
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the number of initializations
    pub fn n_init(mut self, n_init: i32) -> Self {
        self.n_init = n_init;
        self
    }

    /// Set the oversampling factor
    pub fn oversampling_factor(mut self, oversampling_factor: f64) -> Self {
        self.oversampling_factor = oversampling_factor;
        self
    }
}

/// K-Means clustering model
#[derive(Debug)]
pub struct KMeans {
    params: ffi::cuvsKMeansParams_t,
}

impl KMeans {
    /// Create a new K-Means model with the given parameters
    pub fn new(params: Params) -> Result<Self> {
        unsafe {
            let mut c_params = std::mem::MaybeUninit::<ffi::cuvsKMeansParams_t>::uninit();
            check_cuvs(ffi::cuvsKMeansParamsCreate(c_params.as_mut_ptr()))?;
            let mut c_params = c_params.assume_init();
            
            // Set the parameters
            (*c_params).metric = params.metric;
            (*c_params).n_clusters = params.n_clusters;
            (*c_params).init = params.init.into();
            (*c_params).max_iter = params.max_iter;
            (*c_params).tol = params.tol;
            (*c_params).n_init = params.n_init;
            (*c_params).oversampling_factor = params.oversampling_factor;
            (*c_params).batch_samples = params.batch_samples;
            (*c_params).batch_centroids = params.batch_centroids;
            (*c_params).inertia_check = params.inertia_check;
            (*c_params).hierarchical = params.hierarchical;
            (*c_params).hierarchical_n_iters = params.hierarchical_n_iters;
            
            Ok(KMeans { params: c_params })
        }
    }

    /// Fit the K-Means model to the data
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `X` - Training data matrix (n_samples x n_features)
    /// * `sample_weight` - Optional weights for each sample
    /// * `centroids` - Output centroids matrix (n_clusters x n_features)
    /// * `inertia` - Output inertia (sum of squared distances)
    /// * `n_iter` - Output number of iterations performed
    pub fn fit<T: Into<ManagedTensor>>(
        &self,
        res: &Resources,
        X: T,
        sample_weight: Option<T>,
        centroids: &ManagedTensor,
        inertia: &mut f64,
        n_iter: &mut i32,
    ) -> Result<()> {
        let X: ManagedTensor = X.into();
        let sample_weight = sample_weight.map(|sw| sw.into());
        
        unsafe {
            check_cuvs(ffi::cuvsKMeansFit(
                res.0,
                self.params,
                X.as_ptr(),
                sample_weight.as_ref().map(|sw| sw.as_ptr()).unwrap_or(std::ptr::null()),
                centroids.as_ptr(),
                inertia,
                n_iter,
            ))
        }
    }

    /// Predict cluster labels for new data
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `X` - Data matrix to predict (n_samples x n_features)
    /// * `sample_weight` - Optional weights for each sample
    /// * `centroids` - Centroids matrix (n_clusters x n_features)
    /// * `labels` - Output cluster labels (n_samples)
    /// * `normalize_weight` - Whether to normalize weights
    /// * `inertia` - Output inertia
    pub fn predict<T: Into<ManagedTensor>>(
        &self,
        res: &Resources,
        X: T,
        sample_weight: Option<T>,
        centroids: &ManagedTensor,
        labels: &ManagedTensor,
        normalize_weight: bool,
        inertia: &mut f64,
    ) -> Result<()> {
        let X: ManagedTensor = X.into();
        let sample_weight = sample_weight.map(|sw| sw.into());
        
        unsafe {
            check_cuvs(ffi::cuvsKMeansPredict(
                res.0,
                self.params,
                X.as_ptr(),
                sample_weight.as_ref().map(|sw| sw.as_ptr()).unwrap_or(std::ptr::null()),
                centroids.as_ptr(),
                labels.as_ptr(),
                normalize_weight,
                inertia,
            ))
        }
    }

    /// Compute the cluster cost for given data and centroids
    ///
    /// # Arguments
    ///
    /// * `res` - Resources to use
    /// * `X` - Data matrix (n_samples x n_features)
    /// * `centroids` - Centroids matrix (n_clusters x n_features)
    /// * `cost` - Output cluster cost
    pub fn cluster_cost<T: Into<ManagedTensor>>(
        res: &Resources,
        X: T,
        centroids: &ManagedTensor,
        cost: &mut f64,
    ) -> Result<()> {
        let X: ManagedTensor = X.into();
        
        unsafe {
            check_cuvs(ffi::cuvsKMeansClusterCost(
                res.0,
                X.as_ptr(),
                centroids.as_ptr(),
                cost,
            ))
        }
    }
}

impl Drop for KMeans {
    fn drop(&mut self) {
        if let Err(e) = check_cuvs(unsafe { ffi::cuvsKMeansParamsDestroy(self.params) }) {
            write!(stderr(), "failed to call cuvsKMeansParamsDestroy {:?}", e)
                .expect("failed to write to stderr");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::s;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;
    use mark_flaky_tests::flaky;

    #[flaky]
    #[test]
    fn test_kmeans_fit() {
        let res = Resources::new().unwrap();

        // Create a simple dataset with 3 clusters
        let n_samples = 300;
        let n_features = 2;
        let n_clusters = 3;
        
        // Generate data around 3 centers
        let mut dataset_host = ndarray::Array::<f32, _>::zeros((n_samples, n_features));
        
        // Cluster 1: centered around (0, 0)
        for i in 0..100 {
            dataset_host[[i, 0]] = (rand::random::<f32>() - 0.5) * 2.0;
            dataset_host[[i, 1]] = (rand::random::<f32>() - 0.5) * 2.0;
        }
        
        // Cluster 2: centered around (5, 5)
        for i in 100..200 {
            dataset_host[[i, 0]] = 5.0 + (rand::random::<f32>() - 0.5) * 2.0;
            dataset_host[[i, 1]] = 5.0 + (rand::random::<f32>() - 0.5) * 2.0;
        }
        
        // Cluster 3: centered around (10, 0)
        for i in 200..300 {
            dataset_host[[i, 0]] = 10.0 + (rand::random::<f32>() - 0.5) * 2.0;
            dataset_host[[i, 1]] = (rand::random::<f32>() - 0.5) * 2.0;
        }

        let dataset = ManagedTensor::from(&dataset_host).to_device(&res).unwrap();

        // Create kmeans parameters
        let params = Params::new()
            .n_clusters(n_clusters)
            .max_iter(100)
            .tol(1e-4);

        let kmeans = KMeans::new(params).unwrap();

        // Create output tensors
        let mut centroids_host = ndarray::Array::<f32, _>::zeros((n_clusters, n_features));
        let centroids = ManagedTensor::from(&centroids_host).to_device(&res).unwrap();

        let mut inertia = 0.0;
        let mut n_iter = 0;

        // Fit the model
        kmeans.fit(&res, dataset, None::<ManagedTensor>, &centroids, &mut inertia, &mut n_iter).unwrap();

        res.sync_stream().unwrap();

        // Copy centroids back to host
        centroids.to_host(&res, &mut centroids_host).unwrap();
        res.sync_stream().unwrap();

        println!("Fitted kmeans with {} iterations, inertia: {}", n_iter, inertia);
        println!("Centroids: {:?}", centroids_host);

        // Basic sanity checks
        assert!(n_iter > 0);
        assert!(inertia > 0.0);
        assert!(inertia < f64::INFINITY);
    }

    #[flaky]
    #[test]
    fn test_kmeans_predict() {
        let res = Resources::new().unwrap();

        // Create a simple dataset
        let n_samples = 100;
        let n_features = 2;
        let n_clusters = 3;
        
        let dataset_host = ndarray::Array::<f32, _>::random((n_samples, n_features), Uniform::new(0., 10.0));
        let dataset = ManagedTensor::from(&dataset_host).to_device(&res).unwrap();

        // Create kmeans parameters
        let params = Params::new()
            .n_clusters(n_clusters)
            .max_iter(50)
            .tol(1e-4);

        let kmeans = KMeans::new(params).unwrap();

        // Create output tensors for fit
        let mut centroids_host = ndarray::Array::<f32, _>::zeros((n_clusters, n_features));
        let centroids = ManagedTensor::from(&centroids_host).to_device(&res).unwrap();

        let mut inertia = 0.0;
        let mut n_iter = 0;

        // Fit the model
        kmeans.fit(&res, dataset, None::<ManagedTensor>, &centroids, &mut inertia, &mut n_iter).unwrap();

        // Create output tensors for predict
        let mut labels_host = ndarray::Array::<i32, _>::zeros(n_samples);
        let labels = ManagedTensor::from(&labels_host).to_device(&res).unwrap();

        let mut predict_inertia = 0.0;

        // Predict cluster labels
        kmeans.predict(&res, dataset, None::<ManagedTensor>, &centroids, &labels, false, &mut predict_inertia).unwrap();

        res.sync_stream().unwrap();

        // Copy labels back to host
        labels.to_host(&res, &mut labels_host).unwrap();
        res.sync_stream().unwrap();

        println!("Predicted labels: {:?}", labels_host);
        println!("Prediction inertia: {}", predict_inertia);

        // Basic sanity checks
        assert!(predict_inertia > 0.0);
        assert!(predict_inertia < f64::INFINITY);
        
        // Check that all labels are valid cluster indices
        for &label in labels_host.iter() {
            assert!(label >= 0 && label < n_clusters);
        }
    }
} 