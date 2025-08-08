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

use cuvs::{kmeans, Resources};
use ndarray::Array;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create resources
    let res = Resources::new()?;

    // Create a simple dataset with 3 clusters
    let n_samples = 300;
    let n_features = 2;
    let n_clusters = 3;
    
    // Generate data around 3 centers
    let mut dataset = Array::<f32, _>::zeros((n_samples, n_features));
    
    // Cluster 1: centered around (0, 0)
    for i in 0..100 {
        dataset[[i, 0]] = (rand::random::<f32>() - 0.5) * 2.0;
        dataset[[i, 1]] = (rand::random::<f32>() - 0.5) * 2.0;
    }
    
    // Cluster 2: centered around (5, 5)
    for i in 100..200 {
        dataset[[i, 0]] = 5.0 + (rand::random::<f32>() - 0.5) * 2.0;
        dataset[[i, 1]] = 5.0 + (rand::random::<f32>() - 0.5) * 2.0;
    }
    
    // Cluster 3: centered around (10, 0)
    for i in 200..300 {
        dataset[[i, 0]] = 10.0 + (rand::random::<f32>() - 0.5) * 2.0;
        dataset[[i, 1]] = (rand::random::<f32>() - 0.5) * 2.0;
    }

    println!("Dataset shape: {:?}", dataset.shape());
    println!("First 10 samples:");
    for i in 0..10 {
        println!("  Sample {}: [{:.3}, {:.3}]", i, dataset[[i, 0]], dataset[[i, 1]]);
    }

    // Convert to device tensor
    let dataset_tensor = cuvs::ManagedTensor::from(&dataset).to_device(&res)?;

    // Create kmeans parameters
    let params = kmeans::Params::new()
        .n_clusters(n_clusters)
        .max_iter(100)
        .tol(1e-4);

    let kmeans_model = kmeans::KMeans::new(params)?;

    // Create output tensors
    let mut centroids_host = Array::<f32, _>::zeros((n_clusters, n_features));
    let centroids_tensor = cuvs::ManagedTensor::from(&centroids_host).to_device(&res)?;

    let mut inertia = 0.0;
    let mut n_iter = 0;

    // Fit the model
    println!("\nFitting kmeans model...");
    kmeans_model.fit(&res, dataset_tensor, None::<cuvs::ManagedTensor>, &centroids_tensor, &mut inertia, &mut n_iter)?;

    res.sync_stream()?;

    // Copy centroids back to host
    centroids_tensor.to_host(&res, &mut centroids_host)?;
    res.sync_stream()?;

    println!("Fitted kmeans with {} iterations, inertia: {:.6}", n_iter, inertia);
    println!("Centroids:");
    for i in 0..n_clusters {
        println!("  Cluster {}: [{:.3}, {:.3}]", i, centroids_host[[i, 0]], centroids_host[[i, 1]]);
    }

    // Predict cluster labels for the same data
    let mut labels_host = Array::<i32, _>::zeros(n_samples);
    let labels_tensor = cuvs::ManagedTensor::from(&labels_host).to_device(&res)?;

    let mut predict_inertia = 0.0;

    // Convert dataset back to device for prediction
    let dataset_tensor = cuvs::ManagedTensor::from(&dataset).to_device(&res)?;

    println!("\nPredicting cluster labels...");
    kmeans_model.predict(&res, dataset_tensor, None::<cuvs::ManagedTensor>, &centroids_tensor, &labels_tensor, false, &mut predict_inertia)?;

    res.sync_stream()?;

    // Copy labels back to host
    labels_tensor.to_host(&res, &mut labels_host)?;
    res.sync_stream()?;

    println!("Prediction inertia: {:.6}", predict_inertia);
    println!("First 20 cluster assignments:");
    for i in 0..20 {
        println!("  Sample {} -> Cluster {}", i, labels_host[i]);
    }

    // Count samples per cluster
    let mut cluster_counts = vec![0; n_clusters];
    for &label in labels_host.iter() {
        cluster_counts[label as usize] += 1;
    }
    println!("\nCluster sizes:");
    for i in 0..n_clusters {
        println!("  Cluster {}: {} samples", i, cluster_counts[i]);
    }

    Ok(())
} 