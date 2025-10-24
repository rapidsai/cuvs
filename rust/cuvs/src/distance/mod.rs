/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::distance_type::DistanceType;
use crate::dlpack::ManagedTensor;
use crate::error::{check_cuvs, Result};
use crate::resources::Resources;

/// Compute pairwise distances between X and Y
///
/// # Arguments
///
/// * `res` - Resources to use
/// * `x` - A matrix in device memory - shape (m, k)
/// * `y` - A matrix in device memory - shape (n, k)
/// * `distances` - A matrix in device memory that receives the output distances - shape (m, n)
/// * `metric` - DistanceType to use for building the index
/// * `metric_arg` - Optional value of `p` for Minkowski distances
pub fn pairwise_distance(
    res: &Resources,
    x: &ManagedTensor,
    y: &ManagedTensor,
    distances: &ManagedTensor,
    metric: DistanceType,
    metric_arg: Option<f32>,
) -> Result<()> {
    unsafe {
        check_cuvs(ffi::cuvsPairwiseDistance(
            res.0,
            x.as_ptr(),
            y.as_ptr(),
            distances.as_ptr(),
            metric,
            metric_arg.unwrap_or(2.0),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    #[test]
    fn test_pairwise_distance() {
        let res = Resources::new().unwrap();

        // Create a new random dataset to index
        let n_datapoints = 256;
        let n_features = 16;
        let dataset =
            ndarray::Array::<f32, _>::random((n_datapoints, n_features), Uniform::new(0., 1.0));
        let dataset_device = ManagedTensor::from(&dataset).to_device(&res).unwrap();

        let mut distances_host = ndarray::Array::<f32, _>::zeros((n_datapoints, n_datapoints));
        let distances = ManagedTensor::from(&distances_host)
            .to_device(&res)
            .unwrap();

        pairwise_distance(
            &res,
            &dataset_device,
            &dataset_device,
            &distances,
            DistanceType::L2Expanded,
            None,
        )
        .unwrap();

        // Copy back to host memory
        distances.to_host(&res, &mut distances_host).unwrap();

        // Self distance should be 0
        assert_eq!(distances_host[[0, 0]], 0.0);
    }
}
