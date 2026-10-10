/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CI-only stress tests: many host threads, each with its own `Resources`,
//! building and searching indexes concurrently (issue #2233).

use crate::distance::DistanceType;
use crate::neighbors::{brute_force, cagra, ivf_flat, ivf_pq};
use crate::resources::Resources;
use crate::test_utils::DeviceTensor;
use ndarray::s;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn one_round(kind: usize) {
    let res = Resources::new().unwrap();
    let n_datapoints = 1024;
    let n_features = 16;
    let n_queries = 4;
    let k = 10;
    let dataset = ndarray::Array::<f32, _>::random(
        (n_datapoints, n_features),
        Uniform::new(0., 1.0).unwrap(),
    );
    let dataset_device = DeviceTensor::from_host(&res, &dataset).unwrap();
    let queries = dataset.slice(s![0..n_queries, ..]).to_owned();
    let queries = DeviceTensor::from_host(&res, &queries).unwrap();
    let mut distances = DeviceTensor::<f32>::zeros(&res, &[n_queries, k]).unwrap();
    match kind % 4 {
        0 => {
            let bp = ivf_flat::IndexParams::builder().n_lists(64).build().unwrap();
            let index = ivf_flat::Index::build(&res, &bp, &dataset_device).unwrap();
            let sp = ivf_flat::SearchParams::builder().build().unwrap();
            let mut neighbors = DeviceTensor::<i64>::zeros(&res, &[n_queries, k]).unwrap();
            index.search(&res, &sp, &queries, &mut neighbors, &mut distances).unwrap();
            let mut nh = ndarray::Array::<i64, _>::zeros((n_queries, k));
            neighbors.copy_to_host(&res, &mut nh).unwrap();
            assert_eq!(nh[[0, 0]], 0);
        }
        1 => {
            let bp = ivf_pq::IndexParams::builder().n_lists(64).build().unwrap();
            let index = ivf_pq::Index::build(&res, &bp, &dataset_device).unwrap();
            let sp = ivf_pq::SearchParams::builder().build().unwrap();
            let mut neighbors = DeviceTensor::<i64>::zeros(&res, &[n_queries, k]).unwrap();
            index.search(&res, &sp, &queries, &mut neighbors, &mut distances).unwrap();
            let mut nh = ndarray::Array::<i64, _>::zeros((n_queries, k));
            neighbors.copy_to_host(&res, &mut nh).unwrap();
        }
        2 => {
            let bp = cagra::IndexParams::builder().build().unwrap();
            let index = cagra::Index::build(&res, &bp, &dataset_device).unwrap();
            let sp = cagra::SearchParams::builder().build().unwrap();
            let mut neighbors = DeviceTensor::<u32>::zeros(&res, &[n_queries, k]).unwrap();
            index.search(&res, &sp, &queries, &mut neighbors, &mut distances).unwrap();
            let mut nh = ndarray::Array::<u32, _>::zeros((n_queries, k));
            neighbors.copy_to_host(&res, &mut nh).unwrap();
            assert_eq!(nh[[0, 0]], 0);
        }
        _ => {
            let index =
                brute_force::Index::build(&res, DistanceType::L2Expanded, &dataset_device).unwrap();
            let mut neighbors = DeviceTensor::<i64>::zeros(&res, &[n_queries, k]).unwrap();
            index.search(&res, &queries, &mut neighbors, &mut distances).unwrap();
            let mut nh = ndarray::Array::<i64, _>::zeros((n_queries, k));
            neighbors.copy_to_host(&res, &mut nh).unwrap();
            assert_eq!(nh[[0, 0]], 0);
        }
    }
}

fn stress(kinds: &[usize]) {
    let threads = env_usize("CUVS_STRESS_THREADS", 16);
    let iters = env_usize("CUVS_STRESS_ITERS", 20);
    std::thread::scope(|scope| {
        for t in 0..threads {
            scope.spawn(move || {
                for i in 0..iters {
                    one_round(kinds[(t + i) % kinds.len()]);
                }
            });
        }
    });
}

#[test]
#[ignore]
fn stress_ivf_flat_threads() {
    stress(&[0]);
}

#[test]
#[ignore]
fn stress_mixed_threads() {
    stress(&[0, 1, 2, 3]);
}
