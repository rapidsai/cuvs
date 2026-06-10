/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Refinement of approximate nearest neighbor results

use crate::distance_type::DistanceType;
use crate::dlpack::ManagedTensor;
use crate::error::{Result, check_cuvs};
use crate::resources::Resources;

/// Refine nearest neighbor search results.
///
/// Refinement is an operation that follows an approximate nearest neighbors
/// search. The approximate search has already selected `n_candidates` neighbor
/// candidates for each query. This narrows the candidate list down to the `k`
/// nearest neighbors by computing the exact distance between each query and its
/// candidates against the original dataset, then selecting the `k` closest.
///
/// All tensors must reside in the same memory space: either all on the device
/// or all on the host. The dataset and queries may be `f32`, `f16`, `i8`, or
/// `u8` (with matching dtype codes). The candidate and output index tensors
/// must be `i64`, and the output distance tensor must be `f32`.
///
/// # Arguments
///
/// * `res` - Resources to use
/// * `dataset` - A row-major matrix of the original dataset - shape `(n_rows, dims)`
/// * `queries` - A row-major matrix of the queries - shape `(n_queries, dims)`
/// * `candidates` - A row-major `i64` matrix of candidate indices into `dataset`
///   - shape `(n_queries, n_candidates)`, where `n_candidates >= k`
/// * `metric` - DistanceType used to rank candidates
/// * `indices` - Output `i64` matrix that receives the refined indices - shape
///   `(n_queries, k)`. `k` is inferred from this tensor's shape.
/// * `distances` - Output `f32` matrix that receives the refined distances -
///   shape `(n_queries, k)`
///
/// # Example
///
/// ```no_run
/// use cuvs::{ManagedTensor, Resources, Result};
/// use cuvs::distance_type::DistanceType;
/// use cuvs::refine::refine;
/// use ndarray::array;
///
/// fn do_refine() -> Result<()> {
///     let res = Resources::new()?;
///
///     // A tiny dataset with four 2-D points.
///     let dataset = array![[0.0f32, 0.0], [1.0, 0.0], [0.0, 1.0], [5.0, 5.0]];
///     let queries = array![[0.1f32, 0.1]];
///
///     // Approximate candidates - includes the far-away point 3 by mistake.
///     let candidates = array![[3i64, 1, 0]];
///
///     let dataset_d = ManagedTensor::from(&dataset).to_device(&res)?;
///     let queries_d = ManagedTensor::from(&queries).to_device(&res)?;
///     let candidates_d = ManagedTensor::from(&candidates).to_device(&res)?;
///
///     let mut indices_host = ndarray::Array::<i64, _>::zeros((1, 2));
///     let mut distances_host = ndarray::Array::<f32, _>::zeros((1, 2));
///     let indices_d = ManagedTensor::from(&indices_host).to_device(&res)?;
///     let distances_d = ManagedTensor::from(&distances_host).to_device(&res)?;
///
///     refine(
///         &res,
///         &dataset_d,
///         &queries_d,
///         &candidates_d,
///         DistanceType::L2Expanded,
///         &indices_d,
///         &distances_d,
///     )?;
///
///     indices_d.to_host(&res, &mut indices_host)?;
///     distances_d.to_host(&res, &mut distances_host)?;
///     res.sync_stream()?;
///
///     // Point 0 is the true nearest neighbor; the wrong candidate 3 is dropped.
///     assert_eq!(indices_host[[0, 0]], 0);
///     Ok(())
/// }
/// ```
pub fn refine(
    res: &Resources,
    dataset: &ManagedTensor,
    queries: &ManagedTensor,
    candidates: &ManagedTensor,
    metric: DistanceType,
    indices: &ManagedTensor,
    distances: &ManagedTensor,
) -> Result<()> {
    unsafe {
        check_cuvs(ffi::cuvsRefine(
            res.0,
            dataset.as_ptr(),
            queries.as_ptr(),
            candidates.as_ptr(),
            metric,
            indices.as_ptr(),
            distances.as_ptr(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Refinement must repair a candidate list that contains deliberately
    /// wrong entries: after refine, the top-k must equal the exact
    /// brute-force top-k.
    #[test]
    fn test_refine_fixes_wrong_candidates() {
        let res = Resources::new().unwrap();

        // A small, well-separated 2-D dataset. The exact L2 ranking of every
        // query is unambiguous, so we can hard-assert the refined output.
        //
        //   index : point
        //     0   : (0, 0)
        //     1   : (1, 0)
        //     2   : (0, 1)
        //     3   : (2, 2)
        //     4   : (5, 5)
        //     5   : (9, 9)
        let dataset = ndarray::array![
            [0.0f32, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 2.0],
            [5.0, 5.0],
            [9.0, 9.0],
        ];

        // Two queries near distinct clusters.
        //   q0 sits next to point 0; true top-3 = [0, 1, 2]
        //   q1 sits next to point 4; true top-3 = [4, 3, 5] (4 closest, then 3, then 5)
        let queries = ndarray::array![[0.1f32, 0.1], [4.9, 4.9]];

        // Candidate lists are intentionally *wrong order* and include far-away
        // points. Each list is a superset of the true top-3 but jumbled, plus a
        // planted bad candidate (index 5 for q0, index 0 for q1). Refine must
        // re-rank these exactly and select the correct nearest three.
        let candidates = ndarray::array![
            [5i64, 2, 0, 1], // q0: true nearest 0 is buried, 5 is far noise
            [0i64, 5, 3, 4], // q1: true nearest 4 is last, 0 is far noise
        ];

        let n_queries = 2;
        let k = 3;

        let dataset_d = ManagedTensor::from(&dataset).to_device(&res).unwrap();
        let queries_d = ManagedTensor::from(&queries).to_device(&res).unwrap();
        let candidates_d = ManagedTensor::from(&candidates).to_device(&res).unwrap();

        let mut indices_host = ndarray::Array::<i64, _>::zeros((n_queries, k));
        let mut distances_host = ndarray::Array::<f32, _>::zeros((n_queries, k));
        let indices_d = ManagedTensor::from(&indices_host).to_device(&res).unwrap();
        let distances_d = ManagedTensor::from(&distances_host).to_device(&res).unwrap();

        refine(
            &res,
            &dataset_d,
            &queries_d,
            &candidates_d,
            DistanceType::L2Expanded,
            &indices_d,
            &distances_d,
        )
        .unwrap();

        indices_d.to_host(&res, &mut indices_host).unwrap();
        distances_d.to_host(&res, &mut distances_host).unwrap();
        res.sync_stream().unwrap();

        // Exact brute-force top-3, independent of the candidate ordering.
        // q0: distances to (0.1,0.1): 0 -> ~0.14, 1 -> ~0.91, 2 -> ~0.91, ...
        //     point 0 is strictly nearest; 1 and 2 are tied next.
        // q1: distances to (4.9,4.9): 4 -> ~0.14, 3 -> ~4.1, 5 -> ~5.8.
        assert_eq!(
            indices_host[[0, 0]],
            0,
            "q0 nearest must be repaired to index 0, got {:?}",
            indices_host.row(0)
        );
        assert_eq!(
            indices_host[[1, 0]],
            4,
            "q1 nearest must be repaired to index 4, got {:?}",
            indices_host.row(1)
        );

        // The planted noise candidates (5 for q0, 0 for q1) must be evicted
        // from the refined top-k.
        let q0: Vec<i64> = indices_host.row(0).to_vec();
        let q1: Vec<i64> = indices_host.row(1).to_vec();
        assert!(!q0.contains(&5), "q0 must drop far candidate 5, got {:?}", q0);
        assert!(!q1.contains(&0), "q1 must drop far candidate 0, got {:?}", q1);

        // The refined top-3 sets must match the exact brute-force top-3 sets.
        let mut q0_sorted = q0.clone();
        q0_sorted.sort_unstable();
        assert_eq!(q0_sorted, vec![0, 1, 2], "q0 refined set wrong: {:?}", q0);

        let mut q1_sorted = q1.clone();
        q1_sorted.sort_unstable();
        assert_eq!(q1_sorted, vec![3, 4, 5], "q1 refined set wrong: {:?}", q1);

        // Refined distances must be sorted ascending (nearest first) and the
        // first entry must be the small in-cluster distance, not noise.
        assert!(distances_host[[0, 0]] <= distances_host[[0, 1]]);
        assert!(distances_host[[1, 0]] <= distances_host[[1, 1]]);
        assert!(
            distances_host[[0, 0]] < 1.0,
            "q0 nearest distance should be small, got {}",
            distances_host[[0, 0]]
        );
    }
}
