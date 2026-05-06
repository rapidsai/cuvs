/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//! End-to-end example: build a Vamana graph index on the GPU with cuVS,
//! serialize it to disk, then load and search it via Microsoft DiskANN's
//! Rust API (`diskann-providers`).
//!
//! cuVS' `vamana::Index::serialize` writes the canonical DiskANN in-memory
//! file format:
//!
//!   <prefix>          - the Vamana graph (24-byte LE header followed by
//!                       per-node `u32` adjacency lists)
//!   <prefix>.data     - the dataset in DiskANN's binary vector format
//!                       (`u32 npts; u32 dim; T data[npts*dim]`)
//!
//! Those are exactly the layouts that `diskann-providers` expects in
//! `storage::bin::{load_graph, load_from_bin}`, so we can call
//! `load_fp_index` directly on the cuVS-written files.
//!
//! ## Caveats
//!
//! - The Rust DiskANN loader currently requires `num_frozen_pts >= 1`
//!   (`NonZeroUsize`). cuVS does not reserve frozen points, so the loader
//!   reinterprets the *last* vector as a frozen entry-point sentinel and
//!   excludes it from search results. For our random dataset that's fine,
//!   but be aware that point id `npts - 1` will never be returned. The
//!   medoid id stored in the cuVS graph header is also ignored by the
//!   Rust loader; entry points come from the trailing `num_frozen_pts`.
//!
//! Run with: `cargo run --release --example vamana`

use std::env;
use std::fs;
use std::num::NonZeroUsize;
use std::path::PathBuf;

use cuvs::distance_type::DistanceType;
use cuvs::vamana::{Index as VamanaIndex, IndexParams};
use cuvs::{ManagedTensor, Resources, Result as CuvsResult};

use ndarray::Array2;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

// Microsoft DiskANN imports (search side).
use diskann::graph::config::{Builder as ConfigBuilder, MaxDegree};
use diskann::graph::search::Knn;
use diskann::graph::search_output_buffer::IdDistance;
use diskann::provider::DefaultContext;
use diskann::utils::ONE;
use diskann_providers::index::wrapped_async::DiskANNIndex as SyncDiskANNIndex;
use diskann_providers::model::configuration::IndexConfiguration;
use diskann_providers::model::graph::provider::async_::common::{FullPrecision, NoStore};
use diskann_providers::model::graph::provider::async_::inmem::FullPrecisionProvider;
use diskann_providers::storage::FileStorageProvider;
use diskann_vector::distance::Metric;

const N_DATAPOINTS: usize = 50_000;
const N_FEATURES: usize = 64;
const N_QUERIES: usize = 100;
const K: usize = 10;

/// Build a Vamana index with cuVS and serialize it to `<prefix>` and
/// `<prefix>.data` in DiskANN in-memory format.
fn build_and_serialize(dataset: &Array2<f32>, prefix: &str) -> CuvsResult<()> {
    let res = Resources::new()?;
    let dataset_device = ManagedTensor::from(dataset).to_device(&res)?;

    let build_params = IndexParams::new()?
        .set_metric(DistanceType::L2Expanded)
        .set_graph_degree(32)
        .set_visited_size(64)
        .set_alpha(1.2);

    let start = std::time::Instant::now();
    let index = VamanaIndex::build(&res, &build_params, dataset_device)?;
    println!(
        "Built Vamana index on GPU in {:.2?} (R=32, L=64)",
        start.elapsed()
    );

    index.serialize(&res, prefix, /* include_dataset = */ true)
}

/// Load the cuVS-written index using the Microsoft DiskANN Rust API and
/// run a batch of k-NN queries against it.
fn search_with_diskann(
    prefix: &str,
    queries: &Array2<f32>,
    k: usize,
    l_search: usize,
) -> Result<(Vec<u32>, Vec<f32>), Box<dyn std::error::Error>> {
    // R/L used at build time. The values must be at least as large as the
    // graph's actual max degree / search list, but they don't need to match
    // exactly. They feed `IndexConfiguration` and size internal buffers.
    let graph_degree = 32usize;
    let l_build = 64usize;

    let config = ConfigBuilder::new(
        graph_degree,
        MaxDegree::default_slack(),
        l_build,
        Metric::L2.into(),
    )
    .build()?;

    // `max_points` here is the total count the loader will see in the
    // `<prefix>.data` file (cuVS writes every dataset row).
    let index_config = IndexConfiguration::new(
        Metric::L2,
        N_FEATURES,
        N_DATAPOINTS,
        ONE, // num_frozen_pts: smallest legal value; the last point is treated as a sentinel.
        0,   // num_threads: 0 = pick a default
        config,
    );

    let storage = FileStorageProvider;

    // `load_fp_index` returns a `graph::DiskANNIndex<FullPrecisionProvider<f32, NoStore>>`.
    // Wrap it in the synchronous helper so we can call `.search(...)` from non-async code.
    let index: SyncDiskANNIndex<FullPrecisionProvider<f32, NoStore>> =
        SyncDiskANNIndex::load_with_multi_thread_runtime(&storage, &(prefix, index_config))?;

    println!("Loaded the cuVS-written Vamana index via diskann-providers.");

    let n_queries = queries.shape()[0];
    let mut all_ids = vec![0u32; n_queries * k];
    let mut all_dists = vec![0.0f32; n_queries * k];

    let search_params = Knn::new(k, l_search, None)?;

    let start = std::time::Instant::now();
    for (qi, query) in queries.outer_iter().enumerate() {
        let id_slice = &mut all_ids[qi * k..(qi + 1) * k];
        let dist_slice = &mut all_dists[qi * k..(qi + 1) * k];
        let mut buffer = IdDistance::new(id_slice, dist_slice);

        index.search(
            search_params,
            &FullPrecision,
            &DefaultContext,
            query.as_slice().expect("contiguous queries"),
            &mut buffer,
        )?;
    }
    println!(
        "Searched {} queries (k={}, L_search={}) in {:.2?}",
        n_queries,
        k,
        l_search,
        start.elapsed()
    );

    Ok((all_ids, all_dists))
}

/// Brute-force squared-L2 top-k for sanity-checking recall.
fn brute_force_topk(data: &Array2<f32>, queries: &Array2<f32>, k: usize) -> Vec<u32> {
    let n = data.shape()[0];
    let nq = queries.shape()[0];
    let mut out = vec![0u32; nq * k];

    let data_norms: Vec<f32> = data.outer_iter().map(|row| row.dot(&row)).collect();

    for (qi, q) in queries.outer_iter().enumerate() {
        let q_norm: f32 = q.dot(&q);
        let mut dists: Vec<(usize, f32)> = (0..n)
            .map(|i| {
                let row = data.row(i);
                let dot: f32 = row.dot(&q);
                (i, q_norm + data_norms[i] - 2.0 * dot)
            })
            .collect();
        dists.select_nth_unstable_by(k, |a, b| a.1.partial_cmp(&b.1).unwrap());
        dists[..k].sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        for (j, (id, _)) in dists.iter().take(k).enumerate() {
            out[qi * k + j] = *id as u32;
        }
    }
    out
}

fn vamana_example() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Generate dataset.
    let dataset =
        ndarray::Array::<f32, _>::random((N_DATAPOINTS, N_FEATURES), Uniform::new(0., 1.0));
    let queries = ndarray::Array::<f32, _>::random((N_QUERIES, N_FEATURES), Uniform::new(0., 1.0));

    let out_dir: PathBuf = env::temp_dir().join("cuvs_vamana_example");
    fs::create_dir_all(&out_dir)?;
    let prefix = out_dir.join("ann");
    let prefix_str = prefix.to_str().expect("non-utf8 output path").to_string();

    // 2. Build + serialize via cuVS (GPU).
    build_and_serialize(&dataset, &prefix_str)?;
    for entry in fs::read_dir(&out_dir)? {
        let entry = entry?;
        println!(
            "  wrote {} ({} bytes)",
            entry.path().display(),
            entry.metadata()?.len()
        );
    }

    // 3. Load + search via Microsoft DiskANN (CPU).
    let (ids, _dists) = search_with_diskann(&prefix_str, &queries, K, /* L_search = */ 64)?;

    // 4. Recall sanity check vs exact L2 top-k. Note: id == N_DATAPOINTS - 1
    //    can never be returned by the Rust loader (frozen-point caveat above),
    //    so we exclude it from the ground-truth set.
    let gt = brute_force_topk(&dataset, &queries, K);
    let frozen_id = (N_DATAPOINTS - 1) as u32;
    let mut matches = 0usize;
    let mut compared = 0usize;
    for q in 0..N_QUERIES {
        let gt_set: std::collections::HashSet<u32> = gt[q * K..(q + 1) * K]
            .iter()
            .copied()
            .filter(|id| *id != frozen_id)
            .collect();
        let pred_set: std::collections::HashSet<u32> =
            ids[q * K..(q + 1) * K].iter().copied().collect();
        compared += gt_set.len();
        matches += gt_set.intersection(&pred_set).count();
    }
    if compared > 0 {
        println!(
            "recall@{} = {:.3} ({} / {})",
            K,
            matches as f64 / compared as f64,
            matches,
            compared
        );
    }

    println!("First query, top-{} ids: {:?}", K, &ids[..K]);

    Ok(())
}

fn main() {
    if let Err(e) = vamana_example() {
        eprintln!("Failed to run Vamana + DiskANN example: {}", e);
        std::process::exit(1);
    }
}
