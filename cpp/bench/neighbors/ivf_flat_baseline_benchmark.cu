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

// Baseline benchmark for non-JIT branch (production)
// Runs 3 searches to measure performance without JIT-LTO overhead

#include <cuvs/neighbors/ivf_flat.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>

void print_usage(const char* program_name)
{
  std::cout << "Usage: " << program_name
            << " --n_rows <num> --n_dims <num> --n_queries <num> --k <num> [options]\n"
            << "\nRequired arguments:\n"
            << "  --n_rows <num>      Number of vectors in the dataset\n"
            << "  --n_dims <num>      Dimensionality of vectors\n"
            << "  --n_queries <num>   Number of query vectors\n"
            << "  --k <num>           Number of neighbors to find\n"
            << "\nOptional arguments:\n"
            << "  --n_lists <num>     Number of IVF lists (default: sqrt(n_rows))\n"
            << "  --n_probes <num>    Number of probes during search (default: min(n_lists, 50))\n"
            << "  --metric <type>     Distance metric: l2, inner_product, cosine (default: l2)\n"
            << "  --help              Display this help message\n";
}

struct BenchmarkParams {
  int64_t n_rows;
  int64_t n_dims;
  int64_t n_queries;
  uint32_t k;
  uint32_t n_lists   = 0;  // 0 means auto-compute
  uint32_t n_probes  = 0;  // 0 means auto-compute
  std::string metric = "l2";

  bool validate() const
  {
    if (n_rows <= 0 || n_dims <= 0 || n_queries <= 0 || k <= 0) {
      std::cerr << "Error: All dimension parameters must be positive\n";
      return false;
    }
    if (metric != "l2" && metric != "inner_product" && metric != "cosine") {
      std::cerr << "Error: Invalid metric. Must be l2, inner_product, or cosine\n";
      return false;
    }
    return true;
  }
};

bool parse_args(int argc, char** argv, BenchmarkParams& params)
{
  if (argc < 2) {
    print_usage(argv[0]);
    return false;
  }

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];

    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return false;
    } else if (arg == "--n_rows" && i + 1 < argc) {
      params.n_rows = std::stoll(argv[++i]);
    } else if (arg == "--n_dims" && i + 1 < argc) {
      params.n_dims = std::stoll(argv[++i]);
    } else if (arg == "--n_queries" && i + 1 < argc) {
      params.n_queries = std::stoll(argv[++i]);
    } else if (arg == "--k" && i + 1 < argc) {
      params.k = std::stoul(argv[++i]);
    } else if (arg == "--n_lists" && i + 1 < argc) {
      params.n_lists = std::stoul(argv[++i]);
    } else if (arg == "--n_probes" && i + 1 < argc) {
      params.n_probes = std::stoul(argv[++i]);
    } else if (arg == "--metric" && i + 1 < argc) {
      params.metric = argv[++i];
    } else {
      std::cerr << "Error: Unknown argument '" << arg << "'\n";
      print_usage(argv[0]);
      return false;
    }
  }

  return params.validate();
}

cuvs::distance::DistanceType get_metric_type(const std::string& metric)
{
  if (metric == "l2") {
    return cuvs::distance::DistanceType::L2Expanded;
  } else if (metric == "inner_product") {
    return cuvs::distance::DistanceType::InnerProduct;
  } else if (metric == "cosine") {
    return cuvs::distance::DistanceType::CosineExpanded;
  }
  return cuvs::distance::DistanceType::L2Expanded;
}

int main(int argc, char** argv)
{
  BenchmarkParams params;

  if (!parse_args(argc, argv, params)) { return 1; }

  // Auto-compute n_lists and n_probes if not specified
  if (params.n_lists == 0) {
    params.n_lists = std::max(1u, static_cast<uint32_t>(std::sqrt(params.n_rows)));
  }
  if (params.n_probes == 0) { params.n_probes = std::min(params.n_lists, 50u); }

  std::cout << "\n=== IVF Flat Baseline Benchmark (No JIT) ===\n";
  std::cout << "Dataset size:     " << params.n_rows << " x " << params.n_dims << "\n";
  std::cout << "Query size:       " << params.n_queries << "\n";
  std::cout << "k:                " << params.k << "\n";
  std::cout << "n_lists:          " << params.n_lists << "\n";
  std::cout << "n_probes:         " << params.n_probes << "\n";
  std::cout << "metric:           " << params.metric << "\n";
  std::cout << "============================================\n\n";

  try {
    // Initialize RAFT resources
    raft::device_resources handle;
    auto stream = raft::resource::get_cuda_stream(handle);

    // Generate random dataset
    std::cout << "Generating random dataset...\n";
    auto dataset = raft::make_device_matrix<float, int64_t>(handle, params.n_rows, params.n_dims);
    auto queries =
      raft::make_device_matrix<float, int64_t>(handle, params.n_queries, params.n_dims);

    raft::random::RngState rng(42ULL);
    raft::random::uniform(
      handle, rng, dataset.data_handle(), params.n_rows * params.n_dims, 0.0f, 1.0f);
    raft::random::uniform(
      handle, rng, queries.data_handle(), params.n_queries * params.n_dims, 0.0f, 1.0f);
    raft::resource::sync_stream(handle);

    // Build index
    std::cout << "Building IVF Flat index...\n";
    auto build_start = std::chrono::high_resolution_clock::now();

    cuvs::neighbors::ivf_flat::index_params index_params;
    index_params.n_lists                  = params.n_lists;
    index_params.metric                   = get_metric_type(params.metric);
    index_params.adaptive_centers         = false;
    index_params.add_data_on_build        = true;
    index_params.kmeans_trainset_fraction = 1.0;

    auto index = cuvs::neighbors::ivf_flat::build(
      handle, index_params, raft::make_const_mdspan(dataset.view()));
    raft::resource::sync_stream(handle);

    auto build_end = std::chrono::high_resolution_clock::now();
    auto build_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(build_end - build_start).count();
    std::cout << "Build time: " << build_time << " ms\n\n";

    // Prepare output buffers
    auto neighbors = raft::make_device_matrix<int64_t, int64_t>(handle, params.n_queries, params.k);
    auto distances = raft::make_device_matrix<float, int64_t>(handle, params.n_queries, params.k);

    // Search parameters
    cuvs::neighbors::ivf_flat::search_params search_params;
    search_params.n_probes = params.n_probes;

    // Run search 21 times (1 cold + 20 warm, like JIT benchmark)
    constexpr int num_runs = 21;
    std::vector<double> search_times;
    search_times.reserve(num_runs);

    std::cout << "\nRunning " << num_runs << " searches (1 cold + 20 warm)...\n";

    for (int run = 0; run < num_runs; run++) {
      // Synchronize before timing
      raft::resource::sync_stream(handle);

      auto search_start = std::chrono::high_resolution_clock::now();

      cuvs::neighbors::ivf_flat::search(
        handle, search_params, index, queries.view(), neighbors.view(), distances.view());

      // Synchronize after search to ensure completion
      raft::resource::sync_stream(handle);

      auto search_end = std::chrono::high_resolution_clock::now();
      auto search_time_us =
        std::chrono::duration_cast<std::chrono::microseconds>(search_end - search_start).count();

      search_times.push_back(search_time_us / 1000.0);  // Convert to milliseconds

      if (run == 0) {
        std::cout << "Run 1 (cold):  " << search_times[run] << " ms\n";
      } else {
        std::cout << "Run " << (run + 1) << " (warm): " << search_times[run] << " ms\n";
      }
    }

    // Calculate statistics
    double first_run  = search_times[0];
    double warm_total = 0.0;
    double min_warm   = search_times[1];
    double max_warm   = search_times[1];

    // Average of runs 2-4 (warm runs)
    for (int i = 1; i < num_runs; i++) {
      warm_total += search_times[i];
      min_warm = std::min(min_warm, search_times[i]);
      max_warm = std::max(max_warm, search_times[i]);
    }

    double avg_warm_time = warm_total / (num_runs - 1);
    double all_runs_avg  = (first_run + warm_total) / num_runs;

    std::cout << "\n=== Results ===\n";
    std::cout << "First run (cold):        " << first_run << " ms\n";
    std::cout << "Average time (runs 2-21): " << avg_warm_time << " ms\n";
    std::cout << "Min warm time:           " << min_warm << " ms\n";
    std::cout << "Max warm time:           " << max_warm << " ms\n";
    std::cout << "Overall average:         " << all_runs_avg << " ms\n";
    std::cout << "Cold run overhead:       " << (first_run - avg_warm_time) << " ms\n";
    std::cout << "Throughput (after warmup): " << (params.n_queries / (avg_warm_time / 1000.0))
              << " queries/sec\n";
    std::cout << "===============\n";

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
