/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuco/bloom_filter.cuh>
#include <cuvs/core/bitset.hpp>
#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/cagra.hpp>

#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace {

constexpr int k_warmup_runs = 1;
constexpr int k_timed_runs  = 3;

using key_type    = std::uint32_t;
using filter_type = cuco::bloom_filter<key_type>;
using ref_type    = filter_type::ref_type<>;

struct bloom_payload {
  ref_type filter;
};

constexpr std::array<int64_t, 3> k_build_rows{100'000, 500'000, 1'000'000};
constexpr std::array<int64_t, 3> k_build_cols{128, 512, 1024};
constexpr std::array<int64_t, 2> k_search_rows{10'000, 25'000};
constexpr std::array<int64_t, 3> k_search_cols{128, 512, 1024};
constexpr std::array<int, 3> k_valid_pcts{10, 50, 90};
constexpr std::array<int64_t, 3> k_values{64, 256, 1024};

bool is_valid_row(key_type row_id, int valid_pct)
{
  // Deterministic ~valid_pct% membership independent of dataset size.
  return (static_cast<std::uint64_t>(row_id) * 2654435761ULL) % 100ULL <
         static_cast<std::uint64_t>(valid_pct);
}

std::size_t bloom_num_blocks(std::size_t num_valid_rows)
{
  // Scale filter size with the number of inserted keys; keep a reasonable minimum.
  std::size_t blocks = std::max<std::size_t>(256, num_valid_rows / 8);
  blocks             = std::min(blocks, static_cast<std::size_t>(1 << 20));
  return blocks;
}

double compute_recall(std::vector<key_type> const& expected,
                      std::vector<key_type> const& actual,
                      int64_t n_queries,
                      int64_t k,
                      int64_t n_rows)
{
  std::size_t match_count = 0;
  std::size_t total_count = static_cast<std::size_t>(n_queries) * static_cast<std::size_t>(k);
  for (int64_t q = 0; q < n_queries; ++q) {
    for (int64_t ki = 0; ki < k; ++ki) {
      auto const act = actual[static_cast<std::size_t>(q * k + ki)];
      if (act >= static_cast<key_type>(n_rows)) { continue; }
      for (int64_t kj = 0; kj < k; ++kj) {
        if (expected[static_cast<std::size_t>(q * k + kj)] == act) {
          ++match_count;
          break;
        }
      }
    }
  }
  return total_count == 0 ? 0.0
                          : static_cast<double>(match_count) / static_cast<double>(total_count);
}

std::vector<key_type> copy_neighbors_to_host(raft::device_resources const& res,
                                             raft::device_matrix_view<key_type, int64_t> neighbors)
{
  std::vector<key_type> host(neighbors.size());
  auto stream = raft::resource::get_cuda_stream(res);
  raft::copy(host.data(), neighbors.data_handle(), host.size(), stream);
  raft::resource::sync_stream(res);
  return host;
}

struct filter_assets {
  cuvs::core::bitset<std::uint32_t, int64_t> removed_bitset;
  cuvs::neighbors::filtering::bitset_filter<std::uint32_t, int64_t> bitset_filter;
  filter_type bloom;
  rmm::device_uvector<bloom_payload> bloom_payload;
  cuvs::neighbors::filtering::bloom_filter bloom_filter;
  float filtering_rate{0.0f};
};

filter_assets make_filters(raft::device_resources const& res,
                           int64_t n_rows,
                           int valid_pct,
                           rmm::cuda_stream_view stream)
{
  std::vector<key_type> valid_ids_host;
  std::vector<int64_t> removed_ids_host;
  valid_ids_host.reserve(static_cast<std::size_t>(n_rows));
  removed_ids_host.reserve(static_cast<std::size_t>(n_rows));

  for (int64_t i = 0; i < n_rows; ++i) {
    auto const row = static_cast<key_type>(i);
    if (is_valid_row(row, valid_pct)) {
      valid_ids_host.push_back(row);
    } else {
      removed_ids_host.push_back(i);
    }
  }

  auto removed_ids =
    raft::make_device_vector<int64_t, int64_t>(res, static_cast<int64_t>(removed_ids_host.size()));
  if (!removed_ids_host.empty()) {
    raft::copy(removed_ids.data_handle(), removed_ids_host.data(), removed_ids_host.size(), stream);
  }

  auto removed_bitset = cuvs::core::bitset<std::uint32_t, int64_t>(res, removed_ids.view(), n_rows);
  auto bitset_filter =
    cuvs::neighbors::filtering::bitset_filter<std::uint32_t, int64_t>(removed_bitset.view());
  auto bloom          = filter_type{bloom_num_blocks(valid_ids_host.size()), {}, {}, {}, stream};
  auto payload_device = rmm::device_uvector<bloom_payload>{1, stream};
  float const filtering_rate = static_cast<float>(100 - valid_pct) / 100.0f;

  if (!valid_ids_host.empty()) {
    rmm::device_uvector<key_type> valid_ids_device(valid_ids_host.size(), stream);
    raft::copy(valid_ids_device.data(), valid_ids_host.data(), valid_ids_host.size(), stream);
    bloom.add_async(
      valid_ids_device.data(), valid_ids_device.data() + valid_ids_device.size(), stream);
  }

  bloom_payload host_payload{bloom.ref()};
  raft::copy(payload_device.data(), &host_payload, 1, stream);
  auto bloom_filter_obj =
    cuvs::neighbors::filtering::bloom_filter(payload_device.data(), filtering_rate);

  raft::resource::sync_stream(res);
  return filter_assets{std::move(removed_bitset),
                       std::move(bitset_filter),
                       std::move(bloom),
                       std::move(payload_device),
                       std::move(bloom_filter_obj),
                       filtering_rate};
}

struct benchmark_case {
  int64_t build_n_rows;
  int64_t build_n_cols;
  int64_t search_n_rows;
  int64_t search_n_cols;
  int valid_pct;
  int64_t k;
};

struct csv_row {
  benchmark_case config;
  std::string filter_name;
  double build_time_ms;
  double avg_search_latency_ms;
  double avg_latency_per_query_ms;
  double recall;
};

void write_csv_header(std::ostream& os)
{
  os << "build_n_rows,build_n_cols,search_n_rows,search_n_cols,valid_pct,filter_type,"
        "build_time_ms,avg_search_latency_ms,avg_latency_per_query_ms,recall,k,warmup_runs,"
        "timed_runs\n";
}

void write_csv_row(std::ostream& os, csv_row const& row)
{
  os << row.config.build_n_rows << ',' << row.config.build_n_cols << ',' << row.config.search_n_rows
     << ',' << row.config.search_n_cols << ',' << row.config.valid_pct << ',' << row.filter_name
     << ',' << row.build_time_ms << ',' << row.avg_search_latency_ms << ','
     << row.avg_latency_per_query_ms << ',' << row.recall << ',' << row.config.k << ','
     << k_warmup_runs << ',' << k_timed_runs << '\n';
}

template <typename Fn>
double time_cuda_ms(raft::device_resources const& res, int runs, Fn&& fn)
{
  auto stream = raft::resource::get_cuda_stream(res);
  cudaEvent_t start{};
  cudaEvent_t stop{};
  RAFT_CUDA_TRY(cudaEventCreate(&start));
  RAFT_CUDA_TRY(cudaEventCreate(&stop));

  RAFT_CUDA_TRY(cudaEventRecord(start, stream));
  for (int i = 0; i < runs; ++i) {
    fn();
  }
  RAFT_CUDA_TRY(cudaEventRecord(stop, stream));
  RAFT_CUDA_TRY(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  RAFT_CUDA_TRY(cudaEventElapsedTime(&elapsed_ms, start, stop));
  RAFT_CUDA_TRY(cudaEventDestroy(start));
  RAFT_CUDA_TRY(cudaEventDestroy(stop));
  return static_cast<double>(elapsed_ms) / static_cast<double>(runs);
}

void append_cases(std::vector<benchmark_case>& cases,
                  std::vector<int64_t> const& build_rows,
                  std::vector<int64_t> const& build_cols,
                  std::vector<int64_t> const& search_rows,
                  std::vector<int64_t> const& search_cols,
                  std::vector<int> const& valid_pcts,
                  std::vector<int64_t> const& k_sweep)
{
  for (auto build_n_rows : build_rows) {
    for (auto build_n_cols : build_cols) {
      for (auto search_n_rows : search_rows) {
        for (auto search_n_cols : search_cols) {
          if (search_n_cols != build_n_cols) { continue; }
          for (auto valid_pct : valid_pcts) {
            for (auto k : k_sweep) {
              cases.push_back(benchmark_case{
                build_n_rows, build_n_cols, search_n_rows, search_n_cols, valid_pct, k});
            }
          }
        }
      }
    }
  }
}

std::vector<benchmark_case> make_cases(bool quick)
{
  std::vector<benchmark_case> cases;
  if (quick) {
    append_cases(cases, {100'000}, {128}, {10'000}, {128}, {1, 50}, {64});
  } else {
    append_cases(cases,
                 {k_build_rows.begin(), k_build_rows.end()},
                 {k_build_cols.begin(), k_build_cols.end()},
                 {k_search_rows.begin(), k_search_rows.end()},
                 {k_search_cols.begin(), k_search_cols.end()},
                 {k_valid_pcts.begin(), k_valid_pcts.end()},
                 {k_values.begin(), k_values.end()});
  }
  return cases;
}

constexpr std::size_t k_max_bf_bytes       = 20ULL << 30;  // skip BF recall above this estimate
constexpr std::size_t k_max_bf_chunk_bytes = 2ULL << 30;  // cap each BF chunk when computing recall

std::size_t estimate_bf_distance_matrix_bytes(int64_t n_queries, int64_t n_dataset)
{
  return static_cast<std::size_t>(n_queries) * static_cast<std::size_t>(n_dataset) * sizeof(float);
}

bool should_compute_bf_recall(int64_t n_queries, int64_t n_dataset)
{
  return estimate_bf_distance_matrix_bytes(n_queries, n_dataset) <= k_max_bf_bytes;
}

int64_t choose_gt_chunk_queries(int64_t n_dataset)
{
  int64_t chunk = 256;
  while (chunk > 1 && estimate_bf_distance_matrix_bytes(chunk, n_dataset) > k_max_bf_chunk_bytes) {
    chunk /= 2;
  }
  return chunk;
}

std::vector<key_type> brute_force_ground_truth(
  raft::device_resources const& res,
  cuvs::neighbors::brute_force::index<float>& bf_index,
  cuvs::neighbors::brute_force::search_params const& bf_search_params,
  raft::device_matrix_view<const float, int64_t> queries,
  cuvs::neighbors::filtering::bitset_filter<std::uint32_t, int64_t> const& bitset_filter,
  int64_t k,
  int64_t gt_chunk_queries)
{
  int64_t const n_queries = queries.extent(0);
  std::vector<key_type> gt_host(static_cast<std::size_t>(n_queries * k));
  auto stream = raft::resource::get_cuda_stream(res);

  for (int64_t query_offset = 0; query_offset < n_queries; query_offset += gt_chunk_queries) {
    int64_t const chunk_queries = std::min(gt_chunk_queries, n_queries - query_offset);
    auto query_chunk            = raft::make_device_matrix_view<const float, int64_t>(
      queries.data_handle() + query_offset * queries.extent(1), chunk_queries, queries.extent(1));
    auto gt_neighbors = raft::make_device_matrix<int64_t, int64_t>(res, chunk_queries, k);
    auto gt_distances = raft::make_device_matrix<float, int64_t>(res, chunk_queries, k);

    cuvs::neighbors::brute_force::search(res,
                                         bf_search_params,
                                         bf_index,
                                         raft::make_const_mdspan(query_chunk),
                                         gt_neighbors.view(),
                                         gt_distances.view(),
                                         bitset_filter);
    raft::resource::sync_stream(res);

    std::vector<int64_t> chunk_host(static_cast<std::size_t>(chunk_queries * k));
    raft::copy(chunk_host.data(), gt_neighbors.data_handle(), chunk_host.size(), stream);
    raft::resource::sync_stream(res);

    for (int64_t q = 0; q < chunk_queries; ++q) {
      for (int64_t ki = 0; ki < k; ++ki) {
        gt_host[static_cast<std::size_t>((query_offset + q) * k + ki)] =
          static_cast<key_type>(chunk_host[static_cast<std::size_t>(q * k + ki)]);
      }
    }
  }

  return gt_host;
}

}  // namespace

int main(int argc, char** argv)
{
  std::string output_path   = "cagra_filter_benchmark_results.csv";
  bool quick                = false;
  bool compute_ground_truth = false;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--quick") {
      quick = true;
    } else if (arg == "--ground-truth") {
      compute_ground_truth = true;
    } else if (arg == "--skip-ground-truth") {
      compute_ground_truth = false;
    } else if (arg == "--output" && i + 1 < argc) {
      output_path = argv[++i];
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: " << argv[0]
                << " [--quick] [--ground-truth] [--skip-ground-truth] [--output path.csv]\n"
                << "\n"
                << "Brute-force recall is skipped by default. Pass --ground-truth to compute it.\n";
      return 0;
    } else {
      output_path = arg;
    }
  }

  auto cases = make_cases(quick);
  std::cout << "Running " << cases.size() << " benchmark configurations"
            << (quick ? " (quick mode)" : "")
            << (compute_ground_truth ? " (with ground-truth recall)" : " (ground-truth skipped)")
            << std::endl;

  std::ofstream csv(output_path);
  if (!csv) {
    std::cerr << "Failed to open output file: " << output_path << std::endl;
    return 1;
  }
  write_csv_header(csv);

  raft::device_resources res;
  auto stream = raft::resource::get_cuda_stream(res);

  // Large enough for the biggest benchmark configuration (1M x 1024 dataset + index overhead).
  rmm::mr::pool_memory_resource pool_mr(rmm::mr::get_current_device_resource_ref(), 16ULL << 30);
  rmm::mr::set_current_device_resource(pool_mr);

  int64_t prev_build_rows   = -1;
  int64_t prev_build_cols   = -1;
  double last_build_time_ms = 0.0;

  std::optional<cuvs::neighbors::cagra::index<float, key_type>> index;
  std::optional<raft::device_matrix<float, int64_t>> dataset;
  std::optional<cuvs::neighbors::brute_force::index<float>> bf_index;

  cuvs::neighbors::cagra::index_params index_params;
  index_params.metric                    = cuvs::distance::DistanceType::L2Expanded;
  index_params.graph_degree              = 32;
  index_params.intermediate_graph_degree = 64;
  index_params.graph_build_params = cuvs::neighbors::cagra::graph_build_params::nn_descent_params(
    index_params.intermediate_graph_degree);

  cuvs::neighbors::cagra::search_params search_params;
  search_params.algo              = cuvs::neighbors::cagra::search_algo::MULTI_CTA;
  search_params.itopk_size        = 128;
  search_params.thread_block_size = 256;

  cuvs::neighbors::brute_force::index_params bf_index_params;
  cuvs::neighbors::brute_force::search_params bf_search_params;

  for (std::size_t case_idx = 0; case_idx < cases.size(); ++case_idx) {
    auto const& cfg = cases[case_idx];

    if (cfg.build_n_rows != prev_build_rows || cfg.build_n_cols != prev_build_cols) {
      std::cout << "Building CAGRA index: n_rows=" << cfg.build_n_rows
                << " n_cols=" << cfg.build_n_cols << std::endl;

      dataset.emplace(
        raft::make_device_matrix<float, int64_t>(res, cfg.build_n_rows, cfg.build_n_cols));
      raft::random::RngState rng(
        static_cast<std::uint64_t>(cfg.build_n_rows * 17 + cfg.build_n_cols));
      raft::random::uniform(res, rng, dataset->data_handle(), dataset->size(), -1.0f, 1.0f);

      auto build_start = std::chrono::steady_clock::now();
      index.emplace(
        cuvs::neighbors::cagra::build(res, index_params, raft::make_const_mdspan(dataset->view())));
      bf_index.reset();
      raft::resource::sync_stream(res);
      auto build_end = std::chrono::steady_clock::now();
      last_build_time_ms =
        std::chrono::duration<double, std::milli>(build_end - build_start).count();

      prev_build_rows = cfg.build_n_rows;
      prev_build_cols = cfg.build_n_cols;
    }

    int64_t const total_queries = cfg.search_n_rows;
    search_params.max_queries   = total_queries;
    search_params.itopk_size    = static_cast<std::size_t>(cfg.k);

    std::cout << "Case " << (case_idx + 1) << '/' << cases.size()
              << ": build_n_rows=" << cfg.build_n_rows << " search_n_rows=" << total_queries
              << " k=" << cfg.k << " valid_pct=" << cfg.valid_pct << '%' << std::endl;

    try {
      auto queries =
        raft::make_device_matrix<float, int64_t>(res, total_queries, cfg.search_n_cols);
      raft::random::RngState query_rng(static_cast<std::uint64_t>(
        cfg.build_n_rows * 31 + cfg.search_n_rows * 17 + cfg.search_n_cols + cfg.valid_pct));
      raft::random::uniform(res, query_rng, queries.data_handle(), queries.size(), -1.0f, 1.0f);

      auto neighbors = raft::make_device_matrix<key_type, int64_t>(res, total_queries, cfg.k);
      auto distances = raft::make_device_matrix<float, int64_t>(res, total_queries, cfg.k);

      auto filters = make_filters(res, cfg.build_n_rows, cfg.valid_pct, stream);

      bool const run_bf_recall =
        compute_ground_truth && should_compute_bf_recall(total_queries, cfg.build_n_rows);
      std::optional<std::vector<key_type>> gt_host;
      if (run_bf_recall) {
        if (!bf_index.has_value()) {
          bf_index.emplace(cuvs::neighbors::brute_force::build(
            res, bf_index_params, raft::make_const_mdspan(dataset->view())));
        }
        gt_host.emplace(brute_force_ground_truth(res,
                                                 *bf_index,
                                                 bf_search_params,
                                                 queries.view(),
                                                 filters.bitset_filter,
                                                 cfg.k,
                                                 choose_gt_chunk_queries(cfg.build_n_rows)));
      } else if (compute_ground_truth) {
        auto const est_gib =
          static_cast<double>(estimate_bf_distance_matrix_bytes(total_queries, cfg.build_n_rows)) /
          static_cast<double>(1ULL << 30);
        std::cout << "  skipping brute-force recall (estimated " << est_gib
                  << " GiB distance matrix > 20 GiB limit)" << std::endl;
      }

      auto run_cagra_search = [&](cuvs::neighbors::filtering::base_filter const& filter) {
        cuvs::neighbors::cagra::search(res,
                                       search_params,
                                       *index,
                                       raft::make_const_mdspan(queries.view()),
                                       neighbors.view(),
                                       distances.view(),
                                       filter);
        raft::resource::sync_stream(res);
      };

      struct filter_run {
        std::string name;
        cuvs::neighbors::filtering::base_filter const* filter;
      };
      std::vector<filter_run> filter_runs{
        {"bitset", &filters.bitset_filter},
        {"bloom_filter", &filters.bloom_filter},
      };

      for (auto const& fr : filter_runs) {
        for (int w = 0; w < k_warmup_runs; ++w) {
          run_cagra_search(*fr.filter);
        }

        double const avg_search_ms =
          time_cuda_ms(res, k_timed_runs, [&] { run_cagra_search(*fr.filter); });
        double const avg_per_query_ms = avg_search_ms / static_cast<double>(total_queries);

        double const recall = [&]() {
          if (!gt_host.has_value()) { return std::numeric_limits<double>::quiet_NaN(); }
          auto result_host = copy_neighbors_to_host(res, neighbors.view());
          return compute_recall(*gt_host, result_host, total_queries, cfg.k, cfg.build_n_rows);
        }();

        write_csv_row(
          csv, csv_row{cfg, fr.name, last_build_time_ms, avg_search_ms, avg_per_query_ms, recall});
        csv.flush();

        std::cout << "  " << fr.name << ": search_ms=" << avg_search_ms
                  << " per_query_ms=" << avg_per_query_ms << " recall=";
        if (gt_host.has_value()) {
          std::cout << recall;
        } else {
          std::cout << "n/a";
        }
        std::cout << std::endl;
      }
    } catch (std::exception const& ex) {
      std::cerr << "  case failed: " << ex.what() << std::endl;
      for (auto const* filter_name : {"bitset", "bloom_filter"}) {
        write_csv_row(csv,
                      csv_row{cfg,
                              filter_name,
                              last_build_time_ms,
                              std::numeric_limits<double>::quiet_NaN(),
                              std::numeric_limits<double>::quiet_NaN(),
                              std::numeric_limits<double>::quiet_NaN()});
        csv.flush();
      }
    }
  }

  std::cout << "Wrote results to " << output_path << std::endl;
  return 0;
}
