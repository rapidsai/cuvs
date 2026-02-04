/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Benchmark comparing built-in L2 vs Custom UDF L2 for IVF-Flat search
 * Outputs results as CSV for plotting with Python
 */

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <cuvs/neighbors/ivf_flat.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <rmm/device_uvector.hpp>

// Define custom L2 metric using the CUVS_METRIC macro
CUVS_METRIC(custom_l2, { acc += squared_diff(x, y); })

// Raw UDF that matches built-in structure exactly (struct with template specializations)
inline std::string raw_l2_udf()
{
  return R"(
/* Fixed-width integer types for nvrtc */
using int8_t = signed char;
using uint8_t = unsigned char;
using int32_t = int;
using uint32_t = unsigned int;

namespace cuvs { namespace neighbors { namespace ivf_flat { namespace detail {

// Primary template - works for float
template <int Veclen, typename T, typename AccT>
struct euclidean_dist {
  __device__ __forceinline__ void operator()(AccT& acc, AccT x, AccT y)
  {
    const auto diff = x - y;
    acc += diff * diff;
  }
};

// Specialization for int8_t (matching built-in exactly)
template <int Veclen>
struct euclidean_dist<Veclen, int8_t, int32_t> {
  __device__ __forceinline__ void operator()(int32_t& acc, int32_t x, int32_t y)
  {
    if constexpr (Veclen > 1) {
      const auto diff = __vabsdiffs4(x, y);
      acc = __dp4a(diff, diff, static_cast<uint32_t>(acc));
    } else {
      const auto diff = x - y;
      acc += diff * diff;
    }
  }
};

// No __forceinline__ here - matches built-in
template <int Veclen, typename T, typename AccT>
__device__ void compute_dist(AccT& acc, AccT x, AccT y)
{
  euclidean_dist<Veclen, T, AccT>{}(acc, x, y);
}

}}}}
)";
}

namespace {

using namespace cuvs::neighbors;

// ============================================================================
// Clear NVIDIA compute cache for accurate JIT timing
// ============================================================================

void clear_compute_cache()
{
  const char* home = std::getenv("HOME");
  if (home) {
    std::filesystem::path cache_path = std::filesystem::path(home) / ".nv" / "ComputeCache";
    std::error_code ec;
    std::filesystem::remove_all(cache_path, ec);
    // Ignore errors - cache may not exist
  }
}

// ============================================================================
// Timing utilities
// ============================================================================

class Timer {
 public:
  void start() { start_ = std::chrono::high_resolution_clock::now(); }

  double stop_ms()
  {
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start_).count();
  }

 private:
  std::chrono::high_resolution_clock::time_point start_;
};

double median(std::vector<double>& times)
{
  std::sort(times.begin(), times.end());
  size_t n = times.size();
  if (n % 2 == 0) { return (times[n / 2 - 1] + times[n / 2]) / 2.0; }
  return times[n / 2];
}

// ============================================================================
// Data generation
// ============================================================================

template <typename T>
void generate_random_data(std::vector<T>& data, size_t n, std::mt19937& rng)
{
  if constexpr (std::is_same_v<T, float>) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < n; ++i) {
      data[i] = dist(rng);
    }
  } else if constexpr (std::is_same_v<T, int8_t>) {
    std::uniform_int_distribution<int> dist(-127, 127);
    for (size_t i = 0; i < n; ++i) {
      data[i] = static_cast<int8_t>(dist(rng));
    }
  }
}

// ============================================================================
// Benchmark result structure
// ============================================================================

struct SearchResult {
  double first_ms;
  double median_ms;
};

struct BenchmarkResult {
  std::string dtype;
  int64_t k;
  double first_builtin_ms;
  double first_udf_ms;
  double first_raw_ms;
  double median_builtin_ms;
  double median_udf_ms;
  double median_raw_ms;
};

// ============================================================================
// Benchmark runners (separate functions for built-in and UDF)
// ============================================================================

template <typename T>
SearchResult run_builtin_benchmark(
  raft::resources& handle, int64_t num_vectors, int64_t dim, int64_t k, int num_iterations)
{
  // Clear NVIDIA compute cache for accurate JIT timing
  clear_compute_cache();

  auto stream = raft::resource::get_cuda_stream(handle);
  Timer timer;

  // Generate random data
  std::mt19937 rng(42);
  std::vector<T> h_database(num_vectors * dim);
  std::vector<T> h_queries(100 * dim);
  generate_random_data(h_database, h_database.size(), rng);
  generate_random_data(h_queries, h_queries.size(), rng);

  int64_t num_queries = 100;

  // Copy to device
  rmm::device_uvector<T> d_database(num_vectors * dim, stream);
  rmm::device_uvector<T> d_queries(num_queries * dim, stream);
  raft::copy(d_database.data(), h_database.data(), h_database.size(), stream);
  raft::copy(d_queries.data(), h_queries.data(), h_queries.size(), stream);

  auto database_view =
    raft::make_device_matrix_view<const T, int64_t>(d_database.data(), num_vectors, dim);
  auto queries_view =
    raft::make_device_matrix_view<const T, int64_t>(d_queries.data(), num_queries, dim);

  // Build index
  ivf_flat::index_params index_params;
  index_params.n_lists = 1024;
  index_params.metric  = cuvs::distance::DistanceType::L2Expanded;

  auto idx = ivf_flat::build(handle, index_params, database_view);
  raft::resource::sync_stream(handle);

  // Allocate output buffers
  rmm::device_uvector<int64_t> d_indices(num_queries * k, stream);
  rmm::device_uvector<float> d_distances(num_queries * k, stream);

  auto indices_view =
    raft::make_device_matrix_view<int64_t, int64_t>(d_indices.data(), num_queries, k);
  auto distances_view =
    raft::make_device_matrix_view<float, int64_t>(d_distances.data(), num_queries, k);

  // Search params
  ivf_flat::search_params search_params;
  search_params.n_probes = 32;

  SearchResult result;

  // First search (includes JIT compilation)
  timer.start();
  ivf_flat::search(handle, search_params, idx, queries_view, indices_view, distances_view);
  raft::resource::sync_stream(handle);
  result.first_ms = timer.stop_ms();

  // Repeated searches (JIT already cached)
  std::vector<double> times;
  for (int i = 0; i < num_iterations; ++i) {
    timer.start();
    ivf_flat::search(handle, search_params, idx, queries_view, indices_view, distances_view);
    raft::resource::sync_stream(handle);
    times.push_back(timer.stop_ms());
  }

  result.median_ms = median(times);
  return result;
}

template <typename T>
SearchResult run_udf_benchmark(
  raft::resources& handle, int64_t num_vectors, int64_t dim, int64_t k, int num_iterations)
{
  // Clear NVIDIA compute cache for accurate JIT timing
  clear_compute_cache();

  auto stream = raft::resource::get_cuda_stream(handle);
  Timer timer;

  // Generate random data (same seed as built-in for consistency)
  std::mt19937 rng(42);
  std::vector<T> h_database(num_vectors * dim);
  std::vector<T> h_queries(100 * dim);
  generate_random_data(h_database, h_database.size(), rng);
  generate_random_data(h_queries, h_queries.size(), rng);

  int64_t num_queries = 100;

  // Copy to device
  rmm::device_uvector<T> d_database(num_vectors * dim, stream);
  rmm::device_uvector<T> d_queries(num_queries * dim, stream);
  raft::copy(d_database.data(), h_database.data(), h_database.size(), stream);
  raft::copy(d_queries.data(), h_queries.data(), h_queries.size(), stream);

  auto database_view =
    raft::make_device_matrix_view<const T, int64_t>(d_database.data(), num_vectors, dim);
  auto queries_view =
    raft::make_device_matrix_view<const T, int64_t>(d_queries.data(), num_queries, dim);

  // Build index with L2Expanded (kmeans doesn't support CustomUDF)
  // The UDF is only used during search
  ivf_flat::index_params index_params;
  index_params.n_lists = 1024;
  index_params.metric  = cuvs::distance::DistanceType::L2Expanded;

  auto idx = ivf_flat::build(handle, index_params, database_view);
  raft::resource::sync_stream(handle);

  // Allocate output buffers
  rmm::device_uvector<int64_t> d_indices(num_queries * k, stream);
  rmm::device_uvector<float> d_distances(num_queries * k, stream);

  auto indices_view =
    raft::make_device_matrix_view<int64_t, int64_t>(d_indices.data(), num_queries, k);
  auto distances_view =
    raft::make_device_matrix_view<float, int64_t>(d_distances.data(), num_queries, k);

  // Search params with UDF
  ivf_flat::search_params search_params;
  search_params.n_probes   = 32;
  search_params.metric_udf = custom_l2_udf();

  SearchResult result;

  // First search (includes JIT compilation)
  timer.start();
  ivf_flat::search(handle, search_params, idx, queries_view, indices_view, distances_view);
  raft::resource::sync_stream(handle);
  result.first_ms = timer.stop_ms();

  // Repeated searches (JIT already cached)
  std::vector<double> times;
  for (int i = 0; i < num_iterations; ++i) {
    timer.start();
    ivf_flat::search(handle, search_params, idx, queries_view, indices_view, distances_view);
    raft::resource::sync_stream(handle);
    times.push_back(timer.stop_ms());
  }

  result.median_ms = median(times);
  return result;
}

template <typename T>
SearchResult run_raw_udf_benchmark(
  raft::resources& handle, int64_t num_vectors, int64_t dim, int64_t k, int num_iterations)
{
  // Clear NVIDIA compute cache for accurate JIT timing
  clear_compute_cache();

  auto stream = raft::resource::get_cuda_stream(handle);
  Timer timer;

  // Generate random data (same seed as built-in for consistency)
  std::mt19937 rng(42);
  std::vector<T> h_database(num_vectors * dim);
  std::vector<T> h_queries(100 * dim);
  generate_random_data(h_database, h_database.size(), rng);
  generate_random_data(h_queries, h_queries.size(), rng);

  int64_t num_queries = 100;

  // Copy to device
  rmm::device_uvector<T> d_database(num_vectors * dim, stream);
  rmm::device_uvector<T> d_queries(num_queries * dim, stream);
  raft::copy(d_database.data(), h_database.data(), h_database.size(), stream);
  raft::copy(d_queries.data(), h_queries.data(), h_queries.size(), stream);

  auto database_view =
    raft::make_device_matrix_view<const T, int64_t>(d_database.data(), num_vectors, dim);
  auto queries_view =
    raft::make_device_matrix_view<const T, int64_t>(d_queries.data(), num_queries, dim);

  // Build index with L2Expanded (kmeans doesn't support CustomUDF)
  ivf_flat::index_params index_params;
  index_params.n_lists = 1024;
  index_params.metric  = cuvs::distance::DistanceType::L2Expanded;

  auto idx = ivf_flat::build(handle, index_params, database_view);
  raft::resource::sync_stream(handle);

  // Allocate output buffers
  rmm::device_uvector<int64_t> d_indices(num_queries * k, stream);
  rmm::device_uvector<float> d_distances(num_queries * k, stream);

  auto indices_view =
    raft::make_device_matrix_view<int64_t, int64_t>(d_indices.data(), num_queries, k);
  auto distances_view =
    raft::make_device_matrix_view<float, int64_t>(d_distances.data(), num_queries, k);

  // Search params with raw UDF (no point/metric_interface overhead)
  ivf_flat::search_params search_params;
  search_params.n_probes   = 32;
  search_params.metric_udf = raw_l2_udf();

  SearchResult result;

  // First search (includes JIT compilation)
  timer.start();
  ivf_flat::search(handle, search_params, idx, queries_view, indices_view, distances_view);
  raft::resource::sync_stream(handle);
  result.first_ms = timer.stop_ms();

  // Repeated searches (JIT already cached)
  std::vector<double> times;
  for (int i = 0; i < num_iterations; ++i) {
    timer.start();
    ivf_flat::search(handle, search_params, idx, queries_view, indices_view, distances_view);
    raft::resource::sync_stream(handle);
    times.push_back(timer.stop_ms());
  }

  result.median_ms = median(times);
  return result;
}

template <typename T>
BenchmarkResult run_benchmark(raft::resources& handle,
                              const char* dtype_name,
                              int64_t num_vectors,
                              int64_t dim,
                              int64_t k,
                              int num_iterations = 20)
{
  BenchmarkResult result;
  result.dtype = dtype_name;
  result.k     = k;

  // Run built-in benchmark (with fresh cache)
  auto builtin             = run_builtin_benchmark<T>(handle, num_vectors, dim, k, num_iterations);
  result.first_builtin_ms  = builtin.first_ms;
  result.median_builtin_ms = builtin.median_ms;

  // Run UDF benchmark (with fresh cache)
  auto udf             = run_udf_benchmark<T>(handle, num_vectors, dim, k, num_iterations);
  result.first_udf_ms  = udf.first_ms;
  result.median_udf_ms = udf.median_ms;

  // Run raw UDF benchmark (with fresh cache)
  auto raw             = run_raw_udf_benchmark<T>(handle, num_vectors, dim, k, num_iterations);
  result.first_raw_ms  = raw.first_ms;
  result.median_raw_ms = raw.median_ms;

  return result;
}

}  // namespace

int main(int argc, char** argv)
{
  std::string output_file = "udf_benchmark_results.csv";
  if (argc > 1) { output_file = argv[1]; }

  raft::resources handle;

  const int64_t num_vectors           = 1000000;
  const int64_t dim                   = 512;
  const std::vector<int64_t> k_values = {4, 16, 64, 256};
  const int num_iterations            = 20;

  std::vector<BenchmarkResult> results;

  std::cerr << "IVF-Flat UDF Benchmark\n";
  std::cerr << "Dataset: " << num_vectors << " vectors, " << dim << " dimensions\n";
  std::cerr << "Queries: 100, n_probes: 32, n_lists: 1024\n";
  std::cerr << "Iterations for median: " << num_iterations << "\n\n";

  // Float32 benchmarks
  std::cerr << "Running float32 benchmarks...\n";
  for (int64_t k : k_values) {
    std::cerr << "  k=" << k << "... ";
    auto result = run_benchmark<float>(handle, "float32", num_vectors, dim, k, num_iterations);
    results.push_back(result);
    std::cerr << "done\n";
  }

  // Int8 benchmarks
  std::cerr << "Running int8 benchmarks...\n";
  for (int64_t k : k_values) {
    std::cerr << "  k=" << k << "... ";
    auto result = run_benchmark<int8_t>(handle, "int8", num_vectors, dim, k, num_iterations);
    results.push_back(result);
    std::cerr << "done\n";
  }

  // Write CSV
  std::ofstream csv(output_file);
  csv << "dtype,k,first_builtin_ms,first_udf_ms,first_raw_ms,median_builtin_ms,median_udf_ms,"
         "median_raw_ms,udf_ratio,raw_ratio\n";

  for (const auto& r : results) {
    double udf_ratio = r.median_udf_ms / r.median_builtin_ms;
    double raw_ratio = r.median_raw_ms / r.median_builtin_ms;

    csv << r.dtype << "," << r.k << "," << std::fixed << std::setprecision(3) << r.first_builtin_ms
        << "," << r.first_udf_ms << "," << r.first_raw_ms << "," << r.median_builtin_ms << ","
        << r.median_udf_ms << "," << r.median_raw_ms << "," << std::setprecision(4) << udf_ratio
        << "," << raw_ratio << "\n";
  }

  csv.close();
  std::cerr << "\nResults written to: " << output_file << "\n";

  // Also print to stdout for convenience
  std::cout << "dtype,k,first_builtin_ms,first_udf_ms,first_raw_ms,median_builtin_ms,median_udf_ms,"
               "median_raw_ms,udf_ratio,raw_ratio\n";
  for (const auto& r : results) {
    double udf_ratio = r.median_udf_ms / r.median_builtin_ms;
    double raw_ratio = r.median_raw_ms / r.median_builtin_ms;

    std::cout << r.dtype << "," << r.k << "," << std::fixed << std::setprecision(3)
              << r.first_builtin_ms << "," << r.first_udf_ms << "," << r.first_raw_ms << ","
              << r.median_builtin_ms << "," << r.median_udf_ms << "," << r.median_raw_ms << ","
              << std::setprecision(4) << udf_ratio << "," << raw_ratio << "\n";
  }

  return 0;
}
