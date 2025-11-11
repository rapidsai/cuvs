/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "util.hpp"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <sqlite3.h>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace cuvs::bench {

namespace detail {

// Strip parameters from NVTX range names for grouping
// Removes everything inside and including (), <>, and {}
inline std::string strip_nvtx_parameters(const std::string& name)
{
  if (name.empty()) { return ""; }

  std::string result;
  result.reserve(name.size());
  int paren_depth = 0;
  int angle_depth = 0;
  int brace_depth = 0;

  for (char c : name) {
    // Track nesting depth
    if (c == '(') {
      paren_depth++;
    } else if (c == ')') {
      paren_depth--;
    } else if (c == '<') {
      angle_depth++;
    } else if (c == '>') {
      angle_depth--;
    } else if (c == '{') {
      brace_depth++;
    } else if (c == '}') {
      brace_depth--;
    } else if (paren_depth == 0 && angle_depth == 0 && brace_depth == 0) {
      // Only add character if we're not inside any brackets
      result += c;
    }
  }

  // Trim trailing whitespace
  while (!result.empty() && std::isspace(result.back())) {
    result.pop_back();
  }

  return result;
}

// Extract CPU-only NVTX statistics
inline std::map<std::string, double> extract_cpu_stats(sqlite3* db, int64_t algo_bench_domain_id)
{
  std::map<std::string, double> cpu_times;

  const char* cpu_query_sql =
    "SELECT strip_params(text) as base_name, SUM(end - start) as total_duration "
    "FROM NVTX_EVENTS "
    "WHERE end IS NOT NULL "
    "  AND eventType = 59 "
    "  AND domainId != ? "
    "GROUP BY base_name "
    "HAVING base_name IS NOT NULL "
    "ORDER BY total_duration DESC";
  sqlite3_stmt* stmt = nullptr;
  int rc             = sqlite3_prepare_v2(db, cpu_query_sql, -1, &stmt, nullptr);

  if (rc == SQLITE_OK) {
    sqlite3_bind_int64(stmt, 1, algo_bench_domain_id);

    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
      const char* base_name_ptr = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
      std::string base_name     = base_name_ptr ? base_name_ptr : "";
      int64_t duration_ns       = sqlite3_column_int64(stmt, 1);
      cpu_times[base_name]      = duration_ns / 1e9;
    }
    sqlite3_finalize(stmt);
  }

  return cpu_times;
}

// Extract CPU and GPU NVTX statistics with correlation
inline std::pair<std::map<std::string, double>, std::map<std::string, double>>
extract_cpu_gpu_stats(sqlite3* db,
                      int64_t algo_bench_domain_id,
                      const std::vector<std::string>& activity_tables)
{
  std::map<std::string, double> cpu_times;
  std::map<std::string, double> gpu_times;

  if (activity_tables.empty()) { return {cpu_times, gpu_times}; }

  // Build union query for GPU activities
  std::string gpu_union = "(";
  for (size_t i = 0; i < activity_tables.size(); ++i) {
    if (i > 0) gpu_union += " UNION ALL ";
    gpu_union += "SELECT correlationId, start, end FROM " + activity_tables[i];
  }
  gpu_union += ")";

  // Accumulate times by base_name
  std::map<std::string, int64_t> cpu_ns;
  std::map<std::string, int64_t> gpu_ns;

  // Query 1: NVTX events sorted by start, then end
  const char* nvtx_query =
    "SELECT start, end, globalTid, strip_params(text) "
    "FROM NVTX_EVENTS "
    "WHERE end IS NOT NULL "
    "  AND eventType = 59 "
    "  AND domainId != ? "
    "  AND strip_params(text) IS NOT NULL "
    "ORDER BY start, end";

  // Query 2: Runtime+GPU events sorted by start, then end
  std::string runtime_query =
    "SELECT r.start, r.end, r.globalTid, MIN(ga.start), MAX(ga.end) "
    "FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
    "INNER JOIN (" +
    gpu_union +
    ") ga ON r.correlationId = ga.correlationId "
    "GROUP BY r.start, r.end, r.globalTid, r.correlationId "
    "ORDER BY r.start, r.end";

  sqlite3_stmt* nvtx_stmt    = nullptr;
  sqlite3_stmt* runtime_stmt = nullptr;

  if (sqlite3_prepare_v2(db, nvtx_query, -1, &nvtx_stmt, nullptr) != SQLITE_OK) {
    return {cpu_times, gpu_times};
  }
  sqlite3_bind_int64(nvtx_stmt, 1, algo_bench_domain_id);

  if (sqlite3_prepare_v2(db, runtime_query.c_str(), -1, &runtime_stmt, nullptr) != SQLITE_OK) {
    sqlite3_finalize(nvtx_stmt);
    return {cpu_times, gpu_times};
  }

  // Structure to hold runtime events in a sliding window queue
  struct RuntimeEvent {
    int64_t rt_start;
    int64_t rt_end;
    int64_t globalTid;
    int64_t gpu_start;
    int64_t gpu_end;
  };
  std::deque<RuntimeEvent> runtime_queue;
  bool runtime_exhausted = false;

  // Process each NVTX event
  while (sqlite3_step(nvtx_stmt) == SQLITE_ROW) {
    int64_t nvtx_start    = sqlite3_column_int64(nvtx_stmt, 0);
    int64_t nvtx_end      = sqlite3_column_int64(nvtx_stmt, 1);
    int64_t nvtx_tid      = sqlite3_column_int64(nvtx_stmt, 2);
    const char* name      = reinterpret_cast<const char*>(sqlite3_column_text(nvtx_stmt, 3));
    std::string base_name = name ? name : "";

    // Accumulate CPU time
    cpu_ns[base_name] += (nvtx_end - nvtx_start);

    // Remove runtime events that start before this NVTX event starts
    // (they can't match this or any future NVTX events since NVTX is sorted)
    while (!runtime_queue.empty() && runtime_queue.front().rt_start < nvtx_start) {
      runtime_queue.pop_front();
    }

    // Load more runtime events into the queue until we have all that could match
    while (!runtime_exhausted) {
      // Peek: do we need more events?
      if (!runtime_queue.empty() && runtime_queue.back().rt_start > nvtx_end) {
        // We have enough events in queue for this NVTX
        break;
      }

      // Fetch next runtime event
      if (sqlite3_step(runtime_stmt) == SQLITE_ROW) {
        RuntimeEvent evt;
        evt.rt_start  = sqlite3_column_int64(runtime_stmt, 0);
        evt.rt_end    = sqlite3_column_int64(runtime_stmt, 1);
        evt.globalTid = sqlite3_column_int64(runtime_stmt, 2);
        evt.gpu_start = sqlite3_column_int64(runtime_stmt, 3);
        evt.gpu_end   = sqlite3_column_int64(runtime_stmt, 4);
        runtime_queue.push_back(evt);
      } else {
        runtime_exhausted = true;
        break;
      }
    }

    // Scan the queue to find all matching runtime events
    int64_t gpu_min = INT64_MAX;
    int64_t gpu_max = INT64_MIN;
    bool found_any  = false;

    for (const auto& rt : runtime_queue) {
      // Stop if runtime event starts after NVTX event ends
      if (rt.rt_start > nvtx_end) { break; }

      // Check if this runtime event is contained in the NVTX event
      if (rt.globalTid == nvtx_tid && rt.rt_start >= nvtx_start && rt.rt_end <= nvtx_end) {
        gpu_min   = std::min(gpu_min, rt.gpu_start);
        gpu_max   = std::max(gpu_max, rt.gpu_end);
        found_any = true;
      }
    }

    // Record GPU time
    if (found_any && gpu_max > gpu_min) { gpu_ns[base_name] += (gpu_max - gpu_min); }
  }

  sqlite3_finalize(nvtx_stmt);
  sqlite3_finalize(runtime_stmt);

  // Convert to seconds
  for (const auto& [name, ns] : cpu_ns) {
    cpu_times[name] = ns / 1e9;
  }
  for (const auto& [name, ns] : gpu_ns) {
    gpu_times[name] = ns / 1e9;
  }

  return {cpu_times, gpu_times};
}

// Common setup: open database, register functions, find domain ID, discover GPU tables
inline std::tuple<sqlite3*, int64_t, std::vector<std::string>> setup_nvtx_database(
  const std::string& sqlite_file)
{
  sqlite3* db = nullptr;
  int rc      = sqlite3_open_v2(sqlite_file.c_str(), &db, SQLITE_OPEN_READONLY, nullptr);
  if (rc != SQLITE_OK) {
    if (db) sqlite3_close(db);
    return {nullptr, -1, {}};
  }

  // Register custom SQL function to strip parameters from NVTX range names
  auto strip_params_func = [](sqlite3_context* ctx, int argc, sqlite3_value** argv) {
    if (argc != 1) {
      sqlite3_result_null(ctx);
      return;
    }

    const char* text = reinterpret_cast<const char*>(sqlite3_value_text(argv[0]));
    if (!text) {
      sqlite3_result_null(ctx);
      return;
    }

    std::string stripped = strip_nvtx_parameters(text);
    if (stripped.empty()) {
      sqlite3_result_null(ctx);
    } else {
      sqlite3_result_text(ctx, stripped.c_str(), stripped.length(), SQLITE_TRANSIENT);
    }
  };

  sqlite3_create_function(
    db, "strip_params", 1, SQLITE_UTF8, nullptr, strip_params_func, nullptr, nullptr);

  // Find the domainId for "algo benchmark" domain to exclude it
  const char* find_domain_sql =
    "SELECT domainId FROM NVTX_EVENTS WHERE text = 'algo benchmark' LIMIT 1";
  sqlite3_stmt* domain_stmt = nullptr;
  int domain_rc             = sqlite3_prepare_v2(db, find_domain_sql, -1, &domain_stmt, nullptr);

  int64_t algo_bench_domain_id = -1;
  if (domain_rc == SQLITE_OK && sqlite3_step(domain_stmt) == SQLITE_ROW) {
    algo_bench_domain_id = sqlite3_column_int64(domain_stmt, 0);
  }
  if (domain_stmt) sqlite3_finalize(domain_stmt);

  // Check if GPU activity tables exist
  const char* check_tables_sql =
    "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND "
    "(name='CUPTI_ACTIVITY_KIND_RUNTIME' OR name='CUPTI_ACTIVITY_KIND_KERNEL')";
  sqlite3_stmt* check_stmt = nullptr;
  bool has_gpu_tables      = false;
  if (sqlite3_prepare_v2(db, check_tables_sql, -1, &check_stmt, nullptr) == SQLITE_OK) {
    if (sqlite3_step(check_stmt) == SQLITE_ROW) {
      has_gpu_tables = sqlite3_column_int64(check_stmt, 0) == 2;
    }
    sqlite3_finalize(check_stmt);
  }

  std::vector<std::string> activity_tables;
  if (has_gpu_tables) {
    // Find all CUPTI_ACTIVITY_KIND_* tables (except RUNTIME)
    const char* find_tables_sql =
      "SELECT name FROM sqlite_master WHERE type='table' "
      "AND name LIKE 'CUPTI_ACTIVITY_KIND_%' AND name != 'CUPTI_ACTIVITY_KIND_RUNTIME'";
    sqlite3_stmt* tables_stmt = nullptr;

    if (sqlite3_prepare_v2(db, find_tables_sql, -1, &tables_stmt, nullptr) == SQLITE_OK) {
      while (sqlite3_step(tables_stmt) == SQLITE_ROW) {
        const char* table_name = reinterpret_cast<const char*>(sqlite3_column_text(tables_stmt, 0));
        if (table_name) {
          // Check if this table has start, end, and correlationId columns
          std::string check_cols_sql = "PRAGMA table_info(" + std::string(table_name) + ")";
          sqlite3_stmt* cols_stmt    = nullptr;
          bool has_start = false, has_end = false, has_corr = false;

          if (sqlite3_prepare_v2(db, check_cols_sql.c_str(), -1, &cols_stmt, nullptr) ==
              SQLITE_OK) {
            while (sqlite3_step(cols_stmt) == SQLITE_ROW) {
              const char* col_name =
                reinterpret_cast<const char*>(sqlite3_column_text(cols_stmt, 1));
              if (col_name) {
                if (strcmp(col_name, "start") == 0) has_start = true;
                if (strcmp(col_name, "end") == 0) has_end = true;
                if (strcmp(col_name, "correlationId") == 0) has_corr = true;
              }
            }
            sqlite3_finalize(cols_stmt);
          }

          if (has_start && has_end && has_corr) { activity_tables.push_back(table_name); }
        }
      }
      sqlite3_finalize(tables_stmt);
    }
  }

  return {db, algo_bench_domain_id, activity_tables};
}

// Extract NVTX statistics from SQLite database
// Returns pair of (cpu_times, gpu_times) maps with times in seconds
inline std::pair<std::map<std::string, double>, std::map<std::string, double>>
extract_nvtx_stats_from_sqlite(const std::string& sqlite_file)
{
  std::map<std::string, double> cpu_times;
  std::map<std::string, double> gpu_times;

  auto [db, algo_bench_domain_id, activity_tables] = setup_nvtx_database(sqlite_file);
  if (!db) { return {cpu_times, gpu_times}; }

  // Choose extraction method based on GPU table availability
  if (!activity_tables.empty()) {
    std::tie(cpu_times, gpu_times) =
      extract_cpu_gpu_stats(db, algo_bench_domain_id, activity_tables);
  } else {
    cpu_times = extract_cpu_stats(db, algo_bench_domain_id);
  }

  sqlite3_close(db);
  return {cpu_times, gpu_times};
}

// Get process name from PID
inline std::string get_process_name(pid_t pid)
{
  std::ifstream comm_file("/proc/" + std::to_string(pid) + "/comm");
  std::string name;
  if (comm_file.is_open()) { std::getline(comm_file, name); }
  return name;
}

// Get process executable path from PID
inline std::string get_process_exe_path(pid_t pid)
{
  char buffer[PATH_MAX];
  ssize_t len =
    readlink(("/proc/" + std::to_string(pid) + "/exe").c_str(), buffer, sizeof(buffer) - 1);
  if (len != -1) {
    buffer[len] = '\0';
    return std::string(buffer);
  }
  return "";
}

// Get parent PID from a given PID
inline pid_t get_parent_pid(pid_t pid)
{
  std::ifstream stat_file("/proc/" + std::to_string(pid) + "/stat");
  if (stat_file.is_open()) {
    std::string line;
    std::getline(stat_file, line);

    // stat file format: pid (comm) state ppid ...
    size_t last_paren = line.rfind(')');
    if (last_paren != std::string::npos) {
      std::istringstream iss(line.substr(last_paren + 1));
      char state;
      pid_t ppid;
      iss >> state >> ppid;
      return ppid;
    }
  }
  return 0;
}

// Check if a process has 'launch' in its command line
inline bool has_launch_arg(pid_t pid)
{
  std::ifstream cmdline_file("/proc/" + std::to_string(pid) + "/cmdline");
  if (cmdline_file.is_open()) {
    std::string arg;
    while (std::getline(cmdline_file, arg, '\0')) {
      if (arg == "launch") { return true; }
    }
  }
  return false;
}

// Detect if the program was launched by nsys with 'launch' subcommand
// by walking up the process tree
inline std::optional<std::string> detect_nsys_launch()
{
  pid_t parent_pid = getppid();

  // Walk up the process tree (max 10 levels)
  for (int depth = 0; depth < 10 && parent_pid > 1; ++depth) {
    std::string parent_name = get_process_name(parent_pid);

    // Check if this process is nsys
    if (parent_name.find("nsys") != std::string::npos) {
      // Verify it has the 'launch' argument
      if (has_launch_arg(parent_pid)) {
        std::string nsys_exe = get_process_exe_path(parent_pid);
        return nsys_exe.empty() ? std::optional<std::string>(parent_name)
                                : std::optional<std::string>(nsys_exe);
      }
    }

    // Move to the next parent
    pid_t grandparent = get_parent_pid(parent_pid);
    if (grandparent == 0 || grandparent == parent_pid) { break; }
    parent_pid = grandparent;
  }

  return std::nullopt;
}

}  // namespace detail

struct nsys_launcher {
  nsys_launcher()
  {
    std::lock_guard<std::mutex> lock(mtx);
    nsys_exe = detail::detect_nsys_launch();
  }

  bool is_enabled() const
  {
    std::lock_guard<std::mutex> lock(mtx);
    return nsys_exe.has_value();
  }

  bool start(const std::string& output_path) const
  {
    std::lock_guard<std::mutex> lock(mtx);
    if (nsys_exe.has_value()) {
      std::string cmd =
        nsys_exe.value() + " start --export=sqlite -o " + output_path + " >/dev/null 2>&1";
      auto res = system(cmd.c_str());
      if (res != 0) {
        log_warn(
          "Failed to start nsys: %s with error %d. Disabling profiler stats.", cmd.c_str(), res);
        nsys_exe.reset();
        return false;
      }
      return true;
    }
    return false;
  }

  bool stop() const
  {
    std::lock_guard<std::mutex> lock(mtx);
    if (nsys_exe.has_value()) {
      std::string cmd = nsys_exe.value() + " stop >/dev/null 2>&1";
      auto res        = system(cmd.c_str());
      if (res != 0) {
        log_warn(
          "Failed to start nsys: %s with error %d. Disabling profiler stats.", cmd.c_str(), res);
        nsys_exe.reset();
        return false;
      }
      return true;
    }
    return false;
  }

 private:
  mutable std::mutex mtx;
  mutable std::optional<std::string> nsys_exe;
};

/**
 * @brief Returns the nsys executable path if launched via 'nsys launch'.
 *
 * Detects if the program is running under 'nsys launch' by walking up the process tree.
 * Returns std::nullopt for 'nsys profile' or other modes to avoid interference.
 *
 */
inline const nsys_launcher& get_nsys_launcher()
{
  static const nsys_launcher nsys_launcher;
  return nsys_launcher;
}

struct nvtx_stats {
  explicit nvtx_stats(::benchmark::State& state) : state_(state)
  {
    if (state_.thread_index() != 0) { return; }
    if (get_nsys_launcher().is_enabled()) { get_nsys_launcher().start(report_path); }
  }

  ~nvtx_stats()
  {
    if (state_.thread_index() != 0) { return; }
    if (!get_nsys_launcher().is_enabled()) { return; }

    // Stop nsys profiling
    if (!get_nsys_launcher().stop()) { return; }

    auto sql_start = std::chrono::high_resolution_clock::now();
    log_info("Extracting NVTX stats from SQLite database...");

    // Extract NVTX statistics from SQLite database
    std::string sqlite_file     = report_path + ".sqlite";
    std::string nsys_file       = report_path + ".nsys-rep";
    auto [cpu_times, gpu_times] = detail::extract_nvtx_stats_from_sqlite(sqlite_file);

    auto sql_end      = std::chrono::high_resolution_clock::now();
    auto sql_duration = std::chrono::duration_cast<std::chrono::milliseconds>(sql_end - sql_start);
    log_info("NVTX stats SQL query took %d ms (%zu CPU ranges, %zu GPU ranges)",
             static_cast<int>(sql_duration.count()),
             cpu_times.size(),
             gpu_times.size());

    // Insert counters into benchmark state
    for (const auto& [range_name, cpu_time] : cpu_times) {
      state_.counters.insert(
        {{"CPU::" + range_name, {cpu_time, benchmark::Counter::kAvgIterations}}});
    }

    for (const auto& [range_name, gpu_time] : gpu_times) {
      if (gpu_time > 0.0) {
        state_.counters.insert(
          {{"GPU::" + range_name, {gpu_time, benchmark::Counter::kAvgIterations}}});
      }
    }

    // Clean up generated files (ignore errors if files don't exist)
    std::remove(sqlite_file.c_str());
    std::remove(nsys_file.c_str());
  }

 private:
  std::string report_path = std::tmpnam(nullptr);
  ::benchmark::State& state_;
};

};  // namespace cuvs::bench
