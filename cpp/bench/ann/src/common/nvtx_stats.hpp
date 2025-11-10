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

  // Build union subquery for all GPU activity tables
  std::string gpu_union = "(";
  for (size_t i = 0; i < activity_tables.size(); ++i) {
    if (i > 0) gpu_union += " UNION ALL \n";
    gpu_union +=
      "SELECT correlationId, start as gpu_start, end as gpu_end FROM " + activity_tables[i] + "\n";
  }
  gpu_union += ")";

  // Comprehensive query: aggregates CPU times and GPU times together
  // Optimized by:
  // 1. Pre-computing strip_params once with ROWID tracking
  // 2. Computing CPU times separately (no joins, very fast)
  // 3. Pre-joining RUNTIME with GPU activities before range join
  // 4. Using ROWID-based grouping to avoid redundant operations
  // This approach is ~14x faster than the naive double-GROUP-BY approach
  std::string comprehensive_query =
    "WITH nvtx_base AS ( \n"
    "  SELECT \n"
    "    ROWID as id, \n"
    "    globalTid, \n"
    "    start, \n"
    "    end, \n"
    "    text as full_name \n"
    "  FROM NVTX_EVENTS \n"
    "  WHERE end IS NOT NULL \n"
    "    AND eventType = 59 \n"
    "    AND domainId != ? \n"
    "    AND full_name IS NOT NULL \n"
    "), \n"
    "cpu_times AS ( \n"
    "  SELECT \n"
    "    full_name, \n"
    "    SUM(end - start) as total_cpu_duration \n"
    "  FROM nvtx_base \n"
    "  GROUP BY full_name \n"
    "), \n"
    "gpu_activities AS ( \n"
    "  SELECT correlationId, MIN(gpu_start) as gpu_start, MAX(gpu_end) as gpu_end \n"
    "  FROM " +
    gpu_union +
    " \n"
    "  GROUP BY correlationId \n"
    "), \n"
    "runtime_gpu AS ( \n"
    "  SELECT \n"
    "    r.globalTid, \n"
    "    r.start as rt_start, \n"
    "    r.end as rt_end, \n"
    "    ga.gpu_start, \n"
    "    ga.gpu_end \n"
    "  FROM CUPTI_ACTIVITY_KIND_RUNTIME r \n"
    "  INNER JOIN gpu_activities ga ON r.correlationId = ga.correlationId \n"
    "), \n"
    "gpu_per_nvtx AS ( \n"
    "  SELECT \n"
    "    nb.id, \n"
    "    nb.full_name, \n"
    "    MAX(rg.gpu_end) - MIN(rg.gpu_start) as gpu_duration \n"
    "  FROM nvtx_base nb \n"
    "  LEFT JOIN runtime_gpu rg \n"
    "     ON rg.globalTid = nb.globalTid AND rg.rt_start >= nb.start AND rg.rt_end <= nb.end \n"
    "  GROUP BY nb.id, nb.full_name \n"
    "), \n"
    "gpu_times AS ( \n"
    "  SELECT \n"
    "    full_name, \n"
    "    SUM(COALESCE(gpu_duration, 0)) as total_gpu_duration \n"
    "  FROM gpu_per_nvtx \n"
    "  GROUP BY full_name \n"
    "), \n"
    "merged_times AS (SELECT \n"
    "  ct.full_name as full_name, \n"
    "  ct.total_cpu_duration, \n"
    "  COALESCE(gt.total_gpu_duration, 0) as total_gpu_duration \n"
    "FROM cpu_times ct \n"
    "LEFT JOIN gpu_times gt ON ct.full_name = gt.full_name \n"
    ") \n"
    "SELECT \n"
    "  strip_params(full_name) as base_name, \n"
    "  SUM(total_cpu_duration) as total_cpu_duration, \n"
    "  SUM(total_gpu_duration) as total_gpu_duration \n"
    "FROM merged_times \n"
    "GROUP BY base_name \n"
    "ORDER BY total_cpu_duration DESC";
  sqlite3_stmt* stmt = nullptr;
  int rc             = sqlite3_prepare_v2(db, comprehensive_query.c_str(), -1, &stmt, nullptr);

  if (rc == SQLITE_OK) {
    sqlite3_bind_int64(stmt, 1, algo_bench_domain_id);

    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
      const char* base_name_ptr = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
      std::string base_name     = base_name_ptr ? base_name_ptr : "";
      int64_t cpu_duration_ns   = sqlite3_column_int64(stmt, 1);

      // Store CPU time (already summed by SQL)
      cpu_times[base_name] = cpu_duration_ns / 1e9;

      // Store GPU time if available (already summed by SQL)
      if (sqlite3_column_type(stmt, 2) != SQLITE_NULL) {
        int64_t gpu_duration_ns = sqlite3_column_int64(stmt, 2);
        if (gpu_duration_ns > 0) { gpu_times[base_name] = gpu_duration_ns / 1e9; }
      }
    }
    sqlite3_finalize(stmt);
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
