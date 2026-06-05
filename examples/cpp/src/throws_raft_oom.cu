/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuvs/neighbors/cagra.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/make_blobs.cuh>

#include <cuvs/neighbors/hnsw.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <cuvs/util/host_memory.hpp>

#include <rmm/mr/pool_memory_resource.hpp>

#include <cstdio>
#include <cstdlib>  // for exit
#include <fcntl.h>
#include <stdint.h>
#include <sys/mman.h>
#include <unistd.h>

void throw_host_oom_level_1()
{
  raft::resources res;
  raft::resource::set_workspace_to_pool_resource(res, 2 * 1024 * 1024 * 1024ull);
  int n_rows  = 1024 * 1024 * 1024;
  int n_cols  = 3;
  auto matrix = raft::make_host_matrix<float>(res, n_rows, n_cols);
}

void throw_host_oom() { throw_host_oom_level_1(); }

void throw_other_host_exception() { throw std::runtime_error("test exception"); }

void throw_device_oom_level_1()
{
  raft::resources res;
  raft::resource::set_workspace_to_pool_resource(res, 2 * 1024 * 1024 * 1024ull);
  int n_rows  = 1024 * 1024 * 1024;
  int n_cols  = 3;
  auto matrix = raft::make_device_matrix<float>(res, n_rows, n_cols);
}

void throw_device_oom() { throw_device_oom_level_1(); }

// Usage:
// Conda build (requires backtrace library):
// g++ -O2 -g -fPIC -shared -I"$CONDA_PREFIX/include" -o src/intercept_throw.so src/intercept_throw.cpp "$CONDA_PREFIX/lib/libbacktrace.a" -lpthread -ldl
// Run:
// LD_PRELOAD=src/intercept_throw.so ./build/THROWS_RAFT_OMM_EXAMPLE
int main()
{
  // throw_host_oom();
  // throw_other_host_exception();
  throw_device_oom();
  return 0;
}
