/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "common.cuh"

#include <cuvs/neighbors/cagra.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/random/make_blobs.cuh>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <future>

// A helper to split the dataset into chunks
template <typename DeviceMatrixOrView>
auto slice_matrix(DeviceMatrixOrView source,
                  typename DeviceMatrixOrView::index_type offset_rows,
                  typename DeviceMatrixOrView::index_type count_rows) {
  auto n_cols = source.extent(1);
  return raft::make_device_matrix_view<
      typename DeviceMatrixOrView::element_type,
      typename DeviceMatrixOrView::index_type>(
      source.data_handle() + offset_rows * n_cols, count_rows, n_cols);
}

// A helper to measure the execution time of a function
template <typename F, typename... Args>
void time_it(std::string label, F f, Args &&...xs) {
  auto start = std::chrono::system_clock::now();
  f(std::forward<Args>(xs)...);
  auto end = std::chrono::system_clock::now();
  auto t = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  auto t_ms = double(t.count()) / 1000.0;
  std::cout << "[" << label << "] execution time: " << t_ms << " ms"
            << std::endl;
}

void cagra_build_search_variants(
    raft::device_resources const &res,
    raft::device_matrix_view<const float, int64_t> dataset,
    raft::device_matrix_view<const float, int64_t> queries) {
  using namespace cuvs::neighbors;

  // Number of neighbors to search
  int64_t topk = 100;
  // We split the queries set into three subsets for our experiment, one for a
  // sanity check and two for measuring the performance.
  int64_t n_queries_a = queries.extent(0) / 2;
  int64_t n_queries_b = queries.extent(0) - n_queries_a;

  auto queries_a = slice_matrix(queries, 0, n_queries_a);
  auto queries_b = slice_matrix(queries, n_queries_a, n_queries_b);

  // create output arrays
  auto neighbors =
      raft::make_device_matrix<uint32_t>(res, queries.extent(0), topk);
  auto distances =
      raft::make_device_matrix<float>(res, queries.extent(0), topk);
  // slice them same as queries
  auto neighbors_a = slice_matrix(neighbors, 0, n_queries_a);
  auto distances_a = slice_matrix(distances, 0, n_queries_a);
  auto neighbors_b = slice_matrix(neighbors, n_queries_a, n_queries_b);
  auto distances_b = slice_matrix(distances, n_queries_a, n_queries_b);

  // use default index parameters
  cagra::index_params index_params;

  std::cout << "Building CAGRA index (search graph)" << std::endl;
  auto index = cagra::build(res, index_params, dataset);

  std::cout << "CAGRA index has " << index.size() << " vectors" << std::endl;
  std::cout << "CAGRA graph has degree " << index.graph_degree()
            << ", graph size [" << index.graph().extent(0) << ", "
            << index.graph().extent(1) << "]" << std::endl;

  // use default search parameters
  cagra::search_params search_params;
  // get a decent recall by increasing the internal topk list
  search_params.itopk_size = 512;

  // Another copy of search parameters to enable persistent kernel
  cagra::search_params search_params_persistent = search_params;
  search_params_persistent.persistent = true;
  // Persistent kernel only support single-cta search algorithm for now.
  search_params_persistent.algo = cagra::search_algo::SINGLE_CTA;
  // Slightly reduce the kernel grid size to make this example program work
  // smooth on workstations, which use the same GPU for other tasks (e.g.
  // rendering GUI).
  search_params_persistent.persistent_device_usage = 0.95;

  /*
  Define the big-batch setting as a baseline for measuring the throughput.

  Note, this lambda can be used by the standard and the persistent
  implementation interchangeably: the index stays the same, only search
  parameters need some adjustment.
  */
  auto search_batch =
      [&res, &index](bool needs_sync, const cagra::search_params &ps,
                     raft::device_matrix_view<const float, int64_t> queries,
                     raft::device_matrix_view<uint32_t, int64_t> neighbors,
                     raft::device_matrix_view<float, int64_t> distances) {
        cagra::search(res, ps, index, queries, neighbors, distances);
        /*
        To make a fair comparison, standard implementation needs to synchronize
        with the device to make sure the kernel has finished the work.
        Persistent kernel does not make any use of CUDA streams and blocks till
        the results are available. Hence, synchronizing with the stream is a
        waste of time in this case.
         */
        if (needs_sync) {
          raft::resource::sync_stream(res);
        }
      };

  /*
  Define the asynchronous small-batch search setting.
  The same lambda is used for both the standard and the persistent
  implementations.

  There are a few things to remember about this example though:
    1. The standard kernel is launched in the given stream (behind the `res`);
       The persistent kernel is launched implicitly; the public api call does
       not touch the stream and blocks till the results are returned. (Hence the
       optional sync at the end of the lambda.)
    2. When launched asynchronously, the standard kernel should actually have a
       separate raft::resource per-thread to achieve best performance. However,
       this requires extra management of the resource/stream pools, we don't
       include that for simplicity.
       The persistent implementation does not require any special care; you can
       safely pass a single raft::resources to all threads.
    3. This example relies on the compiler implementation to launch the async
       jobs in separate threads. This is not guaranteed, however.
       In the real world, we'd advise to use a custom thread pool for managing
       the requests.
    4. Although the API defines the arguments as device-side mdspans, we advise
       to use the host-side buffers accessible from the device, such as
       allocated by cudaHostAlloc/cudaHostRegister (or any host memory if
       HMM/ATS is enabled).
       This way, you can save some GPU resources by not manually copying the
       data in cuda streams.
  */
  auto search_async =
      [&res, &index](bool needs_sync, const cagra::search_params &ps,
                     raft::device_matrix_view<const float, int64_t> queries,
                     raft::device_matrix_view<uint32_t, int64_t> neighbors,
                     raft::device_matrix_view<float, int64_t> distances) {
        auto work_size = queries.extent(0);
        using index_type = typeof(work_size);
        // Limit the maximum number of concurrent jobs
        constexpr index_type kMaxJobs = 1000;
        std::array<std::future<void>, kMaxJobs> futures;
        for (index_type i = 0; i < work_size + kMaxJobs; i++) {
          // wait for previous job in the same slot to finish
          if (i >= kMaxJobs) {
            futures[i % kMaxJobs].wait();
          }
          // submit a new job
          if (i < work_size) {
            futures[i % kMaxJobs] = std::async(std::launch::async, [&]() {
              cagra::search(res, ps, index, slice_matrix(queries, i, 1),
                            slice_matrix(neighbors, i, 1),
                            slice_matrix(distances, i, 1));
            });
          }
        }
        /* See the remark for search_batch */
        if (needs_sync) {
          raft::resource::sync_stream(res);
        }
      };

  // Launch the baseline search: check the big-batch performance
  time_it("standard/batch A", search_batch, true, search_params, queries_a,
          neighbors_a, distances_a);
  time_it("standard/batch B", search_batch, true, search_params, queries_b,
          neighbors_b, distances_b);

  // Try to handle the same amount of work in the async setting using the
  // standard implementation.
  // (Warning: suboptimal - it uses a single stream for all async jobs)
  time_it("standard/async A", search_async, true, search_params, queries_a,
          neighbors_a, distances_a);
  time_it("standard/async B", search_async, true, search_params, queries_b,
          neighbors_b, distances_b);

  // Do the same using persistent kernel.
  time_it("persistent/async A", search_async, false, search_params_persistent,
          queries_a, neighbors_a, distances_a);
  time_it("persistent/async B", search_async, false, search_params_persistent,
          queries_b, neighbors_b, distances_b);
  /*
Here's an example output, which shows the wall time of processing the same
amount of data in a single batch vs in async mode (1 query per job):
```
CAGRA index has 1000000 vectors
CAGRA graph has degree 64, graph size [1000000, 64]
[standard/batch A] execution time: 854.645 ms
[standard/batch B] execution time: 698.58 ms
[standard/async A] execution time: 19190.6 ms
[standard/async B] execution time: 18292 ms
[I] [15:56:49.756754] Initialized the kernel 0x7ea4e55a5350 in stream
              139227270582864; job_queue size = 8192; worker_queue size = 155
[persistent/async A] execution time: 1285.65 ms
[persistent/async B] execution time: 1316.97 ms
[I] [15:56:55.756952] Destroyed the persistent runner.
```
Note, while the persistent kernel provides minimal latency for each search
request, the wall time to process all the queries in async mode (1 query per
job) is up to 2x slower than the standard kernel with the huge batch
size (100K queries). One reason for this is the non-optimal CTA size: CAGRA
kernels are automatically tuned for latency and so use large CTA sizes when the
batch size is small. Try explicitly setting the search parameter
`thread_block_size` to a small value, such as `64` or `128` if this is an issue
for you. This increases the latency of individual jobs though.
  */
}

int main() {
  raft::device_resources res;

  // Set pool memory resource with 1 GiB initial pool size. All allocations use
  // the same pool.
  rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr(
      rmm::mr::get_current_device_resource(), 1024 * 1024 * 1024ull);
  rmm::mr::set_current_device_resource(&pool_mr);

  // Create input arrays.
  int64_t n_samples = 1000000;
  int64_t n_dim = 128;
  int64_t n_queries = 100000;
  auto dataset =
      raft::make_device_matrix<float, int64_t>(res, n_samples, n_dim);
  auto queries =
      raft::make_device_matrix<float, int64_t>(res, n_queries, n_dim);
  generate_dataset(res, dataset.view(), queries.view());

  // run the interesting part of the program
  cagra_build_search_variants(res, raft::make_const_mdspan(dataset.view()),
                              raft::make_const_mdspan(queries.view()));
}
