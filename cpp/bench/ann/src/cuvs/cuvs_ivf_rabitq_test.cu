/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cassert>
#include <iomanip>
#include <iostream>
#include <queue>
#include <vector>

#include <neighbors/ivf_rabitq/gpu_index/ivf_gpu.cuh>
#include <neighbors/ivf_rabitq/gpu_index/searcher_gpu.cuh>
#include <neighbors/ivf_rabitq/utils/IO.hpp>
#include <neighbors/ivf_rabitq/utils/StopW.hpp>
#include <neighbors/ivf_rabitq/utils/space.hpp>

#include <raft/core/device_resources.hpp>

#include <cuvs/cluster/kmeans.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

namespace {

using namespace cuvs::neighbors::ivf_rabitq::detail;

// search parameters
size_t TOPK          = 10;
size_t ROUND         = 3;
size_t EXPAND_FACTOR = 1;

int test_ivf_rabitq_construct_batch(raft::resources const& handle, int argc, char* argv[])
{
  assert(argc >= 4);
  char* DATASET = argv[1];
  size_t K      = static_cast<size_t>(atoi(argv[2]));
  int B         = atoi(argv[3]);
  // whether to use fast quantize method
  bool fast_quantize_flag = false;
  if (argc > 4) {
    std::string arg_str = argv[4];
    fast_quantize_flag  = (arg_str == "true" || arg_str == "1");
  }
  if (fast_quantize_flag) {
    std::cout << "Using Fast Quantization Config!" << std::endl;
  } else {
    std::cout << "Using Normal Quantization Config!" << std::endl;
  }
  // B must be between 2 and 9, inclusive.
  assert(B >= 2 && B <= 9);

  // number of k-means iterations
  int kmeans_n_iters = 20;
  if (argc > 5) { kmeans_n_iters = atoi(argv[5]); }
  bool clusters_from_file = false;
  if (argc > 6) {
    std::string arg_str = argv[6];
    clusters_from_file  = (arg_str != "false" && arg_str != "0");
  }

  char data_file[500];
  char centroids_file[500];
  char cids_file[500];
  char ivf_file[500];

  sprintf(data_file, "%s/base.fvecs", DATASET);
  if (clusters_from_file) {
    sprintf(centroids_file, "%s/centroid_%ld.fvecs", DATASET, K);
    sprintf(cids_file, "%s/cluster_id_%ld.ivecs", DATASET, K);
  }
  sprintf(ivf_file, "ivf_exhaf%d_gpu_batch.index", B);

  // Load data from file (using your load_vecs template functions).
  raft::host_matrix<float, int64_t> data      = raft::make_host_matrix<float, int64_t>(0, 0);
  raft::host_matrix<float, int64_t> centroids = raft::make_host_matrix<float, int64_t>(0, 0);
  raft::host_matrix<uint32_t, int64_t> cids   = raft::make_host_matrix<uint32_t, int64_t>(0, 0);
  ;  // Assume cids are stored as uint32_t
  load_vecs<float, raft::host_matrix<float, int64_t>>(data_file, data);
  if (clusters_from_file) {
    load_vecs<float, raft::host_matrix<float, int64_t>>(centroids_file, centroids);
    load_vecs<PID, raft::host_matrix<uint32_t, int64_t>>(cids_file, cids);
  }

  size_t N   = data.extent(0);
  size_t DIM = data.extent(1);

  std::cout << "Data loaded:\n\tN: " << N << "\n\tDIM: " << DIM << std::endl;

  StopW stopw;
  if (!clusters_from_file) {
    // Allocate host memory for centroids and cluster IDs
    // Timer for clustering
    StopW clustering_timer;
    stopw.reset();

    // Create RAFT resources handle
    raft::resources handle;
    auto stream = raft::resource::get_cuda_stream(handle);

    // Create device matrices - using int for extents
    auto d_data      = raft::make_device_matrix<float, int>(handle, N, DIM);
    auto d_centroids = raft::make_device_matrix<float, int>(handle, K, DIM);
    auto d_labels    = raft::make_device_vector<uint32_t, int>(handle, N);

    // Perform k-means clustering using cuVS
    std::cout << "\n=== Starting K-means Clustering ===" << std::endl;

    // === Balanced K-means ===
    std::cout << "Using Balanced K-means for better cluster size distribution..." << std::endl;

    // Copy data to device
    cudaMemcpyAsync(d_data.data_handle(),
                    data.data_handle(),
                    N * DIM * sizeof(float),
                    cudaMemcpyHostToDevice,
                    stream);

    // Set up balanced k-means parameters
    cuvs::cluster::kmeans::balanced_params params;
    params.n_iters = kmeans_n_iters;
    params.metric  = cuvs::distance::DistanceType::L2Expanded;

    // Create views
    auto data_view = raft::make_device_matrix_view<const float, int>(d_data.data_handle(), N, DIM);
    auto centroids_view =
      raft::make_device_matrix_view<float, int>(d_centroids.data_handle(), K, DIM);
    auto labels_view = raft::make_device_vector_view<uint32_t, int>(d_labels.data_handle(), N);

    // Perform balanced k-means
    // clustering_timer.reset();
    cuvs::cluster::kmeans::fit_predict(handle, params, data_view, centroids_view, labels_view);
    // cudaStreamSynchronize(stream);
    // float clustering_time = clustering_timer.getElapsedTimeMili();
    //
    // std::cout << "Balanced k-means clustering completed in "
    //           << clustering_time / 1000.0f << " seconds" << std::endl;
    //
    // std::cout << "Clustering results transferred to host memory" << std::endl;

    // Now construct the IVF index with the computed centroids and cluster IDs
    std::cout << "\n=== Constructing IVF Index ===" << std::endl;

    // Create an IVFGPU instance. (Its constructor will allocate device memory as needed.)
    IVFGPU ivf(handle, N, DIM, K, B, true);

    // Construct the index (this function performs necessary host-to-device transfers internally).
    ivf.construct_on_gpu(
      d_data.data_handle(), d_centroids.data_handle(), d_labels.data_handle(), fast_quantize_flag);
    //    ivf.construct(data_device.data_handle(), centroids_device.data_handle(),
    //    labels_device.data_handle(), fast_quantize_flag);
    float minutes = stopw.getElapsedTimeMili() / 1000.0f / 60.0f;
    float seconds = stopw.getElapsedTimeMili() / 1000.0f;
    std::cout << "IVFGPU constructed\n";

    // Save the index to a file.
    ivf.save(ivf_file, true);

    std::cout << "Indexing time: " << seconds << " seconds\n";
  } else {
    load_vecs<float, raft::host_matrix<float, int64_t>>(centroids_file, centroids);
    load_vecs<PID, raft::host_matrix<uint32_t, int64_t>>(cids_file, cids);

    // Now construct the IVF index with the computed centroids and cluster IDs
    std::cout << "\n=== Constructing IVF Index ===" << std::endl;
    stopw.reset();
    // Create an IVFGPU instance. (Its constructor will allocate device memory as needed.)
    IVFGPU ivf(handle, N, DIM, K, B, true);

    // Construct the index (this function performs necessary host-to-device transfers internally).
    ivf.construct(
      data.data_handle(), centroids.data_handle(), cids.data_handle(), fast_quantize_flag);
    float minutes = stopw.getElapsedTimeMili() / 1000.0f / 60.0f;
    float seconds = stopw.getElapsedTimeMili() / 1000.0f;
    std::cout << "IVFGPU constructed\n";

    // Save the index to a file.
    ivf.save(ivf_file, true);

    std::cout << "Indexing time: " << seconds << " seconds\n";
  }

  return 0;
}

template <typename T>
std::vector<T> horizontal_avg_standalone(const std::vector<std::vector<T>>& data)
{
  size_t rows = data.size();
  size_t cols = data[0].size();

  for (auto& row : data) {
    assert(row.size() == cols);
  }

  std::vector<T> avg(cols, 0);
  for (auto& row : data) {
    for (size_t j = 0; j < cols; ++j) {
      avg[j] += row[j];
    }
  }

  for (size_t j = 0; j < cols; ++j) {
    avg[j] /= rows;
  }

  return avg;
}

double get_ratio_standalone(size_t numq,
                            const raft::host_matrix<float, int64_t>& query,
                            const raft::host_matrix<float, int64_t>& data,
                            const raft::host_matrix<uint32_t, int64_t>& gt,
                            PID* ann_results,
                            size_t K,
                            float (*dist_func)(const float*, const float*, size_t))
{
  std::priority_queue<float> gt_distances;
  std::priority_queue<float> ann_distances;

  for (size_t i = 0; i < K; ++i) {
    PID gt_id  = gt(numq, i);
    PID ann_id = ann_results[i];
    if (gt_id > data.extent(0) || ann_id > data.extent(0)) { continue; }
    gt_distances.emplace(dist_func(&query(numq, 0), &data(gt_id, 0), data.extent(1)));
    ann_distances.emplace(dist_func(&query(numq, 0), &data(ann_id, 0), data.extent(1)));
  }

  double ret     = 0;
  size_t valid_k = 0;

  while (!gt_distances.empty()) {
    if (gt_distances.top() > 1e-5) {
      ret += std::sqrt((double)ann_distances.top() / gt_distances.top());
      //            std::cout << "ground truth l2 distance: " << gt_distances.top()
      //                      << ", search result distance: " << ann_distances.top() << std::endl;
      ++valid_k;
    }
    gt_distances.pop();
    ann_distances.pop();
  }

  if (valid_k == 0) { return 1.0 * K; }
  //    printf("ret = %f, valid_k = %zu, K = %zu\n", ret, valid_k, K);
  //    fflush(stdout);
  return ret / valid_k * K;
}

int test_ivf_rabitq_search_batch(raft::resources const& handle, int argc, char* argv[])
{
  cudaSetDevice(0);
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
    std::cerr << "No CUDA-capable devices found." << std::endl;
    return 1;
  }
  std::cout << "Found " << deviceCount << " CUDA device(s)." << std::endl;

  assert(argc >= 4);
  char* DATASET = argv[1];
  int B         = atoi(argv[3]);
  //   int query_bits            = -1;
  bool rabitq_quantize_flag = true;
  std::string mode;
  if (argc > 7) {
    mode = argv[7];  // 4 modes: lut32, lut16, quant8, quant4
  } else {
    mode = "quant4";  // by default, using 4 bit quantization for queries
  }
  if (argc > 8) {
    std::string arg_str  = argv[8];
    rabitq_quantize_flag = (arg_str == "true" || arg_str == "1");
  }
  if (argc > 9) { EXPAND_FACTOR = atoi(argv[9]); }
  assert(B >= 2 && B <= 9);

  char data_file[500];
  char query_file[500];
  char gt_file[500];
  char ivf_file[500];

  sprintf(data_file, "%s/base.fvecs", DATASET);
  sprintf(query_file, "%s/query.fvecs", DATASET);
  sprintf(gt_file, "%s/groundtruth.ivecs", DATASET);
  sprintf(ivf_file, "ivf_exhaf%d_gpu_batch.index", B);
  //    sprintf(ivf_file, "../bin/test_gpu.index", DATASET, B);

  raft::host_matrix<float, int64_t> data  = raft::make_host_matrix<float, int64_t>(0, 0);
  raft::host_matrix<float, int64_t> query = raft::make_host_matrix<float, int64_t>(0, 0);
  raft::host_matrix<uint32_t, int64_t> gt = raft::make_host_matrix<uint32_t, int64_t>(0, 0);

  load_vecs<float, raft::host_matrix<float, int64_t>>(data_file, data);
  load_vecs_k<float, raft::host_matrix<float, int64_t>>(query_file, query, EXPAND_FACTOR);
  load_vecs_k<PID, raft::host_matrix<uint32_t, int64_t>>(gt_file, gt, EXPAND_FACTOR);

  size_t N   = data.extent(0);
  size_t DIM = data.extent(1);
  size_t NQ  = query.extent(0);
  //    NQ = 1;

  std::cout << "data loaded\n";
  std::cout << "\tN: " << N << '\n' << "\tDIM: " << DIM << '\n';
  std::cout << "query loaded\n";
  std::cout << "\tNQ: " << NQ << '\n';

  StopW stopw;
  IVFGPU ivf(handle);
  ivf.load_transposed(ivf_file);

  std::vector<size_t> all_nprobes;
  // ssss
  all_nprobes.push_back(1);
  all_nprobes.push_back(5);
  all_nprobes.push_back(6);
  all_nprobes.push_back(8);
  for (size_t i = 10; i < 200; i += 5) {
    all_nprobes.push_back(i);
  }
  for (size_t i = 200; i < 400; i += 20) {
    all_nprobes.push_back(i);
  }
  for (size_t i = 400; i <= 1000; i += 50) {
    all_nprobes.push_back(i);
  }
  //    all_nprobes.push_back(1);
  //    all_nprobes.push_back(2);
  //    all_nprobes.push_back(10);
  //    all_nprobes.push_back(30);
  //    all_nprobes.push_back(50);
  //    all_nprobes.push_back(100);
  //    all_nprobes.push_back(200);
  //    all_nprobes.push_back(400);
  //    all_nprobes.push_back(700);
  //    all_nprobes.push_back(1000);
  //    all_nprobes.push_back(1500);
  //    all_nprobes.push_back(2000);
  //    all_nprobes.push_back(3000);

  size_t total_count = NQ * TOPK;
  //    StopW stopw;

  raft::host_matrix<float, int64_t> padded_query =
    raft::make_host_matrix<float, int64_t>(NQ, ivf.get_num_padded_dim());
  // padded_query.setZero();
  memset(padded_query.data_handle(), 0, sizeof(float) * NQ * ivf.get_num_padded_dim());
  raft::host_matrix<float, int64_t> rotated_query =
    raft::make_host_matrix<float, int64_t>(NQ, ivf.get_num_padded_dim());
  for (size_t i = 0; i < NQ; ++i) {
    std::memcpy(&padded_query(i, 0), &query(i, 0), sizeof(float) * DIM);
  }

  // adjust nprobes
  for (auto it = all_nprobes.begin(); it != all_nprobes.end();) {
    if (*it > ivf.get_num_centroids()) {
      it = all_nprobes.erase(it);
    } else {
      ++it;
    }
  }
  size_t length = all_nprobes.size();

  std::vector<std::vector<float>> all_qps(ROUND, std::vector<float>(length));
  std::vector<std::vector<float>> all_recall(ROUND, std::vector<float>(length));
  std::vector<std::vector<float>> all_ratio(ROUND, std::vector<float>(length));

  // Create a GPU searcher instance (which uses the device query, etc.).
  SearcherGPU searcher(handle,
                       nullptr,
                       ivf.get_num_padded_dim(),
                       ivf.get_ex_bits(),
                       mode,
                       ivf.quantizer().get_query_scaling_factor(),
                       rabitq_quantize_flag);

  // find the longest cluster to allocate space;
  int max_cluster_length     = 0;
  long int total_num_vectors = 0;
  for (int64_t i = 0; i < ivf.get_cluster_meta_host().extent(0); ++i) {
    total_num_vectors += ivf.get_cluster_meta_host()(i).num;
    max_cluster_length =
      max(max_cluster_length, static_cast<int>(ivf.get_cluster_meta_host()(i).num));
  }
  // TODO: this should be part of the load function
  ivf.set_max_cluster_length(max_cluster_length);
  std::cout << "max cluster length: " << max_cluster_length << std::endl;

  searcher.AllocateSearcherSpace(ivf, NQ, TOPK, 3000, max_cluster_length);
  //   bool multiple_cluster_search = true;

  // prepare CPU side data for offloading computation
  // TODO: Later consider fix the space for computation offloading
  //    searcher.h_est_dis = (float*)malloc(sizeof(float) * max_cluster_length);
  //    searcher.h_ip_results = (float*)malloc(sizeof(float) * max_cluster_length);

  cudaStream_t single_stream = raft::resource::get_cuda_stream(handle);
  // Create a device result pool. (k*nprobe for multiple use)
  for (size_t r = 0; r < ROUND; r++) {
    std::vector<int> probe_hist_global(all_nprobes[0], 0);  // only support length = 1
    for (size_t i = 0; i < length; ++i) {
      size_t nprobe        = all_nprobes[i];
      size_t total_correct = 0;
      double total_ratio   = 0;
      float total_time     = 0;
      float *d_topk_dists, *d_final_dists;
      PID *d_topk_pids, *d_final_pids;
      cudaMallocAsync(&d_topk_dists, NQ * nprobe * TOPK * sizeof(float), single_stream);
      cudaMallocAsync(&d_final_dists, NQ * TOPK * sizeof(float), single_stream);
      cudaMallocAsync(&d_topk_pids, NQ * nprobe * TOPK * sizeof(PID), single_stream);
      cudaMallocAsync(&d_final_pids, NQ * TOPK * sizeof(PID), single_stream);
      cudaDeviceSynchronize();

      stopw.reset();
      // Allocate device memory for query vectors.
      float* d_query = nullptr;
      cudaMallocAsync(&d_query, NQ * ivf.get_num_padded_dim() * sizeof(float), single_stream);

      // Copy query vectors from host to device.
      cudaMemcpyAsync(d_query,
                      padded_query.data_handle(),
                      NQ * ivf.get_num_padded_dim() * sizeof(float),
                      cudaMemcpyHostToDevice,
                      single_stream);

      // Allocate device memory for rotated queries.
      float* d_rotated_query = nullptr;
      cudaMallocAsync(
        &d_rotated_query, NQ * ivf.get_num_padded_dim() * sizeof(float), single_stream);

      // Rotate query and set manually
      ivf.rotator().rotate(d_query, d_rotated_query, NQ);
      searcher.set_query(d_rotated_query);

      if (searcher.get_mode() == "lut32") {
        ivf.BatchClusterSearch(d_rotated_query,
                               TOPK,
                               nprobe,
                               &searcher,
                               NQ,
                               d_topk_dists,
                               d_final_dists,
                               d_topk_pids,
                               d_final_pids);
        cudaDeviceSynchronize();
        total_time += stopw.getElapsedTimeMicro();
        // time stop
      } else if (searcher.get_mode() == "lut16") {
        // test v3 lut using fp16
        ivf.BatchClusterSearchLUT16(d_rotated_query,
                                    TOPK,
                                    nprobe,
                                    &searcher,
                                    NQ,
                                    d_topk_dists,
                                    d_final_dists,
                                    d_topk_pids,
                                    d_final_pids);
        cudaDeviceSynchronize();
        total_time += stopw.getElapsedTimeMicro();
      } else if (searcher.get_mode() == "quant8") {
        ivf.BatchClusterSearchQuantizeQuery(d_rotated_query,
                                            TOPK,
                                            nprobe,
                                            &searcher,
                                            NQ,
                                            d_topk_dists,
                                            d_final_dists,
                                            d_topk_pids,
                                            d_final_pids,
                                            8);
        cudaDeviceSynchronize();
        total_time += stopw.getElapsedTimeMicro();
      } else if (searcher.get_mode() == "quant4") {
        ivf.BatchClusterSearchQuantizeQuery(d_rotated_query,
                                            TOPK,
                                            nprobe,
                                            &searcher,
                                            NQ,
                                            d_topk_dists,
                                            d_final_dists,
                                            d_topk_pids,
                                            d_final_pids,
                                            4);
        cudaDeviceSynchronize();
        total_time += stopw.getElapsedTimeMicro();
      }
      //            //test v2: add a round to compute threshold
      //            // This is abandoned for further use
      //            ivf.BatchClusterSearchPreComputeThresholds(
      //
      //                    d_rotated_query,
      //                    TOPK,
      //                    nprobe,
      //                    &searcher,
      //                    NQ,
      //                    d_topk_dists,
      //                    d_final_dists,
      //                    d_topk_pids,
      //                    d_final_pids,
      //                    single_stream
      //            );

      // First, copy the GPU results to CPU (assuming d_final_pids is on GPU)
      PID* h_final_pids = new PID[NQ * TOPK];
      cudaMemcpy(h_final_pids, d_final_pids, NQ * TOPK * sizeof(PID), cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();
      // Now compute recall and ratio in batch

      // Process all queries in batch
      for (size_t q = 0; q < NQ; q++) {
        // Get the results for query q (starting at position q*TOPK in the flattened array)
        PID* results = &h_final_pids[q * TOPK];

        // Compute ratio for this query
        total_ratio += get_ratio_standalone(q, query, data, gt, results, TOPK, L2SqrCPU_STL);

        // Compute recall for this query
        for (size_t j = 0; j < TOPK; j++) {
          for (size_t k = 0; k < TOPK; k++) {
            if (gt(q, k) == results[j]) {
              total_correct++;
              break;
            }
          }
        }
      }

      // Calculate final metrics
      float recall = static_cast<float>(total_correct) / total_count;
      float ratio  = total_ratio / total_count;

      // std::cout << "nprobe = : " << all_nprobes[i] << " finished, ";
      // std::cout << "Recall: " << recall << ", Ratio: " << ratio << "\n";

      // Clean up
      delete[] h_final_pids;

      cudaFreeAsync(d_topk_dists, single_stream);
      cudaFreeAsync(d_final_dists, single_stream);
      cudaFreeAsync(d_topk_pids, single_stream);
      cudaFreeAsync(d_final_pids, single_stream);
      cudaFreeAsync(d_query, single_stream);

      float qps = NQ / (total_time / 1e6);

      all_qps[r][i]    = qps;
      all_recall[r][i] = recall;
      all_ratio[r][i]  = ratio;
    }
    std::cout << "Round " << r << " finished!\n";
  }

  auto avg_qps    = horizontal_avg_standalone(all_qps);
  auto avg_recall = horizontal_avg_standalone(all_recall);
  auto avg_ratio  = horizontal_avg_standalone(all_ratio);

  // --- Define column widths ---
  const int W_NPROBE = 8;   // Width for nprobe column
  const int W_QPS    = 12;  // Width for QPS column
  const int W_RECALL = 12;  // Width for recall column
  const int W_RATIO  = 10;  // Width for ratio column

  // --- Define floating point precision ---
  const int P_QPS    = 2;  // Precision for QPS (digits after decimal)
  const int P_RECALL = 4;  // Precision for recall
  const int P_RATIO  = 3;  // Precision for ratio

  std::cout << std::left << std::setw(W_NPROBE) << "nprobe" << std::setw(W_QPS) << "QPS"
            << std::setw(W_RECALL) << "recall@" + std::to_string(TOPK) << std::setw(W_RATIO)
            << "ratio" << std::endl;

  // --- Print a separator line ---
  // Repeat '-' character for the total width
  std::cout << std::string(W_NPROBE + W_QPS + W_RECALL + W_RATIO, '-') << std::endl;

  // --- Loop through the data and print each row ---
  for (size_t i = 0; i < length; ++i) {
    size_t nprobe = all_nprobes[i];
    float qps     = avg_qps[i];
    float recall  = avg_recall[i];
    float ratio   = avg_ratio[i];

    // Use std::right for numbers (usually looks better)
    // Use std::fixed and std::setprecision for consistent float formatting
    std::cout << std::left << std::setw(W_NPROBE) << nprobe << std::fixed
              << std::setprecision(P_QPS) << std::left << std::setw(W_QPS) << qps << std::fixed
              << std::setprecision(P_RECALL) << std::left << std::setw(W_RECALL) << recall
              << std::fixed << std::setprecision(P_RATIO) << std::left << std::setw(W_RATIO)
              << ratio << std::endl;
  }

  // --- Find the first nprobe where recall crosses certain thresholds ---
  struct ThresholdInfo {
    float threshold;
    size_t nprobe;
    float recall;
    float qps;
    bool found;
  };

  std::vector<ThresholdInfo> thresholds = {
    {0.90f, 0, 0.0f, 0.0f, false}, {0.95f, 0, 0.0f, 0.0f, false}, {0.99f, 0, 0.0f, 0.0f, false}};

  // For each threshold, find the earliest nprobe that meets recall >= threshold.
  for (auto& t : thresholds) {
    for (size_t i = 0; i < length; ++i) {
      if (avg_recall[i] >= t.threshold) {
        t.nprobe = all_nprobes[i];
        t.recall = avg_recall[i];
        t.qps    = avg_qps[i];
        t.found  = true;
        break;
      }
    }
  }

  std::cout << "\nRecall threshold summary (first nprobe reaching each level):\n";
  for (auto const& t : thresholds) {
    if (t.found) {
      std::cout << "  recall >= " << t.threshold << " at nprobe = " << t.nprobe
                << "  (recall = " << std::fixed << std::setprecision(P_RECALL) << t.recall
                << ", QPS = " << std::setprecision(P_QPS) << t.qps << ")\n";
    } else {
      std::cout << "  recall >= " << t.threshold << " was NOT reached for any nprobe.\n";
    }
  }

  return 0;
}

}  // namespace

int main(int argc, char* argv[])
{
  raft::device_resources handle;
  // Set pool memory resource with 1 GiB initial pool size. All allocations use
  // the same pool.
  rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr(
    rmm::mr::get_current_device_resource(), 1024 * 1024 * 1024ull);
  rmm::mr::set_current_device_resource(&pool_mr);

  int ret = test_ivf_rabitq_construct_batch(handle, argc, argv);
  if (ret) {
    std::cerr << "IVF-RaBitQ index construction failed." << std::endl;
    return ret;
  }
  std::cout << "IVF-RaBitQ index construction complete." << std::endl;

  ret = test_ivf_rabitq_search_batch(handle, argc, argv);
  if (ret) {
    std::cerr << "IVF-RaBitQ search failed." << std::endl;
  } else {
    std::cout << "IVF-RaBitQ search complete." << std::endl;
  }

  return ret;
}
