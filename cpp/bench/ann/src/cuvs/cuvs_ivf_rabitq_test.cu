/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cassert>
#include <iomanip>
#include <iostream>
#include <queue>
#include <vector>

#include <cuvs/neighbors/ivf_rabitq/gpu_index/ivf_gpu.cuh>
#include <cuvs/neighbors/ivf_rabitq/gpu_index/searcher_gpu.cuh>
#include <cuvs/neighbors/ivf_rabitq/utils/IO.hpp>
#include <cuvs/neighbors/ivf_rabitq/utils/StopW.hpp>

#include <raft/core/device_resources.hpp>

#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

namespace {

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

  char data_file[500];
  char centroids_file[500];
  char cids_file[500];
  char ivf_file[500];

  sprintf(data_file, "%s/base.fvecs", DATASET);
  sprintf(centroids_file, "%s/centroid_%ld.fvecs", DATASET, K);
  sprintf(cids_file, "%s/cluster_id_%ld.ivecs", DATASET, K);
  sprintf(ivf_file, "ivf_exhaf%d_gpu_batch.index", B);

  // Load data from file (using your load_vecs template functions).
  FloatRowMat data;
  FloatRowMat centroids;
  UintRowMat cids;  // Assume cids are stored as uint32_t
  load_vecs<float, FloatRowMat>(data_file, data);
  load_vecs<float, FloatRowMat>(centroids_file, centroids);
  load_vecs<PID, UintRowMat>(cids_file, cids);

  size_t N   = data.rows();
  size_t DIM = data.cols();

  std::cout << "Data loaded:\n\tN: " << N << "\n\tDIM: " << DIM << std::endl;

  StopW stopw;
  // Create an IVFGPU instance. (Its constructor will allocate device memory as needed.)
  IVFGPU ivf(handle, N, DIM, K, B, true);

  // Construct the index (this function performs necessary host-to-device transfers internally).
  ivf.construct(data.data(), centroids.data(), cids.data(), fast_quantize_flag);

  float minutes = stopw.getElapsedTimeMili() / 1000.0f / 60.0f;
  float seconds = stopw.getElapsedTimeMili() / 1000.0f;
  std::cout << "IVFGPU constructed\n";

  // Save the index to a file.
  ivf.save(ivf_file, true);

  std::cout << "Indexing time: " << seconds << " seconds\n";

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
                            const FloatRowMat& query,
                            const FloatRowMat& data,
                            const UintRowMat& gt,
                            PID* ann_results,
                            size_t K,
                            float (*dist_func)(const float*, const float*, size_t))
{
  std::priority_queue<float> gt_distances;
  std::priority_queue<float> ann_distances;

  for (size_t i = 0; i < K; ++i) {
    PID gt_id  = gt(numq, i);
    PID ann_id = ann_results[i];
    if (gt_id > data.rows() || ann_id > data.rows()) { continue; }
    gt_distances.emplace(dist_func(&query(numq, 0), &data(gt_id, 0), data.cols()));
    ann_distances.emplace(dist_func(&query(numq, 0), &data(ann_id, 0), data.cols()));
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
  if (argc > 5) {
    mode = argv[5];  // 4 modes: lut32, lut16, quant8, quant4
  } else {
    mode = "quant4";  // by default, using 4 bit quantization for queries
  }
  if (argc > 6) {
    std::string arg_str  = argv[6];
    rabitq_quantize_flag = (arg_str == "true" || arg_str == "1");
  }
  if (argc > 7) { EXPAND_FACTOR = atoi(argv[7]); }
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

  FloatRowMat data;
  FloatRowMat query;
  UintRowMat gt;

  load_vecs<float, FloatRowMat>(data_file, data);
  load_vecs_k<float, FloatRowMat>(query_file, query, EXPAND_FACTOR);
  load_vecs_k<PID, UintRowMat>(gt_file, gt, EXPAND_FACTOR);

  size_t N   = data.rows();
  size_t DIM = data.cols();
  size_t NQ  = query.rows();
  //    NQ = 1;

  std::cout << "data loaded\n";
  std::cout << "\tN: " << N << '\n' << "\tDIM: " << DIM << '\n';
  std::cout << "query loaded\n";
  std::cout << "\tNQ: " << NQ << '\n';

  StopW stopw;
  IVFGPU ivf(handle);
  ivf.load_transposed(handle, ivf_file);

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

  FloatRowMat padded_query(NQ, ivf.padded_dim());
  padded_query.setZero();
  FloatRowMat rotated_query(NQ, ivf.padded_dim());
  for (size_t i = 0; i < NQ; ++i) {
    std::memcpy(&padded_query(i, 0), &query(i, 0), sizeof(float) * DIM);
  }

  // Allocate device memory for query vectors.
  float* d_query = nullptr;
  stopw.reset();
  cudaMalloc(&d_query, NQ * ivf.padded_dim() * sizeof(float));

  // Copy query vectors from host to device.
  cudaMemcpy(
    d_query, padded_query.data(), NQ * ivf.padded_dim() * sizeof(float), cudaMemcpyHostToDevice);

  // Allocate device memory for rotated queries.
  float* d_rotated_query = nullptr;
  cudaMalloc(&d_rotated_query, NQ * ivf.padded_dim() * sizeof(float));

  // Now, use the RotatorGPU::rotate method to rotate the query vectors on GPU.
  // The RotatorGPU::rotate function is defined as:
  //    void RotatorGPU::rotate(const float* d_A, float* d_RAND_A, size_t N) const;
  // where d_A is the input matrix (N x padded_dim) and d_RAND_A is the output.
  ivf.rotator().rotate(d_query, d_rotated_query, NQ);

  float rotate_time = stopw.getElapsedTimeMicro();

  // adjust nprobes
  for (auto it = all_nprobes.begin(); it != all_nprobes.end();) {
    if (*it > ivf.num_clusters()) {
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
  SearcherGPU searcher(
    &rotated_query(0, 0), ivf.padded_dim(), ivf.ex_bits, mode, rabitq_quantize_flag);

  // find the longest cluster to allocate space;
  int max_cluster_length     = 0;
  long int total_num_vectors = 0;
  for (auto i : ivf.h_cluster_meta) {
    total_num_vectors += i.num;
    if (i.num > (unsigned int)max_cluster_length) { max_cluster_length = i.num; }
  }
  // TODO: this should be part of the load function
  ivf.max_cluster_length = max_cluster_length;
  std::cout << "max cluster length: " << max_cluster_length << std::endl;

  searcher.AllocateSearcherSpace(ivf, NQ, TOPK, 3000, max_cluster_length, 0);
  //   bool multiple_cluster_search = true;

  // prepare CPU side data for offloading computation
  // TODO: Later consider fix the space for computation offloading
  //    searcher.h_est_dis = (float*)malloc(sizeof(float) * max_cluster_length);
  //    searcher.h_ip_results = (float*)malloc(sizeof(float) * max_cluster_length);

  cudaStream_t single_stream = 0;
  cudaStreamCreate(&single_stream);
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

      if (searcher.mode == "lut32") {
        stopw.reset();
        ivf.BatchClusterSearch(d_rotated_query,
                               TOPK,
                               nprobe,
                               &searcher,
                               NQ,
                               d_topk_dists,
                               d_final_dists,
                               d_topk_pids,
                               d_final_pids,
                               single_stream);
        cudaDeviceSynchronize();
        total_time += stopw.getElapsedTimeMicro();
        // time stop
      } else if (searcher.mode == "lut16") {
        // test v3 lut using fp16
        stopw.reset();
        ivf.BatchClusterSearchLUT16(d_rotated_query,
                                    TOPK,
                                    nprobe,
                                    &searcher,
                                    NQ,
                                    d_topk_dists,
                                    d_final_dists,
                                    d_topk_pids,
                                    d_final_pids,
                                    single_stream);
        cudaDeviceSynchronize();
        total_time += stopw.getElapsedTimeMicro();
      } else if (searcher.mode == "quant8") {
        stopw.reset();
        ivf.BatchClusterSearchQuantizeQuery(d_rotated_query,
                                            TOPK,
                                            nprobe,
                                            &searcher,
                                            NQ,
                                            d_topk_dists,
                                            d_final_dists,
                                            d_topk_pids,
                                            d_final_pids,
                                            8,
                                            single_stream);
        cudaDeviceSynchronize();
        total_time += stopw.getElapsedTimeMicro();
      } else if (searcher.mode == "quant4") {
        stopw.reset();
        ivf.BatchClusterSearchQuantizeQuery(d_rotated_query,
                                            TOPK,
                                            nprobe,
                                            &searcher,
                                            NQ,
                                            d_topk_dists,
                                            d_final_dists,
                                            d_topk_pids,
                                            d_final_pids,
                                            4,
                                            single_stream);
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

      std::cout << "nprobe = : " << all_nprobes[i] << " finished \n";
      std::cout << "Recall: " << recall << ", Ratio: " << ratio << "\n";

      // Clean up
      delete[] h_final_pids;

      cudaFreeAsync(d_topk_dists, single_stream);
      cudaFreeAsync(d_final_dists, single_stream);
      cudaFreeAsync(d_topk_pids, single_stream);
      cudaFreeAsync(d_final_pids, single_stream);

      float qps = NQ / ((total_time + rotate_time) / 1e6);

      all_qps[r][i]    = qps;
      all_recall[r][i] = recall;
      all_ratio[r][i]  = ratio;
    }
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

  cudaStreamDestroy(single_stream);
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
