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

#pragma once

#include "macros.cuh"
#include "priority_queue.cuh"
#include "vamana_structs.cuh"
#include <cub/cub.cuh>
#include <cuvs/neighbors/vamana.hpp>

#include <cuvs/distance/distance.hpp>
#include <raft/util/warp_primitives.cuh>
#include <rmm/resource_ref.hpp>

#include <chrono>
#include <cstdio>
#include <vector>

namespace cuvs::neighbors::vamana::detail {

/* @defgroup greedy_search_detail greedy search
 * @{
 */

/* Combines edge and candidate lists, removes duplicates, and sorts by distance
 * Uses CUB primitives, so needs to be templated. Called with Macros for supported sizes above */
template <typename accT, typename IdxT, int CANDS>//CANDS：每个查询点的候选数量
__forceinline__ __device__ void sort_visited(
  QueryCandidates<IdxT, accT>* query,
  typename cub::BlockMergeSort<DistPair<IdxT, accT>, 32, (CANDS / 32)>::TempStorage* sort_mem)//指针指向 CUB 库的块合并排序所需临时存储
{
  const int ELTS   = CANDS / 32;//每个线程块有 32 个线程
  using BlockSortT = cub::BlockMergeSort<DistPair<IdxT, accT>, 32, ELTS>;//创建类的别名
  DistPair<IdxT, accT> tmp[ELTS];//临时数组 tmp 存储当前线程处理的候选节点 ID 和距离
  for (int i = 0; i < ELTS; i++) {//查询候选结构中的 ID 和距离复制到临时数组 tmp 中，每个线程处理 ELTS 个元素 提高效率
    tmp[i].idx  = query->ids[ELTS * threadIdx.x + i];
    tmp[i].dist = query->dists[ELTS * threadIdx.x + i];
  }

  __syncthreads();
  BlockSortT(*sort_mem).Sort(tmp, CmpDist());//对临时数组 tmp 进行排序
  __syncthreads();

  for (int i = 0; i < ELTS; i++) {//排序后的 ID 和距离写回到查询候选结构中
    query->ids[ELTS * threadIdx.x + i]   = tmp[i].idx;
    query->dists[ELTS * threadIdx.x + i] = tmp[i].dist;
  }
  __syncthreads();
}

namespace {

/********************************************************************************************
  GPU kernel to perform a batched GreedySearch on a graph. Since this is used for
  Vamana construction, the entire visited list is kept and stored within the query_list.
  Input - graph with edge lists, dataset vectors, query_list_ptr with the ids of dataset
          vectors to be searched. All inputs, including dataset,  must be device accessible.

  Output - the id and dist lists in query_list_ptr will be updated with the nodes visited
           during the GreedySearch.
**********************************************************************************************/
template <typename T,
          typename accT,
          typename IdxT     = uint32_t,
          typename Accessor = raft::host_device_accessor<std::experimental::default_accessor<T>,
                                                         raft::memory_type::host>>
__global__ void GreedySearchKernel(
  raft::device_matrix_view<IdxT, int64_t> graph,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset,
  void* query_list_ptr,//指向查询候选列表的指针，存储每个查询结果
  int num_queries,//查询数量
  int medoid_id,//中心点ID
  int topk,//每个查询要返回的最近邻数量
  cuvs::distance::DistanceType metric,
  int max_queue_size,//搜索队列的最大大小
  int sort_smem_size)//搜索队列的共享内存大小
{
  int n      = dataset.extent(0);//样本数量
  int dim    = dataset.extent(1);//特征维度
  int degree = graph.extent(1);//每个节点度数

  QueryCandidates<IdxT, accT>* query_list =
    static_cast<QueryCandidates<IdxT, accT>*>(query_list_ptr);//转换指针类型

  static __shared__ int topk_q_size;//搜索队列大小  静态分配共享内存
  static __shared__ int cand_q_size;//候选队列大小
  static __shared__ accT cur_k_max;//当前最大搜索距离
  static __shared__ int k_max_idx;//当前搜索距离最大值索引

  static __shared__ Point<T, accT> s_query;//存储当前查询点的特征向量

  union ShmemLayout {//union所有成员共享同一块内存
    // All blocksort sizes have same alignment (16) 所有块排序大小具有相同的对齐方式（16）
    typename cub::BlockMergeSort<DistPair<IdxT, accT>, 32, 1>::TempStorage sort_mem;//CUB 库的 BlockMergeSort 所需的临时存储空间
    T coords;//当前查询点的坐标
    Node<accT> topk_pq;//存储优先队列中的节点
    int neighborhood_arr;//存储当前搜索的邻域
    DistPair<IdxT, accT> candidate_queue;//用于存储候选队列
  };
/*alignof(ShmemLayout) 获取ShmemLayout联合体对齐要求
 (dim - 1)/alignof(ShmemLayout)：计算需要多少个完整的对齐单位来覆盖 dim - 1 的大小； +1：确保至少有一个对齐单位。
*alignof(ShmemLayout)：计算总共需要的字节，确保分配内存大小是alignof(ShmemLayout)的倍数 
- dim：减去原始的 dim，得到为满足对其要求需要添加的字节数。*/
  int align_padding = (((dim - 1) / alignof(ShmemLayout)) + 1) * alignof(ShmemLayout) - dim;

  // Dynamic shared memory used for blocksort, temp vector storage, and neighborhood list
  extern __shared__ __align__(alignof(ShmemLayout)) char smem[];//smem共享内存，用于后续手动管理

  size_t smem_offset = sort_smem_size;  //排序所需临时存储的大小，占用共享内存起始部分 temp sorting memory takes first chunk

  T* s_coords = reinterpret_cast<T*>(&smem[smem_offset]);//获取字节数组 smem 中偏移 smem_offset 处的地址 强制转换为T*
  smem_offset += (dim + align_padding) * sizeof(T);//计算并更新共享内存偏移量，为下一块数据分配内存空间

  Node<accT>* topk_pq = reinterpret_cast<Node<accT>*>(&smem[smem_offset]);//存储优先队列的指针，将 smem 中从 smem_offset 位置开始的内存地址重新解释为 Node<accT> 类型的指针。允许将这块共享内存当作一个 Node<accT> 类型的数组来使用。
  smem_offset += topk * sizeof(Node<accT>);//topk 优先队列的大小；为下一个数据结构分配内存空间

  int* neighbor_array = reinterpret_cast<int*>(&smem[smem_offset]);//存储邻域数组并设置指针
  smem_offset += degree * sizeof(int);

  DistPair<IdxT, accT>* candidate_queue_smem =//为候选队列分配内存并设置指针
    reinterpret_cast<DistPair<IdxT, accT>*>(&smem[smem_offset]);

  s_query.coords = s_coords;//查询点的坐标指针
  s_query.Dim    = dim;//查询点的维度

  PriorityQueue<IdxT, accT> heap_queue;//初始化一个优先队列用于后续搜索

  if (threadIdx.x == 0) { //确保只有线程块中的第一个线程执行初始化代码
    heap_queue.initialize(candidate_queue_smem, max_queue_size, &cand_q_size);//优先队列初始化 &cand_q_size是指向候选队列大小的指针
  }

  static __shared__ int num_neighbors;//存储每个查询点的邻居数量

  for (int i = blockIdx.x; i < num_queries; i += gridDim.x) {
    __syncthreads();

    // resetting visited list ；reset 方法重置当前查询点的查询列表状态
    query_list[i].reset();

    // storing the current query vector into shared memory 将全局内存中特定查询点的数据复制到共享内存中
    // &s_query: 目标共享内存地址；&dataset(0, 0) 源数据集的起始地址；query_list[i].queryId 查询点的索引
    update_shared_point<T, accT>(&s_query, &dataset(0, 0), query_list[i].queryId, dim);

    if (threadIdx.x == 0) {
      topk_q_size = 0;
      cand_q_size = 0;
      s_query.id  = query_list[i].queryId;//设置共享内存中查询点的 ID
      cur_k_max   = 0;//初始化当前最大距离
      k_max_idx   = 0;//初始化当前最大距离的索引
      heap_queue.reset();
    }

    __syncthreads();

    Point<T, accT>* query_vec;//指向查询向量的指针

    // Just start from medoid every time, rather than multiple set_ups 
    query_vec        = &s_query;//当前查询向量指向共享内存中预分配的 s_query
    query_vec->Dim   = dim;
    const T* medoid  = &dataset((size_t)medoid_id, 0);//获取中心点（medoid）的起始地址 dataset(row,col)
    accT medoid_dist = dist<T, accT>(query_vec->coords, medoid, dim, metric);//计算当前查询点与中心点之间的距离

    if (threadIdx.x == 0) { heap_queue.insert_back(medoid_dist, medoid_id); }//中心点距离和中心点 ID 插入到优先队列
    __syncthreads();

    while (cand_q_size != 0) { //候选队列不空
      __syncthreads();

      int cand_num;//从优先队列中取出的候选节点 ID
      accT cur_distance;//该候选节点到查询点的距离
      if (threadIdx.x == 0) {
        Node<accT> test_cand;
        DistPair<IdxT, accT> test_cand_out = heap_queue.pop();//弹出当前队列中距离最小的候选节点（ID和距离）
        test_cand.distance                 = test_cand_out.dist;
        test_cand.nodeid                   = test_cand_out.idx;
        cand_num                           = test_cand.nodeid;
        cur_distance                       = test_cand_out.dist;
      }//dist ID 分别存储到 test_cand 对象及 cand_num 和 cur_distance 变量中
      __syncthreads();
      //将线程块中第一个线程的 cand_num 值广播到该线程块中的所有线程，确保每个线程都拥有相同的 cand_num 值
      cand_num = raft::shfl(cand_num, 0);

      __syncthreads();
      //如果节点已在查询访问列表且新距离不更优，则不再处理这个候选节点
      if (query_list[i].check_visited(cand_num, cur_distance)) { continue; }

      cur_distance = raft::shfl(cur_distance, 0);//广播线程 0 中的 cur_distance 给整个 warp

      // stop condition for the graph traversal process
      bool done      = false;//标记当前节点是否满足提前终止条件
      bool pass_flag = false;//控制是否允许将当前节点插入队列

      if (topk_q_size == topk) {
        // Check the current node with the worst candidate in top-k queue
        if (threadIdx.x == 0) {
          if (cur_k_max <= cur_distance) { done = true; }//新加入的节点距离大于等于队列中最远节点的距离，停止
        }

        done = raft::shfl(done, 0);//广播done给整个warp
        if (done) {
          //当前查询结果仍不足topk个，即使当前节点距离较大，也允许其通过 pass_flag = true 插入队列
          if (query_list[i].size < topk) {
            pass_flag = true;
          }

          else if (query_list[i].size >= topk) {
            break;
          }
        }
      }

      // The current node is closer to the query vector than the worst candidate in top-K queue, so
      // enquee the current node in top-k queue
      Node<accT> new_cand;
      new_cand.distance = cur_distance;
      new_cand.nodeid   = cand_num;
      //判断该节点是否已经在 top-k 队列中存在
      if (check_duplicate(topk_pq, topk_q_size, new_cand) == false) {
        if (!pass_flag) {//未触发提前终止逻辑 则插入
          parallel_pq_max_enqueue<accT>(//将新节点插入到 top-k 优先队列中，并保持队列的最大堆性质
            topk_pq, &topk_q_size, topk, new_cand, &cur_k_max, &k_max_idx);
            
          __syncthreads();
        }
      } else {
        // already visited
        continue;
      }

      num_neighbors = degree;//初始化邻居数量为 degree
      __syncthreads();

      for (size_t j = threadIdx.x; j < degree; j += blockDim.x) {
        // Load neighbors from the graph array and store them in neighbor array (shared memory)
        neighbor_array[j] = graph(cand_num, j);//图节点cand_num的第j个邻居存入共享内存数组
        if (neighbor_array[j] == raft::upper_bound<IdxT>())//upper_bound判断无效值
          atomicMin(&num_neighbors, (int)j);  // warp-wide min to find the number of neighbors 原子操作用于更新 num_neighbors 为当前找到的最小有效邻居索引
      }

      // computing distances between the query vector and neighbor vectors then enqueue in priority
      // queue.计算查询向量和相邻向量之间的距离，在优先级队列中排队。
      enqueue_all_neighbors<T, accT, IdxT>(
        num_neighbors, query_vec, &dataset(0, 0), neighbor_array, heap_queue, dim, metric);

      __syncthreads();

    }  // End cand_q_size != 0 loop

    bool self_found = false;
    // Remove self edges 移除自环边
    for (int j = threadIdx.x; j < query_list[i].size; j += blockDim.x) {
      if (query_list[i].ids[j] == query_vec->id) {
        query_list[i].dists[j] = raft::upper_bound<accT>();//距离设置为无效值
        query_list[i].ids[j]   = raft::upper_bound<IdxT>();//id设置为无效值
        self_found             = true;  // Flag to reduce size by 1
      }
    }
    //将查询结果列表中超出有效数量的部分标记为无效  query_list[i].size 是当前查询点的有效近邻数量
    for (int j = query_list[i].size + threadIdx.x; j < query_list[i].maxSize; j += blockDim.x) {
      query_list[i].ids[j]   = raft::upper_bound<IdxT>();
      query_list[i].dists[j] = raft::upper_bound<accT>();
    }

    __syncthreads();
    if (self_found) query_list[i].size--;//发现自环边，则减少结果数量

    SEARCH_SELECT_SORT(topk);//对结果排序，并选出 top-k 最近邻
  }

  return;
}

}  // namespace

/**
 * @}
 */

}  // namespace cuvs::neighbors::vamana::detail
