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

#include "vamana_structs.cuh"
#include <raft/util/warp_primitives.cuh>
#include <stdio.h>

namespace cuvs::neighbors::vamana::detail {

/***************************************************************************************
***************************************************************************************/
/**
 * @defgroup vamana_priority_queue Vamana Priority queue structure
 * @{
 */

/**
 * @brief Priority Queue structure used by Vamana GreedySearch during
 * graph construction.
 *
 * The structure keeps the nearest visited neighbors seen thus far during
 * search and lets us efficiently find the next node to visit during the search.
 * Stores a total of KVAL pairs, where currently KVAL must be 2i-1 for some integer
 * i since the heap must be complete.
 * This size is determined during vamana build with the "queue_size" parameter (default 127)
 *
 * The queue and all methods are device-side, with a work group size or 32 (one warp).
 * During search, each warp creates their own queue to search a single query at a time.
 * The device memory pointed to by `vals` is assigned during the call to `initialize`.
 * The Vamana GreedySearch call uses shared memory, but any device-accessible memory is applicable.
 *
 *
 * @tparam IdxT type of the vector indices (represent dataset.extent(0))
 * @tparam accT type of distances between vectors (accumuator type)
 *
 */
template <typename IdxT, typename accT>
class PriorityQueue {
 public:
  int KVAL;//优先队列容量
  int insert_pointer;//指示下一个要插入元素的位置
  DistPair<IdxT, accT>* vals;//指向存储队列元素的数组
  DistPair<IdxT, accT> temp;//临时存储操作过程中的元素

  int* q_size;//当前队列的大小
  // Enforce max-heap property on the entries 在条目上强制执行最大堆性质
  __forceinline__ __device__ void heapify()
  {
    int i        = 0;// 初始化当前节点索引
    int swapDest = 0;// 下一步要交换的目标子节点索引

    while (2 * i + 2 < KVAL) { //循环条件：当前节点有左右子节点
      swapDest = 2 * i;
      // 当前节点比左子节点大，且右子节点不比左子节点小 → 选择左子节点交换 swapdest+1
      swapDest +=
        (vals[i].dist > vals[2 * i + 1].dist && vals[2 * i + 2].dist >= vals[2 * i + 1].dist);
      // 当前节点比右子节点大，且左子节点也比右子节点大 → 选择右子节点交换 swapdest+2
        swapDest +=
        2 * (vals[i].dist > vals[2 * i + 2].dist && vals[2 * i + 1].dist > vals[2 * i + 2].dist);

      if (swapDest == 2 * i) return; //不需要交换则直接返回

      swap(&vals[i], &vals[swapDest]);//当前节点与较小的子节点交换

      i = swapDest;//更新当前节点索引
    }
  }
  // Starts the heapify process starting at a particular index 从特定索引位置开始堆化过程
  __forceinline__ __device__ void heapifyAt(int idx)
  {
    int i        = idx;
    int swapDest = 0;

    while (2 * i + 2 < KVAL) {
      swapDest = 2 * i;
      swapDest +=
        (vals[i].dist > vals[2 * i + 1].dist && vals[2 * i + 2].dist <= vals[2 * i + 1].dist);
      swapDest +=
        2 * (vals[i].dist > vals[2 * i + 2].dist && vals[2 * i + 1].dist < vals[2 * i + 2].dist);

      if (swapDest == 2 * i) return;

      swap(&vals[i], &vals[swapDest]);
      i = swapDest;
    }
  }

  // Heapify from the bottom up, used with insert_back 从下到上插入堆排序
  __forceinline__ __device__ void heapifyReverseAt(int idx)
  {
    int i        = idx;
    int swapDest = 0;
    while (i > 0) {
      swapDest = ((i - 1) / 2);//父节点索引
      if (vals[swapDest].dist <= vals[i].dist) return;

      swap(&vals[i], &vals[swapDest]);
      i = swapDest;
    }
  }

  __device__ void reset() //清空优先队列
  {
    *q_size = 0;
    for (int i = 0; i < KVAL; i++) {
      vals[i].dist = raft::upper_bound<accT>();
      vals[i].idx  = raft::upper_bound<IdxT>();
    }
  }

  __device__ void initialize(DistPair<IdxT, accT>* v, int _kval, int* _q_size)
  {
    vals           = v;//队列数据存储地址
    KVAL           = _kval;//队列容量
    insert_pointer = _kval / 2;//插入指针从中间开始（避免频繁调整堆）；
    q_size         = _q_size;//指向队列当前元素数量
    reset();
  }

  // Initialize all nodes of the heap to +infinity
  __device__ void initialize()
  {
    for (int i = 0; i < KVAL; i++) {
      vals[i].idx  = raft::upper_bound<IdxT>();
      vals[i].dist = raft::upper_bound<accT>();
    }
  }
  //优先队列中所有节点的索引（idx）写入全局内存的一维数组 gmem 
  __device__ void write_to_gmem(int* gmem)
  {
    for (int i = 0; i < KVAL; i++) {
      gmem[i] = vals[i].idx;
    }
  }

  // Replace the root of the heap with new pair 新的pair插入堆顶 维护最小堆
  __device__ void insert(accT newDist, IdxT newIdx)
  {
    vals[0].dist = newDist;
    vals[0].idx  = newIdx;

    heapify();
  }

  // Replace a specific element in the heap (and maintain heap properties)
  __device__ void insertAt(accT newDist, IdxT newIdx, int idx)
  {
    vals[idx].dist = newDist;
    vals[idx].idx  = newIdx;

    heapifyAt(idx);
  }

  // Return value of the root of the heap (largest value)
  __device__ accT top() { return vals[0].dist; }

  __device__ IdxT top_node() { return vals[0].idx; }

  __device__ void insert_back(accT newDist, IdxT newIdx) //向一个优先队列中插入新的元素
  {
    if (newDist < vals[insert_pointer].dist) { //新元素的距离值小于当前插入位置的距离值
      if (vals[insert_pointer].idx == raft::upper_bound<IdxT>()) *q_size += 1;//检查插入位置索引，然后插入新元素
      vals[insert_pointer].dist = newDist;
      vals[insert_pointer].idx  = newIdx;
      heapifyReverseAt(insert_pointer);//从当前位置向上调整堆
    }
    insert_pointer++;

    if (insert_pointer == KVAL) insert_pointer = KVAL / 2;
  }

  // Pop root node off and heapify 弹出堆顶之后重新调整堆
  __device__ DistPair<IdxT, accT> pop()
  {
    DistPair<IdxT, accT> result;
    result.dist  = vals[0].dist;
    result.idx   = vals[0].idx;
    vals[0].dist = raft::upper_bound<accT>();
    vals[0].idx  = raft::upper_bound<IdxT>();
    heapify();
    *q_size -= 1;
    return result;
  }
};

/***************************************************************************************
 * Node structure used for simplified lists during GreedySearch.
 * Used for other operations like checking for duplicates, etc.
 * 用于贪心搜索中简化列表的节点结构。同时支持其他操作（如重复项检查等）
 ****************************************************************************************/
template <typename SUMTYPE>
class __align__(16) Node //Node类 内存对齐为 16 字节
{
 public:
  SUMTYPE distance;
  int nodeid;
};

// Less-than operator between two Nodes.
template <typename SUMTYPE>
__host__ __device__ bool operator<(const Node<SUMTYPE>& first, const Node<SUMTYPE>& other)
{
  return first.distance < other.distance;
}

// Less-than operator between two Nodes.
template <typename SUMTYPE>
__host__ __device__ bool operator>(const Node<SUMTYPE>& first, const Node<SUMTYPE>& other)
{
  return first.distance > other.distance;
}

template <typename accT>
//判断新节点 new_node 是否已经存在于已有的节点列表 pq 中
__device__ bool check_duplicate(const Node<accT>* pq, const int size, Node<accT> new_node)
{
  bool found = false;
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    if (pq[i].nodeid == new_node.nodeid) {
      found = true;
      break;
    }
  }
  //将 warp 中每个线程的 found 值打包成一个位掩码
  unsigned mask = raft::ballot(found);

  if (mask == 0)
    return false;

  else
    return true;
}

/*
  Enqueuing a input value into parallel queue with tracker 并行优先队列入队操作
*/
template <typename SUMTYPE>
__inline__ __device__ void parallel_pq_max_enqueue(Node<SUMTYPE>* pq,//优先队列数组z指针
                                                   int* size,//当前队列元素数量
                                                   const int pq_size,//队列最大容量
                                                   Node<SUMTYPE> input_data,//要插入的新节点
                                                   SUMTYPE* cur_max_val,//指向当前队列最大距离
                                                   int* max_idx)//当前队列最大距离索引
{//将一个新节点插入到最大堆结构中
  if (*size < pq_size) { //队列未满
    __syncthreads();
    if (threadIdx.x == 0) {
      pq[*size].distance = input_data.distance;
      pq[*size].nodeid   = input_data.nodeid;
      *size              = *size + 1;
      if (input_data.distance > (*cur_max_val)) {
        *cur_max_val = input_data.distance;
        *max_idx     = *size - 1;
      }
    }
    __syncthreads();
    return;
  } else {//队列已满
    if (input_data.distance >= (*cur_max_val)) {//新节点比当前最大距离还大,return
      __syncthreads();
      return;
    }
    if (threadIdx.x == 0) {//替换最大值位置的节点
      pq[*max_idx].distance = input_data.distance;
      pq[*max_idx].nodeid   = input_data.nodeid;
    }
    //重新查找新的最大值
    int idx         = 0;
    SUMTYPE max_val = pq[0].distance;

    for (int i = threadIdx.x; i < pq_size; i += 32) {
      if (pq[i].distance > max_val) {
        max_val = pq[i].distance;
        idx     = i;
      }
    }
    //Warp线程束内部归约找出全局最大值
    for (int offset = 16; offset > 0; offset /= 2) {
      SUMTYPE new_max_val = raft::shfl_up(max_val, offset);//从上方线程获取最大值
      int new_idx         = raft::shfl_up(idx, offset);//从上方线程获取索引
      if (new_max_val > max_val) {
        max_val = new_max_val;
        idx     = new_idx;
      }
    }

    if (threadIdx.x == 31) {//由最后一个线程更新追踪器状态
      *max_idx     = idx;
      *cur_max_val = max_val;
    }
  }
  __syncthreads();
}

/*
  Compute the distances between the source vector and all nodes in the neighbor_array and enqueue
  them in the PQ 计算源向量与 neighbor_array 中所有节点之间的距离，并将这些节点入队到优先队列（PQ）中
*/
template <typename T, typename accT, typename IdxT>
__forceinline__ __device__ void enqueue_all_neighbors(int num_neighbors,
                                                      Point<T, accT>* query_vec,//指向查询点的指针
                                                      const T* vec_ptr,//数据集的指针
                                                      int* neighbor_array,//存储近邻的数组
                                                      PriorityQueue<IdxT, accT>& heap_queue,//优先队列，用于存储邻居的距离和索引
                                                      int dim,
                                                      cuvs::distance::DistanceType metric)
{
  for (int i = 0; i < num_neighbors; i++) {
    accT dist_out = dist<T, accT>(
      query_vec->coords, &vec_ptr[(size_t)(neighbor_array[i]) * (size_t)(dim)], dim, metric);//查询点与当前邻居的距离

    __syncthreads();
    if (threadIdx.x == 0) { heap_queue.insert_back(dist_out, neighbor_array[i]); }//将计算出的距离和邻居索引插入到优先队列
    __syncthreads();
  }
}

}  // namespace cuvs::neighbors::vamana::detail
