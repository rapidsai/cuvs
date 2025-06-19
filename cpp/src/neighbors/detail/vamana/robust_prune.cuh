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

#include <cub/cub.cuh>
#include <thrust/sort.h>

#include <raft/util/cuda_dev_essentials.cuh>

#include "macros.cuh"
#include "vamana_structs.cuh"

namespace cuvs::neighbors::vamana::detail {

// Load candidates (from query) and previous edges (from nbh_list) into registers (tmp) spanning
// warp 将查询中的候选节点和邻接列表中的旧边加载到每个线程的寄存器中，便于 warp 内多线程并行处理
template <typename accT, typename IdxT = uint32_t, int DEG, int CANDS>
__forceinline__ __device__ void load_to_registers(DistPair<IdxT, accT>* tmp,//输出数组，暂存候选节点和邻居
                                                  QueryCandidates<IdxT, accT>* query,//当前查询点的候选节点
                                                  DistPair<IdxT, accT>* nbh_list)//已有的邻居列表
{
  int cands_per_thread = CANDS / 32;//总候选节点数平均分配给 32 个线程（一个 warp）
  for (int i = 0; i < cands_per_thread; i++) {//query 中加载候选节点到 tmp
    tmp[i].idx  = query->ids[cands_per_thread * threadIdx.x + i];
    tmp[i].dist = query->dists[cands_per_thread * threadIdx.x + i];

    if (cands_per_thread * threadIdx.x + i >= query->size) {//越界检查设置无效值
      tmp[i].idx  = raft::upper_bound<IdxT>();
      tmp[i].dist = raft::upper_bound<accT>();
    }
  }
  int nbh_per_thread = DEG / 32;//邻居总数平均分配给 32 个线程
  for (int i = 0; i < nbh_per_thread; i++) {//将邻居插入 tmp 的后半段
    tmp[cands_per_thread + i] = nbh_list[nbh_per_thread * threadIdx.x + i];
  }
}

/* Combines edge and candidate lists, removes duplicates, and sorts by distance 将已有的邻居列表和候选节点列表合并，去掉重复节点，按距离从小到大排序
 * Uses CUB primitives, so needs to be templated. Called with Macros for supported sizes above  CUB 提供的高性能并行原语 宏调用的，仅支持上面定义的一些固定大小*/
template <typename accT, typename IdxT, int DEG, int CANDS>
__forceinline__ __device__ void sort_edges_and_cands(
  DistPair<IdxT, accT>* new_nbh_list,//输出数组，保存合并后的邻居列表
  QueryCandidates<IdxT, accT>* query,//当前查询点的候选节点
  //指向 TempStorage 类型的指针 sort_mem，用于在 BlockMergeSort 排序操作中提供临时存储空间
  typename cub::BlockMergeSort<DistPair<IdxT, accT>, 32, (DEG + CANDS) / 32>::TempStorage* sort_mem)
{
  const int ELTS   = (DEG + CANDS) / 32;//每个线程要处理的数据元素数量
  using BlockSortT = cub::BlockMergeSort<DistPair<IdxT, accT>, 32, ELTS>;//BlockMergeSort 实现 warp 排序
  DistPair<IdxT, accT> tmp[ELTS];//每个线程线程私有寄存器数组
  //候选节点和已有邻居加载到 tmp
  load_to_registers<accT, IdxT, DEG, CANDS>(tmp, query, new_nbh_list);

  __syncthreads();
  BlockSortT(*sort_mem).Sort(tmp, CmpDist());//对寄存器数组 tmp[] 按距离排序
  __syncthreads();

  // Mark duplicates and re-sort 标记重复项并重新排序
  // Copy last element over and shuffle to check for duplicate between threads 复制末尾元素并进行洗牌操作，以检测线程间的重复数据 因为数据排序后，重复项必然出现在相邻位置
  new_nbh_list[ELTS * threadIdx.x + (ELTS - 1)] = tmp[ELTS - 1];//tmp 数组的最后一个元素复制到 new_nbh_list 对应位置
  if (tmp[ELTS - 1].idx == tmp[ELTS - 2].idx) {//检查 tmp 数组的最后两个元素是否重复 由于洗牌操作可能会影响最后一个元素的值，因为它会从上一个线程获取值
    new_nbh_list[ELTS * threadIdx.x + (ELTS - 1)].idx  = raft::upper_bound<IdxT>();
    new_nbh_list[ELTS * threadIdx.x + (ELTS - 1)].dist = raft::upper_bound<accT>();
  }
  __shfl_up_sync(0xffffffff, tmp[ELTS - 1].idx, 1);//洗牌，每个线程从上一个线程获取 tmp[ELTS - 1].idx 的值
  __syncthreads();

  for (int i = ELTS - 2; i > 0; i--) {//从后向前遍历 tmp 数组，检查相邻元素是否有重复的索引
    if (tmp[i].idx == tmp[i - 1].idx) {
      tmp[i].idx  = raft::upper_bound<IdxT>();
      tmp[i].dist = raft::upper_bound<accT>();
    }
  }
  if (threadIdx.x == 0) {//线程 0 检查 tmp 数组的第一个和最后一个元素是否重复 确保跨线程块边界的元素唯一
    if (tmp[0].idx == tmp[ELTS - 1].idx) {
      tmp[0].idx  = raft::upper_bound<IdxT>();
      tmp[0].dist = raft::upper_bound<accT>();
    }
  }

  tmp[ELTS - 1].idx = new_nbh_list[ELTS * threadIdx.x + (ELTS - 1)].idx;  // copy back to tmp for re-shuffling
  tmp[ELTS - 1].dist = new_nbh_list[ELTS * threadIdx.x + (ELTS - 1)].dist;

  __syncthreads();
  BlockSortT(*sort_mem).Sort(tmp, CmpDist());
  __syncthreads();

  for (int i = 0; i < ELTS; i++) {//去重和排序后的 tmp 数组复制回 new_nbh_list
    new_nbh_list[ELTS * threadIdx.x + i].idx  = tmp[i].idx;
    new_nbh_list[ELTS * threadIdx.x + i].dist = tmp[i].dist;
  }
  __syncthreads();
}

namespace {

/********************************************************************************************
  GPU kernel for RobustPrune operation for Vamana graph creation 用于 Vamana 图构建的 RobustPrune 操作的 GPU 内核
  Input - *graph to be an edgelist of degree number of edges per vector, 输入 - *graph 应为每个向量包含 degree 条边的边列表
  query_list should contain the list of visited nodes during GreedySearch. query_list 应包含贪心搜索过程中访问过的节点列表
  All inputs, including dataset, must be device accessible. 所有输入（包括数据集）必须可在设备上访问

  Output - candidate_ptr contains the new set of *degree* new neighbors that each node
           should have. 输出 - candidate_ptr 包含每个节点应具有的 degree 个新邻居集合
**********************************************************************************************/
template <typename T,
          typename accT,
          typename IdxT     = uint32_t,
          typename Accessor = raft::host_device_accessor<std::experimental::default_accessor<T>,
                                                         raft::memory_type::host>>
__global__ void RobustPruneKernel(
  raft::device_matrix_view<IdxT, int64_t> graph,//当前图的邻接表
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset,//数据集视图
  void* query_list_ptr,//查询列表指针
  int num_queries,//查询数量（要处理的节点数)
  int visited_size,//每个查询点访问过的节点数量
  cuvs::distance::DistanceType metric,
  float alpha,
  int sort_smem_size)//排序所需的共享内存大小
{
  int n      = dataset.extent(0);
  int dim    = dataset.extent(1);
  int degree = graph.extent(1);
  QueryCandidates<IdxT, accT>* query_list =
    static_cast<QueryCandidates<IdxT, accT>*>(query_list_ptr);
  //同一块内存空间中存储不同的数据结构
  union ShmemLayout {
    // All blocksort sizes have same alignment (16) union 中所有成员具有相同的内存对齐要求（16 字节）
    typename cub::BlockMergeSort<DistPair<IdxT, accT>, 32, 3>::TempStorage sort_mem;//CUB 排序所需的临时存储空间
    T coords;//当前查询点坐标
    DistPair<IdxT, accT> nbh_list;//当前查询点的邻居列表
  };

  // Dynamic shared memory used for blocksort, temp vector storage, and neighborhood list
  extern __shared__ __align__(alignof(ShmemLayout)) char smem[];//动态共享内存数组

  int align_padding = raft::alignTo<int>(dim, alignof(ShmemLayout)) - dim;//计算对齐填充的大小，满足ShmemLayout上的对齐要求
  // 共享内存的一部分转换为 T 类型的指针 存储查询点的坐标   &smem[sort_smem_size] 取排序操作临时存储空间结束的位置
  T* s_coords                        = reinterpret_cast<T*>(&smem[sort_smem_size]);
  // 共享内存的一部分转换为 DistPair<IdxT, accT> * 存储新的邻居列表 ;查询点坐标所需的内存大小+ CUB 排序的临时存储空间大小得到邻居列表起始地址
  DistPair<IdxT, accT>* new_nbh_list = reinterpret_cast<DistPair<IdxT, accT>*>(
    &smem[(dim + align_padding) * sizeof(T) + sort_smem_size]);

  static __shared__ Point<T, accT> s_query;//共享内存中的当前查询点对象
  s_query.coords = s_coords;
  s_query.Dim    = dim;

  for (int i = blockIdx.x; i < num_queries; i += gridDim.x) {
    int queryId = query_list[i].queryId;
    //query_list[i].queryId 的向量复制到共享内存中的 s_query 结构体；
    update_shared_point<T, accT>(&s_query, &dataset(0, 0), query_list[i].queryId, dim);

    // Load new neighbors to be sorted with candidates 
    // 图中查询点的邻居节点索引加载到共享内存中的 new_nbh_list 数组
    for (int j = threadIdx.x; j < degree; j += blockDim.x) {
      new_nbh_list[j].idx = graph(queryId, j);
    }
    __syncthreads();
    for (int j = 0; j < degree; j++) {//计算每个邻居与查询点的距离
      if (new_nbh_list[j].idx != raft::upper_bound<IdxT>()) {
        new_nbh_list[j].dist =
          dist<T, accT>(s_query.coords, &dataset((size_t)new_nbh_list[j].idx, 0), dim, metric);
      } else {
        new_nbh_list[j].dist = raft::upper_bound<accT>();
      }
    }
    __syncthreads();

    // combine and sort candidates and existing edges (and removes duplicates)
    // Resulting list is stored in new_nbh_list 组合和排序候选节点和现有边（并删除重复项）。结果列表存储在new_nbh_list中
    PRUNE_SELECT_SORT(degree, visited_size);

    __syncthreads();

    // If less than degree total neighbors, don't need to prune
    // new_nbh_list[degree].idx 是无效值 → 表示前 degree 个邻居都是有效的，没有多余的要剪掉
    if (new_nbh_list[degree].idx == raft::upper_bound<IdxT>()) {
      if (threadIdx.x == 0) {
        int writeId = 0;
        for (; new_nbh_list[writeId].idx != raft::upper_bound<IdxT>(); writeId++) {//复制有效元素
          query_list[i].ids[writeId]   = new_nbh_list[writeId].idx;
          query_list[i].dists[writeId] = new_nbh_list[writeId].dist;
        }
        query_list[i].size = writeId;//有效元素数量
        for (; writeId < degree; writeId++) {//填充无效元素
          query_list[i].ids[writeId]   = raft::upper_bound<IdxT>();
          query_list[i].dists[writeId] = raft::upper_bound<accT>();
        }
      }
    } else {
      // loop through list, writing nearest to visited_list, 遍历列表，将最近邻写入已访问列表
      // while nulling out violating neighbors in shared memory 共享内存中清零不符合条件邻居
      if (threadIdx.x == 0) { //复制第一个元素
        query_list[i].ids[0]   = new_nbh_list[0].idx;
        query_list[i].dists[0] = new_nbh_list[0].dist;
      }

      int writeId = 1;
      for (int j = 1; j < degree + query_list[i].size && writeId < degree; j++) {
        __syncthreads();
        if (new_nbh_list[j].idx == queryId || new_nbh_list[j].idx == raft::upper_bound<IdxT>()) {
          continue;
        }//跳过无效节点(当前节点索引=查询节点ID 或 无效值)
        __syncthreads();
        if (threadIdx.x == 0) {//有效节点从 new_nbh_list 复制到 query_list[i]
          query_list[i].ids[writeId]   = new_nbh_list[j].idx;
          query_list[i].dists[writeId] = new_nbh_list[j].dist;
        }
        writeId++;
        __syncthreads();
        // new_nbh_list[j].idx 对应的数据点加载进共享内存中的 s_query
        update_shared_point<T, accT>(&s_query, &dataset(0, 0), new_nbh_list[j].idx, dim);
        int tot_size = degree + query_list[i].size;
        for (int k = j + 1; k < tot_size; k++) {//从当前节点 j 的下一个节点开始遍历
          T* mem_ptr = const_cast<T*>(&dataset((size_t)new_nbh_list[k].idx, 0));//new_nbh_list[k].idx 在数据集中的地址
          if (new_nbh_list[k].idx != raft::upper_bound<IdxT>()) {
            accT dist_starprime = dist<T, accT>(s_query.coords, mem_ptr, dim, metric);
            // TODO - create cosine and selector fcn

            if (threadIdx.x == 0 && alpha * dist_starprime <= new_nbh_list[k].dist) {//检查候选点距离是否超阈值，剪枝
              new_nbh_list[k].idx = raft::upper_bound<IdxT>();
            }
          }
        }
      }
      __syncthreads();
      if (threadIdx.x == 0) { query_list[i].size = writeId; } // 更新写入数量

      __syncthreads();
      for (int j = writeId + threadIdx.x; j < degree;
           j += blockDim.x) {  // Zero out any unfilled neighbors
        query_list[i].ids[j]   = raft::upper_bound<IdxT>();
        query_list[i].dists[j] = raft::upper_bound<accT>();
      }
    }
  }
}

}  // namespace

}  // namespace cuvs::neighbors::vamana::detail
