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

#include <climits>
#include <cstdint>
#include <cstdio>
#include <float.h>
#include <unordered_set>
#include <vector>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>

#include <cuvs/distance/distance.hpp>

namespace cuvs::neighbors::vamana::detail {

/* @defgroup vamana_structures vamana structures
 * @{
 */

#define FULL_BITMASK 0xFFFFFFFF

// Currently supported values for graph_degree.
static const int DEGREE_SIZES[4] = {32, 64, 128, 256};  // 存储图算法（如近邻搜索）中支持的节点度数

// Object used to store id,distance combination graph construction operations
template <typename IdxT, typename accT>
struct __align__(16) DistPair  //__align__(16)：强制结构体按 16 字节边界对齐
{
  accT dist;  // 距离值（L2）
  IdxT idx;   // 数据点索引

  __device__ __host__ DistPair& operator=(
    const DistPair& other)  //__device__ __host__：可在 CPU 和 GPU 代码中通用
  {
    dist = other.dist;
    idx  = other.idx;
    return *this;
  }

  __device__ __host__ DistPair& operator=(
    const volatile DistPair& other)  // 处理可能被异步修改的内存
  {
    dist = other.dist;
    idx  = other.idx;
    return *this;
  }
};

// Swap the values of two DistPair<SUMTYPE> objects 交换 DistPair 对象的函数模板
template <typename IdxT, typename accT>
__device__ __host__ void swap(DistPair<IdxT, accT>* a, DistPair<IdxT, accT>* b)
{
  DistPair<IdxT, accT> temp;
  temp.dist = a->dist;
  temp.idx  = a->idx;
  a->dist   = b->dist;
  a->idx    = b->idx;
  b->dist   = temp.dist;
  b->idx    = temp.idx;
}

// Structure to sort by distance 按距离（dist）比较 DistPair 对象
struct CmpDist {
  template <typename IdxT, typename accT>
  __device__ bool operator()(const DistPair<IdxT, accT>& lhs, const DistPair<IdxT, accT>& rhs)
  {
    return lhs.dist < rhs.dist;
  }
};

// Used to sort reverse edges by destination 对反向边（reverse edges）按目标节点（destination）排序
template <typename IdxT>
struct CmpEdge {
  __device__ bool operator()(const IdxT& lhs, const IdxT& rhs) { return lhs < rhs; }
};

/*********************************************************************
 * Object representing a Dim-dimensional point, with each coordinate
 * represented by a element of datatype T
 * Note, memory is allocated separately and coords set to offsets
 * 表示 Dim 维点的对象，每个坐标由一个 T
 *数据类型的元素表示。注意，内存是单独分配的，坐标设置为偏移量。
 *********************************************************************/
template <typename T, typename SUMTYPE>
class Point {
 public:
  int id;     // 点的唯一标识符
  int Dim;    // 点的维度
  T* coords;  // 指向坐标数组的指针

  __host__ __device__ Point& operator=(const Point& other)  // 赋值运算符重载
  {
    for (int i = 0; i < Dim; i++) {
      coords[i] = other.coords[i];  // 深拷贝：复制坐标数组内容
    }
    id = other.id;
    return *this;
  }
};

/* L2 fallback for low dimension when ILP is not possible 无法使用ILP（Instruction-Level
 * Parallelism，指令级并行）优化 */
template <typename T, typename SUMTYPE>  // SUMTYPE：累加和类型（通常比坐标类型更大，防止溢出）
__device__ SUMTYPE l2_SEQ(Point<T, SUMTYPE>* src_vec, Point<T, SUMTYPE>* dst_vec)
{
  SUMTYPE partial_sum = 0;
  // threadIdx.x：线程索引 ;blockDim.x：线程块大小
  for (int i = threadIdx.x; i < src_vec->Dim; i += blockDim.x) {
    partial_sum = fmaf((src_vec[0].coords[i] - dst_vec[0].coords[i]),
                       (src_vec[0].coords[i] - dst_vec[0].coords[i]),
                       partial_sum);
  }  // FMA 优化：使用 fmaf（乘加融合运算），一次操作完成：(a-b)² + partial_sum
     //  在线程块内进行归约操作，将所有线程的 partial_sum 值累加到一个线程中
  for (int offset = 16; offset > 0; offset /= 2) {
    partial_sum += __shfl_down_sync(
      FULL_BITMASK, partial_sum, offset);  //__shfl_down_sync 用于在线程块内进行数据交换
  }  // 每次循环都会使线程的 partial_sum 包含与它相隔 offset 个线程的线程的累加结果，offset
     // 不断减小，直到 offset = 1
  return partial_sum;  // 返回线程束内的所有线程的 partial_sum 的和
}

/* L2 optimized with 2-way ILP for DIM >= 64 2路指令级并行（ILP）优化技术，特别适用于高维数据（维度
 * ≥ 64） */
template <typename T, typename SUMTYPE>
__device__ SUMTYPE l2_ILP2(Point<T, SUMTYPE>* src_vec, Point<T, SUMTYPE>* dst_vec)
{
  T temp_dst[2]          = {0, 0};  // 临时存储目标点的两个坐标值
  SUMTYPE partial_sum[2] = {0, 0};  // 存储当前线程计算的两个局部和
  for (int i = threadIdx.x; i < src_vec->Dim; i += 2 * blockDim.x) {  // blockDim.x 通常设置为 32
    temp_dst[0] = dst_vec->coords[i];  // 目标点在坐标 i 处的值存储到 temp_dst[0]
    if (i + 32 < src_vec->Dim)
      temp_dst[1] =
        dst_vec
          ->coords[i +
                   32];  // 如果i+32不超过点的维度，获取目标点在坐标i+32处的值，并存储到temp_dst[1]
    partial_sum[0] =
      fmaf((src_vec[0].coords[i] - temp_dst[0]),  // 计算源点和目标点在坐标 i 处的差值
           (src_vec[0].coords[i] - temp_dst[0]),
           partial_sum[0]);
    if (i + 32 < src_vec->Dim)
      partial_sum[1] = fmaf((src_vec[0].coords[i + 32] - temp_dst[1]),
                            (src_vec[0].coords[i + 32] - temp_dst[1]),
                            partial_sum[1]);
  }
  partial_sum[0] += partial_sum[1];  // 得到当前线程的总和

  for (int offset = 16; offset > 0; offset /= 2) {  // 线程块内归约
    partial_sum[0] += __shfl_down_sync(FULL_BITMASK, partial_sum[0], offset);
  }
  return partial_sum[0];
}

/* L2 optimized with 4-way ILP for optimal performance for DIM >= 128
 * 4路指令级并行（ILP=4）优化，适用于高维数据（维度>=128）*/
template <typename T, typename SUMTYPE>
__device__ SUMTYPE l2_ILP4(Point<T, SUMTYPE>* src_vec, Point<T, SUMTYPE>* dst_vec)
{
  T temp_dst[4]          = {0, 0, 0, 0};  // 缓存目标点的4个坐标值
  SUMTYPE partial_sum[4] = {0, 0, 0, 0};  // 缓存4个元素局部平方和
  for (int i = threadIdx.x; i < src_vec->Dim;
       i += 4 * blockDim.x) {  // 加载目标点坐标（4个连续块），每个线程处理4个元素
    temp_dst[0] = dst_vec->coords[i];
    if (i + 32 < src_vec->Dim) temp_dst[1] = dst_vec->coords[i + 32];
    if (i + 64 < src_vec->Dim) temp_dst[2] = dst_vec->coords[i + 64];
    if (i + 96 < src_vec->Dim) temp_dst[3] = dst_vec->coords[i + 96];

      partial_sum[0] = fmaf((src_vec[0].coords[i] - temp_dst[0]),  // 获取源点和目标点在第i维的坐标差
                            (src_vec[0].coords[i] - temp_dst[0]),
                            partial_sum[0]);
    if (i + 32 < src_vec->Dim)
      partial_sum[1] = fmaf((src_vec[0].coords[i + 32] - temp_dst[1]),
                            (src_vec[0].coords[i + 32] - temp_dst[1]),
                            partial_sum[1]);
    if (i + 64 < src_vec->Dim)
      partial_sum[2] = fmaf((src_vec[0].coords[i + 64] - temp_dst[2]),
                            (src_vec[0].coords[i + 64] - temp_dst[2]),
                            partial_sum[2]);
    if (i + 96 < src_vec->Dim)
      partial_sum[3] = fmaf((src_vec[0].coords[i + 96] - temp_dst[3]),
                            (src_vec[0].coords[i + 96] - temp_dst[3]),
                            partial_sum[3]);
  }
  partial_sum[0] += partial_sum[1] + partial_sum[2] + partial_sum[3];

  for (int offset = 16; offset > 0; offset /= 2) {
    partial_sum[0] += __shfl_down_sync(FULL_BITMASK, partial_sum[0], offset);
  }

  return partial_sum[0];
}

/* Selects ILP optimization level based on dimension 根据输入点的维度自动选择最优的指令级并行(ILP)策略 */
template <typename T, typename SUMTYPE>
__forceinline__ __device__ SUMTYPE l2(Point<T, SUMTYPE>* src_vec, Point<T, SUMTYPE>* dst_vec)
{
  if (src_vec->Dim >= 128) {
    return l2_ILP4<T, SUMTYPE>(src_vec, dst_vec);
  } else if (src_vec->Dim >= 64) {
    return l2_ILP2<T, SUMTYPE>(src_vec, dst_vec);
  } else {
    return l2_SEQ<T, SUMTYPE>(src_vec, dst_vec);
  }
}

/* Convert vectors to point structure to performance distance comparison 将向量转换为点结构以进行距离比较*/
template <typename T, typename SUMTYPE>
__host__ __device__ SUMTYPE l2(const T* src, const T* dest, int dim)
{
  Point<T, SUMTYPE> src_p;//原始指针封装为Point结构
  /*  src 是const T* 类型指向坐标的指针
   const_cast<T*>(src) 将 src 转换为非 const 的 T* 类型。
   将这个非 const 指针赋值给 src_p.coords   */
  src_p.coords = const_cast<T*>(src);
  src_p.Dim    = dim;
  Point<T, SUMTYPE> dest_p;
  dest_p.coords = const_cast<T*>(dest);
  dest_p.Dim    = dim;

  return l2<T, SUMTYPE>(&src_p, &dest_p);
}

// Currently only L2Expanded is supported
template <typename T, typename SUMTYPE>
__host__ __device__ SUMTYPE
dist(const T* src, const T* dest, int dim, cuvs::distance::DistanceType metric)
{
  return l2<T, SUMTYPE>(src, dest, dim);
}

/***************************************************************************************
 * Structure that holds information about and results of a query.保存查询信息和结果的结构
 *  Use by both GreedySearch and RobustPrune, as well as reverse edge lists.
 ***************************************************************************************/
template <typename IdxT, typename accT>
struct QueryCandidates {//管理ANNS中单个查询点的候选结果集
  IdxT* ids;  // 候选元素ID的数组指针
  accT* dists; // 对应的距离值的数组指针
  int queryId;// 当前查询点的唯一标识
  int size;   // 当前存储的候选数量 ∈[0,maxsize]
  int maxSize;// 最大候选容量

  __device__ void reset()
  {
    for (int i = threadIdx.x; i < maxSize; i += blockDim.x) {// 并行初始化所有候选槽位
      ids[i]   = raft::upper_bound<IdxT>();// 设置为ID类型最大值
      dists[i] = raft::upper_bound<accT>();// 设置为距离类型最大值
    }
    size = 0;
  }

  // Checks current list to see if a node as previously been visited
  //检查目标节点是否已在候选列表中，如果不在且列表未满，则将其加入列表
  __inline__ __device__ bool check_visited(IdxT target, accT dist)//target：要检查的节点ID;dist：该节点的距离值
  {
  //同步原语，用于在同一个线程块内的所有线程之间进行同步。确保在该函数被调用之前启动的所有线程都到达同步点，然后才会继续执行后续的指令
    __syncthreads();
    __shared__ bool found;
    found = false;  
    __syncthreads();//确保所有线程看到一致的found状态

    if (size < maxSize) {
      __syncthreads();
      for (int i = threadIdx.x; i < size; i += blockDim.x) {
        if (ids[i] == target) { found = true; } //任一线程找到目标即设为true
      }
      __syncthreads();
      if (!found && threadIdx.x == 0) {//仅由线程0执行添加
        ids[size]   = target;
        dists[size] = dist;
        size++;
      }
      __syncthreads();
    }
    return found;
  }
  // For debugging
  /*
  __inline__ __device__ void print_visited() { //打印当前查询候选集的状态信息 
    printf("queryId:%d, size:%d\n", queryId, size); //printf在GPU上性能开销大
    for(int i=0; i<size; i++) {
      printf("%d (%f), ", ids[i], dists[i]);
    }
    printf("\n");
  }
  */
};

namespace {

/********************************************************************************************
 * Kernels that work on QueryCandidates objects *
 *******************************************************************************************/
// For debugging
template <typename accT, typename IdxT = uint32_t>
__global__ void print_query_results(void* query_list_ptr, int count)//打印多个查询点的候选结果集状态
{
  QueryCandidates<IdxT, accT>* query_list =
    static_cast<QueryCandidates<IdxT, accT>*>(query_list_ptr);

  for (int i = 0; i < count; i++) {
    query_list[i].print_visited();//调用每个查询的调试打印方法
  }
}

// Initialize a list of QueryCandidates objects: assign memory to mpointers and initialize values
template <typename IdxT, typename accT>
//分配内存并初始化候选集的状态
__global__ void init_query_candidate_list(QueryCandidates<IdxT, accT>* query_list,// 待初始化的查询候选列表
                                          IdxT* visited_id_ptr, // 预分配的ID存储空间
                                          accT* visited_dist_ptr,// 预分配的距离存储空间
                                          int num_queries,      // 查询点数量
                                          int maxSize)          // 每个查询的最大候选数
{// 初始化候选集存储空间
  IdxT* ids_ptr  = static_cast<IdxT*>(visited_id_ptr);
  accT* dist_ptr = static_cast<accT*>(visited_dist_ptr);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_queries * maxSize;
       i += blockDim.x + gridDim.x) {// 有点疑问，步长不是blockDim.x * gridDim.x吗？
    ids_ptr[i]  = raft::upper_bound<IdxT>();
    dist_ptr[i] = raft::upper_bound<accT>();
  }
  //对象初始化：为每个查询点设置候选集属性
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_queries;
       i += blockDim.x + gridDim.x) {
    query_list[i].maxSize = maxSize;//为每个查询点的候选集设置最大候选数
    query_list[i].size    = 0;//初始化候选集的当前大小为 0
    query_list[i].ids     = &ids_ptr[i * (size_t)(maxSize)];//为每个查询点的候选集的 ID 和距离数组计算偏移量分配具体的内存位置
    query_list[i].dists   = &dist_ptr[i * (size_t)(maxSize)];
  }
}

// Copy query ID values from input array
template <typename IdxT, typename accT>
//并行初始化结构体数组               指向结构体数组的void*指针 查询ID的输入数组  需要处理的数据总量
__global__ void set_query_ids(void* query_list_ptr, IdxT* d_query_ids, int step_size)
{
  QueryCandidates<IdxT, accT>* query_list =
    static_cast<QueryCandidates<IdxT, accT>*>(query_list_ptr);

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < step_size; i += blockDim.x * gridDim.x) {
    query_list[i].queryId = d_query_ids[i]; // 复制查询ID
    query_list[i].size    = 0;
  }
}

// Compute prefix sums on sizes.计算前缀和 Currently only works with 1 thread
// TODO replace with parallel version
template <typename accT, typename IdxT = uint32_t>
__global__ void prefix_sums_sizes(QueryCandidates<IdxT, accT>* query_list,
                                  int num_queries,
                                  int* total_edges)//指向设备内存中的一个整数指针，用于存储最终的总边数
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {//只让第一个线程完成计算
    int sum = 0;
    for (int i = 0; i < num_queries + 1; i++) {
      sum += query_list[i].size;//第i个查询点的候选集当前大小累加
      query_list[i].size = sum - query_list[i].size;  // 计算不包含当前元素的累加和 exclusive prefix sum
    }
    *total_edges = query_list[num_queries].size;
  }
}

// Device fcn to have a threadblock copy coordinates into shared memory 将全局内存中的数据点高效复制到共享内存
template <typename T, typename accT>
__device__ void update_shared_point(Point<T, accT>* shared_point,
                                    const T* data_ptr,
                                    int id,
                                    int dim)
{
  shared_point->id  = id;
  shared_point->Dim = dim;
  for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
    shared_point->coords[i] = data_ptr[(size_t)(id) * (size_t)(dim) + i];
  }
}

// Update the graph from the results of the query list (or reverse edge list) 将查询列表的结果写入图的边结构
template <typename accT, typename IdxT = uint32_t>
__global__ void write_graph_edges_kernel(raft::device_matrix_view<IdxT, int64_t> graph,//矩阵视图表示图的边结构
                                         void* query_list_ptr,
                                         int degree,
                                         int num_queries)
{
  QueryCandidates<IdxT, accT>* query_list =
    static_cast<QueryCandidates<IdxT, accT>*>(query_list_ptr);//转换指针类型为QC

  for (int i = blockIdx.x; i < num_queries; i += gridDim.x) {//遍历所有查询点
    for (int j = threadIdx.x; j < query_list[i].size; j += blockDim.x) {//遍历当前查询点的候选集
      //当前查询点的第 j 个候选元素的 ID 写入图的边结构中
      graph(query_list[i].queryId, j) = query_list[i].ids[j];
    }
  }
}

// Create src and dest edge lists used to sort and create reverse edges 创建用于排序和生成反向边的源边和目标边列表
template <typename accT, typename IdxT = uint32_t>
__global__ void create_reverse_edge_list(
  void* query_list_ptr, int num_queries, int degree, IdxT* edge_src, IdxT* edge_dest)
{
  QueryCandidates<IdxT, accT>* query_list =
    static_cast<QueryCandidates<IdxT, accT>*>(query_list_ptr);

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_queries;
       i += blockDim.x * gridDim.x) {
    int read_idx   = i * query_list[i].maxSize;//计算当前查询点 i 的候选集在全局存储空间中的起始位置
    int cand_count = query_list[i + 1].size - query_list[i].size;//计算当前查询点的有效候选数量

    for (int j = 0; j < cand_count; j++) {//j遍历当前查询点的有效候选集
      edge_src[query_list[i].size + j]  = query_list[i].queryId;//当前查询点的 ID 写入到边源列表中
      edge_dest[query_list[i].size + j] = query_list[i].ids[j];//将当前查询点的第 j 个候选节点的 ID 写入到边目标列表中
    }
  }
}

// Populate reverse edge QueryCandidates structure based on sorted edge list and unique indices
// values
template <typename T, typename accT, typename IdxT = uint32_t>
__global__ void populate_reverse_list_struct(QueryCandidates<IdxT, accT>* reverse_list,
                                             IdxT* edge_src,
                                             IdxT* edge_dest,
                                             int* unique_indices,//指向目标节点的起始索引列表，快速定位每个目标节点的边
                                             int unique_dests,///目标节点的数量
                                             int total_edges,
                                             int N,
                                             int rev_start,//反向边处理的起始索引
                                             int reverse_batch)//反向边批次大小
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < reverse_batch;
       i += blockDim.x * gridDim.x) {//循环遍历需要处理的反向边批次
    reverse_list[i].queryId = edge_dest[unique_indices[i + rev_start]];//反向边列表中提取目标节点ID，并填充到查询结构体
    //计算目标节点的边数量，如果当前目标节点是最后一个唯一目标节点，则边数量为总边数减去起始索引
    //否则，边数量为下一个唯一目标节点的起始索引减去当前唯一目标节点的起始索引
    if (rev_start + i == unique_dests - 1) {
      reverse_list[i].size = total_edges - unique_indices[i + rev_start];
    } else {
      reverse_list[i].size = unique_indices[i + rev_start + 1] - unique_indices[i + rev_start];
    }
    //确保计算出的边数量不超过 reverse_list[i].maxSize，防止数组越界
    if (reverse_list[i].size > reverse_list[i].maxSize) {
      reverse_list[i].size = reverse_list[i].maxSize;
    }
    //遍历当前目标节点的边，将对应的第j条边的 ID 填充到反向边列表
    for (int j = 0; j < reverse_list[i].size; j++) {
      reverse_list[i].ids[j] = edge_src[unique_indices[i + rev_start] + j];
    }
    //填充剩余的 ids 和 dists 数组元素为最大值（无效）
    for (int j = reverse_list[i].size; j < reverse_list[i].maxSize; j++) {
      reverse_list[i].ids[j]   = raft::upper_bound<IdxT>();
      reverse_list[i].dists[j] = raft::upper_bound<accT>();
    }
  }
}

// Recompute distances of reverse list. Allows us to avoid keeping distances during sort 重新计算反向边列表中的距离
template <typename T,
          typename accT,
          typename IdxT     = uint32_t,
          typename Accessor = raft::host_device_accessor<std::experimental::default_accessor<T>,
                                                         raft::memory_type::host>>
__global__ void recompute_reverse_dists(
  QueryCandidates<IdxT, accT>* reverse_list,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset,
  int unique_dests,
  cuvs::distance::DistanceType metric)
{
  int dim          = dataset.extent(1);
  const T* vec_ptr = dataset.data_handle();//数据集的指针

  for (int i = blockIdx.x; i < unique_dests; i += gridDim.x) {
    for (int j = 0; j < reverse_list[i].size; j++) {//遍历当前目标节点的反向边列表，处理每条边
      reverse_list[i].dists[j] =
        dist<T, accT>(&vec_ptr[(size_t)(reverse_list[i].queryId) * (size_t)dim],//目标节点的特征向量的起始地址
                      &vec_ptr[(size_t)(reverse_list[i].ids[j]) * (size_t)dim],//源节点的特征向量的起始地址
                      dim,
                      metric);
    }
  }
}

}  // namespace

/**
 * @}
 */

}  // namespace cuvs::neighbors::vamana::detail
