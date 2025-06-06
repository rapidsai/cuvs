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

#include "greedy_search.cuh"
#include "robust_prune.cuh"
#include "vamana_structs.cuh"
#include <cuvs/neighbors/vamana.hpp>

#include <raft/cluster/kmeans.cuh>
#include <raft/cluster/kmeans_types.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/matrix/copy.cuh>
#include <raft/matrix/init.cuh>
#include <raft/random/make_blobs.cuh>
#include <raft/sparse/neighbors/cross_component_nn.cuh>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/unique.h>

#include <cuvs/distance/distance.hpp>

#include <chrono>
#include <cstdio>
#include <vector>

namespace cuvs::neighbors::vamana::detail {

/* @defgroup vamana_build_detail vamana build
 * @{
 */

static const int blockD    = 32;//  block size 每个GPU线程块维度
static const int maxBlocks = 10000;//  max number of blocks to launch GPU线程块最大数量

// generate random permutation of inserts - TODO do this on GPU / faster
template <typename IdxT>
//生成一个长度为 N 的随机排列数组 insert_order，用于控制数据点的插入顺序；参数 N：数据点总数
void create_insert_permutation(std::vector<IdxT>& insert_order, uint32_t N)
{
  insert_order.resize(N);//将 insert_order 的大小调整为 N
  for (uint32_t i = 0; i < N; i++) {
    insert_order[i] = (IdxT)i;//将数组初始化为 [0, 1, 2, ..., N-1]，即顺序排列
  }
  for (uint32_t i = 0; i < N; i++) {
    uint32_t temp;
    uint32_t rand_idx      = rand() % N;//生成一个 [0, N-1] 的随机索引
    temp                   = insert_order[i];
    insert_order[i]        = insert_order[rand_idx];//交换 insert_order[i] 和 insert_order[rand_idx]，实现随机打乱
    insert_order[rand_idx] = temp;
  }
}

/********************************************************************************************
 * Main Vamana building function - insert vectors into empty graph in batches
 *      Vamana 构建主函数 - 将向量批量插入空图
 * Pre - dataset contains the vector data, host matrix allocated to store the graph
 *      数据集包含向量数据，以及用于存储图的宿主矩阵
 * Post - graph matrix contains the graph edges of the final Vamana graph
 *      图矩阵包含最终 Vamana 图的图边
 *******************************************************************************************/
template <typename T,//数据类型
          typename accT,//累积计算类型（可能用于中间计算）
          typename IdxT     = uint32_t,//数据索引类型,默认为unsigned int 32位
          typename Accessor = raft::host_device_accessor<std::experimental::default_accessor<T>,
                                                         raft::memory_type::host>>//数据访问器，控制主机或设备内存访问
void batched_insert_vamana(
  raft::resources const& res,
  const index_params& params,//vamana图构建参数
  //mdspan 常用于将多维数据传递给 GPU 核函数，支持高效的内存访问模式
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset,//输入数据集，使用多维视图 mdspan 表示（行主序）
  raft::host_matrix_view<IdxT, int64_t> graph,//图的邻接列表，使用主机矩阵视图 host_matrix_view
  IdxT* medoid_id,//中心点 ID 的指针
  cuvs::distance::DistanceType metric)//距离度量类型（如 L2）
//  int dim)
{
  auto stream = raft::resource::get_cuda_stream(res);//通过 res 获取 CUDA 流，用于异步 GPU 操作
  int N       = dataset.extent(0);//数据集的行数，即数据点的数量
  int dim     = dataset.extent(1);//数据集的列数，即每个数据点的维度
  int degree  = graph.extent(1);//图的列数，即每个数据点的邻居数量(邻接表度数)

  // Algorithm params
  int max_batchsize = (int)(params.max_fraction * (float)N);//根据参数 max_fraction（比例系数）和数据集总量 N 计算最大批次大小，结果转为整数
  if (max_batchsize > (int)dataset.extent(0)) {
    RAFT_LOG_WARN(
      "Max fraction is the fraction of the total dataset, so it cannot be larger 1.0, reducing it "
      "to 1.0");//若计算出的批次大小超过数据集的第一维度（样本总数），发出警告，并将批次大小设为数据集容量上限
    max_batchsize = (int)dataset.extent(0);
  }
  int insert_iters  = (int)(params.vamana_iters);//获取 Vamana 图构建算法的迭代次数
  double base       = (double)(params.batch_base);
  float alpha       = (float)(params.alpha);//获取 Vamana 图构建算法的 alpha 参数，平衡精度与效率
  int visited_size  = params.visited_size;//访问节点数的上限
  int queue_size    = params.queue_size;//搜索队列容量
  int reverse_batch = params.reverse_batchsize;//反向批次大小

//2的幂在二进制中只有最高位是1，其他位均为0;visited_size & (visited_size - 1)：按位与操作后，结果应为0（如果是2的幂）  
  if ((visited_size & (visited_size - 1)) != 0) {
    RAFT_LOG_WARN("visited_size must be a power of 2, rounding up.");//必须是 2 的幂，正在向上取整
    int power = params.graph_degree;//初始值为图节点的度数
    while (power < visited_size)
      power <<= 1;//左移直到找到最近的 2 的幂
    visited_size = power;
  }

  // create gpu graph and set to all -1s  GPU 上分配一个二维矩阵 d_graph，其维度与输入图 graph 相同
  auto d_graph = raft::make_device_matrix<IdxT, int64_t>(res, graph.extent(0), graph.extent(1));
  raft::linalg::map(res, d_graph.view(), raft::const_op<IdxT>{raft::upper_bound<IdxT>()});//raft::const_op 将矩阵所有元素初始化为 IdxT 类型的最大值（upper_bound）

  // Temp storage about each batch of inserts being performed 分配批量插入的临时存储
  auto query_ids      = raft::make_device_vector<IdxT>(res, max_batchsize);//设备向量，存储每个批次的查询 ID，长度为 max_batchsize
  //多维数组，存储每个查询的候选信息（QueryCandidates 结构体），大小为 max_batchsize + 1
  auto query_list_ptr = raft::make_device_mdarray<QueryCandidates<IdxT, accT>>(
    res,
    raft::resource::get_large_workspace_resource(res),//large_workspace_resource 分配内存，可能因数据量较大需要显存池支持
    raft::make_extents<int64_t>(max_batchsize + 1));
  //获取候选列表指针，将多维数组的原始指针转换为 QueryCandidates 结构体指针  
  QueryCandidates<IdxT, accT>* query_list =
    static_cast<QueryCandidates<IdxT, accT>*>(query_list_ptr.data_handle());

  // Results of each batch of inserts during build - Memory is used by query_list structure 为每个批次的插入结果分配存储空间
  //visited_ids：二维设备数组，记录每个批次中每个查询访问过的节点 ID，维度为 (max_batchsize, visited_size)
  auto visited_ids =
    raft::make_device_mdarray<IdxT>(res,
                                    raft::resource::get_large_workspace_resource(res),
                                    raft::make_extents<int64_t>(max_batchsize, visited_size));
  //二维设备数组，记录每个批次中每个查询访问节点的距离，维度同上;visited_size 是单个查询允许访问的最大节点数（需为 2 的幂)
  auto visited_dists =
    raft::make_device_mdarray<accT>(res,
                                    raft::resource::get_large_workspace_resource(res),
                                    raft::make_extents<int64_t>(max_batchsize, visited_size));

  // Assign memory to query_list structures and initiailize 初始化查询候选列表query_list结构（GPU 内核）预分配查询候选列表的内存
  init_query_candidate_list<IdxT, accT><<<256, blockD, 0, stream>>>(query_list,
                                                                    visited_ids.data_handle(),
                                                                    visited_dists.data_handle(),
                                                                    (int)max_batchsize,
                                                                    visited_size);

  // Create random permutation for order of node inserts into graph 生成一个长度为 N 的随机排列向量 insert_order，用于控制节点插入图的顺序
  // 随机插入顺序可避免图结构偏向局部性，提升后续搜索的平衡性
  std::vector<IdxT> insert_order;
  create_insert_permutation<IdxT>(insert_order, (uint32_t)N);

  // Calculate the shared memory sizes of each kernel 计算内核共享内存大小
  int search_smem_sort_size = 0;//贪心搜索（GreedySearch）内核中排序操作需要的共享内存
  int prune_smem_sort_size  = 0;//剪枝（Prune）内核中排序操作需要的共享内存
  //宏 SELECT_SMEM_SIZE 动态计算排序操作所需的共享内存大小
  SELECT_SMEM_SIZES(degree, visited_size);  // Sets above 2 variables to appropriate sizes

  // Total dynamic shared memory used by GreedySearch
  int align_padding          = raft::alignTo(dim, 16) - dim;//将维度对齐到 16 字节
  int search_smem_total_size = static_cast<int>(
    search_smem_sort_size +
    (dim + align_padding) * sizeof(T) +    // 特征向量对齐后的空间
    visited_size * sizeof(Node<accT>) +    // 访问过的节点信息
    degree * sizeof(int) +                 // 度数相关数据
    queue_size * sizeof(DistPair<IdxT, accT>)); //优先级队列；DistPair 结构体，存储节点距离对

  // Total dynamic shared memory size needed by both RobustPrune calls 计算 RobustPrune 内核所需的动态共享内存总量
  int prune_smem_total_size = prune_smem_sort_size +   //prune_smem_sort_size：剪枝排序操作的基础内存
                             (dim + align_padding) * sizeof(T) + //对齐后的特征向量存储空间（T 是数据类型）
                              (degree + visited_size) * sizeof(DistPair<IdxT, accT>);/*剪枝过程中的距离对存储：
degree：当前节点的邻居数量；visited_size：访问过的节点数量；DistPair：存储节点 ID 和距离的结构体*/
//记录共享内存使用情况
  RAFT_LOG_DEBUG("Dynamic shared memory usage (bytes): GreedySearch: %d, RobustPrune: %d",
                 search_smem_total_size,
                 prune_smem_total_size);
//检查剪枝参数是否被宏 SELECT_SMEM_SIZE 支持;
  if (prune_smem_sort_size == 0) {  // If sizes not supported, smem sizes will be 0；说明给定的 graph_degree 或 visited_size 超出预设范围
    RAFT_FAIL("Vamana graph parameters not supported: graph_degree=%d, visited_size:%d\n",
              degree,
              visited_size);
  }

  // Random medoid has minor impact on recall
  // TODO: use heuristic for better medoid selection, issue:
  // https://github.com/rapidsai/cuvs/issues/355
  *medoid_id = rand() % N;//从 N 个节点中随机选择一个作为图的入口点（Medoid）

  // size of current batch of inserts, increases logarithmically until max_batchsize
  int step_size = 1;// 当前插入批次大小（初始为1），后续将按对数增长（如 step_size = min(step_size * base, max_batchsize)
  // Number of passes over dataset (default 1)
  for (int iter = 0; iter < insert_iters; iter++) { //插入迭代的主循环
    // Loop through batches and call the insert and prune kernels
    for (int start = 0; start < N;) { //遍历整个数据集N，循环条件无增量，start在循环内更新
      if (start + step_size > N) {//边界处理：当剩余节点不足完整批次时，调整批次大小
        int new_size = N - start;
        step_size    = new_size;
      }
      RAFT_LOG_DEBUG("Starting batch of inserts indices_start:%d, batch_size:%d", start, step_size);//输出调试日志，记录当前批次的起始位置和大小

      int num_blocks = min(maxBlocks, step_size);//计算线程块数量，maxblocks是预设的GPU最大线程块数；确保不超过批次大小，避免空块

      // Copy ids to be inserted for this batch 将当前批次的节点ID从主机复制到GPU
      //insert.order 前文生成的随机查询顺序向量 复制范围[start, start + step_size)
      raft::copy(query_ids.data_handle(), &insert_order.data()[start], step_size, stream);
      //初始化当前批次查询列表  num_blocks 个线程块，每块 blockD 个线程（如256），使用指定CUDA流（stream）
      set_query_ids<IdxT, accT><<<num_blocks, blockD, 0, stream>>>(
        query_list_ptr.data_handle(), query_ids.data_handle(), step_size);//查询列表结构体指针，当前批次节点id设备指针，当前批次大小

      // Call greedy search to get candidates for every vector being inserted
      // 为当前批次的每个插入节点执行贪心搜索，查找候选邻居节点
      GreedySearchKernel<T, accT, IdxT, Accessor>
        <<<num_blocks, blockD, search_smem_total_size, stream>>>(d_graph.view(),//当前构建的图结构
                                                                 dataset,//原始数据集
                                                                 query_list_ptr.data_handle(),//查询候选列表
                                                                 step_size,//当前批次大小
                                                                 *medoid_id,//图搜索起始点
                                                                 visited_size,//访问节点数上限
                                                                 metric,//距离度量标准（L2）
                                                                 queue_size,//搜索优先级队列大小
                                                                 search_smem_sort_size);//排序操作所需共享内存
      // Run on candidates of vectors being inserted 调用鲁棒剪枝内核 对搜索得到的候选邻居进行剪枝，控制节点度数
      RobustPruneKernel<T, accT, IdxT>
        <<<num_blocks, blockD, prune_smem_total_size, stream>>>(d_graph.view(),
                                                                dataset,
                                                                query_list_ptr.data_handle(),
                                                                step_size,
                                                                visited_size,
                                                                metric,
                                                                alpha,//alpha：剪枝强度参数（α > 1.0）α越大剪枝越激进，图更稀疏,α接近1.0保留更多边，提高召回率
                                                                prune_smem_sort_size);

      // Write results from first prune to graph edge list 将剪枝后的邻居列表写入图结构
      write_graph_edges_kernel<accT, IdxT><<<num_blocks, blockD, 0, stream>>>(
        //目标图结构，包含剪枝后邻居的查询列表，每个节点允许的最大邻居数,当前批次大小
        d_graph.view(), query_list_ptr.data_handle(), degree, step_size);

      // compute prefix sums of query_list sizes - TODO parallelize prefix sums
      // 计算query_list大小的前缀和-TODO并行化前缀和
      // 计算总边数（前缀和）
      auto d_total_edges = raft::make_device_mdarray<int>(
        res, raft::resource::get_workspace_resource(res), raft::make_extents<int64_t>(1));
      prefix_sums_sizes<accT, IdxT> //计算当前批次的总边数（所有节点的邻居数之和）输入：query_list（含每个节点的邻居数）输出：总边数写入 d_total_edges
        <<<1, 1, 0, stream>>>(query_list, step_size, d_total_edges.data_handle());

      int total_edges;
      raft::copy(&total_edges, d_total_edges.data_handle(), 1, stream);//设备上的总边数复制到主机变量 total_edges
      RAFT_CUDA_TRY(cudaStreamSynchronize(stream));//RAFT_CUDA_TRY 是错误检查宏，同步操作（cudaStreamSynchronize）
      //分配边列表存储空间
      auto edge_dest = //edge_dest：目标节点ID数组（邻居节点）
        raft::make_device_mdarray<IdxT>(res,
                                        raft::resource::get_large_workspace_resource(res),
                                        raft::make_extents<int64_t>(total_edges));
      auto edge_src = //edge_src：源节点ID数组（当前节点）
        raft::make_device_mdarray<IdxT>(res,
                                        raft::resource::get_large_workspace_resource(res),
                                        raft::make_extents<int64_t>(total_edges));

      // Create reverse edge list 创建反向边列表 将原始边u-v转换为反向边v-u
      create_reverse_edge_list<accT, IdxT>
        <<<num_blocks, blockD, 0, stream>>>(query_list_ptr.data_handle(),//当前批次节点信息的结构体
                                            step_size,                   //当前批次节点数量
                                            degree,                      //当前批次节点的度数 
                                            edge_src.data_handle(),       //当前批次节点的源节点ID
                                            edge_dest.data_handle());     //当前批次节点的目标节点ID

      void* d_temp_storage      = nullptr;//准备排序的临时存储，空指针与空字节数
      size_t temp_storage_bytes = 0;
      //计算排序操作所需的临时存储空间 调用CUB库  
      cub::DeviceMergeSort::SortPairs(d_temp_storage,//d_temp_storage=nullptr 时，返回所需字节数到 temp_storage_bytes
                                      temp_storage_bytes,
                                      edge_dest.data_handle(),
                                      edge_src.data_handle(),
                                      total_edges,
                                      CmpEdge<IdxT>(),//自定义比较器
                                      stream);

      RAFT_LOG_DEBUG("Temp storage needed for sorting (bytes): %lu", temp_storage_bytes);//记录排序所需内存大小

      auto temp_sort_storage = raft::make_device_mdarray<IdxT>( //分配排序临时空间
        res,
        raft::resource::get_large_workspace_resource(res),
        raft::make_extents<int64_t>(temp_storage_bytes / sizeof(IdxT)));

      // Sort to group reverse edges by destination 按目标节点ID排序反向边列表
      //排序效果：所有指向同一目标节点的边被分组连续存储;eg:[v1→u1,v2→u3, v1→u2] → [v1→u1, v1→u2, v2→u3]
      cub::DeviceMergeSort::SortPairs(temp_sort_storage.data_handle(),
                                      temp_storage_bytes,
                                      edge_dest.data_handle(),
                                      edge_src.data_handle(),
                                      total_edges,
                                      CmpEdge<IdxT>(),
                                      stream);

      // Get number of unique node destinations 计算反向边列表中不同目标节点的数量
      // 对已排序的edge_dest数组，计算连续不重复元素数量  eg input [1,2,2,3,3,3] output 3
      IdxT unique_dests =
        raft::sparse::neighbors::get_n_components(edge_dest.data_handle(), total_edges, stream);

      // Find which node IDs have reverse edges and their indices in the reverse edge list
      // 在反向边列表中查找哪些节点ID具有反向边及其索引
      // edge_dest_vec是thrust库中的变量，是存储在GPU中的动态数组，代码将数据初始化到容器，以便利用thrust库的并行算法
      thrust::device_vector<IdxT> edge_dest_vec(edge_dest.data_handle(),//edge_dest.data_handle()是设备上的指针
                                                edge_dest.data_handle() + total_edges);//数组结束位置：edge_dest.data_handle() + total_edges
      auto unique_indices = raft::make_device_vector<int>(res, total_edges);//创建一个存储在 GPU 设备上的动态数组 unique_indices
      raft::linalg::map_offset(res, unique_indices.view(), raft::identity_op{});//对 unique_indices 中的每个元素应用一个函数（这里是 raft::identity_op），并将结果存储回 unique_indices 中。raft::identity_op 是一个恒等操作
      thrust::unique_by_key(    //去除 edge_dest_vec 和 unique_indices 中连续重复的键值对，对于 edge_dest_vec 中连续相同的键，只保留第一个键及其对应的值，其余重复的键值对会被移除
        edge_dest_vec.begin(), edge_dest_vec.end(), unique_indices.data_handle());

      edge_dest_vec.clear();//立即释放 Thrust 向量占用的设备内存
      edge_dest_vec.shrink_to_fit();//重新分配一个较小的内存块

      // Batch execution of reverse edge creation/application 
      reverse_batch = params.reverse_batchsize;//控制每批处理的数量
      //循环处理所有的唯一目标节点（unique_dests），每次处理reverse_batch个节点
      for (int rev_start = 0; rev_start < (int)unique_dests; rev_start += reverse_batch) {
        if (rev_start + reverse_batch > (int)unique_dests) {
          reverse_batch = (int)unique_dests - rev_start;
        }//如果当前批次的起始位置（rev_start）加上批次大小超过了唯一目标节点的总数，则调整批次大小以避免越界

        // Allocate reverse QueryCandidate list based on number of unique destinations
        /*raft::make_device_mdarray来在设备（GPU）上分配内存，
        用于存储反向查询候选列表（reverse_list_ptr）反向ID（rev_ids）和反向距离（rev_dists）*/ 
        auto reverse_list_ptr = raft::make_device_mdarray<QueryCandidates<IdxT, accT>>(
          res,
          raft::resource::get_large_workspace_resource(res),
          raft::make_extents<int64_t>(reverse_batch));
        auto rev_ids =
          raft::make_device_mdarray<IdxT>(res,
                                          raft::resource::get_large_workspace_resource(res),
                                          raft::make_extents<int64_t>(reverse_batch, visited_size));
        auto rev_dists =
          raft::make_device_mdarray<accT>(res,
                                          raft::resource::get_large_workspace_resource(res),
                                          raft::make_extents<int64_t>(reverse_batch, visited_size));
        //将reverse_list_ptr转换为QueryCandidates指针    
        QueryCandidates<IdxT, accT>* reverse_list =
          static_cast<QueryCandidates<IdxT, accT>*>(reverse_list_ptr.data_handle());
        //初始化反向处理批次的查询候选列表，设置批次大小和访问大小  IdxT：节点ID的数据类型；accT：距离值的数据类型
                            //线程块256，每块线程数blockD（预设值）共享内存0 执行流stream    
        init_query_candidate_list<IdxT, accT><<<256, blockD, 0, stream>>>(reverse_list,//反向查询列表结构体指针,类型是QueryCandidates<IdxT, accT>*
                                                                          rev_ids.data_handle(),//反向节点ID存储数组，类型IdxT*
                                                                          rev_dists.data_handle(),//反向节点距离存储数组，类型accT*
                                                                          (int)reverse_batch,//当前反向批次大小
                                                                          visited_size);//单个查询最大访问节点数

        // May need more blocks for reverse list
        num_blocks = min(maxBlocks, reverse_batch);//根据reverse_batch和maxBlocks计算所需的块数（num_blocks）

        // Populate reverse list ids and candidate lists from edge_src and edge_dest 填充反向邻接表结构
        // 原始图中，只存储出边，而对称化需要填充入边，这个内核就是在为每个节点提收集“谁指向我”的信息
        populate_reverse_list_struct<T, accT, IdxT> //将反向边列表数据填充到查询候选结构体
          <<<num_blocks, blockD, 0, stream>>>(reverse_list,//反向查询列表结构体,类型QueryCandidates<IdxT, accT>*
                                              edge_src.data_handle(),//源节点ID数组（排序后）,类型IdxT*
                                              edge_dest.data_handle(),//目标节点ID数组（排序后）,类型IdxT*
                                              unique_indices.data_handle(),//	唯一目标节点的起始索引，类型int*
                                              unique_dests,//唯一目标节点总数,类型IdxT
                                              total_edges,// 边总数,类型IdxT
                                              dataset.extent(0),// 节点总数,类型int64_T
                                              rev_start,// 逆向起始索引,类型int
                                              reverse_batch);// 逆向批次数,类型int

        // Recompute distances (avoided keeping it during sorting)
        recompute_reverse_dists<T, accT, IdxT> //recompute_reverse_dist内核重新计算距离
          <<<num_blocks, blockD, 0, stream>>>(reverse_list, dataset, reverse_batch, metric);

        // Call 2nd RobustPrune on reverse query_list
        RobustPruneKernel<T, accT, IdxT> //调用RobustPruneKernel内核进行二次剪枝，剪枝反向列表
          <<<num_blocks, blockD, prune_smem_total_size, stream>>>(d_graph.view(),//图结构设备内存视图，表示需要剪枝的图索引
                                                                  raft::make_const_mdspan(dataset),//原始数据集只读视图，计算向量间距离
                                                                  reverse_list_ptr.data_handle(),//指向反向查询列表的设备指针,存储本轮待处理的目标节点
                                                                  reverse_batch,//本次处理的反向查询数量（即 reverse_list_ptr 中的节点数）
                                                                  visited_size,//已访问节点集合的大小（用于防止重复处理）
                                                                  metric,//距离度量类型，(L2)
                                                                  alpha,//剪枝强度
                                                                  prune_smem_sort_size);//共享内存中排序缓冲区的大小

        // Write new edge lists to graph 将剪枝后的新边列表写入图结构
        write_graph_edges_kernel<accT, IdxT><<<num_blocks, blockD, 0, stream>>>(
          d_graph.view(), reverse_list_ptr.data_handle(), degree, reverse_batch);//图结构的设备内存视图,指向反向查询列表的设备指针，剪枝后每个节点保留的邻居数量,本次处理的反向查询节点数量
      }

      start += step_size;//分批处理数据集，每完成一个批次就移动到下一批次起始位置
      step_size *= base;//将当前步长 step_size 乘以 base，以动态调整下一批次的处理量
      if (step_size > max_batchsize) step_size = max_batchsize;//确保步长不会超过最大允许的批次大小 

    }  // Batch of inserts

  }  // insert iterations
//在主机和GPU之间拷贝数据；主机端图数据指针，GPU端图数据指针，要拷贝数据大小
  raft::copy(graph.data_handle(), d_graph.data_handle(), d_graph.size(), stream);

  RAFT_CHECK_CUDA(stream);//检查前面执行的CUDA操作（数据拷贝）是否成功完成
}
//定义模板类，T是数据类型，IdxT是索引类型，Accessor为访问器类型
template <typename T,
          typename IdxT     = uint64_t,
          typename Accessor = raft::host_device_accessor<std::experimental::default_accessor<T>,
                                                         raft::memory_type::host>>
index<T, IdxT> build(
  raft::resources const& res, //res是管理资源的对象
  const index_params& params, //构建索引的参数
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset)//数据集，使用mdspan多维数组视图描述
{
  uint32_t graph_degree = params.graph_degree;//从参数中获取图的每个节点的出边数（度数）

  RAFT_EXPECTS(params.metric == cuvs::distance::DistanceType::L2Expanded,
               "Currently only L2Expanded metric is supported");//检查距离度量类型是否支持L2
  //std::find 在预定义数组 DEGREE_SIZES 中搜索指定的 graph_degree
  const int* deg_size = std::find(std::begin(DEGREE_SIZES), std::end(DEGREE_SIZES), graph_degree);
  RAFT_EXPECTS(deg_size != std::end(DEGREE_SIZES), "Provided graph_degree not currently supported");
  //确保访问大小（visited_size）大于图的度数，构建索引的一个前提条件
  RAFT_EXPECTS(params.visited_size > graph_degree, "visited_size must be > graph_degree");
  /*dataset.extent(0) 表示数据集的样本数量；dataset.extent(1)表示每个样本的特征维度（内在维度）*/
  int dim = dataset.extent(1);//获取数据集的维度信息；

  RAFT_LOG_DEBUG("Creating empty graph structure");
  auto vamana_graph = raft::make_host_matrix<IdxT, int64_t>(dataset.extent(0), graph_degree);//raft::make_host_matrix 在主机端分配内存，用于存储图的边信息

  RAFT_LOG_DEBUG("Running Vamana batched insert algorithm");

  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded;

  IdxT medoid_id;//存储类内聚点（数据集的中心点）的索引
  batched_insert_vamana<T, float, IdxT, Accessor>(//将数据集以批处理的方式插入到 VAMANA 图结构
    res, params, dataset, vamana_graph.view(), &medoid_id, metric);

  try { //尝试构建并返回一个 VAMANA 索引对象
    return index<T, IdxT>(
      res, params.metric, dataset, raft::make_const_mdspan(vamana_graph.view()), medoid_id);//raft::make_const_mdspan 将图结构转换为常量多维数组视图，并提供类内聚点 ID
  } catch (std::bad_alloc& e) {//捕捉GPU内存不足异常
    RAFT_LOG_DEBUG("Insufficient GPU memory to construct VAMANA index with dataset on GPU");
    // We just add the graph. User is expected to update dataset separately (e.g allocating in
    // managed memory). 用户需要自行更新数据集
  } catch (raft::logic_error& e) {
    // The memory error can also manifest as logic_error.
    RAFT_LOG_DEBUG("Insufficient GPU memory to construct VAMANA index with dataset on GPU");
  }
  index<T, IdxT> idx(res, params.metric);//前面的异常处理部分执行失败，这里会构造一个空的 VAMANA 索引对象 idx
  RAFT_LOG_WARN("Constructor not called, returning empty index");
  return idx;
}

/**
 * @}
 */

}  // namespace cuvs::neighbors::vamana::detail
