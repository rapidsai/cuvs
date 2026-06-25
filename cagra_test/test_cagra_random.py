import cupy as cp
import numpy as np
from cuvs.neighbors import cagra

# データセット
n_samples = 10000
dim = 128

# クエリ数
n_queries = 100
k = 10

dataset = cp.random.random((n_samples, dim), dtype=cp.float32)
queries = cp.random.random((n_queries, dim), dtype=cp.float32)

# CAGRA インデックス構築
index_params = cagra.IndexParams(metric="sqeuclidean")
index = cagra.build(index_params, dataset)

# 探索
search_params = cagra.SearchParams()
distances, neighbors = cagra.search(search_params, index, queries, k)

# device_ndarray → NumPy に変換
neighbors_np = np.asarray(neighbors.copy_to_host())
distances_np = np.asarray(distances.copy_to_host())

print("dataset shape:", dataset.shape)
print("queries shape:", queries.shape)
print("neighbors shape:", neighbors_np.shape)
print("distances shape:", distances_np.shape)

print("neighbors[:5]:")
print(neighbors_np[:5])

print("distances[:5]:")
print(distances_np[:5])
