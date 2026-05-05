# Working with ANN Indexes in Python

- [Building an index](#building-an-index)
- [Searching an index](#searching-an-index)

## Building an index

```python
from cuvs.neighbors import cagra

dataset = load_data()
index_params = cagra.IndexParams()

index = cagra.build(build_params, dataset)
```

## Searching an index

```python
from cuvs.neighbors import cagra

queries = load_queries()

search_params = cagra.SearchParams()

index = // ... build index ...

neighbors, distances = cagra.search(search_params, index, queries, k)
```

