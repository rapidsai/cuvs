# cuVS ANN Memory Footprint and Peak Usage

This document caputures:
* Peak memory usage and the total memory footprint of each ANN index. 
* Memory strategies (such as chunking inputs from host)

## Brute-force

### Dataset on device

#### Formula for dataset on device:

$$
\begin{aligned}
n_{\text{vectors}} \times n_{\text{dimensions}} \times \text{sizeof}(T) & +  \textsf{// dataset} \\
\text{sizeof}(T) & \textsf{// norms}
\end{aligned}
$$

#### Formula for peak memory usage on device:

## IVF-Flat

### Dataset on device

#### Formula for index memory footprint (device): 

$$
\begin{aligned}
n_{\text{vectors}} \times n_\text{dimensions} \times \text{sizeof}(T) & + \textsf{// interleaved form} \\
n_{\text{vectors}} \times \text{sizeof}(\text{int-type}) & + \textsf{// list indices} \\
n_{\text{clusters}} \times n_\text{dimensions} \times \text{sizeof}(T) & + \textsf{// cluster centers} \\
n_{\text{clusters}} \times n_{\text{cluster-overhead}} &
\end{aligned}
$$

Each cluster is padded to at least 32 vectors and at most 1,024 vectors. Assuming uniform random distribution of vectors/list, we would have:

$\text{Cluster overhead} = (\text{conservative memory allocation} ? 16 : 512 ) \times \text{dim} \times \text{sizeof}(T) $

Note that each cluster is allocated as a separate allocation. If we use a `cuda_memory_resource`, that would grab memory in `1 MiB` chunks, so on average we might have `0.5 MiB` overhead per cluster. If we us 10's of thousands of clusters, it becomes essential to use pool allocator to avoid this overhead.

$\text{cluster overhead} =  0.5 \text{MiB} \textsf{// if we do not use pool allocator}$

#### Formula for index memory footprint (host):

#### Formula for peak memory usage (device):

$$
\begin{aligned}
\text{index-size} + \text{workspace} \textsf{// where } \\ \text{workspace} = \min(&\\ 
& 1\text{GiB}, \\
& [n_\text{queries} \times \text{sizeof}(T) \times (n_\text{lists} + 1 + n_\text{probes} \times  (k+1)) \\
& + n_\text{queries} \times (n_\text{probes} \times k \times \text{sizeof}(index)]\\
&)
\end{aligned}
$$

## IVF-PQ

### Dataset on device

#### Formula for index memory footprint (device):

Simple approximate formula:

$$\approx n_\text{vectors} \times (pq_\text{dim} \times pq_\text{bits} / 8 + \text{sizeof}(IdxT)) + O(n_\text{clusters})$$

The encoded data in the interleaved form:

$$\approx n_\text{vectors} \times 16 \times \lceil pq_\text{dim} / n_\text{chunks} \rceil \textsf{// where } n_\text{chunks} = \lfloor 128 / pq_\text{bits} \rfloor$$

Ignoring rounding/alignment, this simplifies to:

$$ \approx n_\text{vectors} \times pq_\text{dim} \times pq_\text{bits} / 8 $$

$$ \text{Indices} \approx n_\text{vectors} \times \text{sizeof}(IdxT) $$

Codebooks:

$$ = 4 \times pq_\text{dim} \times pq_\text{len} \times 2^{pq_\text{bits}} \textsf{ // per-subspace (default)} $$
$$ = 4 \times n_\text{clusters} \times pq_\text{len} \times 2^{pq_\text{bits}} \textsf{ // per-cluster} $$

Extras:

$$ \approx n_\text{clusters} \times (20 + 8 \times \text{dim}) $$

#### Formula for index memory footprint (host):

For the index itself - insignificant:

$$\approx O(n_\text{clusters}) $$

When used with refinement, the original data must be available:

$$ n_\text{vectors} \times \text{dim} \times \text{sizeof}(T) $$

#### SEARCH: Formula for peak memory usage (device):

$$ \text{Total usage} = \text{Index} + \text{Queries} + \text{Output indices} + \text{Output distances} +\text{ Workspace} $$

Workspace size is not trivial, a heuristic controls the batch size to make sure the workspace fits the `resource::get_workspace_free_bytes(res)`.

#### BUILD: Formula for peak memory usage (device):

$$
\begin{aligned}
  &\approx n_\text{vectors} / \text{trainset-ratio} \times \text{dim} \times \text{sizeof}(\text{float}) \textsf{ // trainset, may be in managed mem} \\
  & + n_\text{vectors} / \text{trainset-ratio} \times \text{sizeof}(\text{uint32-t}) \textsf{ // labels, may be in managed mem} \\
  &+ n_\text{clusters} \times \text{dim} \times \text{sizeof}(\text{float}) \textsf{ // cluster centers}
\end{aligned}
$$

Note, if there’s not enough space left in the workspace memory resource, IVF-PQ build automatically switches to the managed memory for the training set and labels.

## NN-Descent

#### Formula for index memory footprint (device):

Small graph: 

$$n_\text{vectors} \times 32 \times (\text{sizeof}(\text{IndexT}) + \text{sizeof}(\text{DistT})) $$

$$\textsf{32 is the degree of the small graph, a fixed value}$$

Locks:

$$ n_\text{vectors} \times \text{sizeof}(\text{int}) $$

Edge counter of each list

$$ n_\text{vectors} \times \text{sizeof}(\text{int2}) \times 2 $$

#### Formula for index memory footprint (host):

Full graph: 

$$1.3 \times n_vectors \times node_degree \times (\text{sizeof}(\text{IndexT}) + \text{sizeof}(\text{DistT}))$$

$$\textsf{To speed up graph updates by bucketing neighbors, we allocate 1.3 times the space to reduce collisions}$$

Samples: 

$$n_\text{vectors} \times \text{sizeof}(\text{int}) \times \text{num-samples} \times 5$$

$$\textsf{there are 5 buffers for sampling, num-samples is fixed to 32 for now}$$

Bloom filter for sampling: 

$$n_\text{vectors} \times \text{node-degree} / 32 \times 64$$

Buffer for graph update: 

$$n_\text{vectors} \times 32 \times (\text{sizeof}(\text{int}) + \text{sizeof}(\text{float}))$$

$$\textsf{32 is the degree of the small graph, a fixed value}$$

Edge counter of each list: 

$$n_\text{vectors} \times \text{sizeof}(\text{int2}) \times 2$$

#### Formula for peak memory usage (device):

Small graph: 

$$n_\text{vectors} \times 32 \times (\text{sizeof}(\text{IndexT}) + \text{sizeof}(\text{DistT}))$$

$$\textsf{32 is the degree of the small graph, a fixed value}$$

Locks:

$$n_\text{vectors} \times \text{sizeof}(\text{int})$$

Edge counter of each list: 

$$n_\text{vectors} \times \text{sizeof}(\text{int2}) \times 2$$

Data: 

$$n_\text{vectors} \times \text{ndim} \times \text{sizeof}(\text{--half})$$

Peak memory in bytes `IndexT = uint32_t`, `DistT = float`:

$$n_\text{vectors} \times (\text{ndim} \times 2 + 276)$$

When using the `L2` metric, there will be an additional memory of: $\text{sizeof}(\text{float}) \times n_\text{vectors}$.

## CAGRA

* CAGRA builds a graph that ultimately ends up on the host while it needs to keep the original dataset around (can be on host or device). 
* IVFPQ or NN-DESCENT can be used to build the graph (additions to the peak memory usage calculated as in the respective build algo above). 

### Dataset on device (graph on host)

#### Formula for index memory footprint (device):

$$n_\text{index-vectors} \times n_\text{dims} \times \text{sizeof}(T)$$

#### Formula for index memory footprint (host):

$$\text{graph-degree} \times n_\text{index-vectors} \times \text{sizeof}(T)$$

#### Formula for peak memory usage (device):

### Dataset on host (graph on host)

#### Formula for index memory footprint (device):

#### Formula for index memory footprint (host):

$$n_\text{index-vectors} \times n_\text{dims} \times \text{sizeof}(T) + \text{graph-degree} \times n_\text{index-vectors} \times \text{sizeof}(T)$$

#### Formula for peak memory usage (device):


