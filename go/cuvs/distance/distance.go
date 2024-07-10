package distance

// #include <stdio.h>
// #include <stdlib.h>
// #include <dlpack/dlpack.h>
// #include <cuda_runtime_api.h>
// #include <cuvs/core/c_api.h>
// #include <cuvs/distance/pairwise_distance.h>
// #include <cuvs/neighbors/brute_force.h>
// #include <cuvs/neighbors/ivf_flat.h>
// #include <cuvs/neighbors/cagra.h>
// #include <cuvs/neighbors/ivf_pq.h>
import "C"
import (
	"rapidsai/cuvs/cuvs/common"
	"unsafe"
)

type Distance int

const (
	L2 Distance = iota
	SQEuclidean
	Euclidean
	L1
	Cityblock
	InnerProduct
	Chebyshev
	Canberra
	Cosine
	Lp
	Correlation
	Jaccard
	Hellinger
	BrayCurtis
	JensenShannon
	Hamming
	KLDivergence
	Minkowski
	RusselRao
	Dice
)

var CDistances = map[Distance]int{
	L2:            C.L2SqrtExpanded,
	SQEuclidean:   C.L2Expanded,
	Euclidean:     C.L2SqrtExpanded,
	L1:            C.L1,
	Cityblock:     C.L1,
	InnerProduct:  C.InnerProduct,
	Chebyshev:     C.Linf,
	Canberra:      C.Canberra,
	Cosine:        C.CosineExpanded,
	Lp:            C.LpUnexpanded,
	Correlation:   C.CorrelationExpanded,
	Jaccard:       C.JaccardExpanded,
	Hellinger:     C.HellingerExpanded,
	BrayCurtis:    C.BrayCurtis,
	JensenShannon: C.JensenShannon,
	Hamming:       C.HammingUnexpanded,
	KLDivergence:  C.KLDivergence,
	Minkowski:     C.LpUnexpanded,
	RusselRao:     C.RusselRaoExpanded,
	Dice:          C.DiceExpanded,
}

func PairwiseDistance[T any](Resources common.Resource, x *common.Tensor[T], y *common.Tensor[T], distances *common.Tensor[float32], metric Distance, metric_arg float32) error {

	CMetric := C.cuvsDistanceType(CDistances[metric])

	return common.CheckCuvs(common.CuvsError(C.cuvsPairwiseDistance(C.cuvsResources_t(Resources.Resource), (*C.DLManagedTensor)(unsafe.Pointer(x.C_tensor)), (*C.DLManagedTensor)(unsafe.Pointer(y.C_tensor)), (*C.DLManagedTensor)(unsafe.Pointer(distances.C_tensor)), CMetric, C.float(metric_arg))))

}
