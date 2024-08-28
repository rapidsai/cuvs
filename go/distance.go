package cuvs

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
	"errors"
	"unsafe"
)

type Distance int

const (
	DistanceL2 Distance = iota
	DistanceSQEuclidean
	DistanceEuclidean
	DistanceL1
	DistanceCityblock
	DistanceInnerProduct
	DistanceChebyshev
	DistanceCanberra
	DistanceCosine
	DistanceLp
	DistanceCorrelation
	DistanceJaccard
	DistanceHellinger
	DistanceBrayCurtis
	DistanceJensenShannon
	DistanceHamming
	DistanceKLDivergence
	DistanceMinkowski
	DistanceRusselRao
	DistanceDice
)

var CDistances = map[Distance]int{
	DistanceL2:            C.L2SqrtExpanded,
	DistanceSQEuclidean:   C.L2Expanded,
	DistanceEuclidean:     C.L2SqrtExpanded,
	DistanceL1:            C.L1,
	DistanceCityblock:     C.L1,
	DistanceInnerProduct:  C.InnerProduct,
	DistanceChebyshev:     C.Linf,
	DistanceCanberra:      C.Canberra,
	DistanceCosine:        C.CosineExpanded,
	DistanceLp:            C.LpUnexpanded,
	DistanceCorrelation:   C.CorrelationExpanded,
	DistanceJaccard:       C.JaccardExpanded,
	DistanceHellinger:     C.HellingerExpanded,
	DistanceBrayCurtis:    C.BrayCurtis,
	DistanceJensenShannon: C.JensenShannon,
	DistanceHamming:       C.HammingUnexpanded,
	DistanceKLDivergence:  C.KLDivergence,
	DistanceMinkowski:     C.LpUnexpanded,
	DistanceRusselRao:     C.RusselRaoExpanded,
	DistanceDice:          C.DiceExpanded,
}

func PairwiseDistance[T any](Resources Resource, x *Tensor[T], y *Tensor[T], distances *Tensor[float32], metric Distance, metric_arg float32) error {

	CMetric, exists := CDistances[metric]

	if !exists {
		return errors.New("cuvs: invalid distance metric")
	}

	return CheckCuvs(CuvsError(C.cuvsPairwiseDistance(C.cuvsResources_t(Resources.Resource), (*C.DLManagedTensor)(unsafe.Pointer(x.C_tensor)), (*C.DLManagedTensor)(unsafe.Pointer(y.C_tensor)), (*C.DLManagedTensor)(unsafe.Pointer(distances.C_tensor)), C.cuvsDistanceType(CMetric), C.float(metric_arg))))

}
