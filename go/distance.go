package cuvs

// #include <cuvs/distance/pairwise_distance.h>
import "C"

import (
	"errors"
	"runtime"
	"unsafe"
)

type Distance int

// Supported distance metrics
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

// Maps cuvs Go distances to C distances
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

// Computes the pairwise distance between two vectors.
func PairwiseDistance[T any](Resources Resource, x *Tensor[T], y *Tensor[T], distances *Tensor[float32], metric Distance, metric_arg float32) error {
	CMetric, exists := CDistances[metric]

	if !exists {
		return errors.New("cuvs: invalid distance metric")
	}

	defer func() {
		runtime.KeepAlive(Resources)
		runtime.KeepAlive(x)
		runtime.KeepAlive(y)
		runtime.KeepAlive(distances)
	}()

	return CheckCuvs(CuvsError(C.cuvsPairwiseDistance(C.cuvsResources_t(Resources.Resource), (*C.DLManagedTensor)(unsafe.Pointer(x.C_tensor)), (*C.DLManagedTensor)(unsafe.Pointer(y.C_tensor)), (*C.DLManagedTensor)(unsafe.Pointer(distances.C_tensor)), C.cuvsDistanceType(CMetric), C.float(metric_arg))))
}
