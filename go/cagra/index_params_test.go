package cagra

import (
	"testing"
)

// CompressionParams Tests
func TestCreateCompressionParams(t *testing.T) {
	params, err := CreateCompressionParams()
	if err != nil {
		t.Fatalf("Failed to create CompressionParams: %v", err)
	}
	if params == nil {
		t.Fatal("CreateCompressionParams returned nil params")
	}

	if params.params == nil {
		t.Fatal("CompressionParams internal params are nil")
	}
	if params.params.pq_kmeans_trainset_fraction != 0 {
		t.Fatalf("Error params.params.pq_kmeans_trainset_fraction != 0, got = %v", params.params.pq_kmeans_trainset_fraction)
	}
	if params.params.pq_bits != 8 {
		t.Fatalf("Error params.params.pq_bits != 8, got = %v", params.params.pq_bits)
	}
	if params.params.pq_dim != 0 {
		t.Fatalf("Error params.params.pq_dim != 0, got = %v", params.params.pq_dim)
	}
	if params.params.vq_n_centers != 0 {
		t.Fatalf("Error params.params.vq_n_centers != 0, got = %v", params.params.vq_n_centers)
	}
	if params.params.kmeans_n_iters != 25 {
		t.Fatalf("Error params.params.kmeans_n_iters != 25, got = %v", params.params.kmeans_n_iters)
	}
}

func TestCompressionParamsSetPQBits(t *testing.T) {
	params, err := CreateCompressionParams()
	if err != nil {
		t.Fatalf("Failed to create CompressionParams: %v", err)
	}

	testCases := []struct {
		name  string
		value uint32
	}{
		{"4 bits", 4},
		{"8 bits", 8},
		{"16 bits", 16},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := params.SetPQBits(tc.value)
			if err != nil {
				t.Errorf("SetPQBits failed: %v", err)
			}
			if result != params {
				t.Error("SetPQBits should return the same params instance")
			}
			if uint32(params.params.pq_bits) != tc.value {
				t.Errorf("Expected pq_bits %d, got %d", tc.value, params.params.pq_bits)
			}
		})
	}
}

func TestCompressionParamsSetPQDim(t *testing.T) {
	params, err := CreateCompressionParams()
	if err != nil {
		t.Fatalf("Failed to create CompressionParams: %v", err)
	}

	testCases := []struct {
		name  string
		value uint32
	}{
		{"Zero (auto)", 0},
		{"Small dimension", 32},
		{"Large dimension", 128},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := params.SetPQDim(tc.value)
			if err != nil {
				t.Errorf("SetPQDim failed: %v", err)
			}
			if result != params {
				t.Error("SetPQDim should return the same params instance")
			}
			if uint32(params.params.pq_dim) != tc.value {
				t.Errorf("Expected pq_dim %d, got %d", tc.value, params.params.pq_dim)
			}
		})
	}
}

func TestCompressionParamsSetVQNCenters(t *testing.T) {
	params, err := CreateCompressionParams()
	if err != nil {
		t.Fatalf("Failed to create CompressionParams: %v", err)
	}

	testCases := []struct {
		name  string
		value uint32
	}{
		{"Zero (auto)", 0},
		{"Small centers", 256},
		{"Large centers", 2048},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := params.SetVQNCenters(tc.value)
			if err != nil {
				t.Errorf("SetVQNCenters failed: %v", err)
			}
			if result != params {
				t.Error("SetVQNCenters should return the same params instance")
			}
			if uint32(params.params.vq_n_centers) != tc.value {
				t.Errorf("Expected vq_n_centers %d, got %d", tc.value, params.params.vq_n_centers)
			}
		})
	}
}

func TestCompressionParamsSetKMeansNIters(t *testing.T) {
	params, err := CreateCompressionParams()
	if err != nil {
		t.Fatalf("Failed to create CompressionParams: %v", err)
	}

	testCases := []struct {
		name  string
		value uint32
	}{
		{"Few iterations", 10},
		{"Default iterations", 25},
		{"Many iterations", 100},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := params.SetKMeansNIters(tc.value)
			if err != nil {
				t.Errorf("SetKMeansNIters failed: %v", err)
			}
			if result != params {
				t.Error("SetKMeansNIters should return the same params instance")
			}
			if uint32(params.params.kmeans_n_iters) != tc.value {
				t.Errorf("Expected kmeans_n_iters %d, got %d", tc.value, params.params.kmeans_n_iters)
			}
		})
	}
}

func TestCompressionParamsSetVQKMeansTrainsetFraction(t *testing.T) {
	params, err := CreateCompressionParams()
	if err != nil {
		t.Fatalf("Failed to create CompressionParams: %v", err)
	}

	testCases := []struct {
		name  string
		value float64
	}{
		{"Zero (auto)", 0.0},
		{"Half dataset", 0.5},
		{"Full dataset", 1.0},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := params.SetVQKMeansTrainsetFraction(tc.value)
			if err != nil {
				t.Errorf("SetVQKMeansTrainsetFraction failed: %v", err)
			}
			if result != params {
				t.Error("SetVQKMeansTrainsetFraction should return the same params instance")
			}
			if float64(params.params.vq_kmeans_trainset_fraction) != tc.value {
				t.Errorf("Expected vq_kmeans_trainset_fraction %f, got %f",
					tc.value, params.params.vq_kmeans_trainset_fraction)
			}
		})
	}
}

func TestCompressionParamsSetPQKMeansTrainsetFraction(t *testing.T) {
	params, err := CreateCompressionParams()
	if err != nil {
		t.Fatalf("Failed to create CompressionParams: %v", err)
	}

	testCases := []struct {
		name  string
		value float64
	}{
		{"Zero (auto)", 0.0},
		{"Quarter dataset", 0.25},
		{"Half dataset", 0.5},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := params.SetPQKMeansTrainsetFraction(tc.value)
			if err != nil {
				t.Errorf("SetPQKMeansTrainsetFraction failed: %v", err)
			}
			if result != params {
				t.Error("SetPQKMeansTrainsetFraction should return the same params instance")
			}
			if float64(params.params.pq_kmeans_trainset_fraction) != tc.value {
				t.Errorf("Expected pq_kmeans_trainset_fraction %f, got %f",
					tc.value, params.params.pq_kmeans_trainset_fraction)
			}
		})
	}
}

// IndexParams Tests
func TestCreateIndexParams(t *testing.T) {
	params, err := CreateIndexParams()
	if err != nil {
		t.Fatalf("Failed to create IndexParams: %v", err)
	}
	defer params.Close()

	if params == nil {
		t.Fatal("CreateIndexParams returned nil params")
	}

	if params.params == nil {
		t.Fatal("IndexParams internal params are nil")
	}
}

func TestIndexParamsSetIntermediateGraphDegree(t *testing.T) {
	params, err := CreateIndexParams()
	if err != nil {
		t.Fatalf("Failed to create IndexParams: %v", err)
	}
	defer params.Close()

	testCases := []struct {
		name  string
		value uintptr
	}{
		{"Small degree", 32},
		{"Medium degree", 64},
		{"Large degree", 128},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := params.SetIntermediateGraphDegree(tc.value)
			if err != nil {
				t.Errorf("SetIntermediateGraphDegree failed: %v", err)
			}
			if result != params {
				t.Error("SetIntermediateGraphDegree should return the same params instance")
			}
			if uintptr(params.params.intermediate_graph_degree) != tc.value {
				t.Errorf("Expected intermediate_graph_degree %d, got %d",
					tc.value, params.params.intermediate_graph_degree)
			}
		})
	}
}

func TestIndexParamsSetGraphDegree(t *testing.T) {
	params, err := CreateIndexParams()
	if err != nil {
		t.Fatalf("Failed to create IndexParams: %v", err)
	}
	defer params.Close()

	testCases := []struct {
		name  string
		value uintptr
	}{
		{"Small degree", 16},
		{"Medium degree", 32},
		{"Large degree", 64},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := params.SetGraphDegree(tc.value)
			if err != nil {
				t.Errorf("SetGraphDegree failed: %v", err)
			}
			if result != params {
				t.Error("SetGraphDegree should return the same params instance")
			}
			if uintptr(params.params.graph_degree) != tc.value {
				t.Errorf("Expected graph_degree %d, got %d",
					tc.value, params.params.graph_degree)
			}
		})
	}
}

func TestIndexParamsSetBuildAlgo(t *testing.T) {
	params, err := CreateIndexParams()
	if err != nil {
		t.Fatalf("Failed to create IndexParams: %v", err)
	}
	defer params.Close()

	testCases := []struct {
		name          string
		algo          BuildAlgo
		expectedError bool
	}{
		{"IVF_PQ algorithm", IvfPq, false},
		{"NN_DESCENT algorithm", NnDescent, false},
		{"AUTO_SELECT algorithm", AutoSelect, false},
		{"Invalid algorithm", BuildAlgo(999), true},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := params.SetBuildAlgo(tc.algo)

			if tc.expectedError {
				if err == nil {
					t.Error("Expected error for invalid build_algo, got none")
				}
			} else {
				if err != nil {
					t.Errorf("SetBuildAlgo failed: %v", err)
				}
				if result != params {
					t.Error("SetBuildAlgo should return the same params instance")
				}
			}
		})
	}
}

func TestIndexParamsSetNNDescentNiter(t *testing.T) {
	params, err := CreateIndexParams()
	if err != nil {
		t.Fatalf("Failed to create IndexParams: %v", err)
	}
	defer params.Close()

	testCases := []struct {
		name  string
		value uint32
	}{
		{"Few iterations", 10},
		{"Default iterations", 20},
		{"Many iterations", 50},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := params.SetNNDescentNiter(tc.value)
			if err != nil {
				t.Errorf("SetNNDescentNiter failed: %v", err)
			}
			if result != params {
				t.Error("SetNNDescentNiter should return the same params instance")
			}
			if uint32(params.params.nn_descent_niter) != tc.value {
				t.Errorf("Expected nn_descent_niter %d, got %d",
					tc.value, params.params.nn_descent_niter)
			}
		})
	}
}

func TestIndexParamsSetCompression(t *testing.T) {
	params, err := CreateIndexParams()
	if err != nil {
		t.Fatalf("Failed to create IndexParams: %v", err)
	}
	defer params.Close()

	compression, err := CreateCompressionParams()
	if err != nil {
		t.Fatalf("Failed to create CompressionParams: %v", err)
	}

	// Configure compression params
	compression.SetPQBits(8)
	compression.SetPQDim(64)

	result, err := params.SetCompression(compression)
	if err != nil {
		t.Errorf("SetCompression failed: %v", err)
	}
	if result != params {
		t.Error("SetCompression should return the same params instance")
	}
}

func TestIndexParamsClose(t *testing.T) {
	params, err := CreateIndexParams()
	if err != nil {
		t.Fatalf("Failed to create IndexParams: %v", err)
	}

	err = params.Close()
	if err != nil {
		t.Errorf("Close() failed: %v", err)
	}
}

func TestBuildAlgoConstants(t *testing.T) {
	// Test that BuildAlgo constants are properly defined
	algos := []BuildAlgo{IvfPq, NnDescent, AutoSelect}

	for _, algo := range algos {
		_, exists := cBuildAlgos[algo]
		if !exists {
			t.Errorf("BuildAlgo %v not found in cBuildAlgos map", algo)
		}
	}
}
