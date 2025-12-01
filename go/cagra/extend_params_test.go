package cagra

import (
	"testing"
)

func TestCreateExtendParams(t *testing.T) {
	params, err := CreateExtendParams()
	if err != nil {
		t.Fatalf("Failed to create ExtendParams: %v", err)
	}
	defer params.Close()

	if params == nil {
		t.Fatal("CreateExtendParams returned nil params")
	}

	if params.params == nil {
		t.Fatal("ExtendParams internal params are nil")
	}
}

func TestExtendParamsSetMaxChunkSize(t *testing.T) {
	params, err := CreateExtendParams()
	if err != nil {
		t.Fatalf("Failed to create ExtendParams: %v", err)
	}
	defer params.Close()

	testCases := []struct {
		name          string
		maxChunkSize  uint32
		expectedError bool
	}{
		{
			name:          "Zero value (auto select)",
			maxChunkSize:  0,
			expectedError: false,
		},
		{
			name:          "Small chunk size",
			maxChunkSize:  100,
			expectedError: false,
		},
		{
			name:          "Medium chunk size",
			maxChunkSize:  1000,
			expectedError: false,
		},
		{
			name:          "Large chunk size",
			maxChunkSize:  10000,
			expectedError: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := params.SetMaxChunkSize(tc.maxChunkSize)

			if tc.expectedError && err == nil {
				t.Errorf("Expected error but got none")
			}

			if !tc.expectedError && err != nil {
				t.Errorf("Unexpected error: %v", err)
			}

			if result == nil {
				t.Error("SetMaxChunkSize returned nil")
			}

			// Verify the value was set
			if uint32(params.params.max_chunk_size) != tc.maxChunkSize {
				t.Errorf("Expected max_chunk_size to be %d, got %d",
					tc.maxChunkSize, params.params.max_chunk_size)
			}
		})
	}
}

func TestExtendParamsSetMaxChunkSizeChaining(t *testing.T) {
	params, err := CreateExtendParams()
	if err != nil {
		t.Fatalf("Failed to create ExtendParams: %v", err)
	}
	defer params.Close()

	// Test method chaining
	result, err := params.SetMaxChunkSize(500)
	if err != nil {
		t.Fatalf("SetMaxChunkSize failed: %v", err)
	}

	if result != params {
		t.Error("SetMaxChunkSize should return the same params instance for chaining")
	}
}

func TestExtendParamsClose(t *testing.T) {
	params, err := CreateExtendParams()
	if err != nil {
		t.Fatalf("Failed to create ExtendParams: %v", err)
	}

	err = params.Close()
	if err != nil {
		t.Errorf("Close() failed: %v", err)
	}
}

func TestExtendParamsLifecycle(t *testing.T) {
	// Test complete lifecycle: create -> set -> close
	params, err := CreateExtendParams()
	chunkSize := uint32(2000)
	if err != nil {
		t.Fatalf("Failed to create ExtendParams: %v", err)
	}

	params, err = params.SetMaxChunkSize(chunkSize)
	if err != nil {
		t.Fatalf("SetMaxChunkSize failed: %v", err)
	}

	if uint32(params.params.max_chunk_size) != chunkSize {
		t.Errorf("Expected max_chunk_size to be %d, got %d",
			chunkSize, params.params.max_chunk_size)
	}

	err = params.Close()
	if err != nil {
		t.Errorf("Close() failed: %v", err)
	}
}
