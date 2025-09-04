#!/bin/bash

# Script to debug CUDA illegal memory access errors

echo "=== CUDA Memory Debug Script ==="
echo "Testing IVF Flat with BitwiseHamming distance"
echo ""

# Set environment variables for better debugging
export CUDA_LAUNCH_BLOCKING=1
export CUDA_DEVICE_WAITS_ON_EXCEPTION=1
export RAFT_LOG_LEVEL=DEBUG

# Get test binary path
TEST_BINARY="${1:-./cpp/build/gtests/NEIGHBORS_ANN_IVF_FLAT_TEST}"

if [ ! -f "$TEST_BINARY" ]; then
    echo "Error: Test binary not found at $TEST_BINARY"
    echo "Usage: $0 [path_to_test_binary]"
    exit 1
fi

echo "Using test binary: $TEST_BINARY"
echo ""

# Function to run test with different memory checking tools
run_with_tool() {
    local tool=$1
    local filter=$2
    echo "================================"
    echo "Running with $tool"
    echo "================================"
    
    case $tool in
        "cuda-memcheck")
            cuda-memcheck --leak-check full --report-api-errors all \
                         --tool memcheck --print-limit 100 \
                         $TEST_BINARY --gtest_filter="$filter" 2>&1 | tee cuda_memcheck_output.log
            ;;
        "cuda-memcheck-racecheck")
            cuda-memcheck --tool racecheck --racecheck-report all \
                         $TEST_BINARY --gtest_filter="$filter" 2>&1 | tee cuda_racecheck_output.log
            ;;
        "cuda-memcheck-initcheck")
            cuda-memcheck --tool initcheck \
                         $TEST_BINARY --gtest_filter="$filter" 2>&1 | tee cuda_initcheck_output.log
            ;;
        "compute-sanitizer")
            compute-sanitizer --tool memcheck --leak-check full \
                             --show-backtrace yes \
                             $TEST_BINARY --gtest_filter="$filter" 2>&1 | tee compute_sanitizer_output.log
            ;;
        "standard")
            $TEST_BINARY --gtest_filter="$filter" 2>&1 | tee standard_output.log
            ;;
    esac
    
    echo ""
    echo "Exit code: $?"
    echo ""
}

# Test filter for the failing test case
FILTER="AnnIVFFlatTest/AnnIVFFlatTestF_uint8.AnnIVFFlat/1"

# Check which tools are available
if command -v cuda-memcheck &> /dev/null; then
    echo "cuda-memcheck is available"
    HAS_CUDA_MEMCHECK=1
else
    echo "cuda-memcheck is not available"
    HAS_CUDA_MEMCHECK=0
fi

if command -v compute-sanitizer &> /dev/null; then
    echo "compute-sanitizer is available"
    HAS_COMPUTE_SANITIZER=1
else
    echo "compute-sanitizer is not available"
    HAS_COMPUTE_SANITIZER=0
fi

echo ""

# Run with standard execution first to get baseline
echo "1. Running standard execution with CUDA_LAUNCH_BLOCKING=1..."
run_with_tool "standard" "$FILTER"

# Run with cuda-memcheck if available
if [ $HAS_CUDA_MEMCHECK -eq 1 ]; then
    echo "2. Running with cuda-memcheck..."
    run_with_tool "cuda-memcheck" "$FILTER"
    
    echo "3. Running with cuda-memcheck racecheck..."
    run_with_tool "cuda-memcheck-racecheck" "$FILTER"
    
    echo "4. Running with cuda-memcheck initcheck..."
    run_with_tool "cuda-memcheck-initcheck" "$FILTER"
fi

# Run with compute-sanitizer if available
if [ $HAS_COMPUTE_SANITIZER -eq 1 ]; then
    echo "5. Running with compute-sanitizer..."
    run_with_tool "compute-sanitizer" "$FILTER"
fi

echo ""
echo "=== Debug Summary ==="
echo "Check the following log files for details:"
ls -la *.log 2>/dev/null
echo ""
echo "Look for:"
echo "  - Invalid global/shared memory accesses"
echo "  - Out-of-bounds array accesses"
echo "  - Race conditions"
echo "  - Uninitialized memory reads"

