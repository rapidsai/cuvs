#
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the C++ Google Benchmark backend.
"""

import pytest
import numpy as np
from pathlib import Path

from cuvs_bench.backends import (
    Dataset,
    CppGoogleBenchmarkBackend,
    get_registry,
)


class TestCppGoogleBenchmarkBackend:
    """Tests for CppGoogleBenchmarkBackend."""
    
    def test_backend_auto_registered(self):
        """Test that cpp_gbench backend is auto-registered on import."""
        registry = get_registry()
        assert registry.is_registered("cpp_gbench")
        
        backends = registry.list_backends()
        assert "cpp_gbench" in backends
        assert backends["cpp_gbench"] == CppGoogleBenchmarkBackend
    
    def test_backend_instantiation_missing_executable(self):
        """Test that backend raises error if executable doesn't exist."""
        config = {
            "executable_path": "/nonexistent/path/to/executable"
        }
        
        with pytest.raises(FileNotFoundError, match="executable not found"):
            CppGoogleBenchmarkBackend(config)
    
    def test_backend_name_extraction(self):
        """Test algorithm name extraction from executable path."""
        # Create a dummy executable file for testing
        import tempfile
        with tempfile.NamedTemporaryFile(
            suffix="_ANN_BENCH", 
            delete=False
        ) as f:
            temp_exec = Path(f.name)
        
        try:
            # Make it executable
            temp_exec.chmod(0o755)
            
            config = {"executable_path": str(temp_exec)}
            backend = CppGoogleBenchmarkBackend(config)
            
            # Name should be extracted and lowercased
            expected_name = temp_exec.stem.replace("_ANN_BENCH", "").lower()
            assert backend.name == expected_name
        
        finally:
            temp_exec.unlink(missing_ok=True)
    
    def test_backend_gpu_detection(self):
        """Test GPU detection from backend name."""
        import tempfile
        
        # Test cuVS backend (should support GPU)
        with tempfile.NamedTemporaryFile(
            prefix="CUVS_IVF_FLAT", 
            suffix="_ANN_BENCH", 
            delete=False
        ) as f:
            cuvs_exec = Path(f.name)
        
        try:
            cuvs_exec.chmod(0o755)
            backend = CppGoogleBenchmarkBackend({"executable_path": str(cuvs_exec)})
            assert backend.supports_gpu is True
        finally:
            cuvs_exec.unlink(missing_ok=True)
        
        # Test CPU backend (should not support GPU)
        with tempfile.NamedTemporaryFile(
            prefix="FAISS_CPU_FLAT", 
            suffix="_ANN_BENCH", 
            delete=False
        ) as f:
            cpu_exec = Path(f.name)
        
        try:
            cpu_exec.chmod(0o755)
            backend = CppGoogleBenchmarkBackend({"executable_path": str(cpu_exec)})
            assert backend.supports_gpu is False
        finally:
            cpu_exec.unlink(missing_ok=True)
    
    def test_backend_config_defaults(self):
        """Test that backend uses default config values."""
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_exec = Path(f.name)
        
        try:
            temp_exec.chmod(0o755)
            
            # Minimal config
            config = {"executable_path": str(temp_exec)}
            backend = CppGoogleBenchmarkBackend(config)
            
            # Check defaults
            assert backend.data_prefix == ""
            assert backend.index_prefix == ""
            assert backend.warmup_time == 1.0
        finally:
            temp_exec.unlink(missing_ok=True)
    
    def test_backend_config_custom_values(self):
        """Test that backend respects custom config values."""
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_exec = Path(f.name)
        
        try:
            temp_exec.chmod(0o755)
            
            config = {
                "executable_path": str(temp_exec),
                "data_prefix": "custom_data/",
                "index_prefix": "custom_index/",
                "warmup_time": 2.5
            }
            backend = CppGoogleBenchmarkBackend(config)
            
            assert backend.data_prefix == "custom_data/"
            assert backend.index_prefix == "custom_index/"
            assert backend.warmup_time == 2.5
        finally:
            temp_exec.unlink(missing_ok=True)
    
    def test_get_backend_from_registry(self):
        """Test getting cpp_gbench backend from registry."""
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_exec = Path(f.name)
        
        try:
            temp_exec.chmod(0o755)
            
            registry = get_registry()
            backend = registry.get_backend("cpp_gbench", {
                "executable_path": str(temp_exec)
            })
            
            assert isinstance(backend, CppGoogleBenchmarkBackend)
            assert backend.name == temp_exec.stem.lower()
        finally:
            temp_exec.unlink(missing_ok=True)


class TestCppBackendBuildSkip:
    """Test that build phase correctly skips when index exists."""
    
    def test_build_skip_when_index_exists(self):
        """Test that build is skipped if index exists and force=False."""
        import tempfile
        
        with tempfile.NamedTemporaryFile(delete=False) as exec_file:
            temp_exec = Path(exec_file.name)
        
        with tempfile.NamedTemporaryFile(delete=False) as index_file:
            index_path = Path(index_file.name)
        
        try:
            temp_exec.chmod(0o755)
            
            backend = CppGoogleBenchmarkBackend({
                "executable_path": str(temp_exec)
            })
            
            # Create dummy dataset
            dataset = Dataset(
                name="test",
                base_vectors=np.random.rand(100, 32).astype(np.float32),
                query_vectors=np.random.rand(10, 32).astype(np.float32)
            )
            
            # Build should be skipped since index exists
            result = backend.build(
                dataset=dataset,
                build_params={"nlist": 1024},
                index_path=index_path,
                force=False
            )
            
            assert result.success is True
            assert result.metadata.get("skipped") is True
            assert result.metadata.get("reason") == "index_exists"
            assert result.build_time_seconds == 0.0
        
        finally:
            temp_exec.unlink(missing_ok=True)
            index_path.unlink(missing_ok=True)


# Note: Full integration tests with actual C++ executables should be run separately
# as they require the benchmark executables to be built and available.

