"""
Tests for hdbscan_minimal - Backend selection and parallel backends.
"""
import numpy as np
import scipy.sparse as sp
import pytest

import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from hdbscan_minimal.core import (
    _check_cuda_available,
    _select_parallel_backend,
    _parallel_constrained_boruvka_mst,
    _build_adjacency_list_numba,
    fast_hdbscan_precomputed_with_cannot_link,
)


class TestCUDADetection:
    """Tests for CUDA detection utility."""
    
    def test_cached_result(self):
        """Test that _check_cuda_available returns consistent cached results."""
        result1 = _check_cuda_available()
        result2 = _check_cuda_available()
        
        assert result1 == result2
        assert isinstance(result1, bool)


class TestBackendSelection:
    """Tests for parallel backend selection logic."""
    
    def test_auto_returns_valid_backend(self):
        """Test that 'auto' backend selection returns a valid backend."""
        backend = _select_parallel_backend(100, "auto")
        assert backend in ("cuda", "cpu", "sequential")
    
    def test_explicit_sequential(self):
        """Test explicit sequential backend selection."""
        assert _select_parallel_backend(100, "sequential") == "sequential"
    
    def test_explicit_cpu(self):
        """Test explicit cpu backend selection."""
        assert _select_parallel_backend(100, "cpu") == "cpu"
    
    def test_explicit_cuda(self):
        """Test explicit cuda backend raises if unavailable."""
        try:
            result = _select_parallel_backend(100, "cuda")
            # If CUDA is available, should return "cuda"
            assert result == "cuda"
        except ValueError as e:
            # Expected if CUDA is not available
            assert "CUDA is not available" in str(e)


class TestBackendEquivalence:
    """Tests for CPU vs sequential backend equivalence."""
    
    def test_cpu_vs_sequential_produce_same_result(self):
        """Test that CPU and sequential backends produce identical MST results."""
        np.random.seed(123)
        n = 15

        edges_list = []
        for i in range(n):
            for j in range(i + 1, n):
                weight = np.random.random()
                edges_list.append((i, j, weight))

        u = np.array([e[0] for e in edges_list], dtype=np.int32)
        v = np.array([e[1] for e in edges_list], dtype=np.int32)
        w = np.array([e[2] for e in edges_list], dtype=np.float64)
        n_edges = len(edges_list)

        adj_indptr, adj_neighbors, adj_weights, adj_edge_ids = _build_adjacency_list_numba(u, v, w, n)

        constraint_pairs = [(0, 5), (2, 8), (3, 10)]
        cannot_link = np.zeros((n, n), dtype=np.float64)
        for i, j in constraint_pairs:
            cannot_link[i, j] = 1.0
            cannot_link[j, i] = 1.0
        cannot_link_csr = sp.csr_matrix(cannot_link, dtype=np.int32)
        cl_indptr = np.asarray(cannot_link_csr.indptr, dtype=np.int64)
        cl_indices = np.asarray(cannot_link_csr.indices, dtype=np.int32)

        # Run sequential backend
        mst_edges_seq, cl_edges_seq = _parallel_constrained_boruvka_mst(
            adj_indptr, adj_neighbors, adj_weights, adj_edge_ids,
            cl_indptr, cl_indices, n, n_edges,
            parallel_backend="sequential"
        )

        # Run CPU backend
        mst_edges_cpu, cl_edges_cpu = _parallel_constrained_boruvka_mst(
            adj_indptr, adj_neighbors, adj_weights, adj_edge_ids,
            cl_indptr, cl_indices, n, n_edges,
            parallel_backend="cpu"
        )

        assert np.array_equal(mst_edges_seq, mst_edges_cpu), "MST edges differ"
        assert np.array_equal(cl_edges_seq, cl_edges_cpu), "Cannot-link edges differ"


class TestParameterPropagation:
    """Tests for parallel_backend parameter propagation through API."""
    
    def test_parameter_propagation(self):
        """Test that parallel_backend parameter is propagated through the API."""
        np.random.seed(42)
        n = 10
        X = np.random.randn(n, 2)
        distances = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
        cannot_link = np.zeros((n, n), dtype=np.float64)

        # Test with explicit sequential backend
        labels_seq, _ = fast_hdbscan_precomputed_with_cannot_link(
            distances, cannot_link,
            min_cluster_size=2,
            mst_method="parallel_boruvka",
            parallel_backend="sequential",
        )

        # Test with explicit cpu backend
        labels_cpu, _ = fast_hdbscan_precomputed_with_cannot_link(
            distances, cannot_link,
            min_cluster_size=2,
            mst_method="parallel_boruvka",
            parallel_backend="cpu",
        )

        # Test with auto backend
        labels_auto, _ = fast_hdbscan_precomputed_with_cannot_link(
            distances, cannot_link,
            min_cluster_size=2,
            mst_method="parallel_boruvka",
            parallel_backend="auto",
        )

        # All should produce the same shape
        assert labels_seq.shape == labels_cpu.shape == labels_auto.shape
        # For unconstrained case, results should be identical
        assert np.array_equal(labels_seq, labels_cpu)
        assert np.array_equal(labels_seq, labels_auto)
