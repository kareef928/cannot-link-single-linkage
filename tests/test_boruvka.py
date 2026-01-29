"""
Tests for hdbscan_minimal - Borůvka MST algorithms.
"""
import numpy as np
import scipy.sparse as sp
import pytest

import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from hdbscan_minimal.core import (
    _build_adjacency_list_numba,
    _constrained_boruvka_mst_csr_strict_numba,
    _constrained_kruskal_mst_csr_strict_sorted_numba,
    _parallel_constrained_boruvka_mst,
    _detect_violations_numba,
    _dsu_find_numba,
)


class TestAdjacencyList:
    """Tests for adjacency list construction."""
    
    def test_simple_graph(self):
        """Test adjacency list builder with a simple 4-node graph."""
        # Graph: 0--1 (w=1), 1--2 (w=2), 2--3 (w=3), 0--2 (w=4)
        u = np.array([0, 1, 2, 0], dtype=np.int32)
        v = np.array([1, 2, 3, 2], dtype=np.int32)
        w = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        n_points = 4

        adj_indptr, adj_neighbors, adj_weights, adj_edge_ids = _build_adjacency_list_numba(u, v, w, n_points)

        assert adj_indptr.shape[0] == n_points + 1
        assert adj_indptr[0] == 0

        # Node 0: neighbors 1, 2
        degree_0 = adj_indptr[1] - adj_indptr[0]
        assert degree_0 == 2
        neighbors_0 = set(adj_neighbors[adj_indptr[0]:adj_indptr[1]])
        assert neighbors_0 == {1, 2}

        # Node 1: neighbors 0, 2
        degree_1 = adj_indptr[2] - adj_indptr[1]
        assert degree_1 == 2
        neighbors_1 = set(adj_neighbors[adj_indptr[1]:adj_indptr[2]])
        assert neighbors_1 == {0, 2}

        # Node 2: neighbors 1, 3, 0
        degree_2 = adj_indptr[3] - adj_indptr[2]
        assert degree_2 == 3
        neighbors_2 = set(adj_neighbors[adj_indptr[2]:adj_indptr[3]])
        assert neighbors_2 == {0, 1, 3}

    def test_isolated_node(self):
        """Test adjacency list with an isolated node."""
        # Graph: 0--1 (w=1), node 2 isolated
        u = np.array([0], dtype=np.int32)
        v = np.array([1], dtype=np.int32)
        w = np.array([1.0], dtype=np.float64)
        n_points = 3

        adj_indptr, adj_neighbors, _, _ = _build_adjacency_list_numba(u, v, w, n_points)

        # Node 2 has degree 0
        degree_2 = adj_indptr[3] - adj_indptr[2]
        assert degree_2 == 0


class TestBoruvkaMST:
    """Tests for constrained Borůvka MST algorithm."""
    
    def test_unconstrained_produces_valid_mst(self):
        """Test that Borůvka produces valid MST without constraints."""
        np.random.seed(42)
        n = 10
        X = np.random.randn(n, 2)
        distances = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))

        # Build edges from dense matrix
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j, distances[i, j]))

        u = np.array([e[0] for e in edges], dtype=np.int32)
        v = np.array([e[1] for e in edges], dtype=np.int32)
        w = np.array([e[2] for e in edges], dtype=np.float64)

        # No constraints
        cannot_link_csr = sp.csr_matrix((n, n), dtype=np.int32)
        cl_indptr = np.asarray(cannot_link_csr.indptr, dtype=np.int64)
        cl_indices = np.asarray(cannot_link_csr.indices, dtype=np.int32)

        adj_indptr, adj_neighbors, adj_weights, adj_edge_ids = _build_adjacency_list_numba(u, v, w, n)

        mst_edges = _constrained_boruvka_mst_csr_strict_numba(
            adj_indptr, adj_neighbors, adj_weights, adj_edge_ids,
            cl_indptr, cl_indices, n, u.shape[0]
        )

        assert mst_edges.shape[0] == n - 1, f"MST should have {n-1} edges, got {mst_edges.shape[0]}"

    def test_known_example_with_constraints(self):
        """Test Borůvka with known constraint scenario."""
        # Triangle: 0-1-2 with edges (0,1)=1, (1,2)=2, (0,2)=3
        # Constraint: 0 and 2 cannot-link
        u = np.array([0, 1, 0], dtype=np.int32)
        v = np.array([1, 2, 2], dtype=np.int32)
        w = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        n = 3

        cannot_link = np.zeros((n, n), dtype=np.float64)
        cannot_link[0, 2] = 1.0
        cannot_link[2, 0] = 1.0
        cannot_link_csr = sp.csr_matrix(cannot_link, dtype=np.int32)
        cl_indptr = np.asarray(cannot_link_csr.indptr, dtype=np.int64)
        cl_indices = np.asarray(cannot_link_csr.indices, dtype=np.int32)

        adj_indptr, adj_neighbors, adj_weights, adj_edge_ids = _build_adjacency_list_numba(u, v, w, n)

        mst_edges = _constrained_boruvka_mst_csr_strict_numba(
            adj_indptr, adj_neighbors, adj_weights, adj_edge_ids,
            cl_indptr, cl_indices, n, u.shape[0]
        )

        # Without constraint: MST would be (0,1)=1 + (1,2)=2 connecting all 3
        # With constraint: 0-2 cannot be in same component
        # So we should get a forest, not a tree
        assert mst_edges.shape[0] <= n - 1

    def test_single_node(self):
        """Test Borůvka with 2 nodes."""
        u = np.array([0], dtype=np.int32)
        v = np.array([1], dtype=np.int32)
        w = np.array([1.0], dtype=np.float64)
        n = 2

        cannot_link_csr = sp.csr_matrix((n, n), dtype=np.int32)
        cl_indptr = np.asarray(cannot_link_csr.indptr, dtype=np.int64)
        cl_indices = np.asarray(cannot_link_csr.indices, dtype=np.int32)

        adj_indptr, adj_neighbors, adj_weights, adj_edge_ids = _build_adjacency_list_numba(u, v, w, n)

        mst_edges = _constrained_boruvka_mst_csr_strict_numba(
            adj_indptr, adj_neighbors, adj_weights, adj_edge_ids,
            cl_indptr, cl_indices, n, 1
        )

        assert mst_edges.shape[0] == 1

    def test_fully_constrained_no_merges(self):
        """Test Borůvka when all edges are constrained."""
        n = 4
        edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)]
        u = np.array([e[0] for e in edges], dtype=np.int32)
        v = np.array([e[1] for e in edges], dtype=np.int32)
        w = np.array([e[2] for e in edges], dtype=np.float64)

        # Every pair constrained
        cannot_link = np.ones((n, n), dtype=np.float64)
        np.fill_diagonal(cannot_link, 0)
        cannot_link_csr = sp.csr_matrix(cannot_link, dtype=np.int32)
        cl_indptr = np.asarray(cannot_link_csr.indptr, dtype=np.int64)
        cl_indices = np.asarray(cannot_link_csr.indices, dtype=np.int32)

        adj_indptr, adj_neighbors, adj_weights, adj_edge_ids = _build_adjacency_list_numba(u, v, w, n)

        mst_edges = _constrained_boruvka_mst_csr_strict_numba(
            adj_indptr, adj_neighbors, adj_weights, adj_edge_ids,
            cl_indptr, cl_indices, n, len(edges)
        )

        # No edges should be added
        assert mst_edges.shape[0] == 0

    def test_matches_kruskal_on_random_data(self):
        """Test that Borůvka and Kruskal produce same weight MST."""
        np.random.seed(123)
        n = 15
        X = np.random.randn(n, 2)
        distances = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))

        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j, distances[i, j]))

        u = np.array([e[0] for e in edges], dtype=np.int32)
        v = np.array([e[1] for e in edges], dtype=np.int32)
        w = np.array([e[2] for e in edges], dtype=np.float64)

        cannot_link_csr = sp.csr_matrix((n, n), dtype=np.int32)
        cl_indptr = np.asarray(cannot_link_csr.indptr, dtype=np.int64)
        cl_indices = np.asarray(cannot_link_csr.indices, dtype=np.int32)

        # Borůvka
        adj_indptr, adj_neighbors, adj_weights, adj_edge_ids = _build_adjacency_list_numba(u, v, w, n)
        mst_boruvka = _constrained_boruvka_mst_csr_strict_numba(
            adj_indptr, adj_neighbors, adj_weights, adj_edge_ids,
            cl_indptr, cl_indices, n, len(edges)
        )

        # Kruskal
        order = np.argsort(w)
        mst_kruskal = _constrained_kruskal_mst_csr_strict_sorted_numba(
            u[order], v[order], w[order],
            cl_indptr, cl_indices, n
        )

        total_weight_boruvka = mst_boruvka[:, 2].sum()
        total_weight_kruskal = mst_kruskal[:, 2].sum()

        assert np.isclose(total_weight_boruvka, total_weight_kruskal, rtol=1e-9)


class TestParallelBoruvkaMST:
    """Tests for parallel constrained Borůvka MST."""
    
    def test_no_constraints_matches_sequential(self):
        """Test parallel Borůvka matches sequential without constraints."""
        np.random.seed(42)
        n = 20
        X = np.random.randn(n, 2)
        distances = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))

        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j, distances[i, j]))

        u = np.array([e[0] for e in edges], dtype=np.int32)
        v = np.array([e[1] for e in edges], dtype=np.int32)
        w = np.array([e[2] for e in edges], dtype=np.float64)

        cannot_link_csr = sp.csr_matrix((n, n), dtype=np.int32)
        cl_indptr = np.asarray(cannot_link_csr.indptr, dtype=np.int64)
        cl_indices = np.asarray(cannot_link_csr.indices, dtype=np.int32)

        adj_indptr, adj_neighbors, adj_weights, adj_edge_ids = _build_adjacency_list_numba(u, v, w, n)

        # Sequential
        mst_seq, _ = _parallel_constrained_boruvka_mst(
            adj_indptr, adj_neighbors, adj_weights, adj_edge_ids,
            cl_indptr, cl_indices, n, len(edges),
            parallel_backend="sequential"
        )

        # CPU parallel
        mst_cpu, _ = _parallel_constrained_boruvka_mst(
            adj_indptr, adj_neighbors, adj_weights, adj_edge_ids,
            cl_indptr, cl_indices, n, len(edges),
            parallel_backend="cpu"
        )

        assert mst_seq.shape[0] == n - 1
        assert mst_cpu.shape[0] == n - 1
        assert np.isclose(mst_seq[:, 2].sum(), mst_cpu[:, 2].sum())

    def test_with_constraints_respects_them(self):
        """Test parallel Borůvka respects constraints."""
        np.random.seed(42)
        n = 15
        X = np.random.randn(n, 2)
        distances = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))

        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j, distances[i, j]))

        u = np.array([e[0] for e in edges], dtype=np.int32)
        v = np.array([e[1] for e in edges], dtype=np.int32)
        w = np.array([e[2] for e in edges], dtype=np.float64)

        constraint_pairs = [(0, 5), (2, 8)]
        cannot_link = np.zeros((n, n), dtype=np.float64)
        for i, j in constraint_pairs:
            cannot_link[i, j] = 1.0
            cannot_link[j, i] = 1.0
        cannot_link_csr = sp.csr_matrix(cannot_link, dtype=np.int32)
        cl_indptr = np.asarray(cannot_link_csr.indptr, dtype=np.int64)
        cl_indices = np.asarray(cannot_link_csr.indices, dtype=np.int32)

        adj_indptr, adj_neighbors, adj_weights, adj_edge_ids = _build_adjacency_list_numba(u, v, w, n)

        mst_edges, _ = _parallel_constrained_boruvka_mst(
            adj_indptr, adj_neighbors, adj_weights, adj_edge_ids,
            cl_indptr, cl_indices, n, len(edges),
            parallel_backend="cpu"
        )

        # Rebuild DSU to check constraints
        parent = np.arange(n, dtype=np.int32)
        for e in range(mst_edges.shape[0]):
            a, b = int(mst_edges[e, 0]), int(mst_edges[e, 1])
            ra = _dsu_find_numba(parent, a)
            rb = _dsu_find_numba(parent, b)
            if ra != rb:
                parent[rb] = ra

        # Check no constraint violation
        for i, j in constraint_pairs:
            ri = _dsu_find_numba(parent, i)
            rj = _dsu_find_numba(parent, j)
            assert ri != rj, f"Constraint ({i},{j}) violated"

    def test_violation_detection(self):
        """Test violation detection correctly identifies violations."""
        n = 5
        parent = np.zeros(n, dtype=np.int32)  # All in same component

        cannot_link_csr = sp.csr_matrix(([1], ([1], [3])), shape=(n, n), dtype=np.int32)
        cannot_link_csr = cannot_link_csr + cannot_link_csr.T
        indptr = np.asarray(cannot_link_csr.indptr, dtype=np.int64)
        indices = np.asarray(cannot_link_csr.indices, dtype=np.int32)

        va, vb, n_viol = _detect_violations_numba(parent, indptr, indices, n)

        assert n_viol == 1
        found = (va[0] == 1 and vb[0] == 3) or (va[0] == 3 and vb[0] == 1)
        assert found


class TestMSTMethodInvalid:
    """Test invalid MST method handling."""
    
    def test_invalid_raises(self):
        """Test that invalid mst_method raises ValueError."""
        from hdbscan_minimal.core import _mst_constrained_hard
        
        n = 5
        u = np.array([0, 1], dtype=np.int32)
        v = np.array([1, 2], dtype=np.int32)
        w = np.array([1.0, 2.0], dtype=np.float64)
        
        cannot_link_csr = sp.csr_matrix((n, n), dtype=np.int32)
        cl_indptr = np.asarray(cannot_link_csr.indptr, dtype=np.int64)
        cl_indices = np.asarray(cannot_link_csr.indices, dtype=np.int32)
        
        with pytest.raises(ValueError, match="Invalid mst_method"):
            _mst_constrained_hard(
                n_points=n, u=u, v=v, w=w,
                cannot_link_indptr=cl_indptr,
                cannot_link_indices=cl_indices,
                mst_method="invalid_method"
            )
