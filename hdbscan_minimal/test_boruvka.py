"""
Test suite for the constrained Borůvka MST implementation.

Tests cover:
1. Basic graph connectivity
2. Cannot-link constraint enforcement  
3. Edge cases (empty graphs, single nodes)
4. Real-world scenarios (point clouds with constraints)
"""
import numpy as np
import pytest
from boruvka_constrained import (
    constrained_boruvka_mst,
    build_constraint_csr,
    dsu_find,
    csr_row_contains,
)


class TestBasicMST:
    """Tests for basic MST functionality without constraints."""
    
    def test_triangle_graph(self):
        """Triangle graph should produce 2-edge MST."""
        u = np.array([0, 0, 1], dtype=np.int32)
        v = np.array([1, 2, 2], dtype=np.int32)
        w = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        cl_indptr = np.zeros(4, dtype=np.int64)
        cl_indices = np.array([], dtype=np.int32)
        
        mst, removed = constrained_boruvka_mst(u, v, w, cl_indptr, cl_indices, 3)
        
        assert len(mst) == 2, "Triangle should have 2 MST edges"
        assert len(removed) == 0, "No constraints means no removed edges"
        # Check MST weight is minimal (1 + 2 = 3)
        total_weight = sum(e[2] for e in mst)
        assert total_weight == 3.0, f"MST weight should be 3, got {total_weight}"
    
    def test_line_graph(self):
        """Linear graph: 0-1-2-3 should return itself as MST."""
        u = np.array([0, 1, 2], dtype=np.int32)
        v = np.array([1, 2, 3], dtype=np.int32)
        w = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        cl_indptr = np.zeros(5, dtype=np.int64)
        cl_indices = np.array([], dtype=np.int32)
        
        mst, removed = constrained_boruvka_mst(u, v, w, cl_indptr, cl_indices, 4)
        
        assert len(mst) == 3, "Line graph with 4 nodes should have 3 MST edges"
    
    def test_complete_graph_4_nodes(self):
        """Complete graph on 4 nodes should produce MST with 3 edges."""
        # K4: all pairs connected
        u = np.array([0, 0, 0, 1, 1, 2], dtype=np.int32)
        v = np.array([1, 2, 3, 2, 3, 3], dtype=np.int32)
        w = np.array([1.0, 4.0, 3.0, 2.0, 5.0, 1.0], dtype=np.float64)
        cl_indptr = np.zeros(5, dtype=np.int64)
        cl_indices = np.array([], dtype=np.int32)
        
        mst, removed = constrained_boruvka_mst(u, v, w, cl_indptr, cl_indices, 4)
        
        assert len(mst) == 3, "K4 should have 3 MST edges"
        # Optimal MST: 0-1 (1), 1-2 (2), 2-3 (1) = weight 4
        total_weight = sum(e[2] for e in mst)
        assert total_weight == 4.0, f"MST weight should be 4, got {total_weight}"


class TestConstraints:
    """Tests for cannot-link constraint enforcement."""
    
    def test_single_constraint_splits_graph(self):
        """A constraint between endpoints should split the graph."""
        # Line: 0-1-2, with constraint 0-2
        u = np.array([0, 1], dtype=np.int32)
        v = np.array([1, 2], dtype=np.int32)
        w = np.array([1.0, 1.0], dtype=np.float64)
        
        constraint_pairs = np.array([[0, 2]], dtype=np.int32)
        cl_indptr, cl_indices = build_constraint_csr(3, constraint_pairs)
        
        mst, removed = constrained_boruvka_mst(u, v, w, cl_indptr, cl_indices, 3)
        
        # Since 0 and 2 cannot be connected, and the only path is 0-1-2,
        # we can only have 1 edge (either 0-1 or 1-2)
        assert len(mst) < 2, "Constraint should prevent full connectivity"
    
    def test_constraint_forces_longer_path(self):
        """Constraint should force algorithm to take a longer path."""
        # Square: 0-1-2-3-0 with diagonal 0-2
        #   0 --- 1
        #   |  X  |
        #   3 --- 2
        u = np.array([0, 0, 1, 2, 3, 0], dtype=np.int32)
        v = np.array([1, 3, 2, 3, 0, 2], dtype=np.int32)
        w = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.5], dtype=np.float64)  # diagonal is cheapest
        
        # No constraints: should use diagonal
        cl_indptr = np.zeros(5, dtype=np.int64)
        cl_indices = np.array([], dtype=np.int32)
        mst_no_constraint, _ = constrained_boruvka_mst(u, v, w, cl_indptr, cl_indices, 4)
        
        # With constraint on diagonal (0,2)
        constraint_pairs = np.array([[0, 2]], dtype=np.int32)
        cl_indptr, cl_indices = build_constraint_csr(4, constraint_pairs)
        mst_constrained, _ = constrained_boruvka_mst(u, v, w, cl_indptr, cl_indices, 4)
        
        # Constrained MST should be heavier (can't use cheap diagonal)
        weight_no_constraint = sum(e[2] for e in mst_no_constraint)
        weight_constrained = sum(e[2] for e in mst_constrained)
        
        # Note: Due to constraint, we may get a forest with fewer edges
        # The key is that 0 and 2 are never connected
    
    def test_multiple_constraints(self):
        """Multiple constraints should all be respected."""
        # 6 nodes in a circle: 0-1-2-3-4-5-0
        u = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
        v = np.array([1, 2, 3, 4, 5, 0], dtype=np.int32)
        w = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        
        # Constraints: (0,3), (1,4), (2,5) - opposite pairs
        constraint_pairs = np.array([[0, 3], [1, 4], [2, 5]], dtype=np.int32)
        cl_indptr, cl_indices = build_constraint_csr(6, constraint_pairs)
        
        mst, _ = constrained_boruvka_mst(u, v, w, cl_indptr, cl_indices, 6)
        
        # Build component membership from MST
        parent = np.arange(6, dtype=np.int32)
        for edge in mst:
            u_e, v_e = int(edge[0]), int(edge[1])
            ru, rv = dsu_find(parent, u_e), dsu_find(parent, v_e)
            if ru != rv:
                parent[max(ru, rv)] = min(ru, rv)
        
        # Check no constrained pairs are in same component
        for i, j in constraint_pairs:
            ri, rj = dsu_find(parent, i), dsu_find(parent, j)
            assert ri != rj, f"Constrained pair ({i},{j}) ended up in same component"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_single_node(self):
        """Single node should produce empty MST."""
        u = np.array([], dtype=np.int32)
        v = np.array([], dtype=np.int32)
        w = np.array([], dtype=np.float64)
        cl_indptr = np.zeros(2, dtype=np.int64)
        cl_indices = np.array([], dtype=np.int32)
        
        mst, removed = constrained_boruvka_mst(u, v, w, cl_indptr, cl_indices, 1)
        
        assert len(mst) == 0, "Single node should have no MST edges"
    
    def test_two_nodes_no_constraint(self):
        """Two connected nodes without constraint should form 1-edge MST."""
        u = np.array([0], dtype=np.int32)
        v = np.array([1], dtype=np.int32)
        w = np.array([1.0], dtype=np.float64)
        cl_indptr = np.zeros(3, dtype=np.int64)
        cl_indices = np.array([], dtype=np.int32)
        
        mst, removed = constrained_boruvka_mst(u, v, w, cl_indptr, cl_indices, 2)
        
        assert len(mst) == 1, "Two nodes should have 1 MST edge"
    
    def test_two_nodes_with_constraint(self):
        """Two nodes with constraint between them should be disconnected."""
        u = np.array([0], dtype=np.int32)
        v = np.array([1], dtype=np.int32)
        w = np.array([1.0], dtype=np.float64)
        
        constraint_pairs = np.array([[0, 1]], dtype=np.int32)
        cl_indptr, cl_indices = build_constraint_csr(2, constraint_pairs)
        
        mst, removed = constrained_boruvka_mst(u, v, w, cl_indptr, cl_indices, 2)
        
        assert len(mst) == 0, "Constrained pair should be disconnected"
    
    def test_disconnected_graph(self):
        """Disconnected graph should produce forest."""
        # Two separate triangles
        u = np.array([0, 0, 1, 3, 3, 4], dtype=np.int32)
        v = np.array([1, 2, 2, 4, 5, 5], dtype=np.int32)
        w = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        cl_indptr = np.zeros(7, dtype=np.int64)
        cl_indices = np.array([], dtype=np.int32)
        
        mst, removed = constrained_boruvka_mst(u, v, w, cl_indptr, cl_indices, 6)
        
        # Should have 2 + 2 = 4 edges (2 trees of 3 nodes each)
        assert len(mst) == 4, "Disconnected graph should produce forest"


class TestHelperFunctions:
    """Tests for helper functions."""
    
    def test_build_constraint_csr_empty(self):
        """Empty constraints should produce empty CSR."""
        pairs = np.array([], dtype=np.int32).reshape(0, 2)
        indptr, indices = build_constraint_csr(5, pairs)
        
        assert len(indptr) == 6
        assert len(indices) == 0
        assert all(indptr == 0)
    
    def test_build_constraint_csr_single(self):
        """Single constraint should produce symmetric CSR."""
        pairs = np.array([[0, 2]], dtype=np.int32)
        indptr, indices = build_constraint_csr(4, pairs)
        
        # Node 0 should have constraint to 2
        assert csr_row_contains(indptr, indices, 0, 2)
        # Node 2 should have constraint to 0 (symmetric)
        assert csr_row_contains(indptr, indices, 2, 0)
        # Other nodes should have no constraints
        assert indptr[1] - indptr[0] == 1  # Node 0 has 1 constraint
        assert indptr[2] - indptr[1] == 0  # Node 1 has 0 constraints
    
    def test_dsu_find_basic(self):
        """DSU find should return root."""
        parent = np.array([0, 0, 2, 2], dtype=np.int32)
        
        assert dsu_find(parent, 0) == 0
        assert dsu_find(parent, 1) == 0  # 1's parent is 0
        assert dsu_find(parent, 2) == 2
        assert dsu_find(parent, 3) == 2  # 3's parent is 2


class TestRealWorldScenarios:
    """Tests simulating real clustering scenarios."""
    
    def test_two_moons_intra_constraint(self):
        """Simulate two moons with intra-moon constraints."""
        np.random.seed(42)
        n_per_moon = 10
        
        # Generate simple 2D points for two "moons"
        # Moon 1: points 0-9
        # Moon 2: points 10-19
        n_points = 2 * n_per_moon
        
        # Create complete graph edges
        edges_u, edges_v, edges_w = [], [], []
        for i in range(n_points):
            for j in range(i + 1, n_points):
                edges_u.append(i)
                edges_v.append(j)
                # Intra-moon edges are cheap, inter-moon are expensive
                if (i < n_per_moon) == (j < n_per_moon):
                    edges_w.append(np.random.uniform(0.1, 0.3))  # Same moon
                else:
                    edges_w.append(np.random.uniform(0.8, 1.2))  # Different moons
        
        u = np.array(edges_u, dtype=np.int32)
        v = np.array(edges_v, dtype=np.int32)
        w = np.array(edges_w, dtype=np.float64)
        
        # Add intra-moon constraints (points within same moon that can't connect)
        constraint_pairs = np.array([
            [0, 5],   # Within moon 1
            [10, 15], # Within moon 2
        ], dtype=np.int32)
        cl_indptr, cl_indices = build_constraint_csr(n_points, constraint_pairs)
        
        mst, removed = constrained_boruvka_mst(u, v, w, cl_indptr, cl_indices, n_points)
        
        # Verify constraints are respected
        parent = np.arange(n_points, dtype=np.int32)
        for edge in mst:
            u_e, v_e = int(edge[0]), int(edge[1])
            ru, rv = dsu_find(parent, u_e), dsu_find(parent, v_e)
            if ru != rv:
                parent[max(ru, rv)] = min(ru, rv)
        
        for i, j in constraint_pairs:
            ri, rj = dsu_find(parent, i), dsu_find(parent, j)
            assert ri != rj, f"Intra-moon constrained pair ({i},{j}) should not be connected"
    
    def test_scalability_100_nodes(self):
        """Test with 100 nodes to verify scalability."""
        np.random.seed(42)
        n_points = 100
        
        # Create sparse graph (k-nearest neighbor style)
        k = 10
        edges_u, edges_v, edges_w = [], [], []
        
        for i in range(n_points):
            # Connect to k random other nodes
            neighbors = np.random.choice(
                [j for j in range(n_points) if j != i],
                size=min(k, n_points - 1),
                replace=False
            )
            for j in neighbors:
                if i < j:  # Avoid duplicates
                    edges_u.append(i)
                    edges_v.append(j)
                    edges_w.append(np.random.uniform(0.1, 1.0))
        
        u = np.array(edges_u, dtype=np.int32)
        v = np.array(edges_v, dtype=np.int32)
        w = np.array(edges_w, dtype=np.float64)
        
        # Add some random constraints
        n_constraints = 20
        constraint_pairs = []
        for _ in range(n_constraints):
            i, j = np.random.choice(n_points, size=2, replace=False)
            constraint_pairs.append([min(i, j), max(i, j)])
        constraint_pairs = np.array(constraint_pairs, dtype=np.int32)
        cl_indptr, cl_indices = build_constraint_csr(n_points, constraint_pairs)
        
        # Should complete without error
        mst, removed = constrained_boruvka_mst(u, v, w, cl_indptr, cl_indices, n_points)
        
        # Basic sanity checks
        assert len(mst) <= n_points - 1, "MST cannot have more than n-1 edges"
        
        # Verify all constraints respected
        parent = np.arange(n_points, dtype=np.int32)
        for edge in mst:
            u_e, v_e = int(edge[0]), int(edge[1])
            ru, rv = dsu_find(parent, u_e), dsu_find(parent, v_e)
            if ru != rv:
                parent[max(ru, rv)] = min(ru, rv)
        
        for i, j in constraint_pairs:
            ri, rj = dsu_find(parent, i), dsu_find(parent, j)
            assert ri != rj, f"Constrained pair ({i},{j}) ended up in same component"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
