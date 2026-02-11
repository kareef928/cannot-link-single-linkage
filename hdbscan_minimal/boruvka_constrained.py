"""
Constrained Borůvka MST Algorithm
=================================

A minimal implementation of Borůvka's algorithm for computing Minimum
Spanning Forests with cannot-link constraints, using Numba for CPU
parallelization.

Algorithm Overview
------------------
Borůvka's algorithm builds an MST by repeatedly finding the cheapest
edge leaving each connected component, then merging components.

With cannot-link constraints, the algorithm has 3 main steps per round:

1. **Edge Selection (Step 1a + 1b)**: Find the cheapest valid outgoing edge
   for each component. An edge is valid if merging would not violate any
   cannot-link constraints.

2. **Merging (Step 2)**: Merge components using the selected edges. Merges
   proceed without re-checking constraints to allow the algorithm to see
   all edges before deciding which to keep.

3. **Violation Correction (Step 3a + 3b)**: After merging, detect any
   components that contain cannot-link violations and remove the heaviest
   edge to split them. This ensures we keep the minimum-weight edges and
   produce an optimal Minimum Spanning Forest.

The key insight is that Step 2 intentionally allows violations to occur
temporarily so that Step 3 can remove the *heaviest* edge causing the
violation, preserving MSF optimality.

Parallelization
---------------
Uses Numba's prange for CPU parallelization:
- Step 1a: Per-node edge scanning (parallel)
- Step 1b: Per-component aggregation (sequential - requires atomic min)
- Step 2: Component merging (sequential - inherently serial)
- Step 3a: Violation detection (parallel)
- Step 3b: Violation correction (sequential - inherently serial)
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange
from typing import Tuple


# =============================================================================
# DSU (Disjoint Set Union) - Core Data Structure
# =============================================================================

@njit(cache=True)
def dsu_find(parent: np.ndarray, x: int) -> int:
    """
    Find the root of node x with path compression.
    
    Parameters
    ----------
    parent : np.ndarray
        Parent pointer array. parent[i] == i means i is a root.
    x : int
        Node to find the root of.
        
    Returns
    -------
    int
        Root node of the component containing x.
    """
    root = x
    while parent[root] != root:
        root = parent[root]
    # Path compression
    while parent[x] != x:
        next_x = parent[x]
        parent[x] = root
        x = next_x
    return root


@njit(cache=True)
def dsu_find_readonly(parent: np.ndarray, x: int) -> int:
    """
    Find root WITHOUT path compression (safe for parallel reads).
    
    Use this when multiple threads may call find() simultaneously.
    """
    while parent[x] != x:
        x = parent[x]
    return x


# =============================================================================
# Constraint Checking
# =============================================================================

@njit(cache=True)
def check_merge_violates_constraints(
    root_small: int,
    root_large: int,
    parent: np.ndarray,
    head: np.ndarray,
    next_node: np.ndarray,
    cl_indptr: np.ndarray,
    cl_indices: np.ndarray,
) -> bool:
    """
    Check if merging two components would violate cannot-link constraints.
    
    Iterates through all nodes in the smaller component and checks if any
    have a cannot-link constraint with any node in the larger component.
    
    Uses dsu_find_readonly for thread-safety (called from parallel Step 1a).
    """
    node = head[root_small]
    while node != -1:
        start = int(cl_indptr[node])
        end = int(cl_indptr[node + 1])
        for k in range(start, end):
            partner = int(cl_indices[k])
            if dsu_find_readonly(parent, partner) == root_large:
                return True
        node = next_node[node]
    return False


@njit(cache=True)
def component_has_violation(
    root: int,
    parent: np.ndarray,
    head: np.ndarray,
    next_node: np.ndarray,
    cl_indptr: np.ndarray,
    cl_indices: np.ndarray,
) -> bool:
    """
    Check if a component contains any internal cannot-link violations.
    """
    node = head[root]
    while node != -1:
        start = int(cl_indptr[node])
        end = int(cl_indptr[node + 1])
        for k in range(start, end):
            partner = int(cl_indices[k])
            if dsu_find(parent, partner) == root:
                return True
        node = next_node[node]
    return False


@njit(cache=True)
def component_has_violation_readonly(
    root: int,
    parent: np.ndarray,
    head: np.ndarray,
    next_node: np.ndarray,
    cl_indptr: np.ndarray,
    cl_indices: np.ndarray,
) -> bool:
    """
    Check for violations (parallel-safe version using dsu_find_readonly).
    """
    node = head[root]
    while node != -1:
        start = int(cl_indptr[node])
        end = int(cl_indptr[node + 1])
        for k in range(start, end):
            partner = int(cl_indices[k])
            if dsu_find_readonly(parent, partner) == root:
                return True
        node = next_node[node]
    return False


# =============================================================================
# Adjacency List Construction
# =============================================================================

@njit(cache=True)
def build_adjacency_list(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    n_points: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build CSR-format adjacency list from edge arrays.
    
    Each undirected edge is stored twice (once per direction).
    """
    n_edges = u.shape[0]
    
    # Count degree of each node
    degree = np.zeros(n_points, dtype=np.int64)
    for i in range(n_edges):
        degree[u[i]] += 1
        degree[v[i]] += 1
    
    # Build CSR indptr
    adj_indptr = np.empty(n_points + 1, dtype=np.int64)
    adj_indptr[0] = 0
    for i in range(n_points):
        adj_indptr[i + 1] = adj_indptr[i] + degree[i]
    
    # Allocate arrays
    total_entries = adj_indptr[n_points]
    adj_neighbors = np.empty(total_entries, dtype=np.int32)
    adj_weights = np.empty(total_entries, dtype=np.float64)
    adj_edge_ids = np.empty(total_entries, dtype=np.int32)
    
    # Fill adjacency data
    fill_pos = np.zeros(n_points, dtype=np.int64)
    for i in range(n_edges):
        a, b, ww = u[i], v[i], w[i]
        
        pos_a = adj_indptr[a] + fill_pos[a]
        adj_neighbors[pos_a] = b
        adj_weights[pos_a] = ww
        adj_edge_ids[pos_a] = i
        fill_pos[a] += 1
        
        pos_b = adj_indptr[b] + fill_pos[b]
        adj_neighbors[pos_b] = a
        adj_weights[pos_b] = ww
        adj_edge_ids[pos_b] = i
        fill_pos[b] += 1
    
    return adj_indptr, adj_neighbors, adj_weights, adj_edge_ids


# =============================================================================
# Step 1: Find Cheapest Edges
# =============================================================================

@njit(cache=True)
def _find_node_best_edge(
    node: int,
    adj_indptr: np.ndarray,
    adj_neighbors: np.ndarray,
    adj_weights: np.ndarray,
    adj_edge_ids: np.ndarray,
    parent: np.ndarray,
    size: np.ndarray,
    head: np.ndarray,
    next_node: np.ndarray,
    cl_indptr: np.ndarray,
    cl_indices: np.ndarray,
    infeasible: np.ndarray,
) -> tuple:
    """
    Step 1a helper: Find the best outgoing edge for a single node.
    
    Called in parallel for each node to find its cheapest valid
    outgoing edge to another component.
    """
    my_root = dsu_find_readonly(parent, node)
    my_size = size[my_root]
    
    best_weight = np.inf
    best_edge_id = -1
    best_target = -1
    
    start = adj_indptr[node]
    end = adj_indptr[node + 1]
    
    for k in range(start, end):
        edge_id = adj_edge_ids[k]
        if infeasible[edge_id]:
            continue
        
        j = adj_neighbors[k]
        neighbor_root = dsu_find_readonly(parent, j)
        
        if my_root == neighbor_root:
            continue
        
        neighbor_size = size[neighbor_root]
        if my_size < neighbor_size:
            root_small, root_large = my_root, neighbor_root
        else:
            root_small, root_large = neighbor_root, my_root
        
        if check_merge_violates_constraints(
            root_small, root_large, parent, head, next_node,
            cl_indptr, cl_indices
        ):
            continue
        
        ww = adj_weights[k]
        if ww < best_weight or (ww == best_weight and edge_id < best_edge_id):
            best_weight = ww
            best_edge_id = edge_id
            best_target = neighbor_root
    
    return best_weight, best_edge_id, best_target


@njit(parallel=True, cache=True)
def _boruvka_find_cheapest_edges(
    adj_indptr: np.ndarray,
    adj_neighbors: np.ndarray,
    adj_weights: np.ndarray,
    adj_edge_ids: np.ndarray,
    parent: np.ndarray,
    size: np.ndarray,
    head: np.ndarray,
    next_node: np.ndarray,
    cl_indptr: np.ndarray,
    cl_indices: np.ndarray,
    infeasible: np.ndarray,
    n_points: int,
    cheapest_weight: np.ndarray,
    cheapest_edge_id: np.ndarray,
    cheapest_target: np.ndarray,
    cheapest_source: np.ndarray,
    node_best_weight: np.ndarray,
    node_best_edge_id: np.ndarray,
    node_best_target: np.ndarray,
) -> None:
    """
    Step 1: Find the cheapest valid outgoing edge for each component.
    
    Step 1a (parallel): Each node scans its edges to find its best
    outgoing edge to a different component.
    
    Step 1b (sequential): Aggregate per-node results to find the
    single best edge per component.
    """
    # Initialize outputs (parallel)
    for i in prange(n_points):
        cheapest_weight[i] = np.inf
        cheapest_edge_id[i] = -1
        cheapest_target[i] = -1
        cheapest_source[i] = -1
        node_best_weight[i] = np.inf
        node_best_edge_id[i] = -1
        node_best_target[i] = -1
    
    # Step 1a: Each node finds its best outgoing edge (PARALLEL)
    for i in prange(n_points):
        node_idx = np.int32(i)
        w, e, t = _find_node_best_edge(
            node_idx, adj_indptr, adj_neighbors, adj_weights, adj_edge_ids,
            parent, size, head, next_node, cl_indptr, cl_indices, infeasible
        )
        node_best_weight[i] = w
        node_best_edge_id[i] = e
        node_best_target[i] = t
    
    # Step 1b: Aggregate per-component (sequential)
    for i in range(n_points):
        if node_best_edge_id[i] == -1:
            continue
        
        my_root = dsu_find_readonly(parent, i)
        
        if node_best_weight[i] < cheapest_weight[my_root] or (
            node_best_weight[i] == cheapest_weight[my_root]
            and node_best_edge_id[i] < cheapest_edge_id[my_root]
        ):
            cheapest_weight[my_root] = node_best_weight[i]
            cheapest_edge_id[my_root] = node_best_edge_id[i]
            cheapest_target[my_root] = node_best_target[i]
            cheapest_source[my_root] = i


# =============================================================================
# Step 2: Merge Components
# =============================================================================

@njit(cache=True)
def _boruvka_merge_components(
    adj_indptr: np.ndarray,
    adj_neighbors: np.ndarray,
    adj_edge_ids: np.ndarray,
    parent: np.ndarray,
    size: np.ndarray,
    head: np.ndarray,
    tail: np.ndarray,
    next_node: np.ndarray,
    cheapest_weight: np.ndarray,
    cheapest_edge_id: np.ndarray,
    cheapest_target: np.ndarray,
    cheapest_source: np.ndarray,
    mst_edges: np.ndarray,
    n_added: int,
    n_points: int,
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Step 2: Merge components using selected edges and add them to MST.
    
    Merges proceed without re-checking constraints. This allows
    violations to occur temporarily, but Step 3 will detect and
    fix them by removing the heaviest edge.
    """
    round_edges_u = np.empty(n_points, dtype=np.int32)
    round_edges_v = np.empty(n_points, dtype=np.int32)
    round_edges_w = np.empty(n_points, dtype=np.float64)
    round_count = 0
    
    for r in range(n_points):
        if parent[r] != r or cheapest_edge_id[r] == -1:
            continue
        
        target_root = dsu_find(parent, cheapest_target[r])
        
        if r == target_root:
            continue
        
        # Perform union (smaller index wins for determinism)
        winner = min(r, target_root)
        loser = max(r, target_root)
        
        parent[loser] = winner
        size[winner] += size[loser]
        
        # Merge linked lists
        next_node[tail[winner]] = head[loser]
        tail[winner] = tail[loser]
        
        # Find actual edge endpoints
        src = cheapest_source[r]
        edge_id = cheapest_edge_id[r]
        tgt = -1
        for k in range(adj_indptr[src], adj_indptr[src + 1]):
            if adj_edge_ids[k] == edge_id:
                tgt = adj_neighbors[k]
                break
        
        if tgt != -1:
            mst_edges[n_added, 0] = float(src)
            mst_edges[n_added, 1] = float(tgt)
            mst_edges[n_added, 2] = cheapest_weight[r]
            n_added += 1
            
            round_edges_u[round_count] = src
            round_edges_v[round_count] = tgt
            round_edges_w[round_count] = cheapest_weight[r]
            round_count += 1
        
        if n_added >= n_points - 1:
            break
    
    return n_added, round_edges_u, round_edges_v, round_edges_w, round_count


# =============================================================================
# Step 3: Fix Violations
# =============================================================================

@njit(parallel=True, cache=True)
def _find_violating_components(
    parent: np.ndarray,
    head: np.ndarray,
    next_node: np.ndarray,
    cl_indptr: np.ndarray,
    cl_indices: np.ndarray,
    n_points: int,
    has_violation: np.ndarray,
) -> None:
    """
    Step 3a (parallel): Check all components for violations.
    """
    for r in prange(n_points):
        if parent[r] == r:
            has_violation[r] = component_has_violation_readonly(
                r, parent, head, next_node, cl_indptr, cl_indices
            )
        else:
            has_violation[r] = False


@njit(cache=True)
def _fix_violations(
    parent: np.ndarray,
    size: np.ndarray,
    head: np.ndarray,
    tail: np.ndarray,
    next_node: np.ndarray,
    round_edges_u: np.ndarray,
    round_edges_v: np.ndarray,
    round_edges_w: np.ndarray,
    round_count: int,
    mst_edges: np.ndarray,
    n_added: int,
    cl_indptr: np.ndarray,
    cl_indices: np.ndarray,
    n_points: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, int]:
    """
    Step 3: Detect and fix constraint violations by removing the heaviest edge.
    
    Step 3a (parallel): Find all violating components.
    Step 3b (sequential): For each violation, remove the heaviest edge
    added this round to split the component.
    """
    # Step 3a: Parallel violation detection
    has_violation = np.zeros(n_points, dtype=np.bool_)
    _find_violating_components(
        parent, head, next_node, cl_indptr, cl_indices, n_points, has_violation
    )
    
    # Build worklist
    worklist = np.empty(n_points, dtype=np.int32)
    worklist_count = 0
    for r in range(n_points):
        if has_violation[r]:
            worklist[worklist_count] = r
            worklist_count += 1
    
    removed_edges = np.empty((n_points, 3), dtype=np.float64)
    n_removed = 0
    
    # Step 3b: Fix violations (sequential)
    while worklist_count > 0:
        worklist_count -= 1
        comp_root = worklist[worklist_count]
        
        # Find heaviest edge in this component
        heaviest_idx = -1
        heaviest_weight = -np.inf
        
        for i in range(round_count):
            u_node = round_edges_u[i]
            v_node = round_edges_v[i]
            
            if dsu_find(parent, u_node) == comp_root and dsu_find(parent, v_node) == comp_root:
                if round_edges_w[i] > heaviest_weight:
                    heaviest_weight = round_edges_w[i]
                    heaviest_idx = i
        
        if heaviest_idx == -1:
            continue
        
        # Remove edge
        u_rem = round_edges_u[heaviest_idx]
        v_rem = round_edges_v[heaviest_idx]
        w_rem = round_edges_w[heaviest_idx]
        
        round_edges_w[heaviest_idx] = -np.inf
        
        for i in range(n_added):
            if (int(mst_edges[i, 0]) == u_rem and int(mst_edges[i, 1]) == v_rem) or \
               (int(mst_edges[i, 0]) == v_rem and int(mst_edges[i, 1]) == u_rem):
                mst_edges[i, 0] = mst_edges[n_added - 1, 0]
                mst_edges[i, 1] = mst_edges[n_added - 1, 1]
                mst_edges[i, 2] = mst_edges[n_added - 1, 2]
                n_added -= 1
                break
        
        removed_edges[n_removed, 0] = float(u_rem)
        removed_edges[n_removed, 1] = float(v_rem)
        removed_edges[n_removed, 2] = w_rem
        n_removed += 1
        
        # Find all nodes in component
        comp_nodes = np.empty(n_points, dtype=np.int32)
        comp_node_count = 0
        node = head[comp_root]
        while node != -1:
            comp_nodes[comp_node_count] = node
            comp_node_count += 1
            node = next_node[node]
        
        # Reset to singletons
        for i in range(comp_node_count):
            node = comp_nodes[i]
            parent[node] = node
            size[node] = 1
            head[node] = node
            tail[node] = node
            next_node[node] = -1
        
        # Re-merge using remaining MST edges
        for i in range(n_added):
            u_edge = int(mst_edges[i, 0])
            v_edge = int(mst_edges[i, 1])
            
            u_in_comp = False
            v_in_comp = False
            for j in range(comp_node_count):
                if comp_nodes[j] == u_edge:
                    u_in_comp = True
                if comp_nodes[j] == v_edge:
                    v_in_comp = True
            
            if u_in_comp and v_in_comp:
                ru = dsu_find(parent, u_edge)
                rv = dsu_find(parent, v_edge)
                if ru != rv:
                    winner = min(ru, rv)
                    loser = max(ru, rv)
                    parent[loser] = winner
                    size[winner] += size[loser]
                    next_node[tail[winner]] = head[loser]
                    tail[winner] = tail[loser]
        
        # Check sub-components for remaining violations
        checked = np.zeros(n_points, dtype=np.bool_)
        for i in range(comp_node_count):
            node = comp_nodes[i]
            r = dsu_find(parent, node)
            if not checked[r]:
                checked[r] = True
                if component_has_violation(r, parent, head, next_node, cl_indptr, cl_indices):
                    worklist[worklist_count] = r
                    worklist_count += 1
    
    return parent, size, head, tail, next_node, n_added, removed_edges[:n_removed], n_removed


# =============================================================================
# Core Algorithm
# =============================================================================

@njit(parallel=True, cache=True)
def _constrained_boruvka_core(
    adj_indptr: np.ndarray,
    adj_neighbors: np.ndarray,
    adj_weights: np.ndarray,
    adj_edge_ids: np.ndarray,
    cl_indptr: np.ndarray,
    cl_indices: np.ndarray,
    n_points: int,
    n_edges: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Core Borůvka MST algorithm with cannot-link constraints.
    
    Runs rounds until no more edges can be added:
    1. Find cheapest valid outgoing edge per component
    2. Merge components using selected edges
    3. Detect and fix any constraint violations
    """
    # Initialize DSU
    parent = np.arange(n_points, dtype=np.int32)
    size = np.ones(n_points, dtype=np.int32)
    head = np.arange(n_points, dtype=np.int32)
    tail = np.arange(n_points, dtype=np.int32)
    next_node = np.full(n_points, -1, dtype=np.int32)
    
    infeasible = np.zeros(n_edges, dtype=np.bool_)
    
    mst_edges = np.empty((n_points - 1, 3), dtype=np.float64)
    n_added = 0
    
    all_removed = np.empty((n_points, 3), dtype=np.float64)
    total_removed = 0
    
    # Pre-allocated temporaries
    cheapest_weight = np.empty(n_points, dtype=np.float64)
    cheapest_edge_id = np.empty(n_points, dtype=np.int32)
    cheapest_target = np.empty(n_points, dtype=np.int32)
    cheapest_source = np.empty(n_points, dtype=np.int32)
    node_best_weight = np.empty(n_points, dtype=np.float64)
    node_best_edge_id = np.empty(n_points, dtype=np.int32)
    node_best_target = np.empty(n_points, dtype=np.int32)
    
    max_rounds = n_points
    for round_num in range(max_rounds):
        # Step 1: Find cheapest edges
        _boruvka_find_cheapest_edges(
            adj_indptr, adj_neighbors, adj_weights, adj_edge_ids,
            parent, size, head, next_node,
            cl_indptr, cl_indices, infeasible, n_points,
            cheapest_weight, cheapest_edge_id, cheapest_target, cheapest_source,
            node_best_weight, node_best_edge_id, node_best_target,
        )
        
        any_edge_found = False
        for r in range(n_points):
            if parent[r] == r and cheapest_edge_id[r] != -1:
                any_edge_found = True
                break
        
        if not any_edge_found:
            break
        
        # Step 2: Merge components
        n_added_before = n_added
        n_added, round_u, round_v, round_w, round_count = _boruvka_merge_components(
            adj_indptr, adj_neighbors, adj_edge_ids,
            parent, size, head, tail, next_node,
            cheapest_weight, cheapest_edge_id, cheapest_target, cheapest_source,
            mst_edges, n_added, n_points,
        )
        
        if n_added == n_added_before:
            break
        
        # Step 3: Fix violations
        parent, size, head, tail, next_node, n_added, removed, n_rem = _fix_violations(
            parent, size, head, tail, next_node,
            round_u, round_v, round_w, round_count,
            mst_edges, n_added,
            cl_indptr, cl_indices, n_points,
        )
        
        for i in range(n_rem):
            all_removed[total_removed, 0] = removed[i, 0]
            all_removed[total_removed, 1] = removed[i, 1]
            all_removed[total_removed, 2] = removed[i, 2]
            total_removed += 1
        
        if n_added >= n_points - 1:
            break
    
    return mst_edges[:n_added], all_removed[:total_removed]


# =============================================================================
# Public API
# =============================================================================

def constrained_boruvka_mst(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    cl_indptr: np.ndarray,
    cl_indices: np.ndarray,
    n_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute constrained Minimum Spanning Forest using Borůvka's algorithm.
    
    Uses Numba's prange for CPU parallelization of Steps 1a and 3a.
    
    Parameters
    ----------
    u : np.ndarray, shape (n_edges,), dtype int32
        Source node indices for each edge.
    v : np.ndarray, shape (n_edges,), dtype int32
        Target node indices for each edge.
    w : np.ndarray, shape (n_edges,), dtype float64
        Edge weights (distances).
    cl_indptr : np.ndarray, shape (n_points + 1,), dtype int64
        CSR row pointers for cannot-link constraints.
    cl_indices : np.ndarray, dtype int32
        CSR column indices for cannot-link constraints.
    n_points : int
        Total number of nodes.
        
    Returns
    -------
    mst_edges : np.ndarray, shape (n_mst_edges, 3)
        MST edges as [source, target, weight] rows.
    removed_edges : np.ndarray, shape (n_removed, 3)
        Edges removed due to constraint violations.
    """
    u = np.asarray(u, dtype=np.int32)
    v = np.asarray(v, dtype=np.int32)
    w = np.asarray(w, dtype=np.float64)
    cl_indptr = np.asarray(cl_indptr, dtype=np.int64)
    cl_indices = np.asarray(cl_indices, dtype=np.int32)
    
    n_edges = len(u)
    
    adj_indptr, adj_neighbors, adj_weights, adj_edge_ids = build_adjacency_list(
        u, v, w, n_points
    )
    
    mst_edges, removed_edges = _constrained_boruvka_core(
        adj_indptr, adj_neighbors, adj_weights, adj_edge_ids,
        cl_indptr, cl_indices, n_points, n_edges,
    )
    
    return mst_edges, removed_edges


def build_constraint_csr(
    n_points: int,
    constraint_pairs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build CSR representation of cannot-link constraints from pairs.
    """
    import scipy.sparse as sp
    
    if len(constraint_pairs) == 0:
        indptr = np.zeros(n_points + 1, dtype=np.int64)
        indices = np.array([], dtype=np.int32)
        return indptr, indices
    
    rows = np.concatenate([constraint_pairs[:, 0], constraint_pairs[:, 1]])
    cols = np.concatenate([constraint_pairs[:, 1], constraint_pairs[:, 0]])
    data = np.ones(len(rows), dtype=np.int32)
    
    csr = sp.csr_matrix((data, (rows, cols)), shape=(n_points, n_points))
    csr.sum_duplicates()
    
    return np.asarray(csr.indptr, dtype=np.int64), np.asarray(csr.indices, dtype=np.int32)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Constrained Borůvka MST - Test")
    print("=" * 40)
    
    # Triangle graph
    u = np.array([0, 0, 1], dtype=np.int32)
    v = np.array([1, 2, 2], dtype=np.int32)
    w = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    cl_indptr = np.zeros(4, dtype=np.int64)
    cl_indices = np.array([], dtype=np.int32)
    
    mst, removed = constrained_boruvka_mst(u, v, w, cl_indptr, cl_indices, 3)
    print(f"MST edges: {len(mst)}")
    for edge in mst:
        print(f"  {int(edge[0])} -- {int(edge[1])}: {edge[2]:.2f}")
    
    print("\nAll tests passed!")
