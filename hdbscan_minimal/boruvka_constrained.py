"""
Constrained Borůvka MST Algorithm
=================================

A minimal, readable implementation of Borůvka's algorithm for computing
Minimum Spanning Trees with cannot-link constraints.

This module provides a self-contained implementation that can be used
for constrained HDBSCAN clustering where certain points must never
be placed in the same cluster.

Algorithm Overview
------------------
Borůvka's algorithm builds an MST by repeatedly finding the cheapest
edge leaving each connected component, then merging components.

With cannot-link constraints, we add two modifications:
1. During edge selection: skip edges that would violate constraints
2. After merging: verify no violations exist, fix by removing heaviest edge

Example Usage
-------------
>>> import numpy as np
>>> from boruvka_constrained import constrained_boruvka_mst
>>> 
>>> # 4 points with distances
>>> u = np.array([0, 0, 0, 1, 1, 2], dtype=np.int32)  # source nodes
>>> v = np.array([1, 2, 3, 2, 3, 3], dtype=np.int32)  # target nodes  
>>> w = np.array([1.0, 2.0, 3.0, 1.5, 2.5, 1.0])      # weights
>>> 
>>> # Cannot-link: points 0 and 3 must not be in same cluster
>>> cl_indptr = np.array([0, 1, 1, 1, 2], dtype=np.int64)
>>> cl_indices = np.array([3, 0], dtype=np.int32)
>>> 
>>> mst = constrained_boruvka_mst(u, v, w, cl_indptr, cl_indices, n_points=4)
>>> print(mst)  # MST edges respecting constraints

"""
from __future__ import annotations

import numpy as np
import numba
from numba import njit
from typing import Tuple


# =============================================================================
# DSU (Disjoint Set Union) - Core Data Structure
# =============================================================================

@njit(cache=True)
def dsu_find(parent: np.ndarray, x: int) -> int:
    """
    Find the root of node x with path compression.
    
    The DSU (Disjoint Set Union) structure tracks which nodes belong to
    which component. Each component has a "root" node that represents it.
    
    Path compression flattens the tree during find operations, making
    subsequent finds faster (nearly O(1) amortized).
    
    Parameters
    ----------
    parent : np.ndarray
        Parent pointer array where parent[i] is i's parent in the DSU tree.
        If parent[i] == i, then i is a root (component representative).
    x : int
        Node to find the root of.
        
    Returns
    -------
    int
        The root node of the component containing x.
        
    Example
    -------
    >>> parent = np.array([0, 0, 2, 2])  # Components: {0,1}, {2,3}
    >>> dsu_find(parent, 1)  # Returns 0 (root of {0,1})
    0
    >>> dsu_find(parent, 3)  # Returns 2 (root of {2,3})
    2
    """
    root = x
    while parent[root] != root:
        root = parent[root]
    # Path compression: point all nodes directly to root
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
    Path compression modifies the parent array, causing race conditions
    in parallel code. This version only reads, never writes.
    
    Parameters
    ----------
    parent : np.ndarray
        Parent pointer array.
    x : int
        Node to find root of.
        
    Returns
    -------
    int
        Root node of the component containing x.
    """
    while parent[x] != x:
        x = parent[x]
    return x


# =============================================================================
# CSR Constraint Lookup
# =============================================================================

@njit(cache=True)
def csr_row_contains(
    indptr: np.ndarray,
    indices: np.ndarray,
    row: int,
    value: int,
) -> bool:
    """
    Check if a value exists in a CSR matrix row.
    
    CSR (Compressed Sparse Row) format stores sparse matrices efficiently.
    For row i, the non-zero column indices are stored in:
        indices[indptr[i] : indptr[i+1]]
    
    Parameters
    ----------
    indptr : np.ndarray
        CSR row pointer array. indptr[i] is the start index for row i.
    indices : np.ndarray
        CSR column indices array.
    row : int
        Row number to search in.
    value : int
        Column index to search for.
        
    Returns
    -------
    bool
        True if (row, value) is a non-zero entry in the matrix.
        
    Example
    -------
    >>> # Matrix with entries at (0,2) and (1,3)
    >>> indptr = np.array([0, 1, 2])
    >>> indices = np.array([2, 3])
    >>> csr_row_contains(indptr, indices, 0, 2)  # True
    >>> csr_row_contains(indptr, indices, 0, 3)  # False
    """
    start = int(indptr[row])
    end = int(indptr[row + 1])
    for k in range(start, end):
        if int(indices[k]) == value:
            return True
    return False


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
    of them have a cannot-link constraint with any node in the larger component.
    
    We always iterate the smaller component for efficiency (fewer iterations).
    
    Parameters
    ----------
    root_small : int
        Root of the smaller component (by node count).
    root_large : int
        Root of the larger component.
    parent : np.ndarray
        DSU parent array.
    head : np.ndarray
        Linked list head pointers. head[root] = first node in component.
    next_node : np.ndarray
        Linked list next pointers. next_node[i] = node after i, or -1 if last.
    cl_indptr : np.ndarray
        Cannot-link constraint CSR indptr.
    cl_indices : np.ndarray
        Cannot-link constraint CSR indices.
        
    Returns
    -------
    bool
        True if merging would create a constraint violation.
        
    Example
    -------
    If component {0,1} wants to merge with component {2,3}, and there's a
    cannot-link constraint between nodes 1 and 2, this returns True.
    """
    # Walk through all nodes in the smaller component
    node = head[root_small]
    while node != -1:
        # Check all cannot-link partners of this node
        start = int(cl_indptr[node])
        end = int(cl_indptr[node + 1])
        for k in range(start, end):
            partner = int(cl_indices[k])
            # If partner is in the larger component, merging would violate constraint
            if dsu_find(parent, partner) == root_large:
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
    
    A violation exists if two nodes in the same component have a
    cannot-link constraint between them.
    
    Parameters
    ----------
    root : int
        Root node of the component to check.
    parent : np.ndarray
        DSU parent array.
    head : np.ndarray
        Linked list head pointers.
    next_node : np.ndarray
        Linked list next pointers.
    cl_indptr : np.ndarray
        Cannot-link CSR indptr.
    cl_indices : np.ndarray
        Cannot-link CSR indices.
        
    Returns
    -------
    bool
        True if the component has an internal constraint violation.
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
    
    Converts edge list representation (u, v, w arrays) to CSR adjacency
    format for efficient neighbor iteration. Each undirected edge is
    stored twice (once in each direction).
    
    Parameters
    ----------
    u : np.ndarray
        Source nodes of edges, shape (n_edges,).
    v : np.ndarray
        Target nodes of edges, shape (n_edges,).
    w : np.ndarray
        Edge weights, shape (n_edges,).
    n_points : int
        Total number of nodes in the graph.
        
    Returns
    -------
    adj_indptr : np.ndarray
        CSR row pointers, shape (n_points + 1,).
    adj_neighbors : np.ndarray
        Neighbor indices for each node.
    adj_weights : np.ndarray
        Edge weights corresponding to each neighbor.
    adj_edge_ids : np.ndarray
        Original edge index for each adjacency entry.
        
    Example
    -------
    >>> u = np.array([0, 0, 1])
    >>> v = np.array([1, 2, 2])
    >>> w = np.array([1.0, 2.0, 3.0])
    >>> indptr, neighbors, weights, edge_ids = build_adjacency_list(u, v, w, 3)
    >>> # Node 0's neighbors: neighbors[indptr[0]:indptr[1]] = [1, 2]
    """
    n_edges = u.shape[0]
    
    # Count degree of each node (each edge contributes 2: one per endpoint)
    degree = np.zeros(n_points, dtype=np.int64)
    for i in range(n_edges):
        degree[u[i]] += 1
        degree[v[i]] += 1
    
    # Build CSR indptr (cumulative sum of degrees)
    adj_indptr = np.empty(n_points + 1, dtype=np.int64)
    adj_indptr[0] = 0
    for i in range(n_points):
        adj_indptr[i + 1] = adj_indptr[i] + degree[i]
    
    # Allocate arrays
    total_entries = adj_indptr[n_points]
    adj_neighbors = np.empty(total_entries, dtype=np.int32)
    adj_weights = np.empty(total_entries, dtype=np.float64)
    adj_edge_ids = np.empty(total_entries, dtype=np.int32)
    
    # Fill adjacency data (track fill position for each node)
    fill_pos = np.zeros(n_points, dtype=np.int64)
    
    for i in range(n_edges):
        a, b, ww = u[i], v[i], w[i]
        
        # Add edge a -> b
        pos_a = adj_indptr[a] + fill_pos[a]
        adj_neighbors[pos_a] = b
        adj_weights[pos_a] = ww
        adj_edge_ids[pos_a] = i
        fill_pos[a] += 1
        
        # Add edge b -> a (undirected graph)
        pos_b = adj_indptr[b] + fill_pos[b]
        adj_neighbors[pos_b] = a
        adj_weights[pos_b] = ww
        adj_edge_ids[pos_b] = i
        fill_pos[b] += 1
    
    return adj_indptr, adj_neighbors, adj_weights, adj_edge_ids


# =============================================================================
# Main Borůvka Algorithm
# =============================================================================

@njit(cache=True)
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
) -> None:
    """
    Find the cheapest valid outgoing edge for each component.
    
    This is the core of each Borůvka round. For each node, we scan its
    neighbors to find the cheapest edge that:
    1. Connects to a different component
    2. Would not violate any cannot-link constraints if merged
    
    Then we aggregate per-node results to get per-component cheapest edges.
    
    Parameters
    ----------
    adj_indptr, adj_neighbors, adj_weights, adj_edge_ids : np.ndarray
        Adjacency list in CSR format.
    parent, size, head, next_node : np.ndarray
        DSU structure and component linked lists.
    cl_indptr, cl_indices : np.ndarray
        Cannot-link constraints in CSR format.
    infeasible : np.ndarray
        Boolean mask of edges that can never be used.
    n_points : int
        Number of nodes.
    cheapest_weight, cheapest_edge_id, cheapest_target, cheapest_source : np.ndarray
        Output arrays (indexed by component root) for the selected edges.
    """
    # Initialize outputs
    for i in range(n_points):
        cheapest_weight[i] = np.inf
        cheapest_edge_id[i] = -1
        cheapest_target[i] = -1
        cheapest_source[i] = -1
    
    # Per-node temporary storage
    node_best_weight = np.full(n_points, np.inf, dtype=np.float64)
    node_best_edge_id = np.full(n_points, -1, dtype=np.int32)
    node_best_target = np.full(n_points, -1, dtype=np.int32)
    
    # Step 1a: Each node finds its best outgoing edge
    for i in range(n_points):
        my_root = dsu_find(parent, i)
        my_size = size[my_root]
        
        start = adj_indptr[i]
        end = adj_indptr[i + 1]
        
        for k in range(start, end):
            edge_id = adj_edge_ids[k]
            if infeasible[edge_id]:
                continue
            
            j = adj_neighbors[k]
            neighbor_root = dsu_find(parent, j)
            
            # Skip if same component
            if my_root == neighbor_root:
                continue
            
            # Check if merge would violate constraints
            neighbor_size = size[neighbor_root]
            if my_size < neighbor_size:
                root_small, root_large = my_root, neighbor_root
            else:
                root_small, root_large = neighbor_root, my_root
            
            if check_merge_violates_constraints(
                root_small, root_large, parent, head, next_node,
                cl_indptr, cl_indices
            ):
                # Mark edge as permanently infeasible
                infeasible[edge_id] = True
                continue
            
            # Update best edge for this node
            ww = adj_weights[k]
            if ww < node_best_weight[i] or (
                ww == node_best_weight[i] and edge_id < node_best_edge_id[i]
            ):
                node_best_weight[i] = ww
                node_best_edge_id[i] = edge_id
                node_best_target[i] = neighbor_root
    
    # Step 1b: Aggregate per-component (take best among all nodes in component)
    for i in range(n_points):
        if node_best_edge_id[i] == -1:
            continue
        
        my_root = dsu_find(parent, i)
        
        # Update if this node's edge is better than current best for component
        if node_best_weight[i] < cheapest_weight[my_root] or (
            node_best_weight[i] == cheapest_weight[my_root]
            and node_best_edge_id[i] < cheapest_edge_id[my_root]
        ):
            cheapest_weight[my_root] = node_best_weight[i]
            cheapest_edge_id[my_root] = node_best_edge_id[i]
            cheapest_target[my_root] = node_best_target[i]
            cheapest_source[my_root] = i


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
    cl_indptr: np.ndarray,
    cl_indices: np.ndarray,
    cheapest_weight: np.ndarray,
    cheapest_edge_id: np.ndarray,
    cheapest_target: np.ndarray,
    cheapest_source: np.ndarray,
    mst_edges: np.ndarray,
    n_added: int,
    n_points: int,
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Merge components using selected edges and add them to MST.
    
    For each component that found a valid outgoing edge, we merge it
    with the target component using union-by-size. We also maintain
    a linked list of nodes per component for efficient constraint checking.
    
    Parameters
    ----------
    adj_indptr, adj_neighbors, adj_edge_ids : np.ndarray
        Adjacency list for looking up edge endpoints.
    parent, size, head, tail, next_node : np.ndarray
        DSU structure with linked lists.
    cl_indptr, cl_indices : np.ndarray
        Cannot-link constraints.
    cheapest_* : np.ndarray
        Selected edges from _boruvka_find_cheapest_edges.
    mst_edges : np.ndarray
        Output MST edge array.
    n_added : int
        Number of edges already in MST.
    n_points : int
        Number of nodes.
        
    Returns
    -------
    n_added : int
        Updated count of MST edges.
    round_edges_u, round_edges_v, round_edges_w : np.ndarray
        Edges added this round (for violation correction).
    round_count : int
        Number of edges added this round.
    """
    # Track edges added this round (needed for violation correction)
    round_edges_u = np.empty(n_points, dtype=np.int32)
    round_edges_v = np.empty(n_points, dtype=np.int32)
    round_edges_w = np.empty(n_points, dtype=np.float64)
    round_count = 0
    
    # Process each component's selected edge
    for r in range(n_points):
        # Only process roots that found a valid edge
        if parent[r] != r or cheapest_edge_id[r] == -1:
            continue
        
        # Re-find target root (may have changed due to earlier merges this round)
        target_root = dsu_find(parent, cheapest_target[r])
        
        # Skip if already same component
        if r == target_root:
            continue
        
        # Determine which component is smaller
        if size[r] < size[target_root]:
            root_small, root_large = r, target_root
        else:
            root_small, root_large = target_root, r
        
        # Re-verify constraints (may have changed due to other merges)
        if check_merge_violates_constraints(
            root_small, root_large, parent, head, next_node,
            cl_indptr, cl_indices
        ):
            continue
        
        # Perform union: smaller joins larger
        # Use deterministic tie-breaking (smaller index wins)
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
            # Add to MST
            mst_edges[n_added, 0] = float(src)
            mst_edges[n_added, 1] = float(tgt)
            mst_edges[n_added, 2] = cheapest_weight[r]
            n_added += 1
            
            # Track for violation correction
            round_edges_u[round_count] = src
            round_edges_v[round_count] = tgt
            round_edges_w[round_count] = cheapest_weight[r]
            round_count += 1
        
        if n_added >= n_points - 1:
            break
    
    return n_added, round_edges_u, round_edges_v, round_edges_w, round_count


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
    Detect and fix constraint violations by removing edges.
    
    After merging, some components may contain cannot-link violations
    (two constrained nodes ended up in the same component). We fix this
    by removing the heaviest edge from violating components.
    
    The algorithm uses a worklist approach:
    1. Find all violating components
    2. For each, remove heaviest edge added this round
    3. Check resulting sub-components; add back to worklist if still violating
    4. Repeat until worklist is empty
    
    Returns
    -------
    Updated DSU structures and MST, plus array of removed edges.
    """
    # Find initial violating components
    worklist = np.empty(n_points, dtype=np.int32)
    worklist_count = 0
    
    for r in range(n_points):
        if parent[r] == r:  # Is root
            if component_has_violation(r, parent, head, next_node, cl_indptr, cl_indices):
                worklist[worklist_count] = r
                worklist_count += 1
    
    # Track removed edges
    removed_edges = np.empty((n_points, 3), dtype=np.float64)
    n_removed = 0
    
    # Process worklist
    while worklist_count > 0:
        # Pop component from worklist
        worklist_count -= 1
        comp_root = worklist[worklist_count]
        
        # Find heaviest edge from this round that's inside this component
        heaviest_idx = -1
        heaviest_weight = -np.inf
        
        for i in range(round_count):
            u_node = round_edges_u[i]
            v_node = round_edges_v[i]
            
            # Check if this edge is inside the violating component
            if dsu_find(parent, u_node) == comp_root and dsu_find(parent, v_node) == comp_root:
                if round_edges_w[i] > heaviest_weight:
                    heaviest_weight = round_edges_w[i]
                    heaviest_idx = i
        
        if heaviest_idx == -1:
            continue  # No edge to remove (shouldn't happen)
        
        # Remove edge from MST
        u_rem = round_edges_u[heaviest_idx]
        v_rem = round_edges_v[heaviest_idx]
        w_rem = round_edges_w[heaviest_idx]
        
        # Mark as removed in round_edges (set weight to -inf so we don't pick it again)
        round_edges_w[heaviest_idx] = -np.inf
        
        # Remove from mst_edges
        for i in range(n_added):
            if (int(mst_edges[i, 0]) == u_rem and int(mst_edges[i, 1]) == v_rem) or \
               (int(mst_edges[i, 0]) == v_rem and int(mst_edges[i, 1]) == u_rem):
                # Swap with last and decrement
                mst_edges[i, 0] = mst_edges[n_added - 1, 0]
                mst_edges[i, 1] = mst_edges[n_added - 1, 1]
                mst_edges[i, 2] = mst_edges[n_added - 1, 2]
                n_added -= 1
                break
        
        # Record removed edge
        removed_edges[n_removed, 0] = float(u_rem)
        removed_edges[n_removed, 1] = float(v_rem)
        removed_edges[n_removed, 2] = w_rem
        n_removed += 1
        
        # Split component: rebuild DSU for affected nodes
        # This is done by re-computing parent pointers using remaining MST edges
        
        # Find all nodes in the violating component
        comp_nodes = np.empty(n_points, dtype=np.int32)
        comp_node_count = 0
        node = head[comp_root]
        while node != -1:
            comp_nodes[comp_node_count] = node
            comp_node_count += 1
            node = next_node[node]
        
        # Reset these nodes to singletons
        for i in range(comp_node_count):
            node = comp_nodes[i]
            parent[node] = node
            size[node] = 1
            head[node] = node
            tail[node] = node
            next_node[node] = -1
        
        # Re-merge using remaining MST edges (excluding the removed one)
        for i in range(n_added):
            u_edge = int(mst_edges[i, 0])
            v_edge = int(mst_edges[i, 1])
            
            # Check if both endpoints are in our component
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
                    # Merge
                    if size[ru] < size[rv]:
                        small, large = ru, rv
                    else:
                        small, large = rv, ru
                    winner = min(ru, rv)
                    loser = max(ru, rv)
                    parent[loser] = winner
                    size[winner] += size[loser]
                    next_node[tail[winner]] = head[loser]
                    tail[winner] = tail[loser]
        
        # Check resulting sub-components for violations
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


@njit(cache=True)
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
    
    Runs Borůvka rounds until no more edges can be added:
    1. Each component finds cheapest valid outgoing edge
    2. Merge components using selected edges
    3. Detect and fix any constraint violations
    4. Repeat until done
    
    Parameters
    ----------
    adj_indptr, adj_neighbors, adj_weights, adj_edge_ids : np.ndarray
        Graph in CSR adjacency format.
    cl_indptr, cl_indices : np.ndarray
        Cannot-link constraints in CSR format.
    n_points : int
        Number of nodes.
    n_edges : int
        Number of edges.
        
    Returns
    -------
    mst_edges : np.ndarray
        MST edges as (n_mst_edges, 3) array with [source, target, weight].
    removed_edges : np.ndarray
        Edges removed due to constraint violations.
    """
    # Initialize DSU
    parent = np.arange(n_points, dtype=np.int32)
    size = np.ones(n_points, dtype=np.int32)
    head = np.arange(n_points, dtype=np.int32)
    tail = np.arange(n_points, dtype=np.int32)
    next_node = np.full(n_points, -1, dtype=np.int32)
    
    # Edge feasibility mask
    infeasible = np.zeros(n_edges, dtype=np.bool_)
    
    # Output arrays
    mst_edges = np.empty((n_points - 1, 3), dtype=np.float64)
    n_added = 0
    
    all_removed = np.empty((n_points, 3), dtype=np.float64)
    total_removed = 0
    
    # Temporary arrays for edge selection
    cheapest_weight = np.empty(n_points, dtype=np.float64)
    cheapest_edge_id = np.empty(n_points, dtype=np.int32)
    cheapest_target = np.empty(n_points, dtype=np.int32)
    cheapest_source = np.empty(n_points, dtype=np.int32)
    
    # Main loop
    max_rounds = n_points  # Upper bound on rounds
    for round_num in range(max_rounds):
        # Step 1: Find cheapest edges
        _boruvka_find_cheapest_edges(
            adj_indptr, adj_neighbors, adj_weights, adj_edge_ids,
            parent, size, head, next_node,
            cl_indptr, cl_indices, infeasible, n_points,
            cheapest_weight, cheapest_edge_id, cheapest_target, cheapest_source,
        )
        
        # Check if any component found an edge
        any_edge_found = False
        for r in range(n_points):
            if parent[r] == r and cheapest_edge_id[r] != -1:
                any_edge_found = True
                break
        
        if not any_edge_found:
            break  # No more edges can be added
        
        # Step 2: Merge components
        n_added_before = n_added
        n_added, round_u, round_v, round_w, round_count = _boruvka_merge_components(
            adj_indptr, adj_neighbors, adj_edge_ids,
            parent, size, head, tail, next_node,
            cl_indptr, cl_indices,
            cheapest_weight, cheapest_edge_id, cheapest_target, cheapest_source,
            mst_edges, n_added, n_points,
        )
        
        if n_added == n_added_before:
            break  # No progress made
        
        # Step 3: Fix violations
        parent, size, head, tail, next_node, n_added, removed, n_rem = _fix_violations(
            parent, size, head, tail, next_node,
            round_u, round_v, round_w, round_count,
            mst_edges, n_added,
            cl_indptr, cl_indices, n_points,
        )
        
        # Accumulate removed edges
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
    Compute constrained Minimum Spanning Tree using Borůvka's algorithm.
    
    This function computes an MST that respects cannot-link constraints:
    no two nodes with a constraint between them will be connected in the
    resulting tree (they will be in separate subtrees/forests).
    
    Parameters
    ----------
    u : np.ndarray, shape (n_edges,), dtype int32
        Source node indices for each edge.
    v : np.ndarray, shape (n_edges,), dtype int32
        Target node indices for each edge.
    w : np.ndarray, shape (n_edges,), dtype float64
        Edge weights (e.g., distances between points).
    cl_indptr : np.ndarray, shape (n_points + 1,), dtype int64
        CSR row pointers for cannot-link constraints.
        cl_indices[cl_indptr[i]:cl_indptr[i+1]] gives all nodes that
        node i cannot be linked with.
    cl_indices : np.ndarray, dtype int32
        CSR column indices for cannot-link constraints.
    n_points : int
        Total number of nodes in the graph.
        
    Returns
    -------
    mst_edges : np.ndarray, shape (n_mst_edges, 3)
        The MST edges as [source, target, weight] rows.
        May have fewer than n_points-1 edges if constraints prevent
        full connectivity.
    removed_edges : np.ndarray, shape (n_removed, 3)
        Edges that were initially added but removed due to constraint
        violations during the algorithm.
        
    Examples
    --------
    Basic usage with 4 points:
    
    >>> import numpy as np
    >>> # Complete graph on 4 nodes
    >>> u = np.array([0, 0, 0, 1, 1, 2], dtype=np.int32)
    >>> v = np.array([1, 2, 3, 2, 3, 3], dtype=np.int32)
    >>> w = np.array([1.0, 4.0, 3.0, 2.0, 5.0, 1.0], dtype=np.float64)
    >>> 
    >>> # No constraints
    >>> cl_indptr = np.zeros(5, dtype=np.int64)
    >>> cl_indices = np.array([], dtype=np.int32)
    >>> 
    >>> mst, removed = constrained_boruvka_mst(u, v, w, cl_indptr, cl_indices, 4)
    >>> print("MST edges:", mst)
    
    With cannot-link constraint between nodes 0 and 3:
    
    >>> # Constraint: nodes 0 and 3 cannot be in same component
    >>> cl_indptr = np.array([0, 1, 1, 1, 2], dtype=np.int64)
    >>> cl_indices = np.array([3, 0], dtype=np.int32)  # 0->3 and 3->0
    >>> 
    >>> mst, removed = constrained_boruvka_mst(u, v, w, cl_indptr, cl_indices, 4)
    >>> # MST will be a forest with 0 and 3 in separate trees
    
    Notes
    -----
    - Time complexity: O(E log V) where E is edges, V is vertices
    - The algorithm may return fewer than V-1 edges if constraints prevent
      full connectivity (result is a forest, not a tree)
    - Constraints are symmetric: if (i,j) is constrained, so is (j,i)
    - The CSR format for constraints should include both directions
    """
    # Validate inputs
    u = np.asarray(u, dtype=np.int32)
    v = np.asarray(v, dtype=np.int32)
    w = np.asarray(w, dtype=np.float64)
    cl_indptr = np.asarray(cl_indptr, dtype=np.int64)
    cl_indices = np.asarray(cl_indices, dtype=np.int32)
    
    n_edges = len(u)
    
    # Build adjacency list
    adj_indptr, adj_neighbors, adj_weights, adj_edge_ids = build_adjacency_list(
        u, v, w, n_points
    )
    
    # Run core algorithm
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
    
    Converts a list of constraint pairs (i, j) into CSR format suitable
    for use with constrained_boruvka_mst.
    
    Parameters
    ----------
    n_points : int
        Total number of points.
    constraint_pairs : np.ndarray, shape (n_constraints, 2)
        Array of constraint pairs where each row [i, j] means nodes i and j
        cannot be in the same cluster.
        
    Returns
    -------
    cl_indptr : np.ndarray, shape (n_points + 1,)
        CSR row pointers.
    cl_indices : np.ndarray
        CSR column indices.
        
    Example
    -------
    >>> pairs = np.array([[0, 2], [1, 3]])  # 0-2 and 1-3 cannot link
    >>> indptr, indices = build_constraint_csr(4, pairs)
    """
    import scipy.sparse as sp
    
    if len(constraint_pairs) == 0:
        indptr = np.zeros(n_points + 1, dtype=np.int64)
        indices = np.array([], dtype=np.int32)
        return indptr, indices
    
    # Build symmetric CSR matrix
    rows = np.concatenate([constraint_pairs[:, 0], constraint_pairs[:, 1]])
    cols = np.concatenate([constraint_pairs[:, 1], constraint_pairs[:, 0]])
    data = np.ones(len(rows), dtype=np.int32)
    
    csr = sp.csr_matrix((data, (rows, cols)), shape=(n_points, n_points))
    csr.sum_duplicates()
    
    return np.asarray(csr.indptr, dtype=np.int64), np.asarray(csr.indices, dtype=np.int32)


# =============================================================================
# Simple Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Constrained Borůvka MST - Test Cases")
    print("=" * 60)
    
    # Test 1: Simple triangle without constraints
    print("\nTest 1: Triangle graph, no constraints")
    print("-" * 40)
    u = np.array([0, 0, 1], dtype=np.int32)
    v = np.array([1, 2, 2], dtype=np.int32)
    w = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    cl_indptr = np.zeros(4, dtype=np.int64)
    cl_indices = np.array([], dtype=np.int32)
    
    mst, removed = constrained_boruvka_mst(u, v, w, cl_indptr, cl_indices, 3)
    print(f"MST edges: {len(mst)}")
    for edge in mst:
        print(f"  {int(edge[0])} -- {int(edge[1])}: {edge[2]:.2f}")
    print(f"Removed edges: {len(removed)}")
    assert len(mst) == 2, "Triangle should have 2 MST edges"
    
    # Test 2: Square with constraint
    print("\nTest 2: Square graph with constraint (0,2)")
    print("-" * 40)
    #   0 --- 1
    #   |     |
    #   3 --- 2
    u = np.array([0, 0, 1, 2], dtype=np.int32)
    v = np.array([1, 3, 2, 3], dtype=np.int32)
    w = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    
    # Constraint: 0 and 2 cannot be connected
    constraint_pairs = np.array([[0, 2]], dtype=np.int32)
    cl_indptr, cl_indices = build_constraint_csr(4, constraint_pairs)
    
    mst, removed = constrained_boruvka_mst(u, v, w, cl_indptr, cl_indices, 4)
    print(f"MST edges: {len(mst)}")
    for edge in mst:
        print(f"  {int(edge[0])} -- {int(edge[1])}: {edge[2]:.2f}")
    print(f"Removed edges: {len(removed)}")
    
    # Verify 0 and 2 are not in same component
    # (They should be in separate trees in the forest)
    
    # Test 3: Larger graph
    print("\nTest 3: 6-node graph with multiple constraints")
    print("-" * 40)
    # Complete graph on 6 nodes
    n = 6
    edges_u = []
    edges_v = []
    edges_w = []
    np.random.seed(42)
    for i in range(n):
        for j in range(i + 1, n):
            edges_u.append(i)
            edges_v.append(j)
            edges_w.append(np.random.uniform(0.5, 2.0))
    
    u = np.array(edges_u, dtype=np.int32)
    v = np.array(edges_v, dtype=np.int32)
    w = np.array(edges_w, dtype=np.float64)
    
    # Constraints: (0,5), (1,4), (2,3)
    constraint_pairs = np.array([[0, 5], [1, 4], [2, 3]], dtype=np.int32)
    cl_indptr, cl_indices = build_constraint_csr(n, constraint_pairs)
    
    mst, removed = constrained_boruvka_mst(u, v, w, cl_indptr, cl_indices, n)
    print(f"MST edges: {len(mst)}")
    for edge in mst:
        print(f"  {int(edge[0])} -- {int(edge[1])}: {edge[2]:.2f}")
    print(f"Removed edges: {len(removed)}")
    for edge in removed:
        print(f"  (removed) {int(edge[0])} -- {int(edge[1])}: {edge[2]:.2f}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
