"""
Minimal HDBSCAN with Cannot-Link Constraints
=============================================

This module provides a minimal, self-contained implementation of constrained HDBSCAN
with parallel Borůvka MST and conditional backend selection (CPU/CUDA).

Key Features
------------
- Cannot-link constraints via dense/sparse matrix
- Three MST methods: boruvka, parallel_boruvka, kruskal
- Conditional parallel backend: auto, cuda, cpu, sequential
- Violation detection and correction for parallel algorithm

MST Algorithm Options
---------------------
1. **kruskal**: Classic Kruskal's algorithm with constraint checking
   - O(E log E) time complexity
   - Fully sequential, good for small/sparse graphs
   
2. **boruvka**: Sequential Borůvka's algorithm with constraints
   - O(E log V) time complexity
   - Better cache locality than Kruskal for dense graphs
   
3. **parallel_boruvka**: Parallel Borůvka with backend selection
   - Supports CPU parallel, GPU (CUDA), or sequential execution
   - Best for large dense graphs where parallelism pays off

Parallel Borůvka Algorithm - Step-by-Step
-----------------------------------------
Each round of the algorithm processes these steps:

**Step 1a: Per-Node Minimum Edge Search**
    - Task: Each node finds its cheapest valid outgoing edge
    - Validity: Edge must connect to different component AND not violate cannot-link
    - Parallelization: FULLY PARALLEL (CPU prange / GPU threads)
    - Why parallel: Each node's search is completely independent
    - Output: node_best_neighbor[i], node_best_weight[i] for each node i

**Step 1b: Per-Component Aggregation**
    - Task: For each component (DSU tree), find the globally cheapest edge
    - Challenge: Multiple nodes in same component must agree on one best edge
    - Parallelization:
        * GPU (CUDA): PARALLEL using cuda.atomic.min for thread-safe updates
        * CPU: SEQUENTIAL - Numba lacks atomic primitives, races corrupt results
    - Output: cheapest_weight[root], cheapest_neighbor[root] per component

**Step 2: DSU Merge Operations**
    - Task: Add selected edges to MST and merge components via union-find
    - Parallelization: ALWAYS SEQUENTIAL on both CPU and GPU
    - Output: Updated DSU parent/rank arrays, edges added to MST

**Step 3: Violation Detection & Correction (Component-Based)**
    - Task: Remove edges that would create cannot-link violations
    - When: Only after Step 2 completes (need final component assignments)
    - Algorithm:
        1. Build worklist of components that have violations (PARALLEL per component)
        2. Pop a violating component from worklist
        3. Remove heaviest edge → splits into 2 sub-components
        4. Check each sub-component for violations:
           - If still violating → add back to worklist
           - If clean → done with that sub-component
        5. Repeat until worklist empty
    - Parallelization:
        * Step 3a (Detection): PARALLEL over components - each component checked independently
        * Step 3b (Correction): SEQUENTIAL - removing an edge changes the graph structure
    - Why needed: Parallel edge selection in Step 1 can't see other threads'
                  choices, so two edges might collectively create a violation

Worked Example: Why Each Step Is Parallel or Sequential
-------------------------------------------------------
Consider a graph with 6 nodes and the following edges (sorted by weight):

    Nodes: 0, 1, 2, 3, 4, 5
    Edges: (0,1,1.0), (2,3,1.5), (1,2,2.0), (3,4,2.5), (4,5,3.0), (0,5,3.5)
    Cannot-link constraint: (0, 5) -- nodes 0 and 5 must NOT be in same cluster
    
    Initial state: Each node is its own component
    Components: {0}, {1}, {2}, {3}, {4}, {5}

**STEP 1a: Per-Node Search (PARALLEL IS SAFE)**

    Each node independently finds its cheapest outgoing edge:
    
    Thread 0: Node 0 searches → finds edge (0,1,1.0) to component {1}
    Thread 1: Node 1 searches → finds edge (0,1,1.0) to component {0}
    Thread 2: Node 2 searches → finds edge (2,3,1.5) to component {3}
    Thread 3: Node 3 searches → finds edge (2,3,1.5) to component {2}
    Thread 4: Node 4 searches → finds edge (3,4,2.5) to component {3}
    Thread 5: Node 5 searches → finds edge (4,5,3.0) to component {4}
    
    WHY PARALLEL IS SAFE:
    - Each thread writes to its OWN array slot: node_best[i]
    - No two threads write to the same memory location
    - Read-only access to shared data (edges, components)
    
    Result: node_best = [(1,1.0), (0,1.0), (3,1.5), (2,1.5), (3,2.5), (4,3.0)]

**STEP 1b: Per-Component Aggregation (CPU: SEQUENTIAL REQUIRED)**

    Now we must pick ONE best edge per component. Initially each node is its
    own component, so each component's root equals itself.
    
    SEQUENTIAL EXECUTION (CORRECT):
        Process node 0: root=0, cheapest[0] = (neighbor=1, weight=1.0)
        Process node 1: root=1, cheapest[1] = (neighbor=0, weight=1.0)
        ... each node is its own root, so no conflicts yet
    
    BUT AFTER SOME MERGES, suppose components are: {0,1,2}, {3,4}, {5}
    And nodes 0,1,2 all have root=0. Now consider:
        Node 0 found edge (0→3, weight=2.0)
        Node 1 found edge (1→3, weight=1.8)  ← better!
        Node 2 found edge (2→5, weight=4.0)
    
    PARALLEL EXECUTION (RACE CONDITION - WRONG):
        Time T1: Thread 0 reads cheapest[0] = inf
        Time T1: Thread 1 reads cheapest[0] = inf
        Time T2: Thread 0 writes cheapest[0] = 2.0 (from node 0's edge)
        Time T3: Thread 1 writes cheapest[0] = 1.8 (from node 1's edge)
        
        LUCKY ORDER: Final cheapest[0] = 1.8 ✓
        
        BUT if timing differs:
        Time T2: Thread 1 writes cheapest[0] = 1.8
        Time T3: Thread 0 writes cheapest[0] = 2.0  ← OVERWRITES BETTER VALUE!
        
        UNLUCKY ORDER: Final cheapest[0] = 2.0 ✗ (wrong edge selected!)
    
    SEQUENTIAL EXECUTION (CORRECT):
        Process node 0: cheapest[0] = min(inf, 2.0) = 2.0
        Process node 1: cheapest[0] = min(2.0, 1.8) = 1.8  ← always picks better
        Process node 2: cheapest[0] = min(1.8, 4.0) = 1.8  ← keeps best
        
        Final cheapest[0] = 1.8 ✓ (correct edge selected!)
    
    GPU SOLUTION: cuda.atomic.min performs read-compare-write atomically,
    so parallel execution is safe. CPU Numba has no such primitive.

**STEP 2: DSU Merge (ALWAYS SEQUENTIAL)**

    Selected edges from Step 1: (0,1,1.0), (2,3,1.5), (4,5,3.0)
    
    SEQUENTIAL EXECUTION (CORRECT):
        Merge 0-1: parent[1]=0, components: {0,1}, {2}, {3}, {4}, {5}
        Merge 2-3: parent[3]=2, components: {0,1}, {2,3}, {4}, {5}
        Merge 4-5: parent[5]=4, components: {0,1}, {2,3}, {4,5}
        
        DSU state is consistent. 3 edges added to MST.
    
    PARALLEL EXECUTION (CORRUPTION - WRONG):
        Consider if we also try to merge 1-2 and 3-4 in parallel:
        
        Thread A: Processing edge (1,2)
            find(1) → follows parent[1]=0, returns 0
            find(2) → returns 2
            About to set parent[2] = 0...
            
        Thread B: Processing edge (2,3) simultaneously
            find(2) → returns 2 (not yet updated!)
            find(3) → returns 3
            Sets parent[3] = 2
            
        Thread A: Sets parent[2] = 0
        
        PROBLEM 1 - Lost update:
            Node 3 points to 2, but 2 now points to 0
            Path compression in Thread B didn't see the merge!
            
        PROBLEM 2 - Cycle detection broken:
            Thread C: Processing edge (0,3)
            find(0) → returns 0
            find(3) → follows 3→2→0, returns 0
            Same component! Should skip, but timing could cause:
            find(3) starts before Thread A completes → returns 2 (wrong root)
            We add edge (0,3) even though it creates a CYCLE!
        
        PROBLEM 3 - Size/rank corruption:
            Both threads read size[0]=1, size[2]=1
            Both compute new_size = 2
            Both write size[root] = 2
            Actual merged component has 4 nodes, but size says 2!

**STEP 3: Violation Detection & Correction**

    After Step 2, suppose components are: {0,1,2,3,4,5} (all merged)
    But we have cannot-link(0,5)!
    
    DETECTION (PARALLEL IS SAFE):
        Each constraint pair (i,j) is checked independently:
        Thread for (0,5): find(0)=0, find(5)=0 → SAME! Violation found.
        
        WHY PARALLEL IS SAFE:
        - Only READS from DSU (no writes)
        - Each thread checks ONE constraint pair
        - Writes to separate violation_list slots
    
    CORRECTION (SEQUENTIAL REQUIRED):
        Suppose two violations exist and we must remove edges to fix them:
        Violation 1: (0,5) in same component due to edge (4,5,3.0)
        Violation 2: (1,4) in same component due to edge (3,4,2.5)
        
        SEQUENTIAL EXECUTION (CORRECT):
            Fix violation 1: Remove heaviest edge in path 0↔5
                Path: 0-1-2-3-4-5, heaviest = (4,5,3.0)
                Remove (4,5,3.0) → now {0,1,2,3,4} and {5}
                Check: find(0)≠find(5) ✓ Fixed!
            
            Fix violation 2: Check if still violated
                find(1)=0, find(4)=0 → still same component
                Remove heaviest in path 1↔4 = (3,4,2.5)
                Remove (3,4,2.5) → now {0,1,2,3}, {4}, {5}
        
        PARALLEL EXECUTION (WRONG DECISIONS):
            Thread A: Fixing (0,5) - finds heaviest edge in component
            Thread B: Fixing (1,4) - finds heaviest edge in component
            
            Both see the SAME component {0,1,2,3,4,5}
            Both might identify (4,5,3.0) as heaviest
            Both try to remove it → redundant work, or worse:
            
            Thread A removes (4,5,3.0), creating {0,1,2,3,4} and {5}
            Thread B (not seeing update) removes (3,4,2.5) from original view
            Now we've removed TOO MANY edges!
            
            Or Thread B finds (4,5,3.0) already removed, gets confused
            about the current graph structure.
        
        WHY SEQUENTIAL IS REQUIRED:
        - Removing edge E1 changes which edges exist for violation 2
        - Must re-check if violation still exists after each fix
        - Graph structure changes with each removal

Detailed Example: Why Step 2 AND Step 3b Must Be Sequential
-----------------------------------------------------------
Consider this scenario where parallel execution of Step 2 OR Step 3b fails:

    Graph: 8 nodes arranged as two "diamonds" connected by a bridge
    
         1           5
        /|\\         /|\\
       / | \\       / | \\
      0  |  2-----4  |  6
       \\ | /       \\ | /
        \\|/         \\|/
         3           7
    
    Edges (sorted by weight):
      (0,1,1), (0,3,1), (1,2,2), (2,3,2), (1,3,2.5),  ← left diamond
      (4,5,1), (4,7,1), (5,6,2), (6,7,2), (5,7,2.5),  ← right diamond
      (2,4,3)  ← bridge connecting the diamonds
    
    Cannot-link constraints: (0,2), (4,6)
    
    Initial components: {0},{1},{2},{3},{4},{5},{6},{7}

**ROUND 1 - Step 1: Edge Selection**
    Each component picks cheapest outgoing edge:
    {0}→(0,1,1), {1}→(0,1,1), {2}→(1,2,2), {3}→(0,3,1)
    {4}→(4,5,1), {5}→(4,5,1), {6}→(5,6,2), {7}→(4,7,1)
    
    Selected: (0,1), (0,3), (4,5), (4,7)

**ROUND 1 - Step 2: Merging (WHY SEQUENTIAL IS REQUIRED)**

    SEQUENTIAL (CORRECT):
        Merge (0,1): {0,1}, {2}, {3}, {4}, {5}, {6}, {7}
        Merge (0,3): {0,1,3}, {2}, {4}, {5}, {6}, {7}
        Merge (4,5): {0,1,3}, {2}, {4,5}, {6}, {7}
        Merge (4,7): {0,1,3}, {2}, {4,5,7}, {6}
        
        DSU is consistent. 4 edges added.
    
    PARALLEL (WRONG) - Processing (0,1), (0,3), (1,3) simultaneously:
        Thread A: Merge (0,1)
            find(0)=0, find(1)=1, different → merge
            parent[1]=0, size[0]=2
            
        Thread B: Merge (0,3) at same time
            find(0)=0, find(3)=3, different → merge
            parent[3]=0, size[0]=2  ← RACE! Should be 3!
            
        Thread C: Merge (1,3) at same time
            find(1)=? → might return 1 (not yet updated) or 0
            find(3)=? → might return 3 (not yet updated) or 0
            
            If both return old values: adds edge (1,3) to MST
            But 1 and 3 are already in same component via 0!
            CYCLE CREATED IN MST!

**ROUND 2 - More complex scenario showing Step 3b parallel failure**

    After some rounds, suppose we have:
    Components: {0,1,2,3} and {4,5,6,7}
    MST edges in left: (0,1), (0,3), (1,2)  
    MST edges in right: (4,5), (4,7), (5,6)
    
    Now bridge edge (2,4,3) is selected, merging everything:
    Component: {0,1,2,3,4,5,6,7}
    
    Violations detected:
      - (0,2): both in same component
      - (4,6): both in same component
    
    PARALLEL Step 3b (WRONG):
        Thread A handles violation (0,2):
            Component has edges: (0,1), (0,3), (1,2), (4,5), (4,7), (5,6), (2,4)
            Heaviest edge containing path 0↔2: could be (2,4,3) or (1,2,2)
            Thread A picks (2,4,3), removes it
            Now: {0,1,2,3} and {4,5,6,7}
            
        Thread B handles violation (4,6) AT THE SAME TIME:
            Thread B still sees OLD graph with edge (2,4,3) present!
            Thread B computes path 4↔6 in the FULL component
            Thread B picks (2,4,3) as heaviest (or (5,6,2))
            
            CASE 1: Both remove (2,4,3)
                Redundant work, but result might be okay
                
            CASE 2: Thread A removes (2,4,3), Thread B removes (5,6,2)
                Thread B's decision was based on old graph state
                After Thread A's removal, violation (4,6) is in component {4,5,6,7}
                The correct edge to remove would be (5,6,2) anyway... 
                BUT Thread B might have computed wrong heaviest!
                
            CASE 3: Interleaved DSU updates
                Thread A: parent[4] changes during Thread B's find()
                Thread B gets inconsistent view of components
                Thread B removes wrong edge entirely
    
    SEQUENTIAL Step 3b (CORRECT):
        Worklist: [{0,1,2,3,4,5,6,7}]  ← one big violating component
        
        Pop component, check violations:
            (0,2): find(0)=0, find(2)=0 → YES
            (4,6): find(4)=0, find(6)=0 → YES
        Has violations, find heaviest round edge: (2,4,3)
        Remove (2,4,3)
        
        Now two sub-components: {0,1,2,3} and {4,5,6,7}
        
        Check {0,1,2,3}: has (0,2) violation?
            find(0)=0, find(2)=0 → YES, still violating
            Add to worklist
            
        Check {4,5,6,7}: has (4,6) violation?
            find(4)=4, find(6)=4 → YES, still violating
            Add to worklist
        
        Worklist: [{0,1,2,3}, {4,5,6,7}]
        
        Pop {0,1,2,3}:
            Heaviest round edge in this component: (1,2,2)
            Remove (1,2,2)
            Sub-components: {0,1,3} and {2}
            Check {0,1,3}: (0,2) violation? find(2)=2 ≠ find(0)=0 → NO ✓
            Check {2}: no constraints → NO ✓
            Both clean, don't add to worklist
        
        Pop {4,5,6,7}:
            Heaviest round edge: (5,6,2)
            Remove (5,6,2)
            Sub-components: {4,5,7} and {6}
            Check {4,5,7}: (4,6) violation? find(6)=6 ≠ find(4)=4 → NO ✓
            Check {6}: no constraints → NO ✓
            Both clean
        
        Worklist empty → done!
        
        Final valid MST: (0,1), (0,3), (4,5), (4,7)
        Removed edges: (2,4,3), (1,2,2), (5,6,2)

Backend Selection Logic
-----------------------
The `parallel_backend` parameter controls execution:

- **"auto"**: Detects CUDA availability at runtime
    * If CUDA available and working → uses "cuda"
    * Otherwise → falls back to "cpu"
    
- **"cuda"**: Forces GPU execution
    * Step 1a: GPU parallel (one thread per node)
    * Step 1b: GPU parallel with atomics
    * Step 2: Sequential on GPU (single thread block)
    * Requires: numba.cuda, compatible NVIDIA GPU
    
- **"cpu"**: Forces CPU parallel execution
    * Step 1a: CPU parallel (numba.prange across cores)
    * Step 1b: Sequential (no CPU atomics in Numba)
    * Step 2: Sequential
    * Best for: Multi-core CPUs, medium-large graphs
    
- **"sequential"**: Forces fully sequential execution
    * All steps run sequentially on single CPU core
    * Best for: Debugging, small graphs, reproducibility

Performance Characteristics
---------------------------
- Small graphs (<1000 nodes): kruskal or sequential often fastest (less overhead)
- Medium graphs (1K-100K nodes): cpu backend with parallel_boruvka
- Large graphs (>100K nodes): cuda backend if GPU available
- Very sparse graphs: kruskal (fewer edges to sort)
- Dense graphs: parallel_boruvka (more parallelism opportunity)

Cannot-Link Constraint Handling
-------------------------------
- Constraints stored as CSR sparse matrix for O(1) lookup
- During edge selection: skip edges connecting constrained pairs
- During merging: edges that would merge constrained nodes into same
  component are detected and the heaviest is removed
- Final result: guaranteed no cluster contains cannot-link pairs
"""
from __future__ import annotations

from typing import Callable, Iterable, Literal, Optional, Tuple, Union

import numba
from numba import prange
import numpy as np
import scipy.sparse as sp

from fast_hdbscan.disjoint_set import ds_rank_create, ds_find, ds_union_by_rank
from fast_hdbscan.hdbscan import clusters_from_spanning_tree

Number = Union[int, float]
CSR = sp.csr_matrix

# Type alias for parallel backend selection
ParallelBackend = Literal["auto", "cuda", "cpu", "sequential"]


# ==============================================================================
# CUDA Detection
# ==============================================================================

_CUDA_AVAILABLE: Optional[bool] = None


def _check_cuda_available() -> bool:
    """
    Check if CUDA is available for GPU acceleration.
    Result is cached at module level to avoid repeated checks.
    """
    global _CUDA_AVAILABLE
    if _CUDA_AVAILABLE is None:
        try:
            from numba import cuda
            _CUDA_AVAILABLE = cuda.is_available()
        except Exception:
            _CUDA_AVAILABLE = False
    return _CUDA_AVAILABLE


# ==============================================================================
# DSU (Disjoint Set Union) Functions
# ==============================================================================

@numba.njit(cache=True)
def _dsu_find_numba(parent: np.ndarray, x: int) -> int:
    """DSU find with path compression."""
    root = x
    while parent[root] != root:
        root = parent[root]
    while parent[x] != x:
        px = parent[x]
        parent[x] = root
        x = px
    return root


@numba.njit(cache=True)
def _dsu_find_readonly_numba(parent: np.ndarray, x: int) -> int:
    """
    DSU find WITHOUT path compression (safe for parallel reads).
    
    Use this in parallel phases where multiple threads may call find()
    on the same data structure simultaneously.
    """
    while parent[x] != x:
        x = parent[x]
    return x


# ==============================================================================
# CSR Utility Functions
# ==============================================================================

@numba.njit(cache=True)
def _csr_row_contains_numba(
    indptr: np.ndarray,
    indices: np.ndarray,
    row: int,
    value: int,
) -> bool:
    """Linear membership test in a CSR row."""
    start = int(indptr[row])
    end = int(indptr[row + 1])
    for k in range(start, end):
        if int(indices[k]) == value:
            return True
    return False


@numba.njit(cache=True)
def _component_has_conflict_with_root_numba(
    root_small: int,
    root_large: int,
    parent: np.ndarray,
    head: np.ndarray,
    next_node: np.ndarray,
    cannot_link_indptr: np.ndarray,
    cannot_link_indices: np.ndarray,
) -> bool:
    """
    Check whether merging component(root_small) into component(root_large)
    would violate any cannot-link constraints.
    """
    node = head[root_small]
    while node != -1:
        start = int(cannot_link_indptr[node])
        end = int(cannot_link_indptr[node + 1])
        for k in range(start, end):
            j = int(cannot_link_indices[k])
            if _dsu_find_numba(parent, j) == root_large:
                return True
        node = next_node[node]
    return False


# ==============================================================================
# Adjacency List Construction
# ==============================================================================

@numba.njit(cache=True)
def _build_adjacency_list_numba(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    n_points: int,
) -> tuple:
    """
    Build CSR-style adjacency structure from edge arrays.
    Each undirected edge is stored twice (u->v and v->u).
    """
    n_edges = u.shape[0]

    degree = np.zeros(n_points, dtype=np.int64)
    for i in range(n_edges):
        degree[u[i]] += 1
        degree[v[i]] += 1

    adj_indptr = np.empty(n_points + 1, dtype=np.int64)
    adj_indptr[0] = 0
    for i in range(n_points):
        adj_indptr[i + 1] = adj_indptr[i] + degree[i]

    total_entries = adj_indptr[n_points]
    adj_neighbors = np.empty(total_entries, dtype=np.int32)
    adj_weights = np.empty(total_entries, dtype=np.float64)
    adj_edge_ids = np.empty(total_entries, dtype=np.int32)

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


# ==============================================================================
# Constrained Kruskal MST
# ==============================================================================

@numba.njit(cache=True)
def _constrained_kruskal_mst_csr_strict_sorted_numba(
    u_sorted: np.ndarray,
    v_sorted: np.ndarray,
    w_sorted: np.ndarray,
    cannot_link_indptr: np.ndarray,
    cannot_link_indices: np.ndarray,
    n_points: int,
) -> np.ndarray:
    """
    Strict constrained Kruskal MST (edges must be pre-sorted by weight).
    """
    parent = np.empty(n_points, dtype=np.int32)
    size = np.empty(n_points, dtype=np.int32)
    head = np.empty(n_points, dtype=np.int32)
    tail = np.empty(n_points, dtype=np.int32)
    next_node = np.empty(n_points, dtype=np.int32)

    for i in range(n_points):
        parent[i] = i
        size[i] = 1
        head[i] = i
        tail[i] = i
        next_node[i] = -1

    mst_edges = np.empty((n_points - 1, 3), dtype=np.float64)
    n_added = 0

    for idx_edge in range(u_sorted.shape[0]):
        a = int(u_sorted[idx_edge])
        b = int(v_sorted[idx_edge])
        ww = float(w_sorted[idx_edge])

        # Reject directly forbidden edge
        if _csr_row_contains_numba(cannot_link_indptr, cannot_link_indices, a, b) or \
           _csr_row_contains_numba(cannot_link_indptr, cannot_link_indices, b, a):
            continue

        ra = _dsu_find_numba(parent, a)
        rb = _dsu_find_numba(parent, b)
        if ra == rb:
            continue

        if size[ra] < size[rb]:
            root_small, root_large = ra, rb
        else:
            root_small, root_large = rb, ra

        if _component_has_conflict_with_root_numba(
            root_small, root_large, parent, head, next_node,
            cannot_link_indptr, cannot_link_indices
        ):
            continue

        # Union
        parent[root_small] = root_large
        size[root_large] += size[root_small]
        next_node[tail[root_large]] = head[root_small]
        tail[root_large] = tail[root_small]

        mst_edges[n_added, 0] = float(a)
        mst_edges[n_added, 1] = float(b)
        mst_edges[n_added, 2] = float(ww)
        n_added += 1

        if n_added == n_points - 1:
            break

    return mst_edges[:n_added]


# ==============================================================================
# Constrained Borůvka MST (Sequential)
# ==============================================================================

@numba.njit(cache=True)
def _constrained_boruvka_mst_csr_strict_numba(
    adj_indptr: np.ndarray,
    adj_neighbors: np.ndarray,
    adj_weights: np.ndarray,
    adj_edge_ids: np.ndarray,
    cannot_link_indptr: np.ndarray,
    cannot_link_indices: np.ndarray,
    n_points: int,
    n_edges: int,
) -> np.ndarray:
    """Sequential constrained Borůvka MST algorithm."""
    parent = np.empty(n_points, dtype=np.int32)
    size = np.empty(n_points, dtype=np.int32)
    head = np.empty(n_points, dtype=np.int32)
    tail = np.empty(n_points, dtype=np.int32)
    next_node = np.empty(n_points, dtype=np.int32)

    for i in range(n_points):
        parent[i] = i
        size[i] = 1
        head[i] = i
        tail[i] = i
        next_node[i] = -1

    infeasible = np.zeros(n_edges, dtype=np.bool_)
    cheapest_edge_id = np.empty(n_points, dtype=np.int32)
    cheapest_weight = np.empty(n_points, dtype=np.float64)
    cheapest_target = np.empty(n_points, dtype=np.int32)
    cheapest_source = np.empty(n_points, dtype=np.int32)

    mst_edges = np.empty((n_points - 1, 3), dtype=np.float64)
    n_added = 0

    while True:
        for i in range(n_points):
            cheapest_edge_id[i] = -1
            cheapest_weight[i] = np.inf
            cheapest_target[i] = -1
            cheapest_source[i] = -1

        # Phase 1: Find cheapest valid edge per component
        for i in range(n_points):
            my_root = _dsu_find_numba(parent, i)
            start = adj_indptr[i]
            end = adj_indptr[i + 1]
            
            for k in range(start, end):
                edge_id = adj_edge_ids[k]
                if infeasible[edge_id]:
                    continue

                j = adj_neighbors[k]
                neighbor_root = _dsu_find_numba(parent, j)
                if my_root == neighbor_root:
                    continue

                if size[my_root] < size[neighbor_root]:
                    root_small, root_large = my_root, neighbor_root
                else:
                    root_small, root_large = neighbor_root, my_root

                if _component_has_conflict_with_root_numba(
                    root_small, root_large, parent, head, next_node,
                    cannot_link_indptr, cannot_link_indices
                ):
                    infeasible[edge_id] = True
                    continue

                ww = adj_weights[k]
                if ww < cheapest_weight[my_root] or (
                    ww == cheapest_weight[my_root] and edge_id < cheapest_edge_id[my_root]
                ):
                    cheapest_weight[my_root] = ww
                    cheapest_edge_id[my_root] = edge_id
                    cheapest_target[my_root] = neighbor_root
                    cheapest_source[my_root] = i

        # Phase 2: Merge components
        n_merges = 0
        for r in range(n_points):
            if parent[r] != r or cheapest_edge_id[r] == -1:
                continue

            target_root = _dsu_find_numba(parent, cheapest_target[r])
            if r == target_root:
                continue

            if size[r] < size[target_root]:
                root_small, root_large = r, target_root
            else:
                root_small, root_large = target_root, r

            if _component_has_conflict_with_root_numba(
                root_small, root_large, parent, head, next_node,
                cannot_link_indptr, cannot_link_indices
            ):
                continue

            winner = min(r, target_root)
            loser = max(r, target_root)

            parent[loser] = winner
            size[winner] += size[loser]
            next_node[tail[winner]] = head[loser]
            tail[winner] = tail[loser]

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
                n_merges += 1

            if n_added == n_points - 1:
                break

        if n_merges == 0 or n_added == n_points - 1:
            break

    return mst_edges[:n_added]


# ==============================================================================
# Parallel Edge Selection Functions
# ==============================================================================

@numba.njit(cache=True)
def _parallel_edge_selection_numba(
    adj_indptr: np.ndarray,
    adj_neighbors: np.ndarray,
    adj_weights: np.ndarray,
    adj_edge_ids: np.ndarray,
    parent: np.ndarray,
    size: np.ndarray,
    head: np.ndarray,
    next_node: np.ndarray,
    cannot_link_indptr: np.ndarray,
    cannot_link_indices: np.ndarray,
    infeasible: np.ndarray,
    n_points: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sequential edge selection for constrained Borůvka."""
    cheapest_edge_id = np.full(n_points, -1, dtype=np.int32)
    cheapest_weight = np.full(n_points, np.inf, dtype=np.float64)
    cheapest_target = np.full(n_points, -1, dtype=np.int32)
    cheapest_source = np.full(n_points, -1, dtype=np.int32)
    
    node_best_edge_id = np.full(n_points, -1, dtype=np.int32)
    node_best_weight = np.full(n_points, np.inf, dtype=np.float64)
    node_best_target = np.full(n_points, -1, dtype=np.int32)
    
    for i in range(n_points):
        my_root = _dsu_find_numba(parent, i)
        my_size = size[my_root]
        
        start = adj_indptr[i]
        end = adj_indptr[i + 1]
        
        for k in range(start, end):
            edge_id = adj_edge_ids[k]
            if infeasible[edge_id]:
                continue
                
            j = adj_neighbors[k]
            neighbor_root = _dsu_find_numba(parent, j)
            if my_root == neighbor_root:
                continue
            
            neighbor_size = size[neighbor_root]
            if my_size < neighbor_size:
                root_small, root_large = my_root, neighbor_root
            else:
                root_small, root_large = neighbor_root, my_root
            
            if _component_has_conflict_with_root_numba(
                root_small, root_large, parent, head, next_node,
                cannot_link_indptr, cannot_link_indices
            ):
                infeasible[edge_id] = True
                continue
            
            ww = adj_weights[k]
            if ww < node_best_weight[i] or (
                ww == node_best_weight[i] and edge_id < node_best_edge_id[i]
            ):
                node_best_weight[i] = ww
                node_best_edge_id[i] = edge_id
                node_best_target[i] = neighbor_root
    
    # Aggregate per-component
    for i in range(n_points):
        if node_best_edge_id[i] == -1:
            continue
        my_root = _dsu_find_numba(parent, i)
        if node_best_weight[i] < cheapest_weight[my_root] or (
            node_best_weight[i] == cheapest_weight[my_root]
            and node_best_edge_id[i] < cheapest_edge_id[my_root]
        ):
            cheapest_weight[my_root] = node_best_weight[i]
            cheapest_edge_id[my_root] = node_best_edge_id[i]
            cheapest_target[my_root] = node_best_target[i]
            cheapest_source[my_root] = i
    
    return cheapest_source, cheapest_target, cheapest_weight, cheapest_edge_id


@numba.njit(parallel=True, cache=True)
def _parallel_edge_selection_step1a_cpu(
    adj_indptr: np.ndarray,
    adj_neighbors: np.ndarray,
    adj_weights: np.ndarray,
    adj_edge_ids: np.ndarray,
    parent: np.ndarray,
    size: np.ndarray,
    head: np.ndarray,
    next_node: np.ndarray,
    cannot_link_indptr: np.ndarray,
    cannot_link_indices: np.ndarray,
    infeasible: np.ndarray,
    n_points: int,
    node_best_weight: np.ndarray,
    node_best_edge_id: np.ndarray,
    node_best_target: np.ndarray,
    node_best_source: np.ndarray,
) -> None:
    """
    Step 1a: Per-node edge search (PARALLEL on CPU).
    Uses _dsu_find_readonly_numba for thread safety.
    """
    for i in prange(n_points):
        my_root = _dsu_find_readonly_numba(parent, i)
        my_size = size[my_root]
        
        best_weight = np.inf
        best_edge_id = -1
        best_target = -1
        
        start = adj_indptr[i]
        end = adj_indptr[i + 1]
        
        for k in range(start, end):
            edge_id = adj_edge_ids[k]
            if infeasible[edge_id]:
                continue
            
            j = adj_neighbors[k]
            neighbor_root = _dsu_find_readonly_numba(parent, j)
            if my_root == neighbor_root:
                continue
            
            neighbor_size = size[neighbor_root]
            if my_size < neighbor_size:
                root_small, root_large = my_root, neighbor_root
            else:
                root_small, root_large = neighbor_root, my_root
            
            if _component_has_conflict_with_root_numba(
                root_small, root_large, parent, head, next_node,
                cannot_link_indptr, cannot_link_indices
            ):
                continue
            
            ww = adj_weights[k]
            if ww < best_weight or (ww == best_weight and edge_id < best_edge_id):
                best_weight = ww
                best_edge_id = edge_id
                best_target = neighbor_root
        
        node_best_weight[i] = best_weight
        node_best_edge_id[i] = best_edge_id
        node_best_target[i] = best_target
        node_best_source[i] = i


@numba.njit(cache=True)
def _parallel_edge_selection_step1b_cpu(
    parent: np.ndarray,
    n_points: int,
    node_best_weight: np.ndarray,
    node_best_edge_id: np.ndarray,
    node_best_target: np.ndarray,
    node_best_source: np.ndarray,
    cheapest_weight: np.ndarray,
    cheapest_edge_id: np.ndarray,
    cheapest_target: np.ndarray,
    cheapest_source: np.ndarray,
) -> None:
    """
    Step 1b: Per-component aggregation (SEQUENTIAL on CPU).
    Must be sequential because multiple nodes in same component write to cheapest[root].
    """
    for i in range(n_points):
        if node_best_edge_id[i] == -1:
            continue
        my_root = _dsu_find_numba(parent, i)
        if node_best_weight[i] < cheapest_weight[my_root] or (
            node_best_weight[i] == cheapest_weight[my_root]
            and node_best_edge_id[i] < cheapest_edge_id[my_root]
        ):
            cheapest_weight[my_root] = node_best_weight[i]
            cheapest_edge_id[my_root] = node_best_edge_id[i]
            cheapest_target[my_root] = node_best_target[i]
            cheapest_source[my_root] = node_best_source[i]


@numba.njit(cache=True)
def _parallel_edge_selection_cpu(
    adj_indptr: np.ndarray,
    adj_neighbors: np.ndarray,
    adj_weights: np.ndarray,
    adj_edge_ids: np.ndarray,
    parent: np.ndarray,
    size: np.ndarray,
    head: np.ndarray,
    next_node: np.ndarray,
    cannot_link_indptr: np.ndarray,
    cannot_link_indices: np.ndarray,
    infeasible: np.ndarray,
    n_points: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """CPU edge selection (sequential fallback - parallel version not callable from njit)."""
    node_best_weight = np.full(n_points, np.inf, dtype=np.float64)
    node_best_edge_id = np.full(n_points, -1, dtype=np.int32)
    node_best_target = np.full(n_points, -1, dtype=np.int32)
    node_best_source = np.full(n_points, -1, dtype=np.int32)
    
    cheapest_weight = np.full(n_points, np.inf, dtype=np.float64)
    cheapest_edge_id = np.full(n_points, -1, dtype=np.int32)
    cheapest_target = np.full(n_points, -1, dtype=np.int32)
    cheapest_source = np.full(n_points, -1, dtype=np.int32)
    
    # Step 1a: Sequential per-node edge search
    for i in range(n_points):
        my_root = _dsu_find_numba(parent, i)
        my_size = size[my_root]
        
        best_weight = np.inf
        best_edge_id = np.int32(-1)
        best_target = np.int32(-1)
        
        start = adj_indptr[i]
        end = adj_indptr[i + 1]
        
        for k in range(start, end):
            edge_id = adj_edge_ids[k]
            if infeasible[edge_id]:
                continue
            
            j = adj_neighbors[k]
            neighbor_root = _dsu_find_numba(parent, j)
            if my_root == neighbor_root:
                continue
            
            neighbor_size = size[neighbor_root]
            if my_size < neighbor_size:
                root_small, root_large = my_root, neighbor_root
            else:
                root_small, root_large = neighbor_root, my_root
            
            if _component_has_conflict_with_root_numba(
                root_small, root_large, parent, head, next_node,
                cannot_link_indptr, cannot_link_indices
            ):
                infeasible[edge_id] = True
                continue
            
            ww = adj_weights[k]
            if ww < best_weight or (ww == best_weight and edge_id < best_edge_id):
                best_weight = ww
                best_edge_id = edge_id
                best_target = neighbor_root
        
        node_best_weight[i] = best_weight
        node_best_edge_id[i] = best_edge_id
        node_best_target[i] = best_target
        node_best_source[i] = np.int32(i)
    
    # Step 1b: Sequential aggregation
    _parallel_edge_selection_step1b_cpu(
        parent, n_points,
        node_best_weight, node_best_edge_id, node_best_target, node_best_source,
        cheapest_weight, cheapest_edge_id, cheapest_target, cheapest_source,
    )
    
    return cheapest_source, cheapest_target, cheapest_weight, cheapest_edge_id


# ==============================================================================
# Merge and Violation Handling
# ==============================================================================

@numba.njit(cache=True)
def _merge_edge_into_mst_numba(
    src: int, tgt: int, weight: float, edge_id: int,
    adj_indptr: np.ndarray, adj_neighbors: np.ndarray, adj_edge_ids: np.ndarray,
    parent: np.ndarray, size: np.ndarray, head: np.ndarray, tail: np.ndarray, next_node: np.ndarray,
    mst_edges: np.ndarray, n_added: int,
    round_edges_u: np.ndarray, round_edges_v: np.ndarray, round_edges_w: np.ndarray, round_edges_count: int,
) -> Tuple[int, int, int]:
    """Merge an edge into the MST using lower-root-wins tie-breaking."""
    src_root = _dsu_find_numba(parent, src)
    tgt_root = _dsu_find_numba(parent, tgt)
    
    if src_root == tgt_root:
        return n_added, round_edges_count, src_root
    
    actual_tgt = -1
    for k in range(adj_indptr[src], adj_indptr[src + 1]):
        if adj_edge_ids[k] == edge_id:
            actual_tgt = adj_neighbors[k]
            break
    
    if actual_tgt == -1:
        return n_added, round_edges_count, src_root
    
    winner = min(src_root, tgt_root)
    loser = max(src_root, tgt_root)
    
    parent[loser] = winner
    size[winner] += size[loser]
    next_node[tail[winner]] = head[loser]
    tail[winner] = tail[loser]
    
    mst_edges[n_added, 0] = float(src)
    mst_edges[n_added, 1] = float(actual_tgt)
    mst_edges[n_added, 2] = weight
    
    round_edges_u[round_edges_count] = src
    round_edges_v[round_edges_count] = actual_tgt
    round_edges_w[round_edges_count] = weight
    
    return n_added + 1, round_edges_count + 1, winner


@numba.njit(cache=True)
def _merge_proposed_edges_numba(
    cheapest_source: np.ndarray, cheapest_target: np.ndarray,
    cheapest_weight: np.ndarray, cheapest_edge_id: np.ndarray,
    adj_indptr: np.ndarray, adj_neighbors: np.ndarray, adj_edge_ids: np.ndarray,
    parent: np.ndarray, size: np.ndarray, head: np.ndarray, tail: np.ndarray, next_node: np.ndarray,
    mst_edges: np.ndarray, n_added: int, n_points: int,
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, int]:
    """Merge all proposed edges (without constraint checks - fixes happen in Step 3)."""
    round_edges_u = np.empty(n_points, dtype=np.int32)
    round_edges_v = np.empty(n_points, dtype=np.int32)
    round_edges_w = np.empty(n_points, dtype=np.float64)
    round_count = 0
    
    # Sort roots for determinism
    roots_to_process = np.empty(n_points, dtype=np.int32)
    n_roots = 0
    for r in range(n_points):
        if parent[r] == r and cheapest_edge_id[r] != -1:
            roots_to_process[n_roots] = r
            n_roots += 1
    
    for i in range(n_roots):
        for j in range(i + 1, n_roots):
            if roots_to_process[i] > roots_to_process[j]:
                roots_to_process[i], roots_to_process[j] = roots_to_process[j], roots_to_process[i]
    
    for idx in range(n_roots):
        r = roots_to_process[idx]
        current_root = _dsu_find_numba(parent, r)
        if current_root != r:
            continue
        target_root = _dsu_find_numba(parent, cheapest_target[r])
        if current_root == target_root:
            continue
        
        n_added, round_count, _ = _merge_edge_into_mst_numba(
            cheapest_source[r], target_root, cheapest_weight[r], cheapest_edge_id[r],
            adj_indptr, adj_neighbors, adj_edge_ids,
            parent, size, head, tail, next_node,
            mst_edges, n_added,
            round_edges_u, round_edges_v, round_edges_w, round_count,
        )
    
    return n_added, round_edges_u, round_edges_v, round_edges_w, round_count


@numba.njit(cache=True)
def _detect_violations_numba(
    parent: np.ndarray,
    cannot_link_indptr: np.ndarray,
    cannot_link_indices: np.ndarray,
    n_points: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Detect cannot-link violations in current DSU state."""
    max_violations = cannot_link_indices.shape[0]
    violation_a = np.empty(max_violations, dtype=np.int32)
    violation_b = np.empty(max_violations, dtype=np.int32)
    n_violations = 0
    
    for i in range(n_points):
        i_root = _dsu_find_numba(parent, i)
        start = cannot_link_indptr[i]
        end = cannot_link_indptr[i + 1]
        
        for k in range(start, end):
            j = cannot_link_indices[k]
            if i >= j:
                continue
            j_root = _dsu_find_numba(parent, j)
            if i_root == j_root:
                violation_a[n_violations] = i
                violation_b[n_violations] = j
                n_violations += 1
    
    return violation_a, violation_b, n_violations


@numba.njit(cache=True)
def _check_component_has_violation_numba(
    root: int,
    parent: np.ndarray,
    head: np.ndarray,
    next_node: np.ndarray,
    cannot_link_indptr: np.ndarray,
    cannot_link_indices: np.ndarray,
) -> bool:
    """Check if a component has any internal constraint violations."""
    node = head[root]
    while node != -1:
        start = cannot_link_indptr[node]
        end = cannot_link_indptr[node + 1]
        for k in range(start, end):
            partner = cannot_link_indices[k]
            if _dsu_find_numba(parent, partner) == root:
                return True
        node = next_node[node]
    return False


@numba.njit(parallel=True, cache=True)
def _detect_violating_components_parallel(
    parent: np.ndarray,
    head: np.ndarray,
    next_node: np.ndarray,
    cannot_link_indptr: np.ndarray,
    cannot_link_indices: np.ndarray,
    n_points: int,
) -> np.ndarray:
    """
    Step 3a: Parallel detection of which components have violations.
    
    Each root is checked independently in parallel. Returns a boolean array
    where has_violation[r] = True if component with root r has a violation.
    """
    has_violation = np.zeros(n_points, dtype=np.bool_)
    
    for r in prange(n_points):
        if parent[r] == r:  # Is a root
            # Check this component for violations (read-only, safe for parallel)
            node = head[r]
            found_violation = False
            while node != -1 and not found_violation:
                start = cannot_link_indptr[node]
                end = cannot_link_indptr[node + 1]
                for k in range(start, end):
                    partner = cannot_link_indices[k]
                    # Use readonly find to avoid path compression races
                    partner_root = partner
                    while parent[partner_root] != partner_root:
                        partner_root = parent[partner_root]
                    if partner_root == r:
                        found_violation = True
                        break
                node = next_node[node]
            has_violation[r] = found_violation
    
    return has_violation


@numba.njit(cache=True)
def _rebuild_dsu_from_edges_numba(
    n_points: int,
    mst_edges: np.ndarray,
    n_edges: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rebuild DSU structures from MST edges."""
    parent = np.empty(n_points, dtype=np.int32)
    size = np.empty(n_points, dtype=np.int32)
    head = np.empty(n_points, dtype=np.int32)
    tail = np.empty(n_points, dtype=np.int32)
    next_node = np.empty(n_points, dtype=np.int32)
    
    for i in range(n_points):
        parent[i] = i
        size[i] = 1
        head[i] = i
        tail[i] = i
        next_node[i] = -1
    
    for e in range(n_edges):
        u = np.int32(mst_edges[e, 0])
        v = np.int32(mst_edges[e, 1])
        
        u_root = _dsu_find_numba(parent, u)
        v_root = _dsu_find_numba(parent, v)
        if u_root == v_root:
            continue
        
        winner = min(u_root, v_root)
        loser = max(u_root, v_root)
        
        parent[loser] = winner
        size[winner] += size[loser]
        next_node[tail[winner]] = head[loser]
        tail[winner] = tail[loser]
    
    return parent, size, head, tail, next_node


@numba.njit(cache=True)
def _fix_round_violations_with_initial_worklist_numba(
    parent: np.ndarray, size: np.ndarray, head: np.ndarray, tail: np.ndarray, next_node: np.ndarray,
    round_edges_u: np.ndarray, round_edges_v: np.ndarray, round_edges_w: np.ndarray, round_count: int,
    mst_edges: np.ndarray, n_added: int,
    cannot_link_indptr: np.ndarray, cannot_link_indices: np.ndarray,
    cannot_link_edge_out: np.ndarray, n_cannot_link_edges: int, n_points: int,
    initial_worklist: np.ndarray, initial_worklist_count: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, int]:
    """
    Fix violations given a pre-computed initial worklist (Step 3b - sequential correction).
    
    This version accepts the initial worklist of violating components computed externally
    (potentially in parallel), then does sequential correction.
    """
    if initial_worklist_count == 0 or round_count == 0:
        return parent, size, head, tail, next_node, n_added, cannot_link_edge_out, n_cannot_link_edges
    
    # Sort round edges by weight descending
    sorted_indices = np.arange(round_count, dtype=np.int32)
    for i in range(round_count):
        for j in range(i + 1, round_count):
            if round_edges_w[sorted_indices[i]] < round_edges_w[sorted_indices[j]]:
                sorted_indices[i], sorted_indices[j] = sorted_indices[j], sorted_indices[i]
    
    round_edge_removed = np.zeros(round_count, dtype=np.bool_)
    max_worklist = n_points
    worklist_roots = np.empty(max_worklist, dtype=np.int32)
    
    # Copy initial worklist
    worklist_count = initial_worklist_count
    for i in range(initial_worklist_count):
        worklist_roots[i] = initial_worklist[i]
    
    iterations = 0
    max_iterations = round_count * n_points
    
    while worklist_count > 0 and iterations < max_iterations:
        iterations += 1
        worklist_count -= 1
        current_root = worklist_roots[worklist_count]
        current_root = _dsu_find_numba(parent, current_root)
        
        has_viol = _check_component_has_violation_numba(
            current_root, parent, head, next_node, cannot_link_indptr, cannot_link_indices
        )
        if not has_viol:
            continue
        
        best_edge_idx = -1
        best_weight = -1.0
        
        for idx in range(round_count):
            r = sorted_indices[idx]
            if round_edge_removed[r]:
                continue
            eu, ev, ew = round_edges_u[r], round_edges_v[r], round_edges_w[r]
            eu_root = _dsu_find_numba(parent, eu)
            ev_root = _dsu_find_numba(parent, ev)
            if eu_root == current_root and ev_root == current_root:
                if ew > best_weight:
                    best_weight = ew
                    best_edge_idx = r
        
        if best_edge_idx == -1:
            continue
        
        round_edge_removed[best_edge_idx] = True
        removed_u = round_edges_u[best_edge_idx]
        removed_v = round_edges_v[best_edge_idx]
        removed_w = round_edges_w[best_edge_idx]
        
        temp_mst = np.empty((n_added, 3), dtype=np.float64)
        temp_n = 0
        
        for e in range(n_added):
            eu = np.int32(mst_edges[e, 0])
            ev = np.int32(mst_edges[e, 1])
            ew = mst_edges[e, 2]
            
            is_removed = False
            for rr in range(round_count):
                if round_edge_removed[rr]:
                    ru, rv, rw = round_edges_u[rr], round_edges_v[rr], round_edges_w[rr]
                    if ((eu == ru and ev == rv) or (eu == rv and ev == ru)) and ew == rw:
                        is_removed = True
                        break
            
            if not is_removed:
                temp_mst[temp_n, 0] = mst_edges[e, 0]
                temp_mst[temp_n, 1] = mst_edges[e, 1]
                temp_mst[temp_n, 2] = mst_edges[e, 2]
                temp_n += 1
        
        temp_parent, temp_size, temp_head, temp_tail, temp_next_node = _rebuild_dsu_from_edges_numba(
            n_points, temp_mst, temp_n
        )
        
        subcomp_a_root = _dsu_find_numba(temp_parent, removed_u)
        subcomp_b_root = _dsu_find_numba(temp_parent, removed_v)
        
        viol_a = _check_component_has_violation_numba(
            subcomp_a_root, temp_parent, temp_head, temp_next_node, cannot_link_indptr, cannot_link_indices
        )
        viol_b = _check_component_has_violation_numba(
            subcomp_b_root, temp_parent, temp_head, temp_next_node, cannot_link_indptr, cannot_link_indices
        )
        
        for i in range(n_points):
            parent[i] = temp_parent[i]
            size[i] = temp_size[i]
            head[i] = temp_head[i]
            tail[i] = temp_tail[i]
            next_node[i] = temp_next_node[i]
        
        for e in range(temp_n):
            mst_edges[e, 0] = temp_mst[e, 0]
            mst_edges[e, 1] = temp_mst[e, 1]
            mst_edges[e, 2] = temp_mst[e, 2]
        n_added = temp_n
        
        if not viol_a and not viol_b:
            cannot_link_edge_out[n_cannot_link_edges, 0] = float(removed_u)
            cannot_link_edge_out[n_cannot_link_edges, 1] = float(removed_v)
            cannot_link_edge_out[n_cannot_link_edges, 2] = removed_w
            n_cannot_link_edges += 1
        else:
            if viol_a and worklist_count < max_worklist:
                worklist_roots[worklist_count] = subcomp_a_root
                worklist_count += 1
            if viol_b and worklist_count < max_worklist:
                worklist_roots[worklist_count] = subcomp_b_root
                worklist_count += 1
    
    return parent, size, head, tail, next_node, n_added, cannot_link_edge_out, n_cannot_link_edges


@numba.njit(cache=True)
def _fix_round_violations_numba(
    parent: np.ndarray, size: np.ndarray, head: np.ndarray, tail: np.ndarray, next_node: np.ndarray,
    round_edges_u: np.ndarray, round_edges_v: np.ndarray, round_edges_w: np.ndarray, round_count: int,
    mst_edges: np.ndarray, n_added: int,
    cannot_link_indptr: np.ndarray, cannot_link_indices: np.ndarray,
    cannot_link_edge_out: np.ndarray, n_cannot_link_edges: int, n_points: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, int]:
    """Fix violations by removing heaviest round edges recursively."""
    _, _, n_violations = _detect_violations_numba(parent, cannot_link_indptr, cannot_link_indices, n_points)
    
    if n_violations == 0 or round_count == 0:
        return parent, size, head, tail, next_node, n_added, cannot_link_edge_out, n_cannot_link_edges
    
    # Sort round edges by weight descending
    sorted_indices = np.arange(round_count, dtype=np.int32)
    for i in range(round_count):
        for j in range(i + 1, round_count):
            if round_edges_w[sorted_indices[i]] < round_edges_w[sorted_indices[j]]:
                sorted_indices[i], sorted_indices[j] = sorted_indices[j], sorted_indices[i]
    
    round_edge_removed = np.zeros(round_count, dtype=np.bool_)
    max_worklist = n_points
    worklist_roots = np.empty(max_worklist, dtype=np.int32)
    worklist_count = 0
    
    # Step 3a: Parallel detection of violating components
    # Note: We call the parallel function from the wrapper, not here
    # because njit functions can't call parallel functions directly.
    # For now, use sequential detection inside njit.
    for r in range(n_points):
        if parent[r] == r:
            has_viol = _check_component_has_violation_numba(
                r, parent, head, next_node, cannot_link_indptr, cannot_link_indices
            )
            if has_viol:
                worklist_roots[worklist_count] = r
                worklist_count += 1
    
    iterations = 0
    max_iterations = round_count * n_points
    
    while worklist_count > 0 and iterations < max_iterations:
        iterations += 1
        worklist_count -= 1
        current_root = worklist_roots[worklist_count]
        current_root = _dsu_find_numba(parent, current_root)
        
        has_viol = _check_component_has_violation_numba(
            current_root, parent, head, next_node, cannot_link_indptr, cannot_link_indices
        )
        if not has_viol:
            continue
        
        best_edge_idx = -1
        best_weight = -1.0
        
        for idx in range(round_count):
            r = sorted_indices[idx]
            if round_edge_removed[r]:
                continue
            eu, ev, ew = round_edges_u[r], round_edges_v[r], round_edges_w[r]
            eu_root = _dsu_find_numba(parent, eu)
            ev_root = _dsu_find_numba(parent, ev)
            if eu_root == current_root and ev_root == current_root:
                if ew > best_weight:
                    best_weight = ew
                    best_edge_idx = r
        
        if best_edge_idx == -1:
            continue
        
        round_edge_removed[best_edge_idx] = True
        removed_u = round_edges_u[best_edge_idx]
        removed_v = round_edges_v[best_edge_idx]
        removed_w = round_edges_w[best_edge_idx]
        
        temp_mst = np.empty((n_added, 3), dtype=np.float64)
        temp_n = 0
        
        for e in range(n_added):
            eu = np.int32(mst_edges[e, 0])
            ev = np.int32(mst_edges[e, 1])
            ew = mst_edges[e, 2]
            
            is_removed = False
            for rr in range(round_count):
                if round_edge_removed[rr]:
                    ru, rv, rw = round_edges_u[rr], round_edges_v[rr], round_edges_w[rr]
                    if ((eu == ru and ev == rv) or (eu == rv and ev == ru)) and ew == rw:
                        is_removed = True
                        break
            
            if not is_removed:
                temp_mst[temp_n, 0] = mst_edges[e, 0]
                temp_mst[temp_n, 1] = mst_edges[e, 1]
                temp_mst[temp_n, 2] = mst_edges[e, 2]
                temp_n += 1
        
        temp_parent, temp_size, temp_head, temp_tail, temp_next_node = _rebuild_dsu_from_edges_numba(
            n_points, temp_mst, temp_n
        )
        
        subcomp_a_root = _dsu_find_numba(temp_parent, removed_u)
        subcomp_b_root = _dsu_find_numba(temp_parent, removed_v)
        
        viol_a = _check_component_has_violation_numba(
            subcomp_a_root, temp_parent, temp_head, temp_next_node, cannot_link_indptr, cannot_link_indices
        )
        viol_b = _check_component_has_violation_numba(
            subcomp_b_root, temp_parent, temp_head, temp_next_node, cannot_link_indptr, cannot_link_indices
        )
        
        for i in range(n_points):
            parent[i] = temp_parent[i]
            size[i] = temp_size[i]
            head[i] = temp_head[i]
            tail[i] = temp_tail[i]
            next_node[i] = temp_next_node[i]
        
        for e in range(temp_n):
            mst_edges[e, 0] = temp_mst[e, 0]
            mst_edges[e, 1] = temp_mst[e, 1]
            mst_edges[e, 2] = temp_mst[e, 2]
        n_added = temp_n
        
        if not viol_a and not viol_b:
            cannot_link_edge_out[n_cannot_link_edges, 0] = float(removed_u)
            cannot_link_edge_out[n_cannot_link_edges, 1] = float(removed_v)
            cannot_link_edge_out[n_cannot_link_edges, 2] = removed_w
            n_cannot_link_edges += 1
        else:
            if viol_a and worklist_count < max_worklist:
                worklist_roots[worklist_count] = subcomp_a_root
                worklist_count += 1
            if viol_b and worklist_count < max_worklist:
                worklist_roots[worklist_count] = subcomp_b_root
                worklist_count += 1
    
    return parent, size, head, tail, next_node, n_added, cannot_link_edge_out, n_cannot_link_edges


# ==============================================================================
# Parallel Borůvka MST - Sequential Backend
# ==============================================================================

@numba.njit(cache=True)
def _parallel_constrained_boruvka_mst_sequential_numba(
    adj_indptr: np.ndarray, adj_neighbors: np.ndarray, adj_weights: np.ndarray, adj_edge_ids: np.ndarray,
    cannot_link_indptr: np.ndarray, cannot_link_indices: np.ndarray,
    n_points: int, n_edges: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fully sequential constrained Borůvka MST with violation correction."""
    parent = np.empty(n_points, dtype=np.int32)
    size = np.empty(n_points, dtype=np.int32)
    head = np.empty(n_points, dtype=np.int32)
    tail = np.empty(n_points, dtype=np.int32)
    next_node = np.empty(n_points, dtype=np.int32)
    
    for i in range(n_points):
        parent[i] = i
        size[i] = 1
        head[i] = i
        tail[i] = i
        next_node[i] = -1
    
    infeasible = np.zeros(n_edges, dtype=np.bool_)
    mst_edges = np.empty((n_points - 1, 3), dtype=np.float64)
    n_added = 0
    cannot_link_edge_out = np.empty((n_points - 1, 3), dtype=np.float64)
    n_cannot_link_edges = 0
    
    while True:
        cheapest_source, cheapest_target, cheapest_weight, cheapest_edge_id = \
            _parallel_edge_selection_numba(
                adj_indptr, adj_neighbors, adj_weights, adj_edge_ids,
                parent, size, head, next_node,
                cannot_link_indptr, cannot_link_indices, infeasible, n_points,
            )
        
        any_edge_found = False
        for r in range(n_points):
            if parent[r] == r and cheapest_edge_id[r] != -1:
                any_edge_found = True
                break
        if not any_edge_found:
            break
        
        n_added_before = n_added
        n_added, round_edges_u, round_edges_v, round_edges_w, round_count = \
            _merge_proposed_edges_numba(
                cheapest_source, cheapest_target, cheapest_weight, cheapest_edge_id,
                adj_indptr, adj_neighbors, adj_edge_ids,
                parent, size, head, tail, next_node, mst_edges, n_added, n_points,
            )
        
        if n_added - n_added_before == 0:
            break
        
        parent, size, head, tail, next_node, n_added, cannot_link_edge_out, n_cannot_link_edges = \
            _fix_round_violations_numba(
                parent, size, head, tail, next_node,
                round_edges_u, round_edges_v, round_edges_w, round_count,
                mst_edges, n_added, cannot_link_indptr, cannot_link_indices,
                cannot_link_edge_out, n_cannot_link_edges, n_points,
            )
        
        if n_added >= n_points - 1:
            break
    
    return mst_edges[:n_added], cannot_link_edge_out[:n_cannot_link_edges]


# ==============================================================================
# Parallel Borůvka MST - CPU Backend
# ==============================================================================

def _parallel_constrained_boruvka_mst_cpu(
    adj_indptr: np.ndarray, adj_neighbors: np.ndarray, adj_weights: np.ndarray, adj_edge_ids: np.ndarray,
    cannot_link_indptr: np.ndarray, cannot_link_indices: np.ndarray,
    n_points: int, n_edges: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """CPU parallel constrained Borůvka (Step 1a parallel, Step 3a parallel, rest sequential)."""
    parent = np.empty(n_points, dtype=np.int32)
    size = np.empty(n_points, dtype=np.int32)
    head = np.empty(n_points, dtype=np.int32)
    tail = np.empty(n_points, dtype=np.int32)
    next_node = np.empty(n_points, dtype=np.int32)
    
    for i in range(n_points):
        parent[i] = i
        size[i] = 1
        head[i] = i
        tail[i] = i
        next_node[i] = -1
    
    infeasible = np.zeros(n_edges, dtype=np.bool_)
    mst_edges = np.empty((n_points - 1, 3), dtype=np.float64)
    n_added = 0
    cannot_link_edge_out = np.empty((n_points - 1, 3), dtype=np.float64)
    n_cannot_link_edges = 0
    
    while True:
        cheapest_source, cheapest_target, cheapest_weight, cheapest_edge_id = \
            _parallel_edge_selection_cpu(
                adj_indptr, adj_neighbors, adj_weights, adj_edge_ids,
                parent, size, head, next_node,
                cannot_link_indptr, cannot_link_indices, infeasible, n_points,
            )
        
        any_edge_found = False
        for r in range(n_points):
            if parent[r] == r and cheapest_edge_id[r] != -1:
                any_edge_found = True
                break
        if not any_edge_found:
            break
        
        n_added_before = n_added
        n_added, round_edges_u, round_edges_v, round_edges_w, round_count = \
            _merge_proposed_edges_numba(
                cheapest_source, cheapest_target, cheapest_weight, cheapest_edge_id,
                adj_indptr, adj_neighbors, adj_edge_ids,
                parent, size, head, tail, next_node, mst_edges, n_added, n_points,
            )
        
        if n_added - n_added_before == 0:
            break
        
        # Step 3a: Parallel detection of violating components
        has_violation = _detect_violating_components_parallel(
            parent, head, next_node, cannot_link_indptr, cannot_link_indices, n_points
        )
        
        # Build worklist from parallel detection results
        initial_worklist = np.empty(n_points, dtype=np.int32)
        initial_worklist_count = 0
        for r in range(n_points):
            if has_violation[r]:
                initial_worklist[initial_worklist_count] = r
                initial_worklist_count += 1
        
        # Step 3b: Sequential correction
        parent, size, head, tail, next_node, n_added, cannot_link_edge_out, n_cannot_link_edges = \
            _fix_round_violations_with_initial_worklist_numba(
                parent, size, head, tail, next_node,
                round_edges_u, round_edges_v, round_edges_w, round_count,
                mst_edges, n_added, cannot_link_indptr, cannot_link_indices,
                cannot_link_edge_out, n_cannot_link_edges, n_points,
                initial_worklist, initial_worklist_count,
            )
        
        if n_added >= n_points - 1:
            break
    
    return mst_edges[:n_added], cannot_link_edge_out[:n_cannot_link_edges]


# ==============================================================================
# Backend Selection & Main Entry Point
# ==============================================================================

def _select_parallel_backend(n_points: int, parallel_backend: str) -> str:
    """Select appropriate parallel backend based on preference and hardware."""
    if parallel_backend == "cuda":
        if _check_cuda_available():
            return "cuda"
        raise ValueError(
            "parallel_backend='cuda' requested but CUDA is not available. "
            "Install numba with CUDA support or use parallel_backend='cpu' or 'auto'."
        )
    if parallel_backend == "cpu":
        return "cpu"
    if parallel_backend == "sequential":
        return "sequential"
    # auto
    if _check_cuda_available() and n_points > 10000:
        return "cuda"
    return "cpu"


def _parallel_constrained_boruvka_mst(
    adj_indptr: np.ndarray, adj_neighbors: np.ndarray, adj_weights: np.ndarray, adj_edge_ids: np.ndarray,
    cannot_link_indptr: np.ndarray, cannot_link_indices: np.ndarray,
    n_points: int, n_edges: int,
    parallel_backend: str = "auto",
) -> Tuple[np.ndarray, np.ndarray]:
    """Main entry point for parallel constrained Borůvka MST."""
    backend = _select_parallel_backend(n_points, parallel_backend)
    
    if backend == "cuda":
        # TODO: Implement CUDA backend - fall back to CPU for now
        backend = "cpu"
    
    if backend == "cpu":
        return _parallel_constrained_boruvka_mst_cpu(
            adj_indptr, adj_neighbors, adj_weights, adj_edge_ids,
            cannot_link_indptr, cannot_link_indices, n_points, n_edges,
        )
    
    return _parallel_constrained_boruvka_mst_sequential_numba(
        adj_indptr, adj_neighbors, adj_weights, adj_edge_ids,
        cannot_link_indptr, cannot_link_indices, n_points, n_edges,
    )


# ==============================================================================
# Distance Matrix Utilities
# ==============================================================================

def _ensure_csr_distance_matrix(distances: Union[np.ndarray, CSR]) -> CSR:
    """Validate and return distances as CSR."""
    if sp.isspmatrix_csr(distances):
        distance_matrix_csr = distances.copy().astype(np.float64)
    elif isinstance(distances, np.ndarray):
        if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
            raise ValueError("Precomputed distance matrix must be square (N,N).")
        distance_matrix_csr = sp.csr_matrix(distances.astype(np.float64, copy=False))
    else:
        raise TypeError("distances must be a numpy array or scipy.sparse.csr_matrix.")

    n_points = int(distance_matrix_csr.shape[0])
    if n_points == 0:
        raise ValueError("Precomputed distance matrix must be non-empty.")

    # Remove diagonal structurally
    coo = distance_matrix_csr.tocoo()
    mask_offdiag = coo.row != coo.col
    distance_matrix_csr = sp.coo_matrix(
        (coo.data[mask_offdiag], (coo.row[mask_offdiag], coo.col[mask_offdiag])),
        shape=coo.shape,
    ).tocsr()

    distance_matrix_csr.sum_duplicates()
    distance_matrix_csr.sort_indices()

    if distance_matrix_csr.data.size and np.any(distance_matrix_csr.data < 0):
        raise ValueError("Distances must be non-negative.")

    return distance_matrix_csr


def _symmetrize_min_keep_present(distance_matrix_csr: CSR) -> CSR:
    """Symmetrize sparse distance matrix by min(A_ij, A_ji)."""
    A = distance_matrix_csr.tocsr(copy=True)
    A.sum_duplicates()
    A.sort_indices()

    B = A.T.tocsr(copy=True)
    B.sum_duplicates()
    B.sort_indices()

    n_points = int(A.shape[0])
    if n_points == 0:
        return A

    coo_a = A.tocoo()
    coo_b = B.tocoo()

    rows = np.concatenate([coo_a.row, coo_b.row]).astype(np.int64)
    cols = np.concatenate([coo_a.col, coo_b.col]).astype(np.int64)
    data = np.concatenate([coo_a.data, coo_b.data]).astype(np.float64)

    mask_offdiag = rows != cols
    rows, cols, data = rows[mask_offdiag], cols[mask_offdiag], data[mask_offdiag]

    if data.size == 0:
        return sp.csr_matrix(A.shape, dtype=np.float64)

    key = rows * np.int64(n_points) + cols
    order = np.argsort(key, kind="mergesort")
    key, rows, cols, data = key[order], rows[order], cols[order], data[order]

    group_starts = np.empty(key.shape[0], dtype=bool)
    group_starts[0] = True
    group_starts[1:] = key[1:] != key[:-1]
    idx_start = np.flatnonzero(group_starts)

    data_min = np.minimum.reduceat(data, idx_start)
    rows_u = rows[idx_start]
    cols_u = cols[idx_start]

    sym = sp.coo_matrix((data_min, (rows_u, cols_u)), shape=A.shape).tocsr()
    sym.sum_duplicates()
    sym.sort_indices()
    return sym


def _core_distances_from_sparse_rows(
    distance_matrix_csr: CSR, *, min_samples: int
) -> np.ndarray:
    """Compute HDBSCAN core distances from sparse distance graph."""
    n_points = int(distance_matrix_csr.shape[0])
    core_distances = np.empty(n_points, dtype=np.float64)

    for idx_point in range(n_points):
        start = int(distance_matrix_csr.indptr[idx_point])
        end = int(distance_matrix_csr.indptr[idx_point + 1])
        degree = end - start

        if degree == 0:
            core_distances[idx_point] = np.inf
            continue

        neighbor_distances_sorted = np.sort(distance_matrix_csr.data[start:end])
        if degree >= min_samples:
            core_distances[idx_point] = float(neighbor_distances_sorted[min_samples - 1])
        else:
            core_distances[idx_point] = float(neighbor_distances_sorted[-1])

    return core_distances


def _mutual_reachability_edges_upper_triangle(
    distance_matrix_csr: CSR, *, core_distances: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build edge list with mutual reachability weights."""
    coo = sp.triu(distance_matrix_csr, k=1).tocoo()
    if coo.nnz == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    u = coo.row.astype(np.int64)
    v = coo.col.astype(np.int64)
    d = coo.data.astype(np.float64)

    w = np.maximum.reduce([d, core_distances[u], core_distances[v]])
    return u, v, w


def _kruskal_mst_unconstrained(*, n_points: int, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Standard Kruskal MST."""
    order = np.argsort(w, kind="mergesort")
    u_sorted, v_sorted, w_sorted = u[order], v[order], w[order]

    ds = ds_rank_create(int(n_points))
    mst_edges = np.empty((n_points - 1, 3), dtype=np.float64)
    n_added = 0

    for idx_edge in range(u_sorted.shape[0]):
        a, b, ww = int(u_sorted[idx_edge]), int(v_sorted[idx_edge]), float(w_sorted[idx_edge])
        root_a, root_b = ds_find(ds, a), ds_find(ds, b)
        if root_a == root_b:
            continue
        ds_union_by_rank(ds, root_a, root_b)
        mst_edges[n_added] = [float(a), float(b), ww]
        n_added += 1
        if n_added == n_points - 1:
            break

    return mst_edges[:n_added]


# ==============================================================================
# Constrained MST Entry Points
# ==============================================================================

def _mst_constrained_hard(
    *, n_points: int, u: np.ndarray, v: np.ndarray, w: np.ndarray,
    cannot_link_indptr: np.ndarray, cannot_link_indices: np.ndarray,
    mst_method: str = "boruvka", parallel_backend: str = "auto",
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Hard constrained MST using Borůvka, Parallel Borůvka, or Kruskal."""
    cannot_link_edges = None

    if mst_method == "boruvka":
        u_arr = np.asarray(u, dtype=np.int32)
        v_arr = np.asarray(v, dtype=np.int32)
        w_arr = np.asarray(w, dtype=np.float64)

        adj_indptr, adj_neighbors, adj_weights, adj_edge_ids = _build_adjacency_list_numba(u_arr, v_arr, w_arr, n_points)
        mst_edges = _constrained_boruvka_mst_csr_strict_numba(
            adj_indptr, adj_neighbors, adj_weights, adj_edge_ids,
            cannot_link_indptr, cannot_link_indices, n_points, u_arr.shape[0],
        )

    elif mst_method == "parallel_boruvka":
        u_arr = np.asarray(u, dtype=np.int32)
        v_arr = np.asarray(v, dtype=np.int32)
        w_arr = np.asarray(w, dtype=np.float64)

        adj_indptr, adj_neighbors, adj_weights, adj_edge_ids = _build_adjacency_list_numba(u_arr, v_arr, w_arr, n_points)
        mst_edges, cannot_link_edges = _parallel_constrained_boruvka_mst(
            adj_indptr, adj_neighbors, adj_weights, adj_edge_ids,
            cannot_link_indptr, cannot_link_indices, n_points, u_arr.shape[0],
            parallel_backend=parallel_backend,
        )

    elif mst_method == "kruskal":
        order = np.argsort(w, kind="mergesort")
        u_sorted = np.asarray(u[order], dtype=np.int32)
        v_sorted = np.asarray(v[order], dtype=np.int32)
        w_sorted = np.asarray(w[order], dtype=np.float64)
        mst_edges = _constrained_kruskal_mst_csr_strict_sorted_numba(
            u_sorted, v_sorted, w_sorted, cannot_link_indptr, cannot_link_indices, n_points,
        )

    else:
        raise ValueError(f"Invalid mst_method: {mst_method}. Must be 'boruvka', 'parallel_boruvka', or 'kruskal'.")

    return mst_edges, cannot_link_edges


def _connect_components_with_penalty_edges(*, n_points: int, mst_edges: np.ndarray, penalty: float) -> np.ndarray:
    """Connect forest components with penalty-weight edges."""
    if mst_edges.shape[0] >= n_points - 1:
        return mst_edges

    ds = ds_rank_create(n_points)
    for row in mst_edges:
        root_a, root_b = ds_find(ds, int(row[0])), ds_find(ds, int(row[1]))
        if root_a != root_b:
            ds_union_by_rank(ds, root_a, root_b)

    root_to_rep = {}
    for idx_point in range(n_points):
        root = ds_find(ds, idx_point)
        if root not in root_to_rep:
            root_to_rep[root] = idx_point

    reps = list(root_to_rep.values())
    if len(reps) <= 1:
        return mst_edges

    extra_edges = np.array([[float(reps[0]), float(rep), penalty] for rep in reps[1:]], dtype=np.float64)
    return np.vstack([mst_edges, extra_edges])


# ==============================================================================
# Violation Detection and Post-hoc Cleanup
# ==============================================================================

def find_cannot_link_violations(
    labels: np.ndarray, cannot_link_pairs: Iterable[Tuple[int, int]]
) -> list:
    """Find pairs that violate cannot-link constraints."""
    violations = []
    for i, j in cannot_link_pairs:
        if labels[i] >= 0 and labels[j] >= 0 and labels[i] == labels[j]:
            violations.append((i, j))
    return violations


def split_clusters_to_respect_cannot_link(
    labels: np.ndarray, cannot_link_pairs: Iterable[Tuple[int, int]]
) -> np.ndarray:
    """Post-hoc split of clusters to respect cannot-link constraints."""
    labels = labels.copy()
    pairs_list = list(cannot_link_pairs)
    
    while True:
        violations = find_cannot_link_violations(labels, pairs_list)
        if not violations:
            break
        
        # Split the first violating pair
        i, j = violations[0]
        cluster_id = labels[i]
        
        # Create new cluster for point j
        new_label = labels.max() + 1
        labels[j] = new_label
    
    return labels


# ==============================================================================
# Main Public API
# ==============================================================================

def fast_hdbscan_precomputed(
    distances: Union[np.ndarray, CSR],
    *,
    min_cluster_size: int = 10,
    min_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    HDBSCAN clustering on precomputed distance matrix (no constraints).
    
    Args:
        distances: Precomputed distance matrix (N, N).
        min_cluster_size: Minimum cluster size.
        min_samples: Number of samples for core point definition.
        
    Returns:
        Tuple of (labels, probabilities).
    """
    distance_matrix_csr = _ensure_csr_distance_matrix(distances)
    distance_matrix_csr = _symmetrize_min_keep_present(distance_matrix_csr)
    n_points = distance_matrix_csr.shape[0]
    
    if min_samples is None:
        min_samples = min_cluster_size
    
    core_distances = _core_distances_from_sparse_rows(distance_matrix_csr, min_samples=min_samples)
    u, v, w = _mutual_reachability_edges_upper_triangle(distance_matrix_csr, core_distances=core_distances)
    
    mst_edges = _kruskal_mst_unconstrained(n_points=n_points, u=u, v=v, w=w)
    
    # Connect disconnected components
    if mst_edges.shape[0] < n_points - 1:
        penalty = float(np.percentile(w, 99.9) * 1e6 + 1.0) if w.size > 0 else 1e9
        mst_edges = _connect_components_with_penalty_edges(n_points=n_points, mst_edges=mst_edges, penalty=penalty)
    
    labels, probabilities, *_ = clusters_from_spanning_tree(
        mst_edges, n_points, min_cluster_size=min_cluster_size
    )
    
    return labels, probabilities


def fast_hdbscan_precomputed_with_cannot_link(
    distances: Union[np.ndarray, CSR],
    cannot_link: Union[np.ndarray, CSR],
    *,
    min_cluster_size: int = 10,
    min_samples: Optional[int] = None,
    mst_method: str = "boruvka",
    parallel_backend: str = "auto",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    HDBSCAN clustering with cannot-link constraints.
    
    Args:
        distances: Precomputed distance matrix (N, N).
        cannot_link: Cannot-link constraint matrix (N, N). Non-zero means cannot-link.
        min_cluster_size: Minimum cluster size.
        min_samples: Number of samples for core point definition.
        mst_method: MST algorithm - "boruvka", "parallel_boruvka", or "kruskal".
        parallel_backend: Backend for parallel_boruvka - "auto", "cuda", "cpu", "sequential".
        
    Returns:
        Tuple of (labels, probabilities).
    """
    distance_matrix_csr = _ensure_csr_distance_matrix(distances)
    distance_matrix_csr = _symmetrize_min_keep_present(distance_matrix_csr)
    n_points = distance_matrix_csr.shape[0]
    
    if min_samples is None:
        min_samples = min_cluster_size
    
    # Process cannot-link constraints
    if sp.issparse(cannot_link):
        cannot_link_csr = cannot_link.tocsr().astype(np.int32)
    else:
        cannot_link_csr = sp.csr_matrix(cannot_link.astype(np.int32))
    
    # Symmetrize and remove diagonal
    cannot_link_csr = cannot_link_csr + cannot_link_csr.T
    cannot_link_csr.setdiag(0)
    cannot_link_csr.eliminate_zeros()
    
    cannot_link_indptr = np.asarray(cannot_link_csr.indptr, dtype=np.int64)
    cannot_link_indices = np.asarray(cannot_link_csr.indices, dtype=np.int32)
    
    core_distances = _core_distances_from_sparse_rows(distance_matrix_csr, min_samples=min_samples)
    u, v, w = _mutual_reachability_edges_upper_triangle(distance_matrix_csr, core_distances=core_distances)
    
    mst_edges, _ = _mst_constrained_hard(
        n_points=n_points, u=u, v=v, w=w,
        cannot_link_indptr=cannot_link_indptr,
        cannot_link_indices=cannot_link_indices,
        mst_method=mst_method,
        parallel_backend=parallel_backend,
    )
    
    # Connect disconnected components
    if mst_edges.shape[0] < n_points - 1:
        penalty = float(np.percentile(w, 99.9) * 1e6 + 1.0) if w.size > 0 else 1e9
        mst_edges = _connect_components_with_penalty_edges(n_points=n_points, mst_edges=mst_edges, penalty=penalty)
    
    labels, probabilities, *_ = clusters_from_spanning_tree(
        mst_edges, n_points, min_cluster_size=min_cluster_size
    )
    
    return labels, probabilities
