"""
Core preprocessing algorithms
"""

# Authors: Guillermo Navas-Palencia <guillermo.navas@bbva.com>
# BBVA - Copyright 2018.

import numbers
import struct

import numpy as np
import pandas as pd

from collections import defaultdict
from hashlib import sha1

from ...core.dtypes import is_numpy_int
from ...core.dtypes import is_numpy_float


def fast_array_equal(a, b, n_block=64):
    """
    Slicing operations are fast in numpy, so peform numpy.array_equal by
    block. Worst case: distinct elements are located in last block and
    2 *(n_block + 1) operations are required. In this case, performance
    should be close to numpy.array_equal.
    """
    a, b = np.asarray(a), np.asarray(b)

    if a.shape != b.shape:
        return False

    n = a.shape[0]
    n_block = min(n, n_block)
    block = n // n_block
    last = n % n_block

    for bk in range(n_block):
        i = bk * block
        if not bool((a[i:i+block] == b[i:i+block]).all()):
            return False

    if not bool((a[last:] == b[last:]).all()):
        return False

    return True


def fast_array_constant(a, n_block=64):
    """Check if numpy array is constant by blocks."""
    a = np.asarray(a)
    n = a.shape[0]
    n_block = min(n, n_block)
    block = n // n_block
    last = n % n_block

    constant = a[0]
    for bk in range(n_block):
        i = bk * block
        if np.any(a[i:i+block] != constant):
            return False

    if np.any(a[last:] != constant):
        return False

    return True


def dfs(graph, start):
    """
    Depth-First Search to find all vertices in a subject vertices connected
    component. Pass graph as adjacency list.
    """
    visited, stack = set(), [start]
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            stack.extend(graph[vertex])
    return visited


def connected_pairs(pairs):
    """
    Solve disjoint set problem to find connected components of an undirected
    graph.

    References:
        https://en.wikipedia.org/wiki/Disjoint-set_data_structure
    """

    # undirected graph implementation
    graph = defaultdict(list)
    nodes = set()

    # fill the undirected graph as adjacency list: each edge is stored in both
    # incident nodes adjacent sets.
    for x, y in pairs:
        nodes.add(x)
        nodes.add(y)

        graph[x].append(y)
        graph[y].append(x)

    # travese the graph to find duplicates
    duplicates = []

    while nodes:
        duplicate = dfs(graph, nodes.pop())
        duplicates.append(duplicate)

        # remove duplicate from the set of nodes
        nodes -= duplicate

    return duplicates


def find_duplicates_direct(names, values):
    """Find pairs of duplicated columns."""
    duplicated_columns = []
    stack = list(range(len(names)))

    while(stack):
        i = stack.pop()
        col_ref = values[:, i]
        idx_duplicated = []
        for j in stack:
            col_cmp = values[:, j]
            if fast_array_equal(col_ref, col_cmp):
                duplicated_columns.append((names[i], names[j]))
                idx_duplicated.append(j)
        [stack.remove(k) for k in sorted(idx_duplicated, reverse=True)]

    return duplicated_columns


def find_duplicates(dtype, names, X, n_sampling=100):
    """
    Algorithm for detection of duplicated columns.

    1. Apply hash function to subsample of X and return S hash value.
    2. Generate block of candidates to be duplicated columns, grouping by S.
    3. For each block search exact duplicated columns using a LIFO stack.
       Return list of tuples (column_i, column_j).
    4. Fill undirected graph as adjacency list: each edge is stored in both
       incident nodes adjacency sets.
    5. Traverse the graph to find duplicates (solve disjoint set problem).
        a. Use DFS to find all vertices in a subject vertices connected
        component.
        b. Remvoe duplicate from set of nodes until set is empty.
    """
    if not isinstance(names, (list, np.ndarray)):
        raise ValueError("names must be array-like (int, numpy.ndarray).")

    if not isinstance(X, np.ndarray):
        raise ValueError("X must be numpy array.")

    if isinstance(n_sampling, (numbers.Integral, np.integer)):
        if not 2 <= n_sampling:
            raise ValueError("number of random elements must be at least 2.")

    if isinstance(names, list):
        names = np.array(names)

    n, m = X.shape

    rnd_idx = np.random.randint(0, n, n_sampling)

    if is_numpy_float(dtype) or is_numpy_int(dtype):
        hash_value = np.asarray([int(sha1(X[
            rnd_idx, i].data.tobytes()).hexdigest(), 16) for i in range(m)])
    else:
        hash_value = np.asarray([int(sha1(bytes("".join(
            map(str, X[rnd_idx, i])).encode("utf8"))).hexdigest(), 16)
            for i in range(m)])

    idx = np.argsort(hash_value)
    hash_value = hash_value[idx]
    hash_names = names[idx]

    keys = np.unique(hash_value)
    occurrences = [hash_names[np.where(hash_value == key)] for key in keys]
    d = dict(zip(keys, occurrences))

    # loop through blocks of possible duplicated columns
    duplicated_columns = []
    lst_names = list(names)  # list.index faster than numpy.where
    for block in d.items():
        block_names = block[1]
        if len(block_names) > 1:
            # idx = [np.where(names == k) for k in block_names]
            idx = [lst_names.index(k) for k in block_names]
            block_values = X[:, idx]
            duplicated = find_duplicates_direct(block_names, block_values)
            if duplicated:
                duplicated_columns.append(duplicated)

    return sum(duplicated_columns, [])


def info_measure(x):
    """
    Pack binary array into blocks of 8 bits and multiple each block by its
    relative position.
    """
    u = np.packbits(x) + 1
    r = np.arange(1, len(u)+1)
    return np.dot(u, r)


def information_blocks(df, check_input=False, stats=True):
    """
    Detect information blocks using as a metric the level of available
    information. Common metrics based on entropy and alike are not appropriate
    for this purpose since do not distinguish among positions of 0 and 1.

    1. Generate indicator matrix (0: missing, 1: information).
    2. For each column pack binary array into blocks of 8 bits. Note that for
       each block there are 256 different 8-bit binary numbers.
    3. Compute metric: M=index(b)Tb, where b is the packed array of 8-bit
       binary numbers and index(b) denotes the position of block bi in packed
       array.
    4. Group columns based on M value.
    """
    if check_input:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a dataframe.")

    # level of information for each columns
    names = df.columns.values
    info_columns = [info_measure(pd.isnull(df[c])) for c in names]

    # generate blocks of candidates to be grouped
    keys = np.unique(info_columns)
    groups = [names[np.where(info_columns == key)] for key in keys]
    # sort blocks by size
    blocks = sorted(groups, key=len, reverse=True)

    if stats:
        # generate list with main metrics
        n_blocks = len(blocks)
        largest_block = len(blocks[0])
        ungrouped = sum(1 for b in blocks if len(b) == 1)
        block_stats = [n_blocks-ungrouped, largest_block, ungrouped]

        return blocks, block_stats
    else:
        return blocks


class HyperLogLog(object):
    """HyperLogLog algorithm."""
    def __init__(self,  p=16):
        self.p64 = 2 ** 64
        self.p = p
        self.m = 1 << p
        self.r = np.zeros(1 << p, dtype=np.int)

    def add(self, values):
        """Add values to register."""
        for x in values:
            if isinstance(x, int):
                bx = bytearray(struct.pack("d", x))
            elif isinstance(x, float):
                if float(x).is_integer():
                    bx = bytes(str(int(x)).encode("utf8"))
                else:
                    bx = bytearray(struct.pack("f", x))
            else:
                bx = bytes(str(x).encode("utf8"))

            x = int(sha1(bx).hexdigest(), 16)
            i = x & (self.m - 1)
            w = x >> self.p
            z = (160 - self.p + 1) - w.bit_length()

            self.r[i] = max(self.r[i], z)

    def count(self):
        """Calculate cardinality."""
        if self.m == 16:
            a = 0.673
        elif self.m == 32:
            a = 0.697
        elif self.m == 64:
            a = 0.709
        else:
            a = 0.7213 / (1 + 1.079 / self.m)

        E = a * self.m ** 2 / (1.0 / np.left_shift(1, self.r)).sum()
        if E <= 2.5 * self.m:
            V = np.count_nonzero(self.r == 0)
            if V:
                card = self.m * np.log(self.m / V)
            else:
                card = E
        elif E <= self.p64 / 30:
            card = E
        else:
            card = -self.p64 * np.log(1 - E / self.p64)

        return int(card)

    def merge(self, *hlls):
        """Merge HyperLogLog registers."""
        for hll in hlls:
            if hll.m != self.m:
                raise ValueError("registers must have equal size.")

            self.r = np.maximum(self.r, hll.r)
