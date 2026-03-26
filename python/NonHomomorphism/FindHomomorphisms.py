import more_itertools as miter
import itertools as iter
import networkx as nx
import string

H = nx.petersen_graph()
G = nx.complete_graph(4)


def _char_to_node(c):
    return ord(c) - 97


def _partition_is_independent(G, partition):
    """Check that no two nodes in the same partition cell are adjacent in G."""
    for cell in partition:
        if len(cell) == 1:
            continue
        for u, v in iter.combinations(cell, 2):
            if G.has_edge(_char_to_node(u), _char_to_node(v)):
                return False
    return True


def _partition_respects_non_edges(G, H, partition):
    """Check that non-edges in H don't map to edges in G."""
    for i, j in iter.combinations(range(len(partition)), 2):
        if H.has_edge(i, j):
            continue
        for u in partition[i]:
            for v in partition[j]:
                if G.has_edge(_char_to_node(u), _char_to_node(v)):
                    return False
    return True


def find_a_homomorphism(G, H, num_of_subsets):
    size_of_set = len(G.nodes())
    iterable = string.ascii_lowercase[:size_of_set]

    for partition in miter.set_partitions(iterable, num_of_subsets):
        if _partition_is_independent(G, partition) and _partition_respects_non_edges(G, H, partition):
            return partition

    return []


def handle_one_g6_string(G, g6_string_H):
    H = nx.from_graph6_bytes(bytes(g6_string_H, 'ascii'))
    for j in range(2, len(H) + 1):
        partition = find_a_homomorphism(G, H, j)
        if partition:
            return partition
    return []
