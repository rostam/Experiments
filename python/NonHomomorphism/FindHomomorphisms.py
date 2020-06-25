import more_itertools as miter
import itertools as iter
import networkx as nx
import string

# import matplotlib.pyplot as plt

# G = nx.read_graph6('G.g6')
# H = nx.read_graph6('H.g6')
H = nx.petersen_graph()
G = nx.complete_graph(4)


def find_a_homomorphism(G, H, num_of_subsets):
    G_nodes = G.nodes()
    H_nodes = H.nodes()

    size_of_set = 0
    # num_of_subsets = 0
    if len(G_nodes) >= len(H_nodes):
        size_of_set = len(G_nodes)
        # num_of_subsets = len(H_nodes)
    else:
        size_of_set = len(G_nodes)
        # num_of_subsets = len(G_nodes)

    iterable = string.ascii_lowercase[0:size_of_set]
    is_homomorphism_found = False

    for part in miter.set_partitions(iterable, num_of_subsets):
        is_correct_homomorphism = True
        if is_homomorphism_found:
            break
        for p in part:
            if not is_correct_homomorphism:
                break
            if len(p) == 1:
                continue
            all_combs = iter.combinations(p, 2)
            for a_comb in all_combs:
                v1 = ord(a_comb[0]) - 97
                v2 = ord(a_comb[1]) - 97
                if G.has_edge(v1, v2):
                    is_correct_homomorphism = False
                    break

        if is_correct_homomorphism:
            all_combs = iter.combinations(range(0, len(part)), 2)
            for a_comb in all_combs:
                if is_correct_homomorphism:
                    v1H = a_comb[0]
                    v2H = a_comb[1]
                    if not H.has_edge(v1H, v2H):
                        for p1 in part[a_comb[0]]:
                            if is_correct_homomorphism:
                                for p2 in part[a_comb[1]]:
                                    v1G = ord(p1) - 97
                                    v2G = ord(p2) - 97
                                    if G.has_edge(v1G, v2G):
                                        is_correct_homomorphism = False
                                        break

        if is_correct_homomorphism:
            is_homomorphism_found = True
            return part
            # with open("homomorphism_G_H", 'w') as f:
            #
            #     for p in part:
            #         for a in p:
            #             f.write(a)
            #         f.write(' ')
            #     f.flush()
            #
            # # print([''.join(p) for p in part])
            # break

    if not is_homomorphism_found:
        return []
        # with open("homomorphism_G_H", 'w') as f:
        #     f.write("not found")
        #     f.flush()
        #     f.close()

    # nx.draw(G, with_labels=True)
    # plt.draw()
    # plt.show()


def handle_one_g6_string(G, g6_string_H):
    H = nx.from_graph6_bytes(bytes(g6_string_H, 'ascii'))
    part = []
    for j in range(2, len(H) + 1):
        part = find_a_homomorphism(G, H, j)
        if len(part) != 0:
            break

    return part
    # print(line, len(part))
    # self.assertTrue(len(part) == 0)
    # if len(part) == 0:
        # print(line)
        # f = plt.figure()
        # nx.draw(H, ax=f.add_subplot(111))
        # f.savefig(line + ".png")
        # i = i + 1
        # return
