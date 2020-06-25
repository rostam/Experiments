# import matplotlib.pyplot as plt
# # import networkx as nx
# # import itertools as iter
#
# G = nx.petersen_graph()
# H = nx.complete_graph(3)
# G_nodes = G.nodes()
# H_nodes = H.nodes()
#
# # for n_G in G_nodes:
# #     for n_H in H_nodes:
#
# all_combs = iter.combinations(G_nodes, len(H_nodes))
# found = []
# for c in all_combs:
#     possible_edges = iter.combinations(c, 2)
#     no_problem = True
#     for e in possible_edges:
#         if G.has_edge(e[0], e[1]):
#             if not H.has_edge(e[0], e[1]):
#                 no_problem = False
#                 break
#     if no_problem:
#         found = c
#         break
# for n in c:
#     print(n)
#
#
# # nx.draw(H)
# # plt.draw()
# # plt.show()
