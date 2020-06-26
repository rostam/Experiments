import networkx as nx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import FindHomomorphisms as fh
import csv

for i in range(4, 11):
    with open("gminust_t" + str(i) + ".g6", "r") as f1:
        data = f1.read()
        i = 0
        with open("trianglefree_nontree_graphs_upto_10.csv", "w") as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(['graph6', 'NumOfV', 'NumOfE', 'Diameter', 'MinDeg', 'MaxDeg', 'Coloring'])
            for line in data.split("\n"):
                if line != "":
                    H = nx.from_graph6_bytes(bytes(line, 'ascii'))
                    num_tri = nx.triangles(H)
                    if sum(num_tri.values()) == 0:
                        maxdegree = sorted(H.degree, key=lambda x: x[1], reverse=True)[0][1]
                        mindegree = sorted(H.degree, key=lambda x: x[1])[0][1]
                        if maxdegree == mindegree:
                            col = max(nx.coloring.greedy_color(H, strategy='largest_first').values()) + 1
                            if H.number_of_edges() == 15:
                                csvwriter.writerow(
                                    [line, len(H), H.number_of_edges(), nx.diameter(H), mindegree, maxdegree, col])
