import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import FindHomomorphisms as fh
for i in range(4, 11):
    with open("nontri" + str(i) + ".g6", "r") as f1:
        gs = f1.read().split("\n")
        with open("nontri_t" + str(i) + ".g6", "r") as f2:
            ts = f2.read().split("\n")

            sgs = set(gs)
            sts = set(ts)
            g_minus_t = sgs.difference(sts)
            with open("gminust_t" + str(i) + ".g6", "w") as f2:
                for g in g_minus_t:
                    f2.write(g+"\n")
