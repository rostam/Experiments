import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import FindHomomorphisms as fh
for i in range(2, 11):
    with open("g" + str(i) + ".g6", "r") as f1:
        gs = f1.read().split("\n")
        with open("t" + str(i) + ".g6", "r") as f2:
            ts = f2.read().split("\n")

            sgs = set(gs)
            sts = set(ts)
            g_minus_t = sgs.difference(sts)
            with open("gminust" + str(i) + ".g6", "w") as f2:
                for g in g_minus_t:
                    f2.write(g+"\n")
