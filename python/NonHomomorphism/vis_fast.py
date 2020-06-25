import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

f = plt.figure()
g = nx.from_graph6_bytes(b'Bw')
nx.draw(g, ax=f.add_subplot(111))
f.savefig("res.png")