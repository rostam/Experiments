import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import FindHomomorphisms as fh


# f = plt.figure()
# peterson = nx.petersen_graph()
# g = nx.from_graph6_bytes(b'Bw')
# part = fh.find_a_homomorphism(peterson, g)
# print(part)
# nx.draw(g, ax=f.add_subplot(111))
# f.savefig("res.png")

gs = []
for i in range(2, 8):
    with open("g" + str(i) + ".g6", "r") as f:
        data = f.read()
        for line in data.split("\n"):
            if line != "":
                gs.append(line)

# print(",".join(gs))
print(len(gs))