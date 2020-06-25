import unittest
import networkx as nx
import FindHomomorphisms as fh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class TestStringMethods(unittest.TestCase):
    def test_there_is_homomorphism_from_peterson_to_different_graphs(self):
        peterson = nx.petersen_graph()
        complete3 = nx.complete_graph(3)
        complete4 = nx.complete_graph(4)
        cycle4 = nx.cycle_graph(4)
        part = fh.find_a_homomorphism(peterson, complete3)
        self.assertTrue(len(part) >= 1)
        part = fh.find_a_homomorphism(peterson, complete4)
        self.assertTrue(len(part) >= 1)
        part = fh.find_a_homomorphism(peterson, cycle4)
        self.assertTrue(len(part) == 0)

    def test_there_is_homomorphism_from_peterson_to_all_graphs_upto__vertices(self):
        peterson = nx.petersen_graph()
        for i in range(2, 6):
            with open("g" + str(i) + ".g6", "r") as f:
                data = f.read()
                i = 0
                for line in data.split("\n"):
                    if line != "":
                        # print(line)
                        H = nx.from_graph6_bytes(bytes(line,'ascii'))
                        part = fh.find_a_homomorphism(peterson, H)
                        # self.assertTrue(len(part) == 0)

                        if len(part) != 0:
                            print(line)
                            f = plt.figure()
                            nx.draw(H, ax=f.add_subplot(111))
                            f.savefig(str(i)+".png")
                            i = i + 1


if __name__ == '__main__':
    unittest.main()
