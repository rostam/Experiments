import scipy.sparse as sps
import pymetis
import scipy.io

A=scipy.io.mmread("grid1.mtx")
M = sps.csr_matrix(A)
adjacency_list = [M.getrow(i).indices for i in range(M.shape[0])]
num_clusters = 25
cuts, part_vert = pymetis.part_graph(num_clusters, adjacency=adjacency_list)
print(cuts)
print(part_vert)

