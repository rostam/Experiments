import matplotlib.pylab as plt
import scipy
import scipy.sparse as sps
import pymetis
import numpy as np
import scipy.io
import networkx as nx

A=scipy.io.mmread("grid1.mtx")
M = sps.csr_matrix(A)
adjacency_list = [M.getrow(i).indices for i in range(M.shape[0])]
#node_nd = pymetis.nested_dissection(adjacency=adjacency_list)
num_clusters = 25
cuts, part_vert = pymetis.part_graph(num_clusters, adjacency=adjacency_list)
print(cuts)
print(part_vert)
#perm, iperm = np.array(node_nd[0]), np.array(node_nd[1])
#print M.todense()
#print perm
#plt.spy(M)
#plt.show()
#print(M.indices)
#N = M.todense()
#N[:,:] = N[perm,:]
#N[:,:] = N[:,perm]
#plt.spy(N)
#plt.show()
#G = nx.Graph(N)
#nx.draw(G,pos=nx.spring_layout(G))
#plt.draw()
#plt.show()

