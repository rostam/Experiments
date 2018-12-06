import numpy as np
from scipy.sparse import csc_matrix, linalg as sla
import scipy.io as sio
from matplotlib import pyplot as plt


class gmres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0

    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))


# A = csc_matrix([[1, 2, 0, 4], [1, 0, 0, 1], [1, 0, 2, 1], [2, 2, 1, 0.]])
A = sio.mmread("ex11.mtx")
A = csc_matrix(A)
# plt.spy(A)
# plt.show()
lu = sla.spilu(A)
M_x = lambda x: lu.solve(x)
M = sla.LinearOperator((16614, 16614), M_x)

counter = gmres_counter()

res = sla.gmres(A, np.ones(16614), M=M, callback=counter)
print(res)
print(counter.niter)
# print(lu.L.A)
