"""
Cython versions of Gauss-Seidel Solvers
"""

import numpy as np
cimport numpy as np

def gauss_seidel(double[:, :] f):
    cdef double[:, :] newf = f.copy()
    cdef unsigned int i, j
    for i in range(1, newf.shape[0]-1):
        for j in range(1, newf.shape[1]-1):
            newf[i,j] = 0.25 * (newf[i,j+1] + newf[i,j-1] +
                                newf[i+1,j] + newf[i-1,j])

    return newf


def gauss_seidel_pycollection(f):
    newf = f.copy()
    cdef unsigned int i, j
    for i in range(1,len(newf)-1):
        for j in range(1,len(newf[i])-1):
            newf[i][j] = 0.25 * (newf[i][j+1] + newf[i][j-1] +
                                newf[i+1][j] + newf[i-1][j])

    return newf
