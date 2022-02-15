"""
Author: Johan Ericsson
Date: 2022-02-15
"""

from dgemm import dgemm, dgemm_numpy
import numpy as np


N = 256

# List impl
A = [1.0] * (N * N)
B = [5.0] * (N * N)
C = [0.0] * (N * N)
D = [10.0] * (66000) # Should flush the cache
dgemm(A, B, C, N)

# numpy impl
# A = np.ones((N, N))
# B = 5.0 * np.ones((N, N))
# C = np.zeros((N, N))
# D = np.ones(66000) # cache flush
# dgemm_numpy(A, B, C)

