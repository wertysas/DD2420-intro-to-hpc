"""
Author: Johan Ericsson
Date: 2022-02-14
"""
import numpy as np
import pytest
from dgemm import dgemm, dgemm_numpy


def test_dgemm():
    """
    Since I am a lazy coder I only do this for the list, since list and array.array objects share this routine.
    Bad practice! ^-^
    We also test for equality in a floating point calculation, that's a bit sketchy!
    """
    N = 256
    A = [1.0] * (N*N)
    B = [5.0] * (N*N)
    C = [0.0] * (N*N)
    dgemm(A, B, C, N)
    result = 5.0*N
    for i in range(N*N):
        assert C[i] == result, "dgemm result is wrong"


def test_dgemm_numpy():
    N = 256
    A = np.ones((N, N))
    B = 5.0 * np.ones((N, N))
    C = np.zeros((N, N))
    dgemm_numpy(A, B, C)
    result = np.ones((N,N))*5.0*N
    np.testing.assert_array_equal(C, result, "dgemm result is wrong")
