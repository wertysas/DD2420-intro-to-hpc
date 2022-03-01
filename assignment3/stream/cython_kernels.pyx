"""
This code contains the cython versions of the STREAM kernels
"""


import numpy as np
cimport numpy as np

# List and array versions
def copy_list(a, c, unsigned int STREAM_ARRAY_SIZE):
    cdef unsigned int j
    for j in range(STREAM_ARRAY_SIZE):
        c[j] = a[j]


def scale_list(b, c, unsigned int STREAM_ARRAY_SIZE, double scalar=2.0):
    cdef unsigned int j
    for j in range(STREAM_ARRAY_SIZE):
        b[j] = scalar * c[j]


def sum_list(a, b, c, unsigned int STREAM_ARRAY_SIZE):
    cdef unsigned int j
    for j in range(STREAM_ARRAY_SIZE):
        c[j] = a[j] + b[j]


def triad_list(a, b, c, unsigned int STREAM_ARRAY_SIZE, double scalar=2.0):
    cdef unsigned int j
    for j in range(STREAM_ARRAY_SIZE):
        a[j] = b[j] + scalar * c[j]


# Numpy cythonized kernels

def copy_numpy(double[:] a, double[:] c, double scalar=2.0):
    np.multiply(scalar, a, out=c)


def scale_numpy(double[:] b, double[:] c, double scalar=2.0):
    np.multiply(scalar, c, out=b)


def sum_numpy(double[:] a, double[:] b, double[:] c, double scalar=2.0):
    np.add(a, b, out=c) 


def triad_numpy(double[:] a, double[:] b, double[:] c, double scalar=2.0):
    np.add(b, np.multiply(scalar, c), out=a)

