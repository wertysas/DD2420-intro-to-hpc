"""
Author: Johan Ericsson
Date: 2022-02-14
"""

from time import perf_counter_ns as timer
from array import array
import numpy as np
import matplotlib.pyplot as plt


def dgemm(A, B, C, N):
    """
    DGEMM writes C = C + A*B where A,B,C are NxN matrices.
    We assume that the matrix are stores in one contiguous list/array C-style
    i.e. M_ij = M[i*N + j]
    """
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i*N + j] += A[i*N + k] * B[k*N + j]


def dgemm_numpy(A, B, C):
     C += A.dot(B)


def time_dgemm_list(N):
    A = [1.0] * (N*N)
    B = [5.0] * (N*N)
    C = [0.0] * (N*N)
    t0 = timer()
    dgemm(A, B, C, N)
    t1 = timer()
    return t1 - t0


def time_dgemm_array(N):
    A = array('d', [1.0 for _ in range(N*N)])
    B = array('d', [5.0 for _ in range(N*N)])
    C = array('d', [0.0 for _ in range(N*N)])
    t0 = timer()
    dgemm(A, B, C, N)
    t1 = timer()
    return t1 - t0

def time_dgemm_numpy(N):
    A = np.ones((N, N))
    B = 2 * np.ones((N, N))
    C = np.zeros((N, N))
    t0 = timer()
    dgemm_numpy(A, B, C)
    t1 = timer()
    return t1 - t0


def get_flops_dgemm(time_millis, N):
    return 2 * N**3 * 1e3 / time_millis


if __name__ == '__main__':
    sizes = [16, 32, 64, 128, 192, 256, 320, 384, 448, 512] #[2**i for i in range(4, 10)]
    means_list = []
    var_list = []
    means_array = []
    var_array = []
    means_np_array = []
    var_np_array = []
    for N in sizes:
        print(N)
        list_t = []
        array_t = []
        np_array_t = []
        for j in range(10):
            list_t.append(time_dgemm_list(N))
            array_t.append(time_dgemm_array(N))
            np_array_t.append(time_dgemm_numpy(N))
        x = np.array(list_t) * 1e-6 # We scale from ns to ms
        means_list.append(x.mean())
        var_list.append(x.std())
        x = np.array(array_t) * 1e-6
        means_array.append(x.mean())
        var_array.append(x.std())
        x = np.array(np_array_t) * 1e-6
        means_np_array.append(x.mean())
        var_np_array.append(x.std())

    # Execution time plots
    fig, ax = plt.subplots()
    ax.errorbar(sizes, means_list, yerr=var_list, label='python list', capsize=3)
    ax.errorbar(sizes, means_array, yerr=var_array, label='python array', capsize=3)
    ax.errorbar(sizes, means_np_array, yerr=var_np_array, label='numpy array', capsize=3)
    plt.legend()
    plt.yscale("log")
    plt.title("Matrix DGEMM execution times")
    plt.ylabel("Execution time (milliseconds)")
    plt.xlabel("Matrix dimension N")
    plt.grid(which="both")
    plt.show()

    # FLOPS plot
    fig, ax = plt.subplots()
    ax.plot(sizes, [get_flops_dgemm(means_list[i], sizes[i]) for i in range(len(sizes))], label='python list')
    ax.plot(sizes, [get_flops_dgemm(means_array[i], sizes[i]) for i in range(len(sizes))], label='python array')
    ax.plot(sizes, [get_flops_dgemm(means_np_array[i], sizes[i]) for i in range(len(sizes))], label='numpy array')
    ax.plot(sizes, [5*1e9 for _ in range(len(sizes))], linestyle='dotted', label='theoretical peak (single core')
    plt.legend()
    plt.yscale("log")
    plt.title("Matrix DGEMM FLOPS")
    plt.ylabel("FLOPS")
    plt.xlabel("Matrix dimension N")
    plt.grid(which="both")
    plt.show()
