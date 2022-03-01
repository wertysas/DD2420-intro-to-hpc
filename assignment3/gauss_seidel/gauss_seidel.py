"""
Gauss-Seidel For Poisson Solver
"""

import numpy as np
from time import perf_counter_ns as timer
import matplotlib.pyplot as plt
from matplotlib import rcParams
import cython_gs


def gauss_seidel(f):
    newf = f.copy()
    for i in range(1, newf.shape[0]-1):
        for j in range(1, newf.shape[1]-1):
            newf[i,j] = 0.25 * (newf[i,j+1] + newf[i,j-1] +
                                newf[i+1,j] + newf[i-1,j])

    return newf


def gauss_seidel_pycollection(f):
    newf = f.copy()
    for i in range(1,len(newf)-1):
        for j in range(1,len(newf[i])-1):
            newf[i][j] = 0.25 * (newf[i][j+1] + newf[i][j-1] +
                                newf[i+1][j] + newf[i-1][j])

    return newf


def benchmark_list(grid_size: tuple, iterations=1000):
    # initialisation of f
    f = []
    for i in range(grid_size[0]):
        row = [0.5] * grid_size[1]
        f.append(row)
    # timing
    t0 = timer()
    for _ in range(iterations):
        f = gauss_seidel_pycollection(f)
    time = timer()-t0

    return time


def benchmark_numpy(grid_size: tuple, iterations=1000):
    # initialisation of f
    f = np.ones(shape=(grid_size))
    # timing
    t0 = timer()
    for _ in range(iterations):
        f = gauss_seidel(f)
    time = timer() - t0

    return time


def benchmark_list_cython(grid_size: tuple, iterations=1000):
    # initialisation of f
    f = []
    for i in range(grid_size[0]):
        row = [0.5] * grid_size[1]
        f.append(row)
    # timing
    t0 = timer()
    for _ in range(iterations):
        f = cython_gs.gauss_seidel_pycollection(f)
    time = timer()-t0

    return time


def benchmark_numpy_cython(grid_size: tuple, iterations=1000):
    # initialisation of f
    f = np.ones(shape=(grid_size))
    # timing
    t0 = timer()
    for _ in range(iterations):
        f = cython_gs.gauss_seidel(f)
    time = timer() - t0

    return time

if __name__ == '__main__':
    plot = True
    N = [8, 16, 32, 64, 128, 256]
    grid_sizes = [(n, n) for n in N]
    list_timings = []
    np_timings = []
    list_timings_cython = []
    np_timings_cython = []
    for grid_size in grid_sizes:
        # list timing
        t = benchmark_list(grid_size)
        list_timings.append(t)
        # np timing
        t = benchmark_numpy(grid_size)
        np_timings.append(t)
        # list cython timing
        t = benchmark_list_cython(grid_size)
        list_timings_cython.append(t)
        # np timing
        t = benchmark_numpy_cython(grid_size)
        np_timings_cython.append(t)

    print(N)
    print(list_timings)
    print(np_timings)

    if plot:
        rcParams.update({'font.size': 20})

        fig, ax = plt.subplots(figsize=(16, 10))
        title = 'Gauss Seidel Execution time'
        ax.set_xlabel("Grid size, $N$ ")
        ax.set_ylabel("Exection Time (ns)")
        ax.set(yscale="log")
        plt.grid(True, which="both")
        ax.plot(N, list_timings, "o:", linewidth=2.0, label="list", color='r')
        ax.plot(N, np_timings, "o:", linewidth=2.0, label="numpy array", color='b')
        ax.plot(N, list_timings_cython, "^:", linewidth=2.0, label="list (cython)", color='r')
        ax.plot(N, np_timings_cython, "^:", linewidth=2.0, label="numpy array (cython)", color='b')
        ax.legend()
        ax.set_xticks(N, minor=True)
        fig.suptitle(title)
        fig.tight_layout()
        plt.show()
        fig.savefig("figures/gauss_seidel.pdf", format="pdf")
