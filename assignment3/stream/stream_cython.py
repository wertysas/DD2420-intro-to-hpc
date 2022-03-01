"""
Author: Johan Ericsson
Date: 2022-02-14
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from array import array
from time import perf_counter_ns as timer
import pandas as pd
import cython_kernels

"""
We use arrays of doubles i.e. 8bytes = 64bit this is since python's builtin type float is a 64 bit float
to ensure all collections benchmarked hold data of the same size.
"""


def initialise_lists(STREAM_ARRAY_SIZE):
    a = [1.0 for _ in range(STREAM_ARRAY_SIZE)]
    b = [2.0 for _ in range(STREAM_ARRAY_SIZE)]
    c = [0.0 for _ in range(STREAM_ARRAY_SIZE)]
    return a, b, c


def initialise_arrays(STREAM_ARRAY_SIZE):
    a = array('d', [1.0 for _ in range(STREAM_ARRAY_SIZE)])
    b = array('d', [2.0 for _ in range(STREAM_ARRAY_SIZE)])
    c = array('d', [0.0 for _ in range(STREAM_ARRAY_SIZE)])
    return a, b, c


def initialise_np_arrays(STREAM_ARRAY_SIZE):
    a = 1.0 * np.ones(STREAM_ARRAY_SIZE)     # np.ones returns an array of float64 by default
    b = 2.0 * np.ones(STREAM_ARRAY_SIZE)
    c = np.zeros(STREAM_ARRAY_SIZE)
    return a, b, c


def benchmark(a, b, c, STREAM_ARRAY_SIZE, scalar_=2.0):
    scalar = scalar_
    times = [0] * 4

    # copy
    times[0] = timer()
    cython_kernels.copy_list(a, c, STREAM_ARRAY_SIZE)
    times[0] = timer() - times[0]

    # scale
    times[1] = timer()
    cython_kernels.scale_list(b, c, STREAM_ARRAY_SIZE)
    times[1] = timer() - times[1]
    # sum
    times[2] = timer()
    cython_kernels.sum_list(a, b, c, STREAM_ARRAY_SIZE)
    times[2] = timer() - times[2]

    # triad
    times[3] = timer()
    cython_kernels.triad_list(a, b, c, STREAM_ARRAY_SIZE)
    times[3] = timer() - times[3]

    return times


def benchmark_numpy(a, b, c, scalar_=2.0):
    scalar = scalar_
    times = [0] * 4
    # copy
    times[0] = timer()
    # cython_kernels.copy_numpy(a, c)
    np.multiply(scalar, a, out=c)
    times[0] = timer() - times[0]

    # scale
    times[1] = timer()
    # cython_kernels.scale_numpy(b, c)
    np.multiply(scalar, c, out=b)
    times[1] = timer() - times[1]

    # sum
    times[2] = timer()
    # cython_kernels.sum_numpy(a, b, c)
    np.add(a, b, out=c)
    times[2] = timer() - times[2]

    # triad
    times[3] = timer()
    # cython_kernels.triad_numpy(a, b, c)
    np.add(b, np.multiply(scalar, c), out=a)
    times[3] = timer() - times[3]

    return times    # list of times [copy, scale, sum, triad]


def compute_bandwidths(timings, type_size, array_length, dfs, container):
    """
    @parameter timing: measured time in list [copy_time, add_time, scale_time, triad_time]
    """
    dfs['copy'][container][array_length] = 2 * type_size * array_length * 1e3 / timings[0]
    dfs['scale'][container][array_length] = 2 * type_size * array_length * 1e3 / timings[1]
    dfs['add'][container][array_length] = 3 * type_size * array_length * 1e3 / timings[2]
    dfs['triad'][container][array_length] = 3 * type_size * array_length * 1e3 / timings[3]


if __name__ == '__main__':
    STREAM_ARRAY_TYPE_SIZE = 8  # float64 = 8bytes
    vector_length = [2**i for i in range(4, 16)]  # we grow the size exponentially
    stream_types = ('copy', 'scale', 'add', 'triad')
    containers = ('list', 'array.array', 'np.ndarray')
    data_frames = {}
    for s in stream_types:
        data_frames[s] = pd.DataFrame(index=vector_length, columns=containers)
    for l in vector_length:
        STREAM_ARRAY_SIZE = l

        # List bandwidth computation
        a, b, c = initialise_lists(STREAM_ARRAY_SIZE)
        timings = benchmark(a, b, c, STREAM_ARRAY_SIZE)
        compute_bandwidths(timings, STREAM_ARRAY_TYPE_SIZE, STREAM_ARRAY_SIZE, data_frames, container='list')
        # array bandwidth computation
        a, b, c = initialise_arrays(STREAM_ARRAY_SIZE)
        timings = benchmark(a, b, c, STREAM_ARRAY_SIZE)
        compute_bandwidths(timings, STREAM_ARRAY_TYPE_SIZE, STREAM_ARRAY_SIZE, data_frames, container='array.array')

        # numpy bandwidth computation
        a, b, c = initialise_np_arrays(STREAM_ARRAY_SIZE)
        timings = benchmark_numpy(a, b, c)
        compute_bandwidths(timings, STREAM_ARRAY_TYPE_SIZE, STREAM_ARRAY_SIZE, data_frames, container='np.ndarray')

    # Bandwidth bytes/s = 10^9 * data_size / time_ns => Mbytes/s = 10^3 * data_size / time_ns

    rcParams.update({'font.size': 20})
    colors = ['r', 'g', 'b']
    sns.color_palette("Set2", 3)
    x = 'Container size (number of objects)'
    y = 'Bandwidth (Mbytes/s)'
    for stream_type, df in data_frames.items():
        fig, ax = plt.subplots(figsize=(16, 10))
        title = 'Cythonized STREAM BENCHMARK: ' + stream_type
        ax.set_xlabel("Container size")
        ax.set_ylabel("Bandwidth (Mbyte/s)")
        ax.set(yscale="log")
        plt.grid(True, which="both")
        lines = ax.plot(df.index, df.values, "C1o:", linewidth=2.0, label=df.columns)
        for col, line in zip(colors, lines):
            line.set_color(col)
        ax.legend()
        ax.set_xticks(df.index, minor=True)
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig("figures/stream_"+stream_type+"_cython.pdf", format="pdf")
    plt.show()