"""
Author: Johan Ericsson
Date: 2022-02-14
"""

import numpy as np
from array import array
from time import perf_counter_ns as timer

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
    for j in range(STREAM_ARRAY_SIZE):
        c[j] = a[j]
    times[0] = timer() - times[0]

    # scale
    times[1] = timer()
    for j in range(STREAM_ARRAY_SIZE):
        b[j] = scalar * c[j]
    times[1] = timer() - times[1]
    # sum
    times[2] = timer()
    for j in range(STREAM_ARRAY_SIZE):
        c[j] = a[j] + b[j]
    times[2] = timer() - times[2]

    # triad
    times[3] = timer()
    for j in range(STREAM_ARRAY_SIZE):
        a[j] = b[j] + scalar * c[j]
    times[3] = timer() - times[3]

    return times


def benchmark_numpy(a, b, c, STREAM_ARRAY_SIZE, scalar_=2.0):
    scalar = scalar_
    times = [0] * 4
    # copy
    times[0] = timer()
    np.copyto(c, a)
    times[0] = timer() - times[0]

    # scale
    times[1] = timer()
    b = scalar*c
    times[1] = timer() - times[1]
    # sum
    times[2] = timer()
    c = a + b 
    times[2] = timer() - times[2]

    # triad
    times[3] = timer()
    a = b + scalar * c
    times[3] = timer() - times[3]

    return times    # list of times [copy, scale, sum, triad]
    

def compute_bandwidths(times, type_size, array_length):
    # data sizes computations
    copy_data = 2 * STREAM_ARRAY_TYPE_SIZE * STREAM_ARRAY_SIZE
    add_data = 2 * STREAM_ARRAY_TYPE_SIZE * STREAM_ARRAY_SIZE
    scale_data = 3 * STREAM_ARRAY_TYPE_SIZE * STREAM_ARRAY_SIZE
    triad_data = 3 * STREAM_ARRAY_TYPE_SIZE * STREAM_ARRAY_SIZE


if __name__ == '__main__':
    STREAM_ARRAY_TYPE_SIZE = 8  # float64 = 8bytes
    vector_length = [2**(2*i) for i in range(10)]  # we grow the size exponentially
    times_list = []
    times_array = []
    times_np_array_np = []
    for l in vector_length:
        STREAM_ARRAY_SIZE = l
        a, b, c = initialise_lists(STREAM_ARRAY_SIZE)
        times_list.append(benchmark(a, b, c, STREAM_ARRAY_SIZE))

        a, b, c = initialise_arrays(STREAM_ARRAY_SIZE)
        times_array.append(benchmark(a, b, c, STREAM_ARRAY_SIZE))

        # a, b, c = initialise_np_arrays(STREAM_ARRAY_SIZE)
        # times_np_array = benchmark(a, b, c, STREAM_ARRAY_SIZE)

        a, b, c = initialise_np_arrays(STREAM_ARRAY_SIZE)
        times_np_array_np.append(benchmark_numpy(a, b, c, STREAM_ARRAY_SIZE))



        print("list time:", times_list)
        print("time array:", times_array)
        #print("time numpy array looping", times_np_array)
        print("time numpy array np kernels", times_np_array_np)

        # Bandwidth bytes/s = 10^9 * data_size / time_ns => Mbytes/s = 10^3 * data_size / time_ns

