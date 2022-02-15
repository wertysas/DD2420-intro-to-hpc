"""
Author: Johan Ericsson
Date: 2022-01-26
"""
import numpy
import numpy as np
from timeit import default_timer as timer
import time
import matplotlib.pyplot as plt
"""
Note that default_timer is time.perf_counter()
whilst perf_counter_ns() exists which returns an int instead
"""


def checktick(timer_function):
    M = 200
    timesfound = np.empty((M,), dtype=numpy.int64)
    for i in range(M):
        t1 = timer_function()
        t2 = timer_function()
        while (t2 - t1) == 0:  # if zero then we are below clock granularity, retake timing
            t2 = timer_function()  # get timestamp from timer
        t1 = t2  # this is outside the loop
        timesfound[i] = t1  # record the time stamp
    Delta = np.diff(timesfound)
    minDelta = Delta.min()
    return minDelta


def checktick_float(timer_function):
    M = 200
    timesfound = np.empty((M,))
    for i in range(M):
        t1 = timer_function()
        t2 = timer_function()
        while (t2 - t1) < 1e-16:  # if zero then we are below clock granularity, retake timing
            t2 = timer_function()  # get timestamp from timer
        t1 = t2  # this is outside the loop
        timesfound[i] = t1  # record the time stamp
    Delta = np.diff(timesfound)
    minDelta = Delta.min()
    return int(minDelta * 1e9)


if __name__ == '__main__':
    N = 1000
    timings = [[], [], [], [], []]
    labels = ['time.time_ns', 'time.time', 'time.perf_counter_ns', 'time.perf_counter', 'timeit.default_timer']
    for _ in range(N):
        timings[0].append(checktick(time.time_ns))
        timings[1].append(checktick_float(time.time))
        timings[2].append(checktick(time.perf_counter_ns))
        timings[3].append(checktick_float(time.perf_counter))
        timings[4].append(checktick_float(timer))
    means = []
    print(len(timings))
    print(len(timings[0]))
    for times in timings:
        arr = np.array(times)
        means.append(np.median(arr))

    print(labels)
    print(means)
    plt.rcParams.update({'font.size': 20})
    plt.barh([i for i in range(5)], means) #, yerr=error)
    plt.yticks([i for i in range(5)], labels)
    plt.xlabel("measured granularity (ns)")
    plt.ylabel("Timer function")
    plt.title('Clock Granularity Measurements')
    plt.show()
