"""
Author: Johan Ericsson
Date: 2022-03-14
"""

import numpy as np
import matplotlib.pyplot as plt
from assignment2.dft import DFT
from time import perf_counter_ns as timer


def time_dft(n):
    # generation of a wave with a random error term
    t = np.linspace(0, 1, n)
    s = np.random.normal(0, 0.1, n)
    xre = [np.sin(2*np.pi*ti) + si for ti, si in zip(t, s)]
    xim = [np.cos(2*np.pi*ti) + si for ti, si in zip(t, s)]

    # DFT arrays
    Xre = [0 for _ in range(n)]
    Xim = [0 for _ in range(n)]

    # complex np array
    xnp = np.array(xre) + np.array(xim)*1j

    # DFT list implementation timing
    t0 = timer()
    DFT(xre, xim, Xre, Xim)
    elapsed_time = timer() - t0

    # np.fft.fft timing
    t0 = timer()
    X = np.fft.fft(xnp)
    elapsed_time_np = timer() - t0
    return elapsed_time, elapsed_time_np


N = [8, 16, 32, 64] + [64*i for i in range(2, 16)]
timings = []
timings_np = []

for n in N:
    t, tnp =time_dft(n)
    timings.append(t)
    timings_np.append(tnp)

plt.plot(N, np.log2(timings), label='DFT list implementation')
plt.plot(N, np.log2(timings_np), label='numpy.fft.fft')
plt.legend()
plt.show()
