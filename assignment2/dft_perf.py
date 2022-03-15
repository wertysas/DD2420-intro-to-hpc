"""
Author: Johan Ericsson
Date: 2022-03-14
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from dft import DFT, DFT2
from time import perf_counter_ns as timer

SAVE_FIG = False


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
    mu = np.mean(X)

    # DFT2 timing
    t0 = timer()
    DFT2(xre, xim , Xre, Xim)
    dft2_time = timer() - t0

    return elapsed_time, elapsed_time_np, dft2_time


N = [8] + [64*i for i in range(1, 20)] # should be up to 16 (64*16=1024)
timings = []
timings_np = []
timings_dft2 = []

for n in N:
    t, tnp, t_dft2 =time_dft(n)
    timings.append(t*1e-9)
    timings_np.append(tnp*1e-9)
    timings_dft2.append(t_dft2*1e-9)

# plotting
rcParams.update({'font.size': 20})
fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlabel("Container size")
ax.set_ylabel("Execution time (s)")

ax.plot(N, timings, "o:", label='DFT list implementation', linewidth=2.0)
plt.plot(N, timings_dft2, "o:", label='DFT2 list implementation', linewidth=2.0)
ax.plot(N, timings_np, "o:", label='numpy.fft.fft', linewidth=2.0)
ax.legend()

#ax.set(yscale="log")
plt.grid(True, which="both")
ax.set_xticks(N, minor=False)
fig.suptitle("DFT execution times")
fig.tight_layout()

if SAVE_FIG:
    fig.savefig("figures/dft_execution_times.pdf", format="pdf")

plt.show()
