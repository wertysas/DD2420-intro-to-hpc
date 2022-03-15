"""
Author: Johan Ericsson
Date: 2022-02-14
"""

import numpy as np


def DFT(xre, xim, Xre, Xim):
    N = len(xre)
    for k in range(N):
        Xre[k] = 0
        Xim[k] = 0
        for n in range(N):
            Xre[k] += xre[n] * np.cos(k*2*np.pi*n/N) + xim[n]*np.sin(k*2*np.pi*n/N)
            Xim[k] += -xre[n] * np.sin(k*2*np.pi*n/N) + xim[n]*np.cos(k*2*np.pi*n/N)


def DFT2(xre, xim ,Xre, Xim):
    N = len(xre)
    w = 2*np.pi/N
    for k in range(N):
        Xre[k] = 0
        Xim[k] = 0
        wk = w*k
        for n in range(N):
            c = np.cos(wk*n)
            s = np.sin(wk*n)
            Xre[k] += xre[n]*c + xim[n]*s
            Xim[k] += -xre[n]*s + xim[n]*c



if __name__ == '__main__':
    N = 1024
    t = np.linspace(0, 1, N)
    s = np.random.normal(0, 0.1, N)
    xre = [np.sin(2 * np.pi * ti) + si for ti, si in zip(t, s)]
    xim = [np.cos(2 * np.pi * ti) + si for ti, si in zip(t, s)]

    # DFT calculation
    Xre = [0 for _ in range(N)]
    Xim = [0 for _ in range(N)]
    DFT2(xre, xim, Xre, Xim)
