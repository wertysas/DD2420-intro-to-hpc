"""
Author: Johan Ericsson
Date: 2022-02-14
"""

import numpy as np


def DFT(xre, xim, Xre, Xim):
    assert len(xre) == len(xim), "Error input dimensions does not agree"
    N = len(xre)
    for k in range(N):
        Xre[k] = 0
        Xim[k] = 0
        for n in range(N):
            Xre[k] += xre[n] * np.cos(k*2*np.pi*n/N) + xim[n]*np.sin(k*2*np.pi*n/N)
            Xim[k] += -xre[n] * np.sin(k*2*np.pi*n/N) + xim[n]*np.cos(k*2*np.pi*n/N)

