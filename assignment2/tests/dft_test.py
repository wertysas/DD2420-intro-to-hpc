"""
Author: Johan Ericsson
Date: 2022-03-13
"""

import numpy as np
import pytest
from assignment2.dft import DFT


def test_dft(N=1024):
    # Test signal a complex wave
    t = np.linspace(0, 1, N)
    s = np.random.normal(0, 0.1, N)
    xre = [np.sin(2*np.pi*ti) + si for ti, si in zip(t, s)]
    xim = [np.cos(2*np.pi*ti) + si for ti, si in zip(t, s)]

    # DFT calculation
    Xre = [0 for _ in range(N)]
    Xim = [0 for _ in range(N)]
    DFT(xre, xim, Xre, Xim)

    # numpy.fft reference
    x = np.array(xre) + np.array(xim)*1j
    Xref = np.fft.fft(x)

    np.testing.assert_almost_equal(Xre, Xref.real, decimal=10, err_msg="Error real parts differ")
    np.testing.assert_almost_equal(Xim, Xref.imag, decimal=10, err_msg="Error imaginary parts differ")
