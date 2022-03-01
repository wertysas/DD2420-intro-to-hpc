"""
Setup file for cython stream version
"""

from distutils.core import setup
from Cython.Build import cythonize
import numpy


setup(ext_modules=cythonize("cython_kernels.pyx",
    compiler_directives={"language_level": "3"}), include_dirs=[numpy.get_include()])

