"""Build Cython extensions in-place for development."""

import os

import numpy as np
import scipy
from Cython.Build import cythonize
from setuptools import Extension, setup

scipy_include = os.path.dirname(os.path.dirname(scipy.linalg.__file__))

setup(
    ext_modules=cythonize(
        [
            Extension(
                "tenax.contraction._cython_blas",
                ["src/tenax/contraction/_cython_blas.pyx"],
                include_dirs=[np.get_include(), scipy_include],
            )
        ]
    ),
    package_dir={"": "src"},
)
