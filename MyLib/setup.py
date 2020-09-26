from distutils.core import setup
from Cython.Build import cythonize
import os
import numpy
#os.environ['CFLAGS'] = '-O3 -Wall -std=c++11'
setup(
    name="MyGraph",
    ext_modules=cythonize(
        "MyGraph.pyx",
        language_level=3.6,
    ),
include_dirs = [numpy.get_include()]
)
