from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='Testing stuff',
    ext_modules=cythonize([".\statistical_parts\math_parts\cython_wmw_functions.py",
                           ".\statistical_parts\math_parts\wmw_exact_power_cython.py"],
                          language_level="3", annotate=True),
    zip_safe=False,
    include_dirs=[numpy.get_include()]
)
