from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include 
from numpy.distutils.misc_util import get_info
npymath_info = get_info("npymath")


ext = Extension("_hmmc", sources=["_hmmc.pyx"], **npymath_info)
setup(name="_hmmc", ext_modules=cythonize([ext]))
