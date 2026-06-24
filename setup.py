from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import pybind11

class get_pybind_include:
    def __str__(self):
        return pybind11.get_include()

ext_modules = [
    Extension(
        'core.sim_core',
        ['cpp/sim_core.cpp'],
        include_dirs=[get_pybind_include()],
        language='c++',
        extra_compile_args=['-O3', '-fopenmp'],
        extra_link_args=['-fopenmp'],
    ),
]

setup(
    name='drugnandan',
    version='0.1',
    author='Nandan',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
)