from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "sim_core",
        ["core/sim_core.cpp"],
        include_dirs=["core"],
        language="c++",
    ),
]

setup(
    name="sim_core",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)