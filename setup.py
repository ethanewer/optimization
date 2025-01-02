from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import find_packages, setup

setup(
    name="optimization",
    version="1.0",
    packages=find_packages(),
    ext_modules=[
        Pybind11Extension(
            "lbfgs", ["optimization/lbfgs.cpp"], extra_compile_args=["-O3"]
        )
    ],
    cmdclass={"build_ext": build_ext},
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.26.4",
        "scipy>=1.14.1",
        "numba>=0.60.0",
    ],
    python_requires=">=3.6",
)


# export CXXFLAGS="-std=c++17"
# python setup.py build_ext --inplace
