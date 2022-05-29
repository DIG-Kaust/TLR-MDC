from setuptools import setup, find_packages
import pathlib
import os

setup(
    name='mdctlr',
    packages=[
        'mdctlr',
        'mdctlr/inversiondist',
        'mdctlr/tlrmvm'
        ],
    install_requires=[
    "numpy",
    "scipy",
    "matplotlib",
    "pymorton",
    "hilbertcurve",
    "mpi4py",
    "pylops"
    ],
    version='0.0.1'
)
