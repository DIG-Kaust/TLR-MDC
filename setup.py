from setuptools import setup, find_packages
import pathlib
import os

setup(
    name='mdctlr',
    packages=[
        'mdctlr',
        'mdctlr/inversion',
        'mdctlr/tlrmvm'
        ],
    entry_points = {
        'console_scripts': [
            'mdctlr.caldatasize = mdctlr.calculatedatasize:main',
        ],
    },
    install_requires=[
    "numpy",
    "scipy",
    "matplotlib",
    "pymorton",
    "hilbertcurve",
    "mpi4py"
    ],
    version='0.0.1'
)