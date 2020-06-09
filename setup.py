import os
import platform
import re
import sys

import setuptools

NAME = "scan-tf"

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name=NAME,
    version="0.0.1",
    description="Implementation of SCAN in tensorflow",
    long_description=long_description,
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.6",
)

