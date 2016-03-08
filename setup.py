#!/usr/bin/env python

import sys
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        # Specifying the directory with tests explicitly
        # to prevent Travis CI from running tests from dependencies' eggs
        # (which are copied to the same directory).
        self.test_args = ['-x', 'tests']
        self.test_suite = True

    def run_tests(self):
        import pytest
        errno = pytest.main(self.test_args)
        sys.exit(errno)


setup(
    name="fluxion",
    version="0.0.2+dev",
    description="PDE/SDE integrator",
    long_description=open("README.rst").read(),
    url="https://github.com/fluxion/fluxion-py",
    author="Bogdan Opanchuk",
    author_email="bogdan@opanchuk.net",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.10.0",
        ],
    tests_require=[
        "pytest",
        ],
    cmdclass={'test': PyTest},
    platforms=["any"],
    keywords="numeric integration",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering",
        "Topic :: Education"
    ],
)
