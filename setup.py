#!/usr/bin/env python

from distutils.core import setup
from distutils.command.build_py import build_py as _build_py
from distutils.extension import Extension

from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy

import subprocess
import glob
#import cython_gsl



install_requires = [
    'pandas', 
    'numpy',
    'ipython',
    'matplotlib',
    'dill',
    'regex',
    'HTSeq', 
    'cython',
    'quadprog',
    ]

test_requires = [
    'nose',
    ]


setup(name='HSModel',
      version='0.1',
      description='',
      author='swag',
      author_email='smukhopa@ionisph.com',
      url='',
      packages=['HSModel' ],
      package_dir={'HSModel': 'HSModel'},
      install_requires=install_requires,
      tests_require=test_requires,
      test_suite="nose.collector",
      ext_modules = cythonize("./HSModel/*.pyx"), 
      include_dirs=[numpy.get_include()],
      extra_compile_args=["-O3"]
     )

