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

class ModelBuildingSetup(_build_py):
    """ compile and build the scikit learn datasources that will be needed for 
        prediction algorithms as part of the installation process """

    def run(self):
        print "building models..."
        subprocess.check_call(['./scripts/_buildmodels', '--datadir', 'seqpred/data'])
        _build_py.run(self)


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


setup(name='IPyBAT-ngstools',
      version='0.1',
      description='',
      author='Chris Hart',
      author_email='chart@isisph.com',
      url='http://www.isisiph.com',
      packages=['ngstools' ],
      package_dir={'ngstools': 'ngstools'},
      scripts= glob.glob('scripts/*'),
      install_requires=install_requires,
      tests_require=test_requires,
      test_suite="nose.collector",
      ext_modules = cythonize("./ngstools/*.pyx"), 
      include_dirs=[numpy.get_include()],
      extra_compile_args=["-O3"]
     )

# CHRIS: I want to include GSL and try to build extensions as follows, right now C++ compiler optimization is not used and gsl is disabled! 
#setup(
#  name = 'Test app',
#  ext_modules=[
#    Extension('test',
#              sources=['test.pyx'],
#              extra_compile_args=['-O3'],
#              language='c++')
#    ],
#  cmdclass = {'build_ext': build_ext}
#)

# Setup examples 

#extensions = [
#    Extension("primes", ["primes.pyx"],
#        include_dirs = [...],
#        libraries = [...],
#        library_dirs = [...]),
#    # Everything but primes.pyx is included here.
#    Extension("*", ["*.pyx"],
#        include_dirs = [...],
#        libraries = [...],
#        library_dirs = [...]),
#]
#setup(
#    name = "My hello app",
#    ext_modules = cythonize(extensions),
#)

