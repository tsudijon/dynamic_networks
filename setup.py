
import sys
import os
import platform

from setuptools import setup
from setuptools.extension import Extension

# Ensure Cython is installed before we even attempt to install linmdtw
try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
except:
    print("You don't seem to have Cython installed. Please get a")
    print("copy from www.cython.org or install it with `pip install Cython`")
    sys.exit(1)

## Get version information from _version.py
import re
VERSIONFILE="linmdtw/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

# Use README.md as the package long description  
with open('README.md') as f:
    long_description = f.read()


setup(
    name="tdadynamicnetworks-tsudijon",
    version=verstr,
    description="A pipeline for detecting periodicity in node-weighted dynamic networks via topological data analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Tim Sudijono",
    author_email="timothysudijono@gmail.com",
    license='Apache2',
    packages=['tdadynamicnetworks'],
    install_requires=[
        'Cython',
        'numpy',
        'matplotlib',
        'scipy'
    ],
    extras_require={
        'testing': [ # `pip install -e ".[testing]"``
            'pytest'  
        ],
        'docs': [ # `pip install -e ".[docs]"`
            'linmdtw_docs_config'
        ],
        'examples': []
    },
    cmdclass={'build_ext': CustomBuildExtCommand},
    python_requires='>=3.6',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8'
    ],
    keywords='tda, dynamic networks, periodicity'
)