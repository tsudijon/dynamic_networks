
from setuptools import setup


## Get version information from _version.py
import re
VERSIONFILE="tdadynamicnetworks/_version.py"
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
    name="tdadynamicnetworks",
    version=verstr,
    description="A pipeline for detecting periodicity in node-weighted dynamic networks via topological data analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Tim Sudijono",
    author_email="timothysudijono@gmail.com",
    license='Apache2',
    packages=['tdadynamicnetworks'],
    install_requires=[
        'POT',
        'ripser',
        'gudhi',
        'persim',
        'numpy',
        'matplotlib',
        'scipy',
        'intervaltree',
        'haversine'
    ],
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