#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from codecs import open
from os import path
import re

here = path.abspath(path.dirname(__file__))

# Get the version string
with open(path.join(here, 'DL_Models', '__init__.py'), encoding='utf-8') as f:
    version = re.search(r'__version__ = \'(.*?)\'', f.read()).group(1)

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    readme = f.read()

setup(
    name='eeg-dl',
    version=version,
    description='A Deep Learning library for EEG Signals Classification',
    long_description=readme,
    long_description_content_type='text/markdown',

    # The project's main homepage.
    url='https://github.com/SuperBruceJia/EEG-DL',

    # Author details
    author='Shuyue Jia',
    author_email='shuyuej@ieee.org',

    # Choose your license
    license='MIT',

    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 3 or both.
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],

    # What does your project relate to?
    keywords='eeg tensorflow deep learning',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['DL_Models', 'DL_Models.*']),

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['numpy>=1.14.0',
                      'six',
                      'pandas',
                      'scipy',
                      'matplotlib',
                      'pandas',
                      'sklearn']
)
