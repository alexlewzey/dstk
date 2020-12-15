#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

setup(
    author="Alexander Lewzey",
    author_email='a.lewzey@hotmail.co.uk',
    python_requires='>=3.5',
    description="A collection of general purpose helper modules",
    entry_points={
        'console_scripts': [
            'dstk=dstk.cli:main',
        ],
    },
    install_requires=[
        'slibtk @ git+https://github.com/alexlewzey/slibtk.git#egg=slibtk',
        'pandas',
        'scikit-learn',
        'tqdm',
        'matplotlib',
        'plotly',
        'seaborn',
        'umap-learn',
        'lightgbm',
        'google-cloud-bigquery',
        'xlwings',
        'fuzzywuzzy',
        'python-Levenshtein',
        'pyxlsb',
        'holidays',
        'openpyxl',
        'pyperclip',
    ],
    license="BSD license",
    keywords='dstk',
    name='dstk',
    packages=find_packages(include=['dstk', 'dstk.*']),
    test_suite='tests',
    url='https://github.com/alexlewzey/dstk',
    version='0.1.0',
)
