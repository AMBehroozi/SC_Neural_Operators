#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ ]

test_requirements = [ ]

setup(
    author="Abdolmehdi Behroozi",
    author_email='amb10399@psu.edu',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Python Boilerplate contains all the boilerplate you need to create a Python package.",
    install_requires=requirements,
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='fno_swe_2d',
    name='fno_swe_2d',
    packages=find_packages(include=['fno_swe_2d', 'fno_swe_2d.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/AMBehroozi/fno_swe_2d',
    version='0.1.0',
    zip_safe=False,
)
