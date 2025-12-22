from setuptools import setup, find_packages
import subprocess

setup(
    name='autoprot',
    version='0.1.0',
    description='Automatic selection of protonation states for small molecules',
    url='.',
    authors=[
        {'name':'Soren von Bulow', 'email':'soeren.buelow@bindresearch.org'},
    ],
    license='GNU GPL3',
    packages=find_packages(),
    install_requires=[
        'rdkit',
        'numpy',
        'pyyaml',
        'cairosvg',
        'matplotlib',
    ],

    classifiers=[
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3>=3.7,<3.11',
    ],
)
