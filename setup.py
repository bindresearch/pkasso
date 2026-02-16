from setuptools import setup, find_packages
import subprocess

setup(
    name='autoprot',
    version='0.2.1',
    description='Automatic selection of protonation states for small molecules',
    url='.',
    authors=[
        {'name':'Soeren von Buelow', 'email':'soeren.buelow@bindresearch.org'},
    ],
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'rdkit',
        'numpy',
        'scipy',
        'pyyaml',
        'cairosvg',
        'svgutils',
        'matplotlib',
        'torch',
        'torch_geometric',
        'pandas',
        'pytest',
        'pytest-cov',
    ],

    classifiers=[
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3>=3.7',
    ],

    package_data={'' : ['data/*.csv']},
)
