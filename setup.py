from setuptools import setup, find_packages

setup(
    name='partial_symmetries_graphs',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'networkx',
        'matplotlib',
        'numpy',
    ],
)