from setuptools import setup
from setuptools import find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="molreps",
    version="0.1.0",
    author="Patrick Reiser",
    author_email="patrick.reiser@kit.edu",
    description="Molecular feature generation for machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    install_requires=['numpy', "scikit-learn","pandas"],
    extras_require={
        "ase": ["ase>=3.0.0"],
        "matplotlib": ["matplotlib>=2.0.0"],
    },
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    keywords=["materials", "science", "machine learning", "molecules","features","descriptors"]
)

