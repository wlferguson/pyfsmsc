from setuptools import setup

setup(
    name="pyfsmsc",
    version="1.0.0",
    description="A fluids and soft materials structure characterization package",
    maintainer="William Ferguson",
    maintainer_email="wferguso@andrew.cmu.edu",
    license="MIT",
    url="https://github.com/wlferguson/pyfsmsc",
    packages=[
        "pyfsmsc",
        "pyfsmsc.helpfunctions",
        "pyfsmsc.reciprocalspace",
        "pyfsmsc.realspace",
        "pyfsmsc.shapemetrics",
    ],
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "numba",
        "pytest",
        "netCDF4",
        "scikit-learn",
        "umap-learn",
        "black",
        "flake8",
        "flake8-docstrings",
    ],
    long_description="""Package
      includes utilties to characterize the structures of simulated fluids and soft materials.""",
)
