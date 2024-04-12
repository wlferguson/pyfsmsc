from setuptools import setup

setup(
    name="pyfsmsc",
    version="1.0.0",
    description="A fluids and soft materials structure characterization package",
    maintainer="William Ferguson",
    maintainer_email="wferguso@andrew.cmu.edu",
    license="MIT",
    packages=["pyfsmsc",
              'pyfsmsc.helpfunctions',
              'pyfsmsc.reciprocalspace',
              'pyfsmsc.realspace',
              'pyfsmsc.shapemetrics'],
    long_description="""This package
      includes utilties to characterize the structures of simulated fluids and soft materials.""",
)
