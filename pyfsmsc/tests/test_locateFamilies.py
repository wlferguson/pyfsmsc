"""Test utilities for finding microstructure families from shape data."""

import pytest
import pyfsmsc
import numpy as np
import scipy.interpolate as interp
from pyfsmsc.shapemetrics.locateFamilies import locateFamilies
import sklearn
from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.pyplot as plt


def test_locateFamilies():
    """Test unsupervised machine learning technique to find microstructural families.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    simulation1 = pd.read_csv("examples/ionomers/shapes/microstructure1.csv")
    simulation2 = pd.read_csv("examples/ionomers/shapes/microstructure2.csv")
    simulation3 = pd.read_csv("examples/ionomers/shapes/microstructure3.csv")

    totds = pd.concat([simulation3, simulation2, simulation1])
    inp = totds[
        [
            "clusterID",
            "clusterSize",
            "xcm",
            "ycm",
            "zcm",
            "rgxx",
            "rgyy",
            "rgzz",
            "rgxy",
            "rgxz",
            "rgyz",
            "rg",
            "l1",
            "l2",
            "l3",
            "asphericity",
            "acylindricity",
            "anisotropy",
            "aspectratio",
        ]
    ]

    assert locateFamilies(inp) == 3
