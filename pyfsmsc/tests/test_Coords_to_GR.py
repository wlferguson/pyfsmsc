"""Tests the coordinates to GR utility."""

import pyfsmsc
import pytest
import netCDF4 as nc
import scipy.interpolate as interp
from pyfsmsc.realspace.Coords_to_GR import Coords_to_GR
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
import sklearn
from sklearn.metrics import r2_score


warnings.filterwarnings("ignore")


def test_Coords_to_GR():
    """Test conversion of coordinates to real space structure, G(r).

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    fn = "examples/colloids/colloidNC"  # read netCDF4 data

    rCut = 5
    nHis = 80
    frame = 0

    # calculate RDF for several frames for time averaging
    re1, gAll1 = Coords_to_GR(fn, rCut, nHis, frame)  # call function
    frame = 1
    re2, gAll2 = Coords_to_GR(fn, rCut, nHis, frame)
    frame = 2
    re3, gAll3 = Coords_to_GR(fn, rCut, nHis, frame)
    frame = 3
    re4, gAll4 = Coords_to_GR(fn, rCut, nHis, frame)
    frame = 4
    re5, gAll5 = Coords_to_GR(fn, rCut, nHis, frame)
    frame = 5
    re6, gAll6 = Coords_to_GR(fn, rCut, nHis, frame)

    # time average the data
    re = (re1[1:] + re2[1:] + re3[1:] + re4[1:] + re5[1:] + re6[1:]) / 6
    gAll = (gAll1 + gAll2 + gAll3 + gAll4 + gAll5 + gAll6) / 6

    # load in reference data from OVITO control
    data = pd.read_csv("examples/colloids/colloidGRref", header=None)
    rCont = data.iloc[:, 0]
    grCont = data.iloc[:, 1]

    # interpolate for shared values
    f = interp.interp1d(rCont, grCont, fill_value="extrapolate")
    new_y1 = f(re)

    # determine if the difference between method and control is in passing threshold
    R2 = sklearn.metrics.r2_score(new_y1, gAll)

    assert R2 > 0.90
