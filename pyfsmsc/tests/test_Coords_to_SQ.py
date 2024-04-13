"""Tests the coordinates to SQ utility."""

import pyfsmsc
import pytest
import netCDF4 as nc
from pyfsmsc.reciprocalspace.Coords_to_SQ import Coords_to_SQ
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import warnings

warnings.filterwarnings("ignore")


def test_Coords_to_SQ():
    """Test if coords to SQ works.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    fn = "examples/ionomerCoords.csv"
    data = pd.read_csv(fn, header=None)

    q, Sq = Coords_to_SQ(30, 55, data[data[3] == 2].iloc[:, 0:3].to_numpy())

    plt.scatter(q[1:], Sq[1:], s=1, alpha=0.2)
    b = np.concatenate((q, Sq), axis=1)
    ind = np.lexsort((b[:, 1], b[:, 0]))
    b = b[ind]
    fltSq = savgol_filter(b[1:-1, 1], 300, 1)

    plt.plot(b[1:-1, 0], fltSq, color="orange")
    plt.ylim(0, 25)

    fn = "examples/savitzyTest"
    data = pd.read_csv(fn, header=None, delimiter=" ")

    assert np.max(b - data.to_numpy()) < 10 ** (-3)
