"""Test utitlies related to the reciprocal space structure."""

import pytest
import pyfsmsc
import numpy as np
import pyfsmsc.reciprocal
import scipy.interpolate as interp
import sklearn
from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.pyplot as plt


def test_RDF_to_SQ():
    """Test conversion of real space structural information into reciprocal space data.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    dataSq = pd.read_csv("examples/TestSq2.csv")

    q = dataSq.iloc[:, 1]
    Sq = dataSq.iloc[:, 0]

    # ---- TESTING FOURIER DEFINITION OF G(R) ---- #

    density = 5500 / 10**3 * 0.9
    rmin = 0.1
    rmax = 5
    nrs = 1000

    data = pd.read_csv("examples/GEM_Gr.csv", header=None)
    rCont = data.iloc[:, 0]
    grCont = data.iloc[:, 1]

    plt.plot(rCont, grCont, linewidth=3, color="black")

    r, Gr = pyfsmsc.SQ_to_RDF(q, Sq, density, rmin, rmax, nrs)

    plt.plot(
        r,
        Gr,
        linewidth=0,
        marker="o",
        markersize=10,
        markerfacecolor="white",
        color="red",
        markevery=50,
    )

    f = interp.interp1d(rCont, grCont, fill_value="extrapolate")
    new_y1 = f(r)

    R2 = sklearn.metrics.r2_score(new_y1, Gr)

    plt.legend(["Control", "Utility"])
    plt.text(4, 1.03, r"$R^{2} =$" + str(round(R2, 3)))
    plt.ylim(0.84, 1.1)
    plt.xlim(0, 5)
    plt.ylabel("g(r)", size=19)
    plt.xlabel(r"$r / \sigma$", size=19)

    assert R2 > 0.95
