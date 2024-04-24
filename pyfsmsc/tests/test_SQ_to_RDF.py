"""Test utitlies related to the reciprocal space structure."""

import pytest
import pyfsmsc
import numpy as np
from pyfsmsc.reciprocalspace.sq2rdf import SQ_to_RDF
import scipy.interpolate as interp
import sklearn
from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.pyplot as plt


def test_SQ_to_RDF():
    """Test conversion of reciprocal space structural information into real space data.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    dataSq = pd.read_csv("examples/colloids/colloidSQ")  # read data

    q = dataSq.iloc[:, 1]
    Sq = dataSq.iloc[:, 0]

    # data from simulation
    density = 5500 / 10**3 * 0.9
    rmin = 0.1
    rmax = 5
    nrs = 1000

    data = pd.read_csv(
        "examples/colloids/colloidGRref", header=None
    )  # reference data to compare to
    rCont = data.iloc[:, 0]
    grCont = data.iloc[:, 1]

    plt.plot(rCont, grCont, linewidth=3, color="black")

    r, Gr = SQ_to_RDF(q, Sq, density, rmin, rmax, nrs)  # call the function

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

    f = interp.interp1d(
        rCont, grCont, fill_value="extrapolate"
    )  # interpolate to shared values
    new_y1 = f(r)

    R2 = sklearn.metrics.r2_score(
        new_y1, Gr
    )  # calculate score between control and calculated score

    plt.legend(["Control", "Utility"])
    plt.text(4, 1.03, r"$R^{2} =$" + str(round(R2, 3)))
    plt.ylim(0.84, 1.1)
    plt.xlim(0, 5)
    plt.ylabel("g(r)", size=19)
    plt.xlabel(r"$r / \sigma$", size=19)

    # pass the test if above threshold
    assert R2 > 0.95
