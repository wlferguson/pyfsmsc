"""Test utitlies related to the real space structure."""

import pytest
import pyfsmsc
import numpy as np
import scipy.interpolate as interp
from pyfsmsc.realspace.rdf2sq import RDF_to_SQ
import sklearn
from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.pyplot as plt


def test_RDF_to_SQ():
    """Test conversion of reciprocal space structural information into real space data.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    data = pd.read_csv("examples/colloids/colloidGR", header=None)  # read data

    control = pd.read_csv("examples/colloids/colloidSQref")  # read control / compare data
    qCont = control.iloc[:, 0].dropna().to_numpy()
    SqCont = control.iloc[:, 1].dropna().to_numpy()

    r = data.iloc[:, 0]
    gr = data.iloc[:, 1]

    # parameters for the simulation
    density = 5500 / 10**3 * 0.9
    qmin = 1
    qmax = 8
    nqs = 1000

    q, Sq = RDF_to_SQ(r, gr, density, qmin, qmax, nqs)  # call the function

    f = interp.interp1d(qCont, SqCont, fill_value="extrapolate")  # interpolate for shared values
    new_y1 = f(q)

    plt.plot(q, new_y1, linewidth=3, color="black")
    plt.plot(
        q,
        Sq,
        linewidth=0,
        marker="o",
        markerfacecolor="white",
        markersize=10,
        color="red",
        markevery=50,
    )

    R2 = sklearn.metrics.r2_score(new_y1, Sq)  # calculate score between control and calculated data

    plt.legend(["Control", "Utility"])
    plt.text(0.8, 0.825, r"$R^{2} =$" + str(round(R2, 3)))
    plt.ylabel("S(q)", size=19)
    plt.xlabel(r"$q \sigma$", size=19)

    # pass if above threshold
    assert R2 > 0.95
