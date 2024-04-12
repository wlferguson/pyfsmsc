"""Test utitlies related to the real space structure."""

import pytest
import pyfsmsc
import numpy as np
import pyfsmsc.reciprocal
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
    data = pd.read_csv("examples/GEM_Gr.csv", header=None)

    control = pd.read_csv("examples/LikosSq.csv")
    qCont = control.iloc[:, 0].dropna().to_numpy()
    SqCont = control.iloc[:, 1].dropna().to_numpy()

    r = data.iloc[:, 0]
    gr = data.iloc[:, 1]

    density = 5500 / 10**3 * 0.9
    qmin = 1
    qmax = 8
    nqs = 1000

    q, Sq = pyfsmsc.RDF_to_SQ(r, gr, density, qmin, qmax, nqs)

    control = pd.read_csv("examples/LikosSq.csv")

    f = interp.interp1d(qCont, SqCont, fill_value="extrapolate")
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

    R2 = sklearn.metrics.r2_score(new_y1, Sq)

    plt.legend(["Control", "Utility"])
    plt.text(0.8, 0.825, r"$R^{2} =$" + str(round(R2, 3)))
    plt.ylabel("S(q)", size=19)
    plt.xlabel(r"$q \sigma$", size=19)

    assert R2 > 0.95