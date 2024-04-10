"""Includes utilities for calculating structure information."""

import requests
import numpy as np
from numba import jit
import pandas as pd
from scipy.signal import savgol_filter


def hello(name):
    """Welcomes user to the package."""
    print(f"Welcome: {name}")


def RDF_to_SQ(r, gr, density, qmin, qmax, nqs):
    """Convert radial distribution data into scattering data."""
    rs = r
    dr = r[1] - r[0]
    RDFs = gr - 1
    qs = np.linspace(qmin, qmax, num=nqs)
    Sqs = np.zeros(qs.shape[0])
    for qind, q in enumerate(qs):
        s = 0
        for rind, r in enumerate(rs):
            s += dr * r * np.sin(q * r) * RDFs[rind]
            Sqs[qind] = 1 + 4 * np.pi * density * s / q
    return qs, Sqs


def Coords_to_SQ(nmax, L, df):
    """Convert atomic coordinates into scattering data."""
    nk = nmax
    kCard = (nk) ** 3
    ds1 = np.zeros((kCard, 3))
    index = 0

    for nx in range(0, nk):
        for ny in range(0, nk):
            for nz in range(0, nk):
                ds1[index][0] = nx
                ds1[index][1] = ny
                ds1[index][2] = nz
                index += 1
    ds1 = 2 * np.pi / L * ds1
    ds3 = computeSq(ds1, df)
    ds3 = ds3 / 16000
    qmagVec = np.zeros((kCard, 1))
    for rows in range(0, kCard):
        qmagVec[rows] = (ds1[rows][0] ** 2 +
                         ds1[rows][1] ** 2 +
                         ds1[rows][2] ** 2)**(0.5)

    return qmagVec, ds3


@jit(nopython=True)
def computeSq(A, B):
    """Perform rigorous calculations of scattering."""
    ds3 = np.zeros((A.shape[0], 1))
    sumInit1 = 0
    sumInit2 = 0
    for row in range(0, A.shape[0]):
        for i in range(0, B.shape[0]):
            sumInit1 = sumInit1 + np.cos(np.dot(A[row], B[i]))
            sumInit2 = sumInit2 + np.sin(np.dot(A[row], B[i]))
        tot = sumInit1**2 + sumInit2**2
        ds3[row] = tot
        sumInit1 = 0
        sumInit2 = 0
    return ds3
