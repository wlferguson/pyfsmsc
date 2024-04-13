"""Include utilities for calculating scattering information in reciprocal space."""

import numpy as np
from numba import jit
import pandas as pd
from scipy.signal import savgol_filter
import warnings

warnings.filterwarnings("ignore")


def Coords_to_SQ(nmax, L, df) -> tuple:
    """Convert atomic coordinates into scattering data.

    Parameters
    ----------
    nmax : int
        Maximum integer index for reciprocal vector scaling.
    L : float
        Size of the periodic simulation box.
    df : ndarray
        2D array containing atomic coordinates of simulation in `float` type.

    Returns
    -------
    qmagVec : ndarray
        1D array containing magnitude of reciprocal space vector, q, of `float` type.
    ds3 : ndarray
        1D array containing non-averaged scattering data, S(q), of simulation in `float` type.
    """
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
    ds3 = waveInteractions(ds1, df)
    ds3 = ds3 / df.shape[0]
    qmagVec = np.zeros((kCard, 1))
    for rows in range(0, kCard):
        qmagVec[rows] = (ds1[rows][0] ** 2 + ds1[rows][1] ** 2 + ds1[rows][2] ** 2) ** (
            0.5
        )

    return qmagVec, ds3


@jit(nopython=True)
def waveInteractions(A, B) -> np.ndarray:
    """Perform rigorous calculations of wave calculations with JIT compilation.

    Parameters
    ----------
    A : ndarray
        2D array containing atomic coordinates of simulation in `float` type.
    B : float
        2D array containing allowable q vectors of simulation box in `float` type.

    Returns
    -------
    ds3 : ndarray
        1D array containing non-averaged scattering data, S(q), of simulation in `float` type.
    """
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
