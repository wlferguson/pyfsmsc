"""Include utilities for calculating reciprocal space scattering from coordinates."""

import numpy as np
from numba import jit
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


def Coords_to_SQ(nmax, L, df):
    """Convert atomic coordinates into scattering data.

    Parameters
    ----------
    nmax : int
        Maximum integer index for reciprocal vectors.
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

    ds1 = generateWaves(nmax, L)
    ds3 = waveInteractions(ds1, df)
    ds3 = ds3 / df.shape[0]
    qmagVec = np.zeros((nmax**3, 1))
    # for rows in range(0, nmax**3):
    #    qmagVec[rows] = (ds1[rows][0] ** 2 + ds1[rows][1] ** 2 + ds1[rows][2] ** 2) ** (
    #        0.5
    #    )

    qmagVec = np.linalg.norm(ds1, axis=1)

    return qmagVec, ds3


def generateWaves(nmax, L):
    """Create compatible scattering vectors for the system.

    Parameters
    ----------
    nmax : int
        Maximum integer index for reciprocal vector scaling.
    L : float
        Size of the periodic simulation box.

    Returns
    -------
    ds1 : ndarray
        1D array containing allowable scattering vectors for system geometry.
    """

    nk = nmax
    ds1 = np.zeros(((nk) ** 3, 3))
    index = 0

    for nx in range(0, nk):
        for ny in range(0, nk):
            for nz in range(0, nk):
                ds1[index][0] = nx
                ds1[index][1] = ny
                ds1[index][2] = nz
                index += 1
    ds1 = 2 * np.pi / L * ds1

    return ds1


@jit(nopython=True)
def waveInteractions(q, C):
    """Perform wave interaction with particle calculation with JIT compilation.

    Parameters
    ----------
    q : ndarray
        2D array containing allowable q vectors of simulation box in `float` type.
    C : float
        2D array containing atomic coordinates of simulation in `float` type.

    Returns
    -------
    ds3 : ndarray
        1D array containing non-averaged scattering data, S(q), of simulation in `float` type.
    """
    ds3 = np.zeros((q.shape[0], 1))
    sumInit1 = 0
    sumInit2 = 0
    for row in range(0, q.shape[0]):
        for i in range(0, C.shape[0]):
            sumInit1 = sumInit1 + np.cos(np.dot(q[row], C[i]))
            sumInit2 = sumInit2 + np.sin(np.dot(q[row], C[i]))
        tot = sumInit1**2 + sumInit2**2
        ds3[row] = tot
        sumInit1 = 0
        sumInit2 = 0

    return ds3
