"""Utility for reciprocal to real space conversion."""

import pytest
import numpy as np
from numba import jit
import pandas as pd
from scipy.signal import savgol_filter


def SQ_to_RDF(q, sq, density, rmin, rmax, nrs) -> tuple:
    """Convert recriprocal space scattering data into real space radial distribution data.

    Parameters
    ----------
    q : ndarray
        1D array containing reciprocal space vector magnitudes, q, of `float` type.
    sq : ndarray
        1D array containing static structure factor values, S(q), of `float` type.
    density : float
        Number density of the species with scattering data.
    rmin : float
        Minimum x-value for the real space conversion.
    rmax : float
        Maximum x-value for the real space conversion.
    rmax : int
        Number of points to map recriprocal space to real space.

    Returns
    -------
    rs : ndarray
        1D array containing real space vector magnitudes, r, of `float` type.
    Grs : ndarray
        1D array containing radial distribution values, G(r), of `float` type.
    """
    qs = q
    dq = q[1] - q[0]
    SQs = sq - 1
    rs = np.linspace(rmin, rmax, num=nrs)
    Grs = np.zeros(rs.shape[0])
    for rind, r in enumerate(rs):
        g = 0
        for qind, q in enumerate(qs):
            g += dq * q * np.sin(q * r) * SQs[qind]
            Grs[rind] = 1 + 1 / ((np.pi) ** 2 * density * 2) * g / r
    return rs, Grs
