"""Include utilities for calculating scattering information in real space."""

import numpy as np
from numba import jit
import pandas as pd
from scipy.signal import savgol_filter


def realgreet(name) -> None:
    """Welcomes user to the real space folder.

    Parameters
    ----------
    name : str
        The user's name.

    Returns
    -------
    None
    """
    print(f"Welcome to the real space utilties: {name}")


def RDF_to_SQ(r, gr, density, qmin, qmax, nqs) -> tuple:
    """Convert real space scattering data into reciprocal space scattering data.

    Parameters
    ----------
    r : ndarray
        1D array containing real space vector magnitudes, r, of `float` type.
    gr : ndarray
        1D array containing radial distribution, G(r), of `float` type.
    density : float
        Number density of the species with scattering data.
    qmin : float
        Minimum x-value for the reciprocal space conversion.
    qmax : float
        Maximum x-value for the reciprocal space conversion.
    qmax : int
        Number of points to map real space to reciprocal space.

    Returns
    -------
    qs : ndarray
        1D array containing reciprocal space vector magnitudes, q, of `float` type.
    Sqs : ndarray
        1D array containing reciprocal distribution values, S(q), of `float` type.
    """
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
