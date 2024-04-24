"""Include utilities for converting real space scattering to reciprocal space."""

import numpy as np
import pandas as pd


def RDF_to_SQ(r, gr, density, qmin, qmax, nqs):
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
