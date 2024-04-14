"""Include utilities for calculating real space scattering from coordinates."""

import numpy as np
from netCDF4 import Dataset


def Coords_to_GR(fn, rCut, nHis, frame) -> tuple:
    """Convert coordinates into real space scattering data.
    Code is modified starting from base code created by advisor T.O'C.

    Parameters
    ----------
    fn : str
        1D array containing real space vector magnitudes, r, of `float` type.
    rCut : float
        1D array containing radial distribution, G(r), of `float` type.
    nHis : int
        Number density of the species with scattering data.
    frame : int
        Minimum x-value for the reciprocal space conversion.

    Returns
    -------
    re : ndarray
        1D array containing real space radial vectors, r, of `float` type.
    gAll : ndarray
        1D array containing radial distribution function, G(r), of `float` type.
    """
    ds = Dataset(fn)
    X = ds["coordinates"][frame, :]
    X = X[ds["atom_types"][frame, :] == 1]
    L = ds["cell_lengths"][0]

    Npart = X.shape[0]

    gAll = np.zeros(nHis)
    idx = np.arange(Npart, dtype=int)

    for i in range(Npart):
        dXij = neigh_distances(i, X, idx)
        Rij = np.linalg.norm(adjust_half_box(dXij, L), axis=1)
        gi, re = np.histogram(
            Rij[Rij < rCut], bins=nHis, range=(0, rCut), density=False
        )
        gAll += gi
    gAll = gAll / Npart

    Vb = 4 / 3 * np.pi * (re[1:] ** 3 - re[:-1] ** 3)
    nid = Npart / (L[0] * L[1] * L[2])
    gAll = gAll / (Vb * nid)

    return re, gAll


def neigh_distances(i, X, idx) -> np.ndarray:
    """Compute neighbors from a reference particle.
    Code is modified starting from base code created by advisor T.O'C.

    Parameters
    ----------
    i : str
        Index of reference particle.
    X : float
        3D coordinates of all the particles in the system.
    idx : int
        Index of all the particles in the system.

    Returns
    -------
    dXij : ndarray
        2D array 3D distances of the reference particle with neighbors, of `float` type.
    """
    xi = X[i]
    Xj = X[idx != i, :]
    dXij = Xj - xi
    return dXij


def adjust_half_box(dXij, L) -> np.ndarray:
    """Adjusts for periodicity of box by correcting particles outside +/- L/2.
    Code is modified starting from base code created by advisor T.O'C.

    Parameters
    ----------
    dXij : str
        2D array 3D distances of the reference particle with neighbors, of `float` type.
    L : float
        Length of the box.

    Returns
    -------
    dXij : ndarray
        2D array 3D distances of the adjusted reference particle with neighbors, of `float` type.
    """
    for d in range(3):
        mask = dXij[:, d] > 0.5 * L[d]
        dXij[mask, d] -= L[d]
        mask = dXij[:, d] <= -0.5 * L[d]
        dXij[mask, d] += L[d]
    return dXij
