"""Include utilities for calculating reciprocal space scattering from coordinates."""

import numpy as np
from numba import jit
import pandas as pd
import warnings
from netCDF4 import Dataset
import numpy.ma as ma

warnings.filterwarnings("ignore")

def Coords_to_FSQ(fn, type, n):
    """Convert atomic coordinates into intemediate scattering data.

    Parameters
    ----------
    fn : str
        Path to netCDF4 file.
    type : int
        Atom type for scattering.
    nmax : int
        Maximum integer index for reciprocal vectors.

    Returns
    -------
    qmagVec : ndarray
        1D array containing magnitude of reciprocal space vector, q, of `float` type.
    ds3 : ndarray
        1D array containing non-averaged scattering data, F(q), of simulation in `float` type.
    """

    ds = Dataset(fn)
    # initial frame
    frame = 0

    X0 = ds["coordinates"][frame, :]  # grab coordinates

    type = 2
    X0 = X0[ds["atom_types"][frame, :] == type]  # get atom type

    t = ds['time']
    L = ds["cell_lengths"][0]  # cell coordinates

    q = 2*np.pi/L*np.array([n,n,n])

    timesteps = len(t) - 1

    x = np.linspace(0.01, int(timesteps/100), timesteps)

    len(t)
    fme = len(t)

    fR = np.array([])
    fI = np.array([])

    for t in range(0,timesteps):

        Xf = ds["coordinates"][t, :]  # grab coordinates
        Xf = Xf[ds["atom_types"][t, :] == type]  # get atom type
        rjr0 = Xf - X0

        fR = np.append(fR, np.cos(np.sum(-q*(rjr0),axis=1)).mean())
        fI = np.append(fI, np.sin(np.sum(-q*(rjr0),axis=1)).mean())

    Fs = (fR**2 + fI**2)**0.5

    x=np.linspace(500000*0.01*1/10**(6), 500000*0.01*fme/10**(6), fme-1)

    return x, Fs

