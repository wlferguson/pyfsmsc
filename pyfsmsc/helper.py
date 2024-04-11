"""Include utilities for calculating structure information."""

import requests
import numpy as np
from numba import jit
import pandas as pd
from scipy.signal import savgol_filter
import netCDF4 as nc
from netCDF4 import Dataset


def helpergreet(name) -> None:
    """Welcomes user to the real space folder.

    Parameters
    ----------
    name : str
        The user's name.

    Returns
    -------
    None
    """
    print(f"Welcome to the helper utilties: {name}")


def loadNCAtoms(fn, frame) -> pd.DataFrame:
    """Convert netcdf4 file into pandas dataframe for ease of manipulation.

    Parameters
    ----------
    fn : str
        The path is the netCDF4 trajectory.
    frame : int
        The frame in the netCDF4 trajectory being accessed.

    Returns
    -------
    df : pdDataframe
        Atomic coordinates and types of atoms.
    """
    ds = nc.Dataset(fn)

    frame = frame

    ds["coordinates"][frame].shape

    posType = np.zeros(
        (ds["coordinates"][frame].shape[0], ds["coordinates"][frame].shape[1] + 1)
    )

    for i in range(0, 160000):
        posType[i][0] = ds["coordinates"][frame][i][0]
        posType[i][1] = ds["coordinates"][frame][i][1]
        posType[i][2] = ds["coordinates"][frame][i][2]
        posType[i][3] = ds["atom_types"][frame, i]

    df = pd.DataFrame(posType)

    return df