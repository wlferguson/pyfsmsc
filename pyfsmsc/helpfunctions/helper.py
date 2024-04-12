"""Include utilities for calculating structure information."""

import requests
import numpy as np
from numba import jit
import pandas as pd
from scipy.signal import savgol_filter
import netCDF4 as nc
from netCDF4 import Dataset


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

    posType[:, 0:3] = ds["coordinates"][frame]
    posType[:, 3] = ds["atom_types"][frame]

    df = pd.DataFrame(posType)

    return df
