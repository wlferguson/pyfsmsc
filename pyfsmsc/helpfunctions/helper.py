"""Include support utilities for structure calculations."""

import numpy as np
import pandas as pd
import netCDF4 as nc
from netCDF4 import Dataset


def loadNCAtoms(fn, frame):
    """Convert netcdf4 file into pandas dataframe.

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
    ds = nc.Dataset(fn)  # read netCDF4

    ds["coordinates"][frame].shape  # get dimension of data structure

    posType = np.zeros(
        (ds["coordinates"][frame].shape[0], ds["coordinates"][frame].shape[1] + 1)
    )  # create data structure to write to

    posType[:, 0:3] = ds["coordinates"][frame]  # copy coordinates
    posType[:, 3] = ds["atom_types"][frame]  # copy atom types

    df = pd.DataFrame(posType)

    headers = ["x", "y", "z", "type"]  # assign headers
    df.columns = headers
    df["type"] = df["type"].astype(int)  # convert type to int

    return df
