"""Include support utilities for structure calculations."""

import numpy as np
import pandas as pd
import netCDF4 as nc
from netCDF4 import Dataset


def loadNCAtoms(fn, frame) -> pd.DataFrame:
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
    ds = nc.Dataset(fn)

    ds["coordinates"][frame].shape

    posType = np.zeros(
        (ds["coordinates"][frame].shape[0], ds["coordinates"][frame].shape[1] + 1)
    )

    posType[:, 0:3] = ds["coordinates"][frame]
    posType[:, 3] = ds["atom_types"][frame]

    df = pd.DataFrame(posType)

    headers = ["x", "y", "z", "type"]
    df.columns = headers
    df["type"] = df["type"].astype(int)

    return df
