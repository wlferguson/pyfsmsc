"""Test utitlies related to the loading data."""

import pyfsmsc
import pytest
from pyfsmsc.helpfunctions.helper import loadNCAtoms
import netCDF4 as nc
import numpy as np
import pandas as pd


def test_loadNCAtoms():
    """Test if conversion of .nc to pd.Dataframe is correct.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    frame = -1
    fn = "examples/ionomers/ionomerNC"  # load data

    loaded = loadNCAtoms(fn, frame)  # call the function
    ds = nc.Dataset(fn)

    indexCheck = np.random.randint(0, ds["coordinates"][frame].shape[0], 10)  # pick random atoms to compare to netCDF4 data

    assert np.array_equal(
        ds["coordinates"][frame, indexCheck], loaded.iloc[indexCheck, 0:3].to_numpy()  # compare if coordinate copy process works
    )
    assert np.array_equal(
        ds["atom_types"][frame, indexCheck], loaded.iloc[indexCheck, 3].to_numpy()  # compare if atom copy is correct
    )
