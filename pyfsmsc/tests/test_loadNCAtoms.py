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
    fn = "examples/ionomers/ionomerNC"

    loaded = loadNCAtoms(fn, frame)
    ds = nc.Dataset(fn)

    indexCheck = np.random.randint(0, ds["coordinates"][frame].shape[0], 10)

    assert np.array_equal(
        ds["coordinates"][frame, indexCheck], loaded.iloc[indexCheck, 0:3].to_numpy()
    )
    assert np.array_equal(
        ds["atom_types"][frame, indexCheck], loaded.iloc[indexCheck, 3].to_numpy()
    )
