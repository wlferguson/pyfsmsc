"""Test utilities for microstructure shape and morphology."""

import pytest
import pyfsmsc
import numpy as np
from pyfsmsc.shapemetrics.shapeMetrics import findMicrostructures
from pyfsmsc.shapemetrics.shapeMetrics import computeGyTensor
from pyfsmsc.shapemetrics.shapeMetrics import computeShapeMetrics
import scipy.interpolate as interp
import sklearn
from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.pyplot as plt


def test_shapeMetrics():
    """Test if discovered microstructures and shapes match reference data.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    fn = "examples/ionomers/ionomerNC"

    df, clusterID = findMicrostructures(fn)  # calculate microstructures
    df = computeGyTensor(df, clusterID)  # calculate gyration tensor
    microstructures = computeShapeMetrics(df)  # calculate shape metrics

    OVITOControl = pd.read_csv("examples/ionomers/microstructureControl", header=None)  # read control data to reference

    # Determine if both these techniques find the same number of microstructures.
    assert microstructures.shape[0] == OVITOControl.shape[0]

    # Determine if both of these techniques find microstructures of the same size!
    truth = np.array_equal(
        microstructures["clusterSize"].sort_values().unique(),
        OVITOControl[1].sort_values().unique(),
    )

    assert truth

    # Determine if they both have the same eigenvalues through radius of gyration number!
    RgUtility = microstructures["rg"].sort_values()
    RgOVITO = OVITOControl[2].sort_values()
    RgOVITO = RgOVITO.reset_index(drop=True)
    RgUtility = RgUtility.reset_index(drop=True)
    assert np.max(abs(RgOVITO - RgUtility)) < 10 ** (-3)
