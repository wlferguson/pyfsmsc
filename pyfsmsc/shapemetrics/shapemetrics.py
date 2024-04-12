"""Include utilities for calculating scattering information in reciprocal space."""

import netCDF4 as nc
from netCDF4 import Dataset
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
from numpy import linalg as LA


def shapeMetrics(fn) -> pd.DataFrame:
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
    # Step 1) Loading Dependencies and Data
    ds = nc.Dataset(fn)

    # frame = frame

    # Step 2) Generating Data Structure with Cluster ID and Atomic Coordinates
    vals = atoms = np.arange(0, 0)

    for i in range(0, (ds["identifier"]).shape[1]):
        if ds["c_clst"][0, i] != 0:
            vals = np.append(vals, ds["c_clst"][0, i])
            atoms = np.append(atoms, i)

    atomCluster = np.vstack((vals, atoms)).T
    df = pd.DataFrame(atomCluster, columns=["clusterID", "atomID"])

    Clusters = df["clusterID"]
    Clusters = Clusters.astype(int)
    sizeClusters = np.arange(0, 0)
    for i in Clusters:
        sizeClusters = np.append(sizeClusters, len(df[df["clusterID"] == i]))
    df = df.assign(clusterSize=sizeClusters)

    df["clusterID"] = df["clusterID"].astype(int)
    df["atomID"] = df["atomID"].astype(int)

    # Step 3) Appending Atomic Coordinates and COM to Data Structure
    atomID = df["atomID"]
    coords = np.zeros((len(df["atomID"]), 3))
    index = 0
    for i in atomID:  # getting the atomic coordinates
        coords[index] = ds["coordinates"][0][i]
        index += 1
    df = df.assign(
        atomCoordx=coords[:, 0], atomCoordy=coords[:, 1], atomCoordz=coords[:, 2]
    )

    m = 1
    xcmv = ycmv = zcmv = np.arange(0, 0)

    clusterID = df["clusterID"]
    for item in clusterID:  # computing the COM
        ID = item
        xcm = sum(df[df["clusterID"] == ID]["atomCoordx"] * m) / len(
            df[df["clusterID"] == ID]
        )
        ycm = sum(df[df["clusterID"] == ID]["atomCoordy"] * m) / len(
            df[df["clusterID"] == ID]
        )
        zcm = sum(df[df["clusterID"] == ID]["atomCoordz"] * m) / len(
            df[df["clusterID"] == ID]
        )
        xcmv = np.append(xcmv, xcm)
        ycmv = np.append(ycmv, ycm)
        zcmv = np.append(zcmv, zcm)
    df = df.assign(xcm=xcmv, ycm=ycmv, zcm=zcmv)

    # Step 4) Computing the Gyration Tensor and Eigenvalues

    Rgxx = Rgyy = Rgzz = np.arange(0, 0)
    Rgxy = Rgxz = Rgyz = np.arange(0, 0)
    L1 = L2 = L3 = np.arange(0, 0)
    Rgv = np.arange(0, 0)

    for item in clusterID:
        ID = item
        rgxx = sum(
            (df[df["clusterID"] == ID]["atomCoordx"] - df[df["clusterID"] == ID]["xcm"])
            ** 2
        ) / len(df[df["clusterID"] == ID])
        rgyy = sum(
            (df[df["clusterID"] == ID]["atomCoordy"] - df[df["clusterID"] == ID]["ycm"])
            ** 2
        ) / len(df[df["clusterID"] == ID])
        rgzz = sum(
            (df[df["clusterID"] == ID]["atomCoordz"] - df[df["clusterID"] == ID]["zcm"])
            ** 2
        ) / len(df[df["clusterID"] == ID])
        rgxy = sum(
            (df[df["clusterID"] == ID]["atomCoordx"] - df[df["clusterID"] == ID]["xcm"])
            * (
                df[df["clusterID"] == ID]["atomCoordy"]
                - df[df["clusterID"] == ID]["ycm"]
            )
        ) / len(df[df["clusterID"] == ID])
        rgxz = sum(
            (df[df["clusterID"] == ID]["atomCoordx"] - df[df["clusterID"] == ID]["xcm"])
            * (
                df[df["clusterID"] == ID]["atomCoordz"]
                - df[df["clusterID"] == ID]["zcm"]
            )
        ) / len(df[df["clusterID"] == ID])
        rgyz = sum(
            (df[df["clusterID"] == ID]["atomCoordy"] - df[df["clusterID"] == ID]["ycm"])
            * (
                df[df["clusterID"] == ID]["atomCoordz"]
                - df[df["clusterID"] == ID]["zcm"]
            )
        ) / len(df[df["clusterID"] == ID])

        tensor = np.array([rgxx, rgxy, rgxz, rgxy, rgyy, rgyz, rgxz, rgyz, rgzz])
        tensor = tensor.reshape(3, 3)

        eigenvalues, eigenvectors = LA.eig(tensor)
        Rg = sum(eigenvalues) ** (0.5)
        Rgxx = np.append(Rgxx, rgxx)
        Rgyy = np.append(Rgyy, rgyy)
        Rgzz = np.append(Rgzz, rgzz)
        Rgxy = np.append(Rgxy, rgxy)
        Rgxz = np.append(Rgxz, rgxz)
        Rgyz = np.append(Rgyz, rgyz)

        eigenvalues = np.sort(eigenvalues)
        L1 = np.append(L1, eigenvalues[0])
        L2 = np.append(L2, eigenvalues[1])
        L3 = np.append(L3, eigenvalues[2])
        Rgv = np.append(Rgv, Rg)

    df = df.assign(
        rgxx=Rgxx,
        rgyy=Rgyy,
        rgzz=Rgzz,
        rgxy=Rgxy,
        rgxz=Rgxz,
        rgyz=Rgyz,
        l1=L1,
        l2=L2,
        l3=L3,
        rg=Rgv,
    )

    # Step 5) Computing Shape Metrics and Cleaning Up

    data = df

    shapedata = data.iloc[data["clusterID"].drop_duplicates().index][
        [
            "clusterID",
            "clusterSize",
            "xcm",
            "ycm",
            "zcm",
            "rgxx",
            "rgyy",
            "rgzz",
            "rgxx",
            "rgyy",
            "rgzz",
            "rgxy",
            "rgxz",
            "rgyz",
            "l1",
            "l2",
            "l3",
            "rg",
        ]
    ]
    eig1 = data.iloc[data["clusterID"].drop_duplicates().index]["l1"]
    eig2 = data.iloc[data["clusterID"].drop_duplicates().index]["l2"]
    eig3 = data.iloc[data["clusterID"].drop_duplicates().index]["l3"]

    b2 = eig3 - 0.5 * (eig1 + eig2)
    c2 = eig2 - eig1
    k2 = (b2**2 + 0.75 * c2**2) / (eig1 + eig2 + eig3) ** 2
    aspect = np.sqrt(eig3 / eig1)

    shapedata = shapedata.assign(
        asphericity=b2, acylindricity=c2, anisotropy=k2, aspectratio=aspect
    )

    return shapedata
