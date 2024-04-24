"""Include utilities for unsupervised machine learning of microstructure families."""

import umap
from sklearn.preprocessing import StandardScaler
import sklearn.cluster as cluster
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score


def locateFamilies(inp):
    """Find microstructural families given shape data using unsupervised machine learning.

    Parameters
    ----------
    inp : pdDataframe
        2D array containing shape information about microstructures.

    Returns
    -------
    re : ndarray
        1D array containing real space radial vectors, r, of `float` type.
    """
    plt.rcParams["figure.figsize"] = [6, 6]

    cluster_data = inp[
        [
            "rg",
            "asphericity",
            "acylindricity",
            "aspectratio",
            "clusterSize",
            "anisotropy",
        ]
    ].values

    scaled_cluster_data = StandardScaler().fit_transform(
        cluster_data
    )  # scale data for ML
    standard_embedding = umap.UMAP(
        n_neighbors=30, min_dist=0.1, n_components=2, random_state=25
    ).fit_transform(
        scaled_cluster_data
    )  # create UMAP embedding

    plt.rcParams["figure.figsize"] = [12, 6]

    # We feed in the embedding from the UMAP section
    kmeans_labels = cluster.KMeans(n_clusters=3, max_iter=100).fit_predict(
        standard_embedding
    )  # calculate KMeans clustering for a set number of clusters

    # plotting
    fig, axs = plt.subplots(1, 2)

    axs[0].scatter(
        standard_embedding[:, 0], standard_embedding[:, 1], c="black", s=1.5, alpha=0.6
    )
    axs[0].tick_params(
        left=False, right=False, labelleft=False, labelbottom=False, bottom=False
    )
    axs[0].set_title("Unlabeled Data", fontsize=14)

    axs[1].scatter(
        standard_embedding[:, 0],
        standard_embedding[:, 1],
        c=kmeans_labels,
        s=1.5,
        alpha=0.6,
    )
    axs[1].tick_params(
        left=False, right=False, labelleft=False, labelbottom=False, bottom=False
    )
    axs[1].set_title("K-Means", fontsize=14)

    plt.rcParams["figure.figsize"] = [6, 6]

    # Write to these lists to plot
    inertias = []
    silhouettelist = []

    # The data K-Means will work with comes from the UMAP Reduction
    X = standard_embedding

    for k in range(2, 11):  # hyperparameter tuning for optimal clusters
        kmeanModel = KMeans(n_clusters=k).fit(X)
        kmeanModel2 = KMeans(n_clusters=k)
        kmeanModel2.fit_predict(X)
        inertias.append(kmeanModel.inertia_)
        silhouette = silhouette_score(X, kmeanModel2.labels_)
        silhouettelist.append(silhouette)

    fig, ax = plt.subplots()

    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.tick_params(axis="both", which="minor", labelsize=15)

    Inertia = plt.plot(np.arange(2, 11, 1), inertias, "bo-", label="Inertia")
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    plt.xlabel("Number of Clusters, N", fontsize=15)
    plt.ylabel("Inertia, I", fontsize=15, color="Blue")

    ax2 = ax.twinx()
    Silhouette = ax2.plot(
        np.arange(2, 11, 1), silhouettelist, "ro-", label="Silhouette"
    )
    ax2.set_ylim([0, 1])
    ax2.tick_params(axis="both", which="major", labelsize=15)
    ax2.tick_params(axis="both", which="minor", labelsize=15)

    plt.xlabel("Number of Clusters, N", fontsize=15)
    plt.ylabel("Silhouette Score, S", fontsize=15, color="Red")
    plt.xticks(np.arange(2, 11, 1))

    lines = Inertia + Silhouette
    ax.legend(lines, [label.get_label() for label in lines])
    clusters = np.arange(2, 11, 1)

    return clusters[np.argmax(silhouettelist)]
