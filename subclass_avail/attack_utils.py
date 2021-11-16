"""
General utility functions to perform the attack
"""

from numpy import ndarray
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# Clustering functions, see https://github.com/google-research/mixmatch

def pca_whiten(x, pca_dim):
    """ Perform PCA decomposition

    Args:
        x (ndarray): data matrix
        pca_dim (int): number of dimensions to keep

    Returns:
        (ndarray, PCA): transformed matrix, PCA object
    """

    pca = PCA(n_components=pca_dim)
    new_x = pca.fit_transform(x)
    return new_x, pca


def kmeans_cluster(x, clusters):
    """ Perform K-means clustering

    Args:
        x (ndarray): data matrix
        clusters (int): number of centroids

    Returns:
        (ndarray, KMeans): clustering labels, K-means object
    """

    km = KMeans(n_clusters=clusters)
    km.fit(x)
    return km.labels_, km


def clustering(x, cluster_alg='km', clusters=100, pca_dim=None):
    """

    Args:
        x (ndarray): data matrix
        cluster_alg (str): type of clustering algorithm to use
        clusters (int): number of centroids for K-means
        pca_dim (int): number of dimensions for PCA decomposition

    Returns:
        (ndarray, function): clustering labels, clustering predictor
    """

    if pca_dim is not None:
        new_x, pca = pca_whiten(x, pca_dim)
        pca_fn = pca.transform
    else:
        new_x = x
        pca_fn = lambda inp: inp

    if cluster_alg == 'km':
        labels, km = kmeans_cluster(new_x, clusters)
        pred_fn = lambda inp: km.predict(pca_fn(inp))
    else:
        raise NotImplementedError

    return labels, pred_fn
