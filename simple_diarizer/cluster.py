import numpy as np

import scipy.cluster.hierarchy as hcluster
from scipy.sparse.csgraph import laplacian
from scipy.ndimage import gaussian_filter
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics import pairwise_distances


def similarity_matrix(embeds, metric="cosine"):
    return pairwise_distances(embeds, metric=metric)


def cluster_AHC(embeds, n_clusters=None, threshold=None, metric="cosine", **kwargs):
    """
    Cluster embeds using Agglomerative Hierarchical Clustering
    """
    if n_clusters is None:
        assert threshold, "If num_clusters is not defined, threshold must be defined"

    S = similarity_matrix(embeds, metric=metric)

    if n_clusters is None:
        cluster_model = AgglomerativeClustering(
            n_clusters=None,
            affinity="precomputed",
            linkage="average",
            compute_full_tree=True,
            distance_threshold=threshold,
        )

        return cluster_model.fit_predict(S)
    else:
        cluster_model = AgglomerativeClustering(
            n_clusters=n_clusters, affinity="precomputed", linkage="average"
        )

        return cluster_model.fit_predict(S)


##########################################
# Spectral clustering
# A lot of these methods are lifted from
# https://github.com/wq2012/SpectralCluster
##########################################


def cluster_SC(embeds, n_clusters=None, threshold=None, enhance_sim=True, **kwargs):
    """
    Cluster embeds using Spectral Clustering
    """
    if n_clusters is None:
        assert threshold, "If num_clusters is not defined, threshold must be defined"

    S = compute_affinity_matrix(embeds)
    if enhance_sim:
        S = sim_enhancement(S)

    if n_clusters is None:
        (eigenvalues, eigenvectors) = compute_sorted_eigenvectors(S)
        # Get number of clusters.
        k = compute_number_of_clusters(eigenvalues, 100, threshold)

        # Get spectral embeddings.
        spectral_embeddings = eigenvectors[:, :k]

        # Run K-Means++ on spectral embeddings.
        # Note: The correct way should be using a K-Means implementation
        # that supports customized distance measure such as cosine distance.
        # This implemention from scikit-learn does NOT, which is inconsistent
        # with the paper.
        kmeans_clusterer = KMeans(
            n_clusters=k, init="k-means++", max_iter=300, random_state=0
        )
        labels = kmeans_clusterer.fit_predict(spectral_embeddings)
        return labels
    else:
        cluster_model = SpectralClustering(
            n_clusters=n_clusters, affinity="precomputed"
        )

        return cluster_model.fit_predict(S)


def diagonal_fill(A):
    """
    Sets the diagonal elemnts of the matrix to the max of each row
    """
    np.fill_diagonal(A, 0.0)
    A[np.diag_indices(A.shape[0])] = np.max(A, axis=1)
    return A


def gaussian_blur(A, sigma=1.0):
    """
    Does a gaussian blur on the affinity matrix
    """
    return gaussian_filter(A, sigma=sigma)


def row_threshold_mult(A, p=0.95, mult=0.01):
    """
    For each row multiply elements smaller than the row's p'th percentile by mult
    """
    percentiles = np.percentile(A, p * 100, axis=1)
    mask = A < percentiles[:, np.newaxis]

    A = (mask * mult * A) + (~mask * A)
    return A


def symmetrization(A):
    """
    Symmeterization: Y_{i,j} = max(S_{ij}, S_{ji})
    """
    return np.maximum(A, A.T)


def diffusion(A):
    """
    Diffusion: Y <- YY^T
    """
    return np.dot(A, A.T)


def row_max_norm(A):
    """
    Row-wise max normalization: S_{ij} = Y_{ij} / max_k(Y_{ik})
    """
    maxes = np.amax(A, axis=1)
    return A / maxes


def sim_enhancement(A):
    func_order = [
        diagonal_fill,
        gaussian_blur,
        row_threshold_mult,
        symmetrization,
        diffusion,
        row_max_norm,
    ]
    for f in func_order:
        A = f(A)
    return A


def compute_affinity_matrix(X):
    """Compute the affinity matrix from data.
    Note that the range of affinity is [0,1].
    Args:
        X: numpy array of shape (n_samples, n_features)
    Returns:
        affinity: numpy array of shape (n_samples, n_samples)
    """
    # Normalize the data.
    l2_norms = np.linalg.norm(X, axis=1)
    X_normalized = X / l2_norms[:, None]
    # Compute cosine similarities. Range is [-1,1].
    cosine_similarities = np.matmul(X_normalized, np.transpose(X_normalized))
    # Compute the affinity. Range is [0,1].
    # Note that this step is not mentioned in the paper!
    affinity = (cosine_similarities + 1.0) / 2.0
    return affinity


def compute_sorted_eigenvectors(A):
    """Sort eigenvectors by the real part of eigenvalues.
    Args:
        A: the matrix to perform eigen analysis with shape (M, M)
    Returns:
        w: sorted eigenvalues of shape (M,)
        v: sorted eigenvectors, where v[;, i] corresponds to ith largest
           eigenvalue
    """
    # Eigen decomposition.
    eigenvalues, eigenvectors = np.linalg.eig(A)
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real
    # Sort from largest to smallest.
    index_array = np.argsort(-eigenvalues)
    # Re-order.
    w = eigenvalues[index_array]
    v = eigenvectors[:, index_array]
    return w, v


def compute_number_of_clusters(eigenvalues, max_clusters=None, stop_eigenvalue=1e-2):
    """Compute number of clusters using EigenGap principle.
    Args:
        eigenvalues: sorted eigenvalues of the affinity matrix
        max_clusters: max number of clusters allowed
        stop_eigenvalue: we do not look at eigen values smaller than this
    Returns:
        number of clusters as an integer
    """
    max_delta = 0
    max_delta_index = 0
    range_end = len(eigenvalues)
    if max_clusters and max_clusters + 1 < range_end:
        range_end = max_clusters + 1
    for i in range(1, range_end):
        if eigenvalues[i - 1] < stop_eigenvalue:
            break
        delta = eigenvalues[i - 1] / eigenvalues[i]
        if delta > max_delta:
            max_delta = delta
            max_delta_index = i
    return max_delta_index
