import numpy as np
import scipy
from sklearn.cluster import SpectralClustering

# NME low-level operations
# These functions are taken from the Kaldi scripts.

# Prepares binarized(0/1) affinity matrix with p_neighbors non-zero elements in each row
def get_kneighbors_conn(X_dist, p_neighbors):
    X_dist_out = np.zeros_like(X_dist)
    for i, line in enumerate(X_dist):
        sorted_idx = np.argsort(line)
        sorted_idx = sorted_idx[::-1]
        indices = sorted_idx[:p_neighbors]
        X_dist_out[indices, i] = 1
    return X_dist_out


# Thresolds affinity matrix to leave p maximum non-zero elements in each row
def Threshold(A, p):
    N = A.shape[0]
    Ap = np.zeros((N, N))
    for i in range(N):
        thr = sorted(A[i, :], reverse=True)[p]
        Ap[i, A[i, :] > thr] = A[i, A[i, :] > thr]
    return Ap


# Computes Laplacian of a matrix
def Laplacian(A):
    d = np.sum(A, axis=1) - np.diag(A)
    D = np.diag(d)
    return D - A


# Calculates eigengaps (differences between adjacent eigenvalues sorted in descending order)
def Eigengap(S):
    S = sorted(S)
    return np.diff(S)


# Computes parameters of normalized eigenmaps for automatic thresholding selection
def ComputeNMEParameters(A, p, max_num_clusters):
    # p-Neighbour binarization
    Ap = get_kneighbors_conn(A, p)
    # Symmetrization
    Ap = (Ap + np.transpose(Ap)) / 2
    # Laplacian matrix computation
    Lp = Laplacian(Ap)
    # Get max_num_clusters+1 smallest eigenvalues
    S = scipy.sparse.linalg.eigsh(
        Lp,
        k=max_num_clusters + 1,
        which="SA",
        tol=1e-6,
        return_eigenvectors=False,
        mode="buckling",
    )
    # Get largest eigenvalue
    Smax = scipy.sparse.linalg.eigsh(
        Lp, k=1, which="LA", tol=1e-6, return_eigenvectors=False, mode="buckling"
    )
    # Eigengap computation
    e = Eigengap(S)
    g = np.max(e[:max_num_clusters]) / (Smax + 1e-10)
    r = p / g
    k = np.argmax(e[:max_num_clusters])
    return (e, g, k, r)


"""
Performs spectral clustering with Normalized Maximum Eigengap (NME)
Parameters:
   A: affinity matrix (matrix of pairwise cosine similarities or PLDA scores between speaker embeddings)
   num_clusters: number of clusters to generate (if None, determined automatically)
   max_num_clusters: maximum allowed number of clusters to generate
   pmax: maximum count for matrix binarization (should be at least 2)
   pbest: best count for matrix binarization (if 0, determined automatically)
Returns: cluster assignments for every speaker embedding   
"""


def NME_SpectralClustering(
    A, num_clusters=None, max_num_clusters=10, pbest=0, pmin=3, pmax=20
):
    print(num_clusters,max_num_clusters)
    if pbest == 0:
        print("Selecting best number of neighbors for affinity matrix thresolding:")
        rbest = None
        kbest = None
        for p in range(pmin, pmax + 1):
            e, g, k, r = ComputeNMEParameters(A, p, max_num_clusters)
            print("p={}, g={}, k={}, r={}, e={}".format(p, g, k, r, e))
            if rbest is None or rbest > r:
                rbest = r
                pbest = p
                kbest = k
        print("Best number of neighbors is {}".format(pbest))
        num_clusters = num_clusters if num_clusters is not None else (kbest + 1)
        # Handle some edge cases in AMI SDM
        num_clusters = 4 if num_clusters == 1 else num_clusters
        return NME_SpectralClustering_sklearn(
            A, num_clusters, pbest
        )
    if num_clusters is None:
        print("Compute number of clusters to generate:")
        e, g, k, r = ComputeNMEParameters(A, pbest, max_num_clusters)
        print("Number of clusters to generate is {}".format(k + 1))
        return NME_SpectralClustering_sklearn(A, k + 1, pbest)
    return NME_SpectralClustering_sklearn(A, num_clusters, pbest)


"""
Performs spectral clustering with Normalized Maximum Eigengap (NME) with fixed threshold and number of clusters
Parameters:
   A: affinity matrix (matrix of pairwise cosine similarities or PLDA scores between speaker embeddings)
   OLVec: 0/1 vector denoting which segments are overlap segments
   num_clusters: number of clusters to generate
   pbest: best count for matrix binarization
Returns: cluster assignments for every speaker embedding   
"""


def NME_SpectralClustering_sklearn(A, num_clusters, pbest):
    print("Number of speakers is {}".format(num_clusters))
    # Ap = Threshold(A, pbest)
    Ap = get_kneighbors_conn(A, pbest)  # thresholded and binarized
    Ap = (Ap + np.transpose(Ap)) / 2
    
    
    model = SpectralClustering(
        n_clusters=num_clusters, affinity="precomputed", random_state=0
    )
    labels = model.fit_predict(Ap)
    return labels
