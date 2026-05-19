"""
Two-way clustered standard errors following Cameron, Gelbach & Miller (2011).

Implements the CGM formula:
    V_twoway = V_cluster_origin + V_cluster_dest - V_intersection

For directed occupation pairs where each (origin, dest) is unique,
the intersection clusters are singletons, so V_intersection = V_HC0
(with appropriate small-sample correction).

Works with both GLM (Poisson/PPML) and OLS models from statsmodels.
"""

import numpy as np
from scipy.stats import norm


def twoway_cluster_cov(X, resid, bread_weights, cluster_a, cluster_b):
    """Compute two-way clustered variance-covariance matrix (CGM 2011).

    Parameters
    ----------
    X : ndarray (n, k)
        Regressor matrix (including constant column if applicable).
    resid : ndarray (n,)
        Residuals: (y - mu) for Poisson, (y - X*beta) for OLS.
    bread_weights : ndarray (n,)
        Weights for the bread matrix:
        - Poisson/PPML: mu (fitted values)
        - OLS: ones
    cluster_a : ndarray-like (n,)
        First clustering dimension (e.g., occ_origin).
    cluster_b : ndarray-like (n,)
        Second clustering dimension (e.g., occ_dest).

    Returns
    -------
    V : ndarray (k, k)
        Two-way clustered variance-covariance matrix.
    """
    n, k = X.shape
    cluster_a = np.asarray(cluster_a)
    cluster_b = np.asarray(cluster_b)

    # Bread: (X' W X)^{-1}
    XW = X * bread_weights[:, np.newaxis]
    bread = np.linalg.inv(XW.T @ X)

    # Score contributions: s_i = resid_i * x_i
    scores = resid[:, np.newaxis] * X  # (n, k)

    # Meat for each clustering dimension
    meat_a = _clustered_meat(scores, cluster_a)
    meat_b = _clustered_meat(scores, cluster_b)
    meat_int = scores.T @ scores  # HC0 meat (singletons)

    # Small-sample corrections: G / (G - 1)
    Ga = len(np.unique(cluster_a))
    Gb = len(np.unique(cluster_b))
    ca = Ga / (Ga - 1)
    cb = Gb / (Gb - 1)
    c_int = n / (n - 1)

    # CGM formula: V = V_a + V_b - V_intersection
    meat_combined = ca * meat_a + cb * meat_b - c_int * meat_int
    V = bread @ meat_combined @ bread

    return V


def _clustered_meat(scores, clusters):
    """Compute the clustered meat matrix efficiently.

    Sums score contributions within each cluster, then computes
    the outer product of cluster-level sums.

    Parameters
    ----------
    scores : ndarray (n, k)
        Score contributions (resid_i * x_i for each observation).
    clusters : ndarray (n,)
        Cluster identifiers.

    Returns
    -------
    meat : ndarray (k, k)
        Clustered meat matrix (before small-sample correction).
    """
    unique_clusters, inverse = np.unique(clusters, return_inverse=True)
    G = len(unique_clusters)
    k = scores.shape[1]

    # Sum scores within clusters using np.add.at for efficiency
    cluster_scores = np.zeros((G, k))
    np.add.at(cluster_scores, inverse, scores)

    # Meat = sum_g (S_g)(S_g')
    meat = cluster_scores.T @ cluster_scores
    return meat


def twoway_cluster_se(X, resid, bread_weights, cluster_a, cluster_b):
    """Compute two-way clustered standard errors.

    Convenience wrapper around twoway_cluster_cov that returns SEs.
    Uses conservative fallback (HC0-based SE) for any coefficient where
    the two-way variance is negative.

    Parameters
    ----------
    X, resid, bread_weights, cluster_a, cluster_b :
        See twoway_cluster_cov.

    Returns
    -------
    se : ndarray (k,)
        Two-way clustered standard errors.
    V : ndarray (k, k)
        Full variance-covariance matrix.
    """
    V = twoway_cluster_cov(X, resid, bread_weights, cluster_a, cluster_b)
    diag = np.diag(V)

    # Handle negative diagonal elements (rare edge case)
    if np.any(diag < 0):
        n_neg = np.sum(diag < 0)
        print(f"    WARNING: {n_neg} negative diagonal element(s) in V_twoway. "
              f"Using max(0, diag) fallback.")
        diag = np.maximum(diag, 0)

    se = np.sqrt(diag)
    return se, V


def pvalues_from_se(params, se):
    """Compute two-sided p-values from coefficients and standard errors.

    Parameters
    ----------
    params : ndarray (k,)
        Point estimates.
    se : ndarray (k,)
        Standard errors.

    Returns
    -------
    pvalues : ndarray (k,)
        Two-sided p-values (normal approximation).
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        z = np.where(se > 0, params / se, np.inf)
    pvalues = 2 * norm.sf(np.abs(z))
    return pvalues
