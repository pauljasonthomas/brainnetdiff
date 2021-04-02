#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020--, CoNeCT Lab, University of Illinois at Chicago .

Distributed under the terms of the Modified BSD License.

The full license is in the file COPYING.txt, distributed with this software.
"""
__all__ = ['nbs_ttest_compute',
           'nbs_ttest_compute_null',
           'nbs_cor_compute_null',
           'nbs_cor_compute']

import numpy as np
from bct.algorithms.clustering import get_components
from scipy.stats import t as spt
from scipy.stats import rankdata
from tqdm import tqdm
import networkx as nx
from networkx.algorithms.components import connected_components


def _tstat_compute(x, y):

    """Return t statistics for comparing rows

    Parameters
    ----------
    x : numpy array
        size M x S0, where M is the number of edges,
        and S0 is the number of subjects in group 0
    y : numpy array
        size M x S1, where M is the number of edges,
        and S1 is the number of subjects in group 1
    tail : {'left', 'right', 'both'}
        enables specification of particular alternative hypothesis
        'left' : mean population of  group 0 < mean population of group 1
        'right' : mean population of group 1 < mean population of group 0
        'both' : means are unequal (default)

    Returns
    -------
    t_stat : numpy array
        size M, contains t-statistics for each row of X and Y
    """
    nx = x.shape[1]
    ny = y.shape[1]
    mean_xy = np.mean(x, axis = 1) - np.mean(y, axis = 1)
    var_x = np.var(x, axis = 1, ddof=1)
    var_y = np.var(y, axis = 1, ddof=1)
    s = np.sqrt( ((nx - 1) * var_x + (ny - 1) * var_y) / (nx + ny - 2))
    denom = s * np.sqrt((1 / nx) + (1 / ny))
    denom[denom == 0] = 1000 #to prevent division by zero
    t_stat = np.divide(mean_xy, denom)
    return t_stat


def nbs_ttest_compute_null(X, g0_idx, g1_idx, nperms):

    """ Returns null distribution of t-statistics for each edge

    Parameters
    ----------
    X : numpy array
        size N x N x S, where N is the number of nodes and S is the total
        number of subjects
    g0_idx : numpy array
        size S0, indicates the indices of X for group 0
    g1_idx : numpy array
        size S1, indicates the indices of X for group 1
    nperms : scalar
        indicates the number of permutations to perform
    tail : string
        indicates type of t-test (see tstat_perm_compute)

    Returns
    -------
    tstat_null : numpy array
        size M x nperms, contains t-statistics from group permutations for
        each edge

    """
    n = X.shape[0]
    s = X.shape[2]
    ixes = np.where(np.triu(np.ones((n, n)), 1))
    m = np.size(ixes, axis=1)

    xmat = np.zeros((m, s))

    for i in range(s):

        xmat[:, i] = X[:, :, i][ixes].squeeze()

    tstat_null = np.zeros((m, nperms))

    print('generating null distribution')

    for i in tqdm(range(nperms)):

        xperm = xmat[:, np.random.permutation(xmat.shape[1])]
        x = xperm[:, g0_idx]
        y = xperm[:, g1_idx]

        tstat_null[:, i] = _tstat_compute(x, y)

    return tstat_null





def nbs_ttest_compute(X, g0_idx, g1_idx, thresh, nperms, tail, null_tdist = None, verbose=False):

    n = X.shape[0]
    s = X.shape[2]
    ixes = np.where(np.triu(np.ones((n, n)), 1))
    m = np.size(ixes, axis=1)

    xmat = np.zeros((m, s))

    for i in range(s):

        xmat[:, i] = X[:, :, i][ixes].squeeze()

    x = xmat[:, g0_idx]
    y = xmat[:, g1_idx]

    # perform t-test at each edge
    t_stat = _tstat_compute(x, y)

    if tail == "both":
        t_stat = abs(t_stat)
    elif tail == "left":
        t_stat = -t_stat

    # threshold
    ind_t, = np.where(t_stat > thresh)

    if len(ind_t) == 0:
        print("unsustainable threshold")
        return np.array([1]), None, None, None

    # suprathreshold adjacency matrix
    adj = np.zeros((n, n))
    adj[(ixes[0][ind_t], ixes[1][ind_t])] = 1
    adj = adj + adj.T

    G = nx.from_numpy_matrix(adj)
    S = [list(G.subgraph(c).copy()._adj.keys()) \
         for c in sorted(connected_components(G),
                          key=len, reverse=True)]
    AC = np.zeros(n)
    SZ = np.zeros(len(S))
    for i, s in enumerate(S):
        AC[s] = i + 1
        SZ[i] = len(s)

    sz_links = np.array([len(G.subgraph(c).copy().edges) \
                         for c in sorted(connected_components(G),
                         key=len, reverse=True)])
    ind_sz, = np.where(SZ > 1)
    nr_components = np.size(ind_sz)

    if np.size(sz_links):
        max_sz = np.max(sz_links)
    else:
        # max_sz=0
        print("unsustainable threshold")
        return np.array([1]), AC, SZ, adj

    print('Maximum component size:')
    print('number of nodes = ', np.max(SZ))
    print('number of edges = ', max_sz)

    # estimate empirical null distribution of maximum component size by
    # generating k independent permutations
    print('estimating null distribution with %i permutations' % nperms)

    null = np.zeros((nperms,))
    hit = 0

    #generate null distribution
    if null_tdist is None:

        null_tdist = nbs_ttest_compute_null(X, g0_idx, g1_idx, nperms, tail = 'both')

    nperms = null_tdist.shape[1]

    for u in tqdm(range(nperms)):

        t_stat_perm = np.array(null_tdist[:,u]).reshape((m,))

        ind_t, = np.where(t_stat_perm > thresh)

        adj_perm = np.zeros((n, n))
        adj_perm[(ixes[0][ind_t], ixes[1][ind_t])] = 1
        adj_perm = adj_perm + adj_perm.T

        G = nx.from_numpy_matrix(adj_perm)
        mcc = max(nx.algorithms.components.connected_components(G), key=len)
        S = G.subgraph(mcc)
        null[u] = len(S.edges)

        # compare to the true dataset
        if null[u] >= max_sz:
            hit += 1

        if verbose:

            if (u % (nperms / 10) == 0 or u == nperms - 1):
                print('permutation %i of %i.  p-value so far is %.3f' % (u, nperms,
                                                                     hit / (u + 1)))

    pvals = np.zeros((nr_components,))
    # calculate p-vals
    for i in range(nr_components):
        pvals[i] = np.size(np.where(null >= sz_links[i])) / nperms

    return pvals, AC, SZ, adj


def _critical_rho(n, alpha = 0.05):

    """computes critical rho value given n and p = alpha

    Parameters
    ----------
    n : scalar
        number of samples
    alpha : scalar
        p-value threshold

    Returns
    -------
    r_crit : scalar
        critical rho for given n and alpha

    """

    df = n - 2
    t_crit = spt.ppf(alpha, df)
    r_crit = np.sqrt( (t_crit**2) / ( (t_crit**2) + df ) )

    return r_crit


def _cor_stat_compute(a, b):

    """computes spearman's rho statistic

    Parameters
    ----------
    a : 2D array of shape M x N
    b : 1D array length M

    Returns
    -------
    rho : scalar
        rho statistic for cor(a, b)
    """
    n = a.shape[1]
    b = b.reshape(-1, 1)

    A = a.T
    B = b.T

    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    # Finally get corr coeff
    cc = np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))
    return cc.reshape((n,))


def nbs_cor_compute_null(X, Y, nperms, cor_type = 'spearman'):

    """ Returns null distribution of t-statistics for each edge
    Parameters
    ----------
    X : numpy array
        size N x N x S, where N is the number of nodes and S is the total
        number of subjects
    Y : numpy array
        size S, array of trait value for correlation with edges in NBS
    nperms : scalar
        indicates the number of permutations to perform
    cor_type : {'spearman, pearson'}
        indicates type of correlation to use
    Returns
    -------
    cstat_null : numpy array
        size M x nperms, contains correlation statistics from group permutations for each edge
    """
    n = X.shape[0]
    s = X.shape[2]
    ixes = np.where(np.triu(np.ones((n, n)), 1))
    m = np.size(ixes, axis=1)

    xmat = np.zeros((m, s))

    for i in range(s):

        xmat[:, i] = X[:, :, i][ixes].squeeze()

    xmat = xmat.T

    if cor_type == 'spearman':

        xmat = np.apply_along_axis(rankdata, 0, xmat)
        y = rankdata(Y)

    else:

        y = Y

    cstat_null = np.zeros((m, nperms))

    print('generating null distribution')

    for i in tqdm(range(nperms)):

        yperm = y[np.random.permutation(y.shape[0])]

        cstat_null[:, i] = _cor_stat_compute(xmat, yperm)

    return cstat_null


def nbs_cor_compute(X, Y, thresh, nperms, method, cor_type = 'spearman', null_cdist = None, verbose=False, alpha = 0.05):

    n = X.shape[0]
    s = X.shape[2]
    ixes = np.where(np.triu(np.ones((n, n)), 1))
    m = np.size(ixes, axis=1)

    xmat = np.zeros((m, s))

    for i in range(s):

        xmat[:, i] = X[:, :, i][ixes].squeeze()

    xmat = xmat.T

    if cor_type == 'spearman':

        xmat = np.apply_along_axis(rankdata, 0, xmat)
        y = rankdata(Y)

    else:

        y = Y

    # perform correlation at each edge
    r_stat = _cor_stat_compute(xmat, y)

    r_stat[np.isnan(r_stat)] = 0

    # determine threshold
    if thresh is None:

        thresh = _critical_rho(s, alpha)

    # transform r to absolute, positive or negative only values
    if method == "abs":

        r_stat = abs(r_stat)

    elif method == "neg":

        r_stat = -r_stat

    ind_r, = np.where(r_stat > thresh)

    if len(ind_r) == 0:
        print("unsustainable threshold")
        return np.array([1]), None, None, None

    # suprathreshold adjacency matrix
    adj = np.zeros((n, n))
    adj[(ixes[0][ind_r], ixes[1][ind_r])] = 1
    adj = adj + adj.T

    G = nx.from_numpy_matrix(adj)
    S = [list(G.subgraph(c).copy()._adj.keys()) \
         for c in sorted(connected_components(G),
                          key=len, reverse=True)]
    AC = np.zeros(n)
    SZ = np.zeros(len(S))
    for i, s in enumerate(S):
        AC[s] = i + 1
        SZ[i] = len(s)

    sz_links = np.array([len(G.subgraph(c).copy().edges) \
                         for c in sorted(connected_components(G),
                         key=len, reverse=True)])
    ind_sz, = np.where(SZ > 1)
    nr_components = np.size(ind_sz)

    if np.size(sz_links):
        max_sz = np.max(sz_links)
    else:
        # max_sz=0
        print("unsustainable threshold")
        return np.array([1]), AC, SZ, adj

    print('Maximum component size:')
    print('number of nodes = ', np.max(SZ))
    print('number of edges = ', max_sz)

    # estimate empirical null distribution of maximum component size by
    # generating k independent permutations
    print('estimating null distribution with %i permutations' % nperms)

    null = np.zeros((nperms,))
    hit = 0

    #generate null distribution
    if null_cdist is None:

        print("computing null distribution")

        null_cdist = nbs_cor_compute_null(X, Y, nperms, cor_type = cor_type)

    nperms = null_cdist.shape[1]

    null_cdist[np.isnan(null_cdist)] = 0

    # transform r for absolute or negative only values
    if method == "abs":

        null_cdist = abs(null_cdist)

    elif method == "neg":

        null_cdist = -null_cdist

    #permutation analysis
    for u in tqdm(range(nperms)):

        r_stat_perm = np.array(null_cdist[:,u]).reshape((m,))

        ind_r, = np.where(r_stat_perm > thresh)

        adj_perm = np.zeros((n, n))
        adj_perm[(ixes[0][ind_r], ixes[1][ind_r])] = 1
        adj_perm = adj_perm + adj_perm.T

        G = nx.from_numpy_matrix(adj_perm)
        mcc = max(nx.algorithms.components.connected_components(G), key=len)
        S = G.subgraph(mcc)
        null[u] = len(S.edges)

        # compare to the true dataset
        if null[u] >= max_sz:
            hit += 1

        if verbose:

            if (u % (nperms / 10) == 0 or u == nperms - 1):
                print('permutation %i of %i.  p-value so far is %.3f' % (u, nperms,
                                                                     hit / (u + 1)))

    pvals = np.zeros((nr_components,))
    # calculate p-vals
    for i in range(nr_components):
        pvals[i] = np.size(np.where(null >= sz_links[i])) / nperms

    return pvals, AC, SZ, adj
