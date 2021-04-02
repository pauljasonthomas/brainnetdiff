#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2021--, CoNeCT Lab, University of Illinois at Chicago.

Distributed under the terms of the Modified BSD License.

The full license is in the file COPYING.txt, distributed with this software.
"""

import numpy as np
from scipy.sparse.csgraph import laplacian, shortest_path
from scipy.linalg import expm
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.linalg import eigh
from scipy.io import loadmat
from tqdm import tqdm
import os
import glob
from math import floor
from scipy.stats import rankdata
import pandas as pd

def _laplacian_compute(X, norm=True):
    lX = np.zeros(X.shape)
    for i in range(X.shape[2]):
        lX[:,:,i] = laplacian(X[:, :, i].copy(), normed=norm)
    return lX

def _eigenmodes_compute(lX):
    eX = np.zeros(lX.shape)
    evals = []
    for i in range(lX.shape[2]):
        w, v = eigh(lX[:,:,i].copy())
        eX[:,:,i] = v
        evals.append(w)
    return eX, evals

def ctm_corr(X, Y):
    """Computes the pearson correlation between two connectome adjacency matrices as in https://doi.org/10.1016/j.media.2020.101799
    """
    n = X.shape[0]
    def _bar(A, n):
        return (2/(n**2 - n)) * A[np.triu_indices(n, k=1)].sum()
    def _sym_norm(A, abar, n):
        return ((A[np.triu_indices(n, k=1)] - abar)**2).sum()
    xbar = _bar(X, n)
    ybar = _bar(Y, n)
    ind = np.triu_indices(n, k=1)
    numer = ((X[ind] - xbar) * (Y[ind] - ybar)).sum()
    denom = np.sqrt(_sym_norm(X, xbar, n) * _sym_norm(Y, ybar, n))
    return numer / denom

def reorder_regions(cv):
    uq, uqn = pd.factorize(cv)
    rv = []
    cl = []
    for i in range(uq.max() + 1):
        for j, k in enumerate(uq):
            if k == i:
                rv.append(j)
    for i in range(uq.max() + 1):
        cl.append(np.where(uq[rv] == i)[0])
    return rv, cl, uqn

def reorder_ctm(X, rv):
    if len(X.shape) == 3:
        return X[np.ix_(rv, rv, np.arange(X.shape[2]))]
    elif len(X.shape) == 2:
        return X[np.ix_(rv, rv)]
    else:
        raise ValueError('X must be 2- or 3-dimensional array')
    return

def fdr(p_vals):
    ranked_p_values = rankdata(p_vals)
    fdr = p_vals * len(p_vals) / ranked_p_values
    fdr[fdr > 1] = 1
    return fdr

def lexpm(ex, ev, t):
    return ex @ np.diag(np.exp(-t * ev)) @ ex.T

def norm_matrix(X):
    if len(X.shape) == 3:
        N, _, S = X.shape
        Xn = np.zeros((N,N,S))
        for i in range(S):
            for j in range(N):
                Xn[:,j,i] = X[:,j,i] / np.linalg.norm(X[:,j,i])
        return Xn
    else:
        N, _ = X.shape
        Xn = np.zeros((N,N))
        for j in range(N):
            Xn[:,j] = X[:,j] / np.linalg.norm(X[:,j])
        return Xn

def make_ctm_array(path, node_idcs = None):
    os.chdir(path)
    filenames = glob.glob("*.mat")
    filenames = sorted(filenames)
    ctms = []
    for i in filenames:
        ctms.append(loadmat(i)['connectivity'])
    if node_idcs is None:
        ctm_array = np.zeros([ctms[0].shape[0], ctms[0].shape[1], len(ctms)])
        for i in range(len(ctms)):
            ctm_array[:,:,i] = ctms[i]
        subjects = [i.split('.')[0] for i in filenames]
    else:
        c_idx = np.array(node_idcs)
        r_idx = c_idx[:, None]
        ctm_array = np.zeros([len(c_idx), len(c_idx), len(ctms)])
        for i in range(len(ctms)):
            ctm_array[:,:,i] = ctms[i][r_idx, c_idx]
        subjects = [i.split('.')[0] for i in filenames]
    os.chdir(os.getcwd())
    return [ctm_array, subjects]


def norm_ctm(X, method='ssq', p=0.5):
    """Normalize 2- or 3-way connectome array

    Parameters
    ----------
    X : numpy array
        size N x N x Z, where N is the number of ROIs,
        and Z is the number of subjects, or N x N array.
    method : string
        Normalization method to use. 'ssq' (default) normalizes each
        subject connectome to unit mass 1. 'max' normalizes each subject
        connectome by max value.
    p : scalar
        Exponentiation constant for 'ssq'. 0.5 (default) for sqrt sum
        squares, 1 for sum of squares normalization

    Returns
    -------
    Xn : numpy array numpy array of size X normalized according
    to method.
    """
    dim = X.shape
    if method == 'ssq':
        if len(dim) == 2:
            Xn = X / (np.sum(X**2)**p)
        elif len(dim) == 3:
            Xn = np.zeros(dim)
            for k in range(dim[2]):
                Xn[:,:,k] = X[:,:,k] / (np.sum(X[:,:,k]**2)**p)
    elif method == 'max':
        if len(dim) == 2:
            Xn = X / np.max(X)
        elif len(dim) == 3:
            Xn = np.zeros(dim)
            for k in range(dim[2]):
                Xn[:,:,k] = X[:,:,k] / np.max(X[:,:,k])
    return Xn

def is_connected(A):
    sp = shortest_path(A, method = 'D', directed = False, unweighted = False)
    return False if np.isinf(np.max(sp)) else True

def dens_thresh(X, D):
    """Return density thresholded ctms

    Parameters
    ----------
    X : numpy array
        size N x N x Z, where N is the number of ROIs,
        and Z is the number of subjects
    D : scalar (from 0-1)
           percent binary density to threshold connectomes at
           this function will prune connections until a connectome
           is at density = D or until it just becomes disconnected,
           whichever happens first

    Returns
    -------
    dX : numpy array of size X
         density thresholded connectomes
    density : list (numeric)
              density for each subject
    disc : list (bolean)
           T/F for each subject indicating if connected (T)
    """
    density = []
    disc = []
    n = X.shape[0]
    s = X.shape[2]
    ixes = np.where(np.triu(np.ones((n, n)), 1))
    m = np.size(ixes, axis=1)
    dX = np.zeros((n, n, s))
    for i in range(s):
        print("processing subject", i+1, "of", s)
        x1 = np.copy(X[:,:,i])
        x = x1[ixes]
        #calculate current density:
        dens = np.count_nonzero(x) / m
        con = is_connected(x1)
        if not con:
            print('subject', i+1, 'disconnected')
            density.append(dens)
            disc.append(True)
            dX[:,:,i] = x1
        elif dens <= D and con:
            print('subject', i+1, 'at or below threshold and connected')
            density.append(dens)
            disc.append(False)
            dX[:,:,i] = x1
        else:
            #find initial density guess:
            y = np.zeros(m); Y = np.zeros((n, n))
            ne = floor(D * m)
            asx = np.argsort(x)
            con = False
            while con is False:
                y = np.zeros(m); Y = np.zeros((n, n))
                y[asx[-ne:]] = x[asx[-ne:]]
                Y[ixes] = y; Y += Y.T
                con = is_connected(Y)
                dens = np.count_nonzero(y) / m
                if con:
                    break
                ne += 1
            print('target density: ', D, 'actual:', dens)
            dX[:,:,i] = np.copy(Y)
            density.append(dens)
            disc.append(con)
    return dX, density, disc

def get_ctm_density(X):
    """Return binary density of ctm(s)

    Parameters
    ----------
    X : numpy array
        size N x N (x Z), where N is the number of ROIs,
        and Z is the number of subjects, if 3-way array

    Returns
    -------
    density : scalar or list
              density for each subject or for ctm
    con : Boolean or list
            indicates whether ctm(s) are connected (True)
    """
    n = (X.shape[0]**2) - X.shape[0]
    if len(X.shape) == 3:
        density = []; disc = []
        for i in range(X.shape[2]):
            density.append(np.count_nonzero(X[:,:,i]) / n)
            disc.append(is_connected(X[:,:,i]))
        print("mean binary density =", sum(density) / len(density))
        print("subjects connected: ", disc.count(True), "/", len(disc))
    elif len(X.shape) == 2:
        density = np.count_nonzero(X) / n
        disc = is_connected(X)
        print("binary density =", density, "connected: ", disc)
    return density, disc


def fmri_thresh(X, thresh = 0.05, negrm = False):
    if len(X.shape) == 3:
        N, _, S = X.shape
        Xn = np.zeros((N, N, S))
        for i in range(S):
            xn = np.zeros((N,N))
            np.copyto(xn, X[:,:,i])
            thr = abs(xn).max() * thresh
            xn[abs(xn) < thr] = 0
            Xn[:,:,i] = xn
    else:
        N, _ = X.shape
        Xn = np.zeros((N,N))
        np.copyto(Xn, X)
        thr = abs(Xn).max() * thresh
        Xn[abs(Xn) < thr] = 0
    if negrm:
        Xn[Xn < 0] = 0
    return Xn

def fmri_reg(X, thresh = 1e-3):
    if len(X.shape) == 3:
        N, _, S = X.shape
        Xr = np.zeros((N, N, S))
        fex, fev = eigenmodes_compute(X)
        for i in tqdm(range(S)):
            fevi = fev[i]
            fevi[fevi < thresh] = 0
            Xr[:,:,i] = fex[:,:,i] @ np.diag(fevi) @ fex[:,:,i].T
    else:
        N, _ = X.shape
        Xr = np.zeros((N,N))
        ev, ex = eigh(X)
        ev[ev < thresh] = 0
        Xr = ex @ np.diag(ev) @ ex.T
    return Xr

def vectorize_utril_ctm(X):
    N, _, S = X.shape
    ixes = np.where(np.triu(np.ones((N,N)), 1))
    M = np.size(ixes, axis=1)
    xmat = np.zeros((M, S))
    for i in range(S):
        xmat[:, i] = X[:, :, i][ixes].squeeze()
    return xmat

def small_worldness(A):
    """
    from https://en.wikipedia.org/wiki/Clustering_coefficient
    """
    n = A.shape[0]
    deg = A.sum(axis = 1)
    numer = sum([A[i,j]*A[j,k]*A[j,k] for i in range(n) for j in range(n) for k in range(n)])
    denom = np.sum(deg * (deg - 1))
    return numer/denom if denom != 0 else 0
