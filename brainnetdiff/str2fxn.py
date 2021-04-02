#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2021--, CoNeCT Lab, University of Illinois at Chicago .

Distributed under the terms of the Modified BSD License.

The full license is in the file COPYING.txt, distributed with this software.

NOTES:

"""

import numpy as np
from scipy.linalg import expm
from scipy.linalg import eigh
from tqdm import tqdm
from scipy.optimize import minimize
import copy
from scipy.stats import pearsonr


class Struc2Func:
    """
    Description:
        Class for computing structural-functional mapping based on [1].

    Reference:
        [1] F. Abdelnour, M. Dayan, O. Devinsky, T. Thesen, and A. Raj.
        Functional brain connectivity is predictable from anatomic network’s
        laplacian eigen-structure. NeuroImage, 172:728–739, 2018.

    Parameters
    ----------
    FC : numpy array
        3-way array of functional connectomes of shape N x N x S where N is  number ROIs and S is number of subjects.
    eX : numpy array
        3-way array of structural connectome eigenvectors of shape N x N x S.
    eX : list
        list of 1-way arrays of structural connectome eigenvalues length S and shape (N,), respectively.

    Returns
    -------
    Object of class Struc2Func
    """

    def __init__(self, FC, eX, evals):
        self.FC = FC
        self.eX = eX
        self.evals = evals
        self.params = None
        self.s2f_cor_values = None

    def s2f_fit_(self, indices=[2,None], opt_args=None, max_iter=100, xtol=1e-6):
        """Wrapper function for computing structural to functional mappings

        Parameters
        ----------
        indices : list
            list of length 2 with starting and ending indices of the structural connectome laplacian eigenmodes used for mapping. Default value is [2, None] to indicate using all but the first two eigenmodes.
        opt_args : dict
            dict of kwargs for scipy.optimize.minimize function calls. Default is value is None which defaults to {'maxiter': 100, 'gtol': 1e-6, 'disp': False}.
        max_iter : int
            max number of iterations for minimization. default value is 100.
        xtol : float
            tolerance for minimization termination. default value is 1e-6.

        Returns
        -------
        self : returns an instance of self
        """
        self.ind1 = indices[0]
        self.ind2 = indices[1]
        self.opt_args = opt_args
        self.max_iter = max_iter
        self.xtol = xtol
        self._objective_helper()
        return

    def s2f_map_(self):
        """Wrapper function for computing estimated functional connectomes and correlation between estimated and empirical functional connectomes
        """
        self._s2f_map()
        self._s2f_corr()
        return

    def get_params(self):
        """Method for returning parameter values from structural to functional mapping. Columns are 'a', 'beta t' and 'b' parameters."""
        if self.params is None:
            raise ValueError('No mapping parameters found. Must first run structural to functional mapping to compute parameters.')
        else:
            return self.params

    def get_cor_values(self):
        """Method for returning correlation values between estimated and empirical functional connectomes."""
        if self.s2f_cor_values is None:
            raise ValueError('No correlation values found. Must first run structural to functional fit and mapping.')
        else:
            return self.s2f_cor_values

    def save_params(self, filename):
        """Method for saving parameter values from structural to functional mapping as a .npy file."""
        if self.params is None:
            raise ValueError('No mapping parameters found. Must first run structural to functional mapping to compute parameters.')
        else:
            np.save(filename, self.params)
        return

    def load_params(self, filename, indices=[2,None]):
        """Method for loading previously computed parameter values from structural to functional mapping from .npy file.

        Parameters
        ----------
        filename : string
            filename of previously saved parameters .npy file.
        indices : list
            list of length 2 with starting and ending indices of the structural connectome laplacian eigenmodes used for previously saved mapping. Default value is [2, None] to indicate using all but the first two eigenmodes.

        Returns
        -------
        self : returns an instance of self
        """
        self.ind1 = indices[0]
        self.ind2 = indices[1]
        self.params = np.load(filename)
        return

    def _objective_helper(self):
        S = self.FC.shape[2]
        self.params = np.zeros((S,3))
        if self.opt_args is None:
            opt_args = {'maxiter': 100, 'gtol': 1e-6, 'disp': False}
        for i in tqdm(range(S)):
            xf = self.FC[:,:,i].copy()
            ex = self.eX[:,self.ind1:self.ind2,i].copy()
            ev = self.evals[i][self.ind1:self.ind2].copy()
            ev = np.diag(ev)
            errors = []
            b = 1
            A = [1, 0]
            iter = 0
            while iter < self.max_iter:
                opt1 = minimize(Struc2Func._obj1, b,
                                args = (A, xf, ex, ev),
                                method='BFGS', options = self.opt_args)
                b = opt1.x
                opt2 = minimize(Struc2Func._obj2, A,
                                args = (b, xf, ex, ev),
                                method='BFGS', options = self.opt_args)
                A = opt2.x
                errors.append(Struc2Func._obj1(b, A, xf, ex, ev))
                if len(errors) > 1:
                    tol = errors[-1] - errors[-2]
                    if tol < self.xtol:
                        break
                iter += 1
            self.params[i,:] = np.array([A[0], A[1], b[0]])
        return

    def _obj1(b, A, xf, ex, ev):
        I =  np.identity(ex.shape[0])
        S = A[0] * ex @ expm(-A[1] * ev) @ ex.T + b * I
        return np.linalg.norm(S - xf)**2 / np.linalg.norm(xf)**2

    def _obj2(A, b, xf, ex, ev):
        I =  np.identity(ex.shape[0])
        S = A[0] * ex @ expm(-A[1] * ev) @ ex.T + b * I
        return np.linalg.norm(S - xf)**2 / np.linalg.norm(xf)**2

    def _s2f_map(self):
        N, _, S = self.FC.shape
        self.Fp = np.zeros((N, N, S))
        for i in range(S):
            ex = self.eX[:,self.ind1:self.ind2,i].copy()
            ev = self.evals[i][self.ind1:self.ind2].copy()
            ev = np.diag(ev)
            a, A, b = self.params[i]
            I = np.identity(N)
            self.Fp[:,:,i] = a * ex @ expm(-A * ev) @ ex.T + b * I
        return

    def _s2f_corr(self):
        N, _, S = self.Fp.shape
        self.s2f_cor_values = np.zeros(S)
        for i in range(S):
            self.s2f_cor_values[i] = Struc2Func.ctm_corr(self.Fp[:,:,i], self.FC[:,:,i])
        return

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
