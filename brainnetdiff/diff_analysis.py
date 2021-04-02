#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2021--, CoNeCT Lab, University of Illinois at Chicago .

Distributed under the terms of the Modified BSD License.

The full license is in the file COPYING.txt, distributed with this software.

NOTES:

"""

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.linalg import eigh
from tqdm import tqdm
from brainnetdiff.nbs import *
from brainnetdiff.utils import (small_worldness, reorder_regions, reorder_ctm, fdr)
from brainnetdiff import Struc2Func
from scipy.stats import ttest_ind, spearmanr, pearsonr
from statsmodels.stats.multitest import fdrcorrection
import networkx as nx
from scipy.sparse.csgraph import laplacian, dijkstra

class BrainNetDiff:
    """
    Description:
        Class for conducting multimodal brain network diffusion embedding analyses.

    Reference:


    Parameters
    ----------
    SC : numpy array
        3-way array of structural connectomes of shape N x N x S where N is  number ROIs and S is number of subjects.
    FC : numpy array
        3-way array of functional connectomes of shape N x N x S.
    roi_names : list
        string list with ROI names of length N
    roi_regions : list
        string list with ROI cortical region labels of length N

    Returns
    -------
    Object of class BrainNetDiff
    """

    def __init__(self, SC, FC, roi_names, roi_regions=None):
        self.SC = SC
        self.FC = FC
        self.roi_names = roi_names
        self.roi_regions = roi_regions
        self._laplacian_compute()
        self._eigenmodes_compute()
        self.s2f_params = None
        self.nbs_df = None
        self.nbs_type = None
        self.null_dist = None
        self.nbs_ttest_res = []
        self.nbs_cor_res = []
        self.nbs_cor_trait = None
        self.g1_idx = None
        self.g2_idx = None

        self.diff_sim_option = None
        self.sn_diff_stats = []
        self.hX = None
        self.dhX = None
        self.s2f_idx = 1000
        self.nbs_type = None
        self.nbs_cor_trait = None
        self.HK = None
        self.nbs_idx = None
        self.gtms = {}

    def _laplacian_compute(self):
        self.lX = np.zeros(self.SC.shape)
        for i in range(self.SC.shape[2]):
            self.lX[:, :, i] = laplacian(self.SC[:,:,i].copy(), normed = True)
        return

    def _eigenmodes_compute(self):
        self.eX = np.zeros(self.lX.shape)
        self.evals = []
        for i in range(self.lX.shape[2]):
            w, v = eigh(self.lX[:,:,i].copy())
            self.eX[:,:,i] = v
            self.evals.append(w)
        return

    def s2f_fit(self, s2f_indices=[2,None]):
        """Method for calling structural to functional mapping function
        """
        s2f = Struc2Func(self.FC, self.eX, self.evals)
        s2f.s2f_fit_(indices=s2f_indices)
        self.s2f = s2f
        return

    def s2f_map(self):
        """Method for calling structural to functional mapping function
        """
        self.s2f.s2f_map_()
        self.s2f_params = self.s2f.get_params()
        return

    def save_mapping(self, filename):
        """Function for saving structural to functional mapping parameters
        """
        self.s2f.save_params(filename)
        return

    def load_mapping(self, filename, indices=[2,None]):
        """Function for saving structural to functional mapping parameters
        """
        self.s2f = Struc2Func(self.FC, self.eX, self.evals)
        self.s2f.load_params(filename, indices=indices)
        self.s2f_params = self.s2f.get_params()
        return

    def get_s2f_corr(self):
        """Method for returning correlation values between estimated and empirical functional connectomes."""
        return self.s2f.get_cor_values()

    def SDD_compute(self, indices=[1,None], beta_t=None, distance_metric='euclidean'):
        """compute structural diffusion distance (SDD matrices)
        Parameters
        ----------
        indices : list
            list of length 2 with starting and ending indices of the structural connectome laplacian eigenmodes used computing distance in the SDD diffusion embedding. Default value is [1, None] to indicate using all but the first (constant) eigenmode.
        beta_t : float or array like or None
            diffusion depth 'time' at which to compute embeddings for SDD generation. Default value of None results in use of beta t mapping parameter from structural to functional mapping. Use a float for the same value for each subject, or an array of length S for subject specific time values.
        method : string
            indicates which distance metric to use for scipy pdist.default is 'euclidean'
        Returns
        -------
        self : returns an instance of self
        """

        N, _, S = self.eX.shape
        self.SDD = np.zeros(self.eX.shape)
        self.EM = np.zeros(self.eX.shape)
        ind1 = indices[0]
        ind2 = indices[1]
        if ind2 is None:
            ind2 = S
        if beta_t is None:
            if self.s2f_params is None:
                raise ValueError('Must have s2f_params from s2f mapping or supply non-None value for time parameter for diffusion embedding')
            else:
                self.bt = self.s2f_params[:,1]
        elif hasattr(beta_t, '__len__'):
            self.bt = beta_t
        else:
            self.bt = np.repeat(beta_t, S)
        for i in range(S):
            ex, ev = self.eX[:,:,i].copy(), self.evals[i].copy()
            ev_weighted = np.exp(-self.bt[i] * ev)
            tmp = np.zeros((N,N))
            for j in range(N):
                tmp[:,j] = ex[:,j] * ev_weighted[j]
            self.SDD[:,:,i] = squareform(pdist(tmp[:, ind1:ind2],
                                         metric = distance_metric))
            self.EM[:,:,i] = tmp
        return

    def nbs_ttest(self, group1_idx, group2_idx, thresh, tail='left', nperms=1000):
        """compute NBS on SDD connectomes using ttest at each edge
        Parameters
        ----------
        group1_idx : list or array
            array or list of the indices of the first group's SDD connectomes for NBS ttest.
        group2_idx : list or array
            array or list of the indices of the second group's SDD connectomes for NBS ttest.
        thresh : float
            t-statistic threshold for NBS
        tail : {'left', 'right', 'both'}
            enables specification of particular alternative hypothesis
            'left' : mean population of  group 1 < mean population of group 2 (default)
            'right' : mean population of group 2 < mean population of group 1
            'both' : means are unequal
        nperms : int
            integer number of permutations to use for NBS null distribution generation. default is 1000.
        Returns
        -------
        self : returns an instance of self
        """
        if self.g1_idx is None:
            self.g1_idx = group1_idx
        if self.g2_idx is None:
            self.g2_idx = group2_idx
        if self.null_dist is None or self.nbs_type is not 'ttest':
            self.null_dist = nbs_ttest_compute_null(self.SDD.copy(), group1_idx, group2_idx, nperms)
            self.nbs_type = 'ttest'
        pvals, AC, SZ, adj = nbs_ttest_compute(self.SDD.copy(), group1_idx, group2_idx, thresh, nperms, tail, null_tdist = self.null_dist)
        self.nbs_ttest_res.append([pvals, AC, SZ, adj, tail, thresh])
        return

    def nbs_cor(self, trait, trait_name, thresh, cor_dir='pos',
                cor_type = 'spearman', nperms=1000):
        """compute NBS on SDD connectomes using correlation on each edge
        Parameters
        ----------
        NOTES: still under development
        trait : numpy array
            size S, array of trait value for correlation with edges in NBS
        trait_name : string
            name of trait
        thresh : float
            threshold for correlation statistic for NBS
        nperms : scalar
            indicates the number of permutations to perform
        cor_dir : {'pos', 'neg', 'both'}
            enables specification of particular direction of correlation to use.
            'pos' : positive only
            'neg' : negative only
            'both' : either positive or negative.
        cor_type : {'spearman, pearson'}
            indicates type of correlation to use
        nperms : int
            integer number of permutations to use for NBS null distribution generation. default is 1000.
        Returns
        -------
        self : returns an instance of self
        """
        print('Warning: all features of brainnetdiff may not work with nbs correlation - still underdevelopment. Thank you for your patience :)')
        if trait_name == self.nbs_cor_trait:
            diff_trait = False
        else:
            diff_trait = True
        if self.null_dist is None or self.nbs_type != 'cor' or diff_trait:
            self.null_dist = nbs_cor_compute_null(self.SDD, trait, nperms, cor_type)
            self.nbs_cor_trait = trait_name
            self.nbs_type = 'cor'
        pvals, AC, SZ, adj = nbs_cor_compute(self.SDD, trait, thresh, nperms, cor_dir, cor_type, null_cdist = self.null_dist)
        self.nbs_cor_res.append([pvals, AC, SZ, adj, cor_dir, thresh,
                                 trait_name])
        return

    def save_null_dist(self, filename, rm_null_dist=True):
        """Method for saving NBS null distribution as a .npy file.

        Parameters
        ----------
        filename : string
            filename for .npy file.
        rm_null_dist : {True, False}
            if True (default), will delete current null distribution after saving

        Returns
        -------
        self : returns an instance of self
        """
        np.save(filename, self.null_dist)
        if rm_null_dist:
            self.null_dist = None
        return

    def load_null_dist(self, filename, nbs_type):
        """Method for loading previously computed parameter values from structural to functional mapping from .npy file.

        Parameters
        ----------
        filename : string
            filename of previously saved null distribution .npy file.
        nbs_type : {'ttest', 'cor'}
            specifies the type of NBS used to generate null distribution.
        Returns
        -------
        self : returns an instance of self
        """
        self.null_dist = np.load(filename)
        self.nbs_type = nbs_type
        return

    def make_nbs_df(self):
        """Method for making pandas data frame for summarizing results from NBS analyses.
        """
        if self.nbs_type == 'ttest':
            tails = []; t_thresh = []; sn_size = []; pval = []; sn_rois = []; nbs_res_data = []; sn_id = []; sn_regions = []; trait = []
            nbs_res_data.append(['params', 'adj', 'node_idx', 'node_names',
                                 'roi_regions', 'pval'])
            counter = 1
            for i, data in enumerate(self.nbs_ttest_res):
                cv = data[1]
                cs = data[2]
                if cs is None:
                    pass
                else:
                    csi, = np.where(cs > 1)
                    for j, sn in enumerate(csi):
                        sn_idx, = np.where(cv == (sn + 1))
                        sz = len(sn_idx)
                        sn_names = [self.roi_names[n] for n in sn_idx]
                        if self.roi_regions is None:
                            reg_names = [None for n in sn_idx]
                        else:
                            reg_names = [self.roi_regions[n] for n in sn_idx]
                        sn_regions += [reg_names]
                        sn_rois += [sn_names]
                        tails += [data[4]]
                        trait += [None]
                        t_thresh += [data[5]]
                        sn_size += [sz]
                        pval += [data[0][j]]
                        sn_id += [counter]
                        params = [data[4], data[5]]
                        nbs_res_data.append([params, data[3], sn, sn_idx,
                                             sn_names, reg_names, data[0][j]])
                        counter += 1
            df = pd.DataFrame(data = list(zip(sn_id, tails, t_thresh, trait,
                                              sn_size, pval, sn_rois,
                                              sn_regions)),
                              columns = ['nbs_idx', 'alt_hypo', 'stat_thresh', 'trait', 'size', 'pval', 'rois', 'regions'])
            self.nbs_df = df
            self.nbs_res_data = nbs_res_data
        elif self.nbs_type == 'cor':
            tails = []; r_thresh = []; sn_size = []; trait = [];
            pval = []; sn_rois = []; nbs_res_data = []; sn_id = [];
            sn_regions = []
            nbs_res_data.append(['params', 'adj', 'node_idx', 'node_names',
                                 'roi_regions','pval'])
            counter = 1
            for i, data in enumerate(self.nbs_cor_res):
                cv = data[1]
                cs = data[2]
                if cs is None:
                    pass
                else:
                    csi, = np.where(cs > 1)
                    for j, sn in enumerate(csi):
                        sn_idx, = np.where(cv == (sn + 1))
                        sz = len(sn_idx)
                        sn_names = [self.roi_names[n] for n in sn_idx]
                        sn_rois += [sn_names]
                        if self.roi_regions is None:
                            reg_names = [None for n in sn_idx]
                        else:
                            reg_names = [self.roi_regions[n] for n in sn_idx]
                        sn_regions += [reg_names]
                        tails += [data[4]]
                        r_thresh += [data[5]]
                        trait += [data[6]]
                        sn_size += [sz]
                        pval += [data[0][j]]
                        sn_id += [counter]
                        params = [data[4], data[5], data[6]]
                        nbs_res_data.append([params, data[3], sz, sn_idx,
                                             sn_names, reg_names, data[0][j]])
                        counter += 1
            df = pd.DataFrame(data = list(zip(sn_id, tails, r_thresh, trait,
                                              sn_size, pval, sn_rois,
                                              sn_regions)),
                              columns = ['nbs_idx', 'alt_hypo', 'stat_thresh', 'trait', 'size', 'pval', 'rois', 'regions'])
            self.nbs_df = df
            self.nbs_res_data = nbs_res_data
        return

    def get_nbs_df(self):
        """Method for getting pandas data frame for summarizing results from NBS analyses.

        Parameters
        ----------
        Returns
        -------
        nbs_df : pandas dataframe
            returns a dataframe containing results relating to NBS analyses
        """
        return self.nbs_df

    def diffusion_analysis(self, nbs_idx, heat_val=1, by_region=True, group1_idx=None, group2_idx=None, roi2region=False):
        """Method for computing heat kernels based on diffusion depth used for SDD given a subnetwork identified by NBS and specified by nbs_idx. Then computes, for each possible starting node in the subnetwork, the distribution of heat at every subnetwork node after the diffusion depth. Finally, conducts t-tests on heat distribution at each node in the subnetwork, for each initial node which heat was placed, with groups either previously specified by group1_idx and group2_idx from nbs_ttest or supplied seperately to this method.

        Parameters
        ----------
        nbs_idx : int
            specifies the subnetwork to use for diffusion analysis as indexed in the column 'nbs_idx' of the pandas dataframe obtained using the 'get_nbs_df()' method
        heat_val : {float, array like}
            specifies the heat applied to each node for computation of distribution of heat at each node after the given diffusion depth. Supply a float (default = 1) to compute heat distribution for this value for each subject. Otherwise supply an array-like object of length S to apply a unique value for each subject.
        group1_idx : {None, list or array}
            array or list of the indices of the first group's heat kernels (based on order of originally supplied structural connectomes, SC).
            None (default) uses indices supplied to nbs_ttest method.
        group2_idx : {None, list or array}
            array or list of the indices of the second group's heat kernels (based on order of originally supplied structural connectomes, SC).
            None (default) uses indices supplied to nbs_ttest method.
        by_region : boolean
            Specifies whether to perform heat kernel computations on region averages of rois as given by 'roi_regions' argument when creating brainnetdiff object. True (default) performs computations at each roi.
        roi2region : boolean
            Allows for computation with starting nodes as individual rois while ending distributions are computed as regional averages

        Returns
        -------
        self : returns an instance of self
        """
        if group1_idx is None or group2_idx is None:
            if self.g1_idx is None or self.g2_idx is None:
                raise ValueError('Group indices must be previously given in nbs_ttest() or as kwargs in this method by group1_idx and group2_idx')
            else:
                g1_idx = self.g1_idx
                g2_idx = self.g2_idx
        else:
            g1_idx = group1_idx
            g2_idx = group2_idx
        if g1_idx is None or g2_idx is None:
            raise ValueError('Both group 1 and 2 indicies must be previously given in nbs_ttest() or as kwargs in this method by group1_idx and group2_idx')
        if by_region or roi2region:
            if roi2region is None:
                raise ValueError('must supply roi region labels as roi_regions to do regional average-based analysis')
        if by_region and roi2region:
            raise ValueError('by_region and roi2region cannot both be True, select only one of these arguments to be True.')
        if not hasattr(heat_val, '__len__'):
            heat = np.array([heat_val] * self.SC.shape[2])
        self.by_region = by_region
        self.roi2reg = roi2region
        self.nbs_idx = nbs_idx
        if self.HK is None:
            self._hk_compute()
        self._subnetdiff_data()
        if by_region:
            Ht = self._extract_reorder(self.HK)
        else:
            Ht = self.HK.copy()
        self.rHt = self._get_heat_distbn(Ht, heat, by_region, roi2reg=False)
        if roi2region:
            _ = self._extract_reorder(self.HK)
            self.r2r_rHt = self._get_heat_distbn(Ht, heat, by_region=False, roi2reg=True)
            self.tstat, self.pval = self._ttest_heat_sn(g1_idx, g2_idx, self.r2r_rHt)
        else:
            self.tstat, self.pval = self._ttest_heat_sn(g1_idx, g2_idx, self.rHt)
        self.heat_df = self._make_diff_df(self.tstat, self.pval, utril=True,
                                          by_region=by_region,
                                          roi2reg=roi2region)
        return

    def _hk_compute(self):
        N, _, S = self.SC.shape
        self.HK = np.zeros(self.SC.shape)
        for i in range(S):
            ex = self.eX[:,:,i].copy()
            ev = self.evals[i].copy()
            t = self.bt[i].copy()
            self.HK[:,:,i] = ex @ np.diag(np.exp(-t * ev)) @ ex.T
        return

    def _subnetdiff_data(self):
        #returns sn_ind, list of region names, and the adj (binary) for sn
        sn_res = self.nbs_res_data[self.nbs_idx]
        sn_ind = sn_res[3]
        sn_roi_names = sn_res[4]
        region_names = sn_res[5]
        n_adj = np.copy(sn_res[1][np.ix_(sn_ind, sn_ind)])
        n_adj[n_adj > 0] = 1
        self.sn_ind = sn_ind
        self.sn_region_names = region_names
        self.sn_roi_names = sn_roi_names
        self.n_adj = n_adj
        return

    def _extract_reorder(self, hX):
        #extract the subnetwork data and reorder according to region
        S = self.SC.shape[2]
        N = len(self.sn_ind)
        Ht = np.zeros((N, N, S))
        Ht[:] = hX[np.ix_(self.sn_ind, self.sn_ind, np.arange(S))]
        self.rv, self.cl,_ = reorder_regions(self.sn_region_names)
        return reorder_ctm(Ht, self.rv)

    def _get_heat_distbn(self, Ht, heat_0, by_region=True, roi2reg=False):
        N,_,S = Ht.shape
        for i in range(S):
            Ht[:,:,i] *= heat_0[i]
        if by_region:
            rHt = np.zeros((len(self.cl), len(self.cl), S))
            for i, ii in enumerate(self.cl):
                for j, jj in enumerate(self.cl):
                    rHt[i,j,:] = Ht[np.ix_(ii, jj, np.arange(S))].sum(0).sum(0)
        elif roi2reg:
            rHt = np.zeros((len(self.sn_ind), len(self.cl), S))
            for i, ii in enumerate(self.sn_ind):
                for j, jj in enumerate(self.cl):
                    rHt[i,j,:] = Ht[ii,jj,:].sum(0)
        else:
            rHt = np.zeros((len(self.sn_ind), len(self.sn_ind), S))
            for i, ii in enumerate(self.sn_ind):
                for j, jj in enumerate(self.sn_ind):
                    rHt[i,j,:] = Ht[ii,jj,:]
        return rHt

    def _ttest_heat_sn(self, g1, g2, rHt, avg=False):
        N,_,S = rHt.shape
        if avg:
            rHtm = (rHt + rHt.transpose(1,0,2)) / 2
            tstat, pval = ttest_ind(rHtm[:,:,g1], rHtm[:,:,g2], axis = 2)
            return tstat, pval
        tstat, pval = ttest_ind(rHt[:,:,g1], rHt[:,:,g2], axis = 2)
        return tstat, pval

    def hk_correlation(self, trait, subset_ind=None, cor_type='spearman', mcc=True):
        """Method for computing correlations between the given trait and heat kernel values computed from the diffusion_analysis() method.

        Parameters
        ----------
        trait : array-like
            array or list of values to correlate with heat kernel values. Must be of length S or len(subset_ind), if given.
        subset_ind : array-like or None
            specifies the indicies of subjects to be used for correlations. Must be of same length as trait. Default (None) assumes all subjects are used.
        cor_type : {'pearson', 'spearman'}
            specifies the type of correlation to be used. Default is 'spearman'.
        mcc : boolean
            True (default) will compute FDR corrected p-values in the 'qval' column of the pandas dataframe obtained with the get_correlation_df() method.

        Returns
        -------
        self : returns an instance of self
        """
        if self.roi2reg:
            cstat, pval = self._cor_heat_sn(trait, self.r2r_rHt, subset_ind,
                                            cor_type, avg=False)
        else:
            cstat, pval = self._cor_heat_sn(trait, self.rHt, subset_ind,
                                            cor_type, avg=True)
        self.cor_heat_df = self._make_diff_df(cstat, pval, utril=True,
                                              by_region=self.by_region,
                                              roi2reg=self.roi2reg)
        self.cor_heat_df['stat_idx'] = np.arange(self.cor_heat_df.shape[0])
        if mcc:
            self.cor_heat_df['qval'] = fdrcorrection(self.cor_heat_df.pval.values)[1]
        if self.by_region:
            sn_reg = reorder_regions([self.sn_region_names[i] for i in self.rv])[2]
            self.roi_map = {k : v for v, k in enumerate(sn_reg)}
            self.cor_heat_df['roi1_ind'] = [self.roi_map[i] for i in self.cor_heat_df.roi1.values]
            self.cor_heat_df['roi2_ind'] = [self.roi_map[i] for i in self.cor_heat_df.roi2.values]
        elif self.roi2reg:
            sn_roi = [self.roi_names[i] for i in self.sn_ind]
            sn_reg = reorder_regions([self.sn_region_names[i] for i in self.rv])[2]
            self.roi_map = {k : v for v, k in enumerate(sn_roi)}
            self.reg_map = {k : v for v, k in enumerate(sn_reg)}
            self.cor_heat_df['roi1_ind'] = [self.roi_map[i] for i in self.cor_heat_df.roi1.values]
            self.cor_heat_df['roi2_ind'] = [self.reg_map[i] for i in self.cor_heat_df.roi2.values]
        else:
            sn_roi = [self.roi_names[i] for i in self.sn_ind]
            self.roi_map = {k : v for v, k in enumerate(sn_roi)}
            self.cor_heat_df['roi1_ind'] = [self.roi_map[i] for i in self.cor_heat_df.roi1.values]
            self.cor_heat_df['roi2_ind'] = [self.roi_map[i] for i in self.cor_heat_df.roi2.values]
        return

    def get_correlation_df(self):
        """Method for returning pandas dataframe of correlations computed by the hk_correlation() method.
        """
        return self.cor_heat_df

    def _cor_heat_sn(self, trait, rHt, subset_ind=None, cor_type='spearman',
                     avg=False, roi2reg=False):
        N,M,S = rHt.shape
        cstat = np.zeros((N,M))
        pval = np.zeros((N,M))
        if avg:
            rHtm = (rHt + rHt.transpose(1,0,2)) / 2
        else:
            rHtm = np.zeros(rHt.shape)
            rHtm[:] = rHt
        if subset_ind is not None:
            rHtm = rHtm[:,:,subset_ind]
        if cor_type == 'pearson':
            for i in range(N):
                for j in range(M):
                    cstat[i,j], pval[i,j] = pearsonr(rHtm[i,j,:], trait)
        elif cor_type == 'spearman':
            for i in range(N):
                for j in range(M):
                    cstat[i,j], pval[i,j] = spearmanr(rHtm[i,j,:], trait)
        return cstat, pval

    def _make_diff_df(self, tstat, pval, utril=False, by_region=True,
                      include_regions=True, roi2reg=False):
        roi1 = []; roi2 = []; stat = []; pvalue = []
        if by_region:
            reg_names_reord = reorder_regions([self.sn_region_names[i] for i in self.rv])[2]
        else:
            reg_names_reord = [self.roi_names[i] for i in self.sn_ind]

        if roi2reg:
            roi_names2 = [self.roi_names[i] for i in self.sn_ind]
            reg_names2 = reorder_regions([self.sn_region_names[i] for i in self.rv])[2]
            for i, ii in enumerate(roi_names2):
                for j, jj in enumerate(reg_names2):
                    roi1 += [ii]
                    roi2 += [jj]
                    stat += [tstat[i,j]]
                    pvalue += [pval[i,j]]
        else:
            if utril:
                for i, ii in enumerate(reg_names_reord):
                    for j, jj in enumerate(reg_names_reord):
                        if i > j:
                            roi1 += [ii]
                            roi2 += [jj]
                            stat += [tstat[i,j]]
                            pvalue += [pval[i,j]]
            else:
                for i, ii in enumerate(reg_names_reord):
                    for j, jj in enumerate(reg_names_reord):
                        roi1 += [ii]
                        roi2 += [jj]
                        stat += [tstat[i,j]]
                        pvalue += [pval[i,j]]
        self.reg_names_reord = reg_names_reord
        if include_regions and not by_region:
            reg1 = []; reg2 = []
            if roi2reg:
                reg_names2 = reorder_regions([self.sn_region_names[i] for i in self.rv])[2]
                for i, ii in enumerate(self.sn_region_names):
                    for j, jj in enumerate(reg_names2):
                        reg1 += [ii]
                        reg2 += [jj]
            else:
                if utril:
                    for i, ii in enumerate(self.sn_region_names):
                        for j, jj in enumerate(self.sn_region_names):
                            if i > j:
                                reg1 += [ii]
                                reg2 += [jj]
                else:
                    for i, ii in enumerate(self.sn_region_names):
                        for j, jj in enumerate(self.sn_region_names):
                            reg1 += [ii]
                            reg2 += [jj]
            df = pd.DataFrame(list(zip(roi1, roi2, reg1, reg2, stat, pvalue)),
                              columns = ['roi1', 'roi2','reg1', 'reg2', 'stat', 'pval'])
        else:
            df = pd.DataFrame(list(zip(roi1, roi2, stat, pvalue)),
                              columns = ['roi1', 'roi2', 'stat', 'pval'])
        return df

    def hk_modulation(self, min_type='heat', group_min=True, control_idx=None, patient_idx=None, region_idx=None):
        """Method for determining optimal region for supplemental heat modulation for patient subnetwork heat kernels based on minimizing the difference between each patient and mean control group. Also computes modified subnetwork heat kernels after addition of optimal heat.

        Parameters
        ----------
        min_type : {'heat', 'norm'}
            specifies whether to determine region optimality by minimizing either the amount of heat (default; 'heat') or the frobenius norm ('norm') of the error between modified patient heat kernels and mean control heat kernel.
        group_min : boolean
            specifies whether to return patient specific optimal regions (False) or the mean optimal region across all patients (default: True).
        control_idx : {None, list or array}
            array or list of the indices of the control group's heat kernels (based on order of originally supplied structural connectomes, SC).
            None (default) uses indices supplied to nbs_ttest method by group1_idx parameter.
        patient_idx : {None, list or array}
            array or list of the indices of the patient/intervention group's heat kernels (based on order of originally supplied structural connectomes, SC). None (default) uses indices supplied to nbs_ttest method by group2_idx parameter.
        region_idx : {None, int}
            If an integer is supplied, the value for group_min is ignored and the optimal heat is calculated for the specified subnetwork region only.

        Returns
        -------
        self : returns an instance of self
        """
        if control_idx is None or patient_idx is None:
            if self.g1_idx is None or self.g2_idx is None:
                raise ValueError('Group indices must be previously given in nbs_ttest() or as kwargs in this method by control_idx and patient_idx')
            else:
                self.hc_idx = self.g1_idx
                self.pt_idx = self.g2_idx
        else:
            self.hc_idx = control_idx
            self.pt_idx = patient_idx
        if self.hc_idx is None or self.pt_idx is None:
            raise ValueError('Both control and patient indicies must be previously given in nbs_ttest() or as kwargs in this method by control_idx and patient_idx')
        if self.roi2reg:
            raise ValueError('Cannot compute heat supplement optimization with assymetric heat kernels. Use roi2reg=False when running the diffusion_analysis() method.')
        self.opt_heat, self.opt_idx = self._opt_heat(self.rHt, self.hc_idx, min_type, group_min, pos_only=True, region_idx=region_idx)
        self.nrHt = self._add_heat(self.rHt, self.opt_heat, self.opt_idx, self.hc_idx)
        tstat, pval = self._ttest_heat_sn(self.hc_idx, self.pt_idx, self.nrHt, avg=True)
        self.add_heat_df = self._make_diff_df(tstat, pval, utril=True, by_region=self.by_region)
        self.group_min = group_min
        return

    def _opt_heat(self, rHt, hc_idx, min_type, group_min, pos_only=True, region_idx=None):
        avg_hc = rHt[:,:,hc_idx].mean(2)
        N,_,S = rHt.shape
        opt_heat = np.zeros((N,S))
        norm_val = np.zeros((N,S))
        if pos_only:
            for i in range(hc_idx[-1]+1, rHt.shape[2]):
                k = rHt[:,:,i]
                H = avg_hc - k
                H[H < 0] = 0
                for j in range(N):
                    K = np.outer(np.ones(N), k[j,:])
                    opt_heat[j,i] = np.trace(K.T @ H) / np.trace(K.T @ K)
                    norm_val[j,i] = ((H - opt_heat[j,i] * K)**2).sum()
        else:
            for i in range(hc_idx[-1]+1, rHt.shape[2]):
                k = rHt[:,:,i]
                H = k - avg_hc
                for j in range(N):
                    K = np.outer(np.ones(N), k[j,:])
                    opt_heat[j,i] = np.trace(K.T @ H) / np.trace(K.T @ K)
                    norm_val[j,i] = ((H - opt_heat[j,i] * K)**2).sum()
        self.norm_val_mat = norm_val
        self.opt_heat_mat = opt_heat
        if region_idx is not None:
            ind = np.zeros(S)
            optHeat = opt_heat[region_idx,:]
            for i in range(hc_idx[-1]+1, rHt.shape[2]):
                ind[i] = region_idx
        else:
            if group_min:
                ind = np.zeros(S)
                if min_type == 'heat':
                    idx = np.argmin(opt_heat.sum(1))
                elif min_type == 'norm':
                    idx = np.argmin(norm_val.sum(1))
                optHeat = opt_heat[idx,:]
                for i in range(hc_idx[-1]+1, rHt.shape[2]):
                    ind[i] = idx
            else:
                optHeat = np.zeros(S)
                ind = np.zeros(S)
                if min_type == 'heat':
                    sort_ind = np.argsort(opt_heat, axis = 0)
                elif min_type == 'norm':
                    sort_ind = np.argsort(norm_val, axis = 0)
                for i in range(hc_idx[-1]+1, rHt.shape[2]):
                    lst = [opt_heat[j,i] for j in sort_ind[:,i]]
                    idx = [j for j, x in enumerate(lst) if x > 0]
                    if len(idx) == 0:
                        optHeat[i] = 0
                        ind[i] = 0
                    else:
                        optHeat[i] = opt_heat[sort_ind[idx[0],i],i]
                        ind[i] = sort_ind[idx[0],i]
        ind = ind.astype(int)
        return optHeat, ind

    def _add_heat(self, rHt, opt_heat, ind, hc_idx):
        N,_,S = rHt.shape
        nrHt = np.zeros(rHt.shape)
        nrHt[:] = rHt
        for i in range(hc_idx[-1]+1, S):
            K = np.outer(np.ones(N), nrHt[ind[i],:,i])
            nrHt[:,:,i] = nrHt[:,:,i] + opt_heat[i] * K
        return nrHt

    def get_opt_modulation_df(self):
        """Method for returning pandas dataframe with optimal nodes, and norm values for patient group from hk modulation analysis.

        Parameters
        ----------

        Returns
        -------
        df : pandas dataframe with data pertaining to optimal heat modulation
        """
        roi = [self.reg_names_reord[i] for i in self.opt_idx]
        roi = [roi[i] for i in self.pt_idx]
        if self.by_region:
            reg = roi
        else:
            reg = self.sn_region_names
        opt_heat = self.opt_heat[self.pt_idx]
        norm_val = np.zeros(len(self.pt_idx))
        node_idx = np.zeros(len(self.pt_idx))
        for i,j in enumerate(self.pt_idx):
            norm_val[i] = self.norm_val_mat[self.opt_idx[j], j]
            node_idx[i] = self.opt_idx[j]
        return pd.DataFrame(data=list(zip(roi, node_idx, reg, opt_heat, norm_val)), columns=['node', 'node_idx', 'region', 'heat_value', 'norm_value'])

    def hk_modulation_compare(self, thresh=0.05, use_mcc=True):
        """Method comparing pre vs post heat kernel modulation heat kernel values and computing pandas dataframe that contains the statistic and plotting information relavant to this comparison. thresh and use_mcc parameters determine number of 'corrected' heat kernel values and how subsequent plots are made.

        Parameters
        ----------
        thresh : float
            p- or q-value threshold for determining significance of t-test between mean heat kernel values of controls vs patients both at baseline and post-heat kernel modulation.
        use_mcc :
            specifies whether to use corrected (default: True) or uncorrected p-values from t-tests.

        Returns
        -------
        self : returns an instance of self
        """
        self.use_mcc = use_mcc
        hdf1 = self.heat_df
        hdf1['qval'] = fdrcorrection(hdf1.pval.values)[1]
        hdf2 = self.add_heat_df
        hdf1['mod_stat'] = hdf2.stat.values
        hdf1['mod_pval'] = hdf2.pval.values
        hdf1['mod_qval'] = fdrcorrection(hdf1.mod_pval.values)[1]
        if use_mcc:
            hdf1['h0'] = self._df_sig_code(hdf1.stat.values, hdf1.qval.values, thresh)
            hdf1['h1'] = self._df_sig_code(hdf1.mod_stat.values,
                                           hdf1.mod_qval.values, thresh)
        else:
            hdf1['h0'] = self._df_sig_code(hdf1.stat.values, hdf1.pval.values, thresh)
            hdf1['h1'] = self._df_sig_code(hdf1.mod_stat.values,
                                       hdf1.mod_pval.values, thresh)
        if self.by_region:
            sn_reg = reorder_regions([self.sn_region_names[i] for i in self.rv])[2]
            self.roi_map = {k : v for v, k in enumerate(sn_reg)}
            hdf1['roi1_ind'] = [self.roi_map[i] for i in hdf1.roi1.values]
            hdf1['roi2_ind'] = [self.roi_map[i] for i in hdf1.roi2.values]
        elif self.roi2reg:
            sn_roi = [self.roi_names[i] for i in self.sn_ind]
            sn_reg = reorder_regions([self.sn_region_names[i] for i in self.rv])[2]
            self.roi_map = {k : v for v, k in enumerate(sn_roi)}
            self.reg_map = {k : v for v, k in enumerate(sn_reg)}
            hdf1['roi1_ind'] = [self.roi_map[i] for i in hdf1.roi1.values]
            hdf1['roi2_ind'] = [self.reg_map[i] for i in hdf1.roi2.values]
        else:
            sn_roi = [self.roi_names[i] for i in self.sn_ind]
            self.roi_map = {k : v for v, k in enumerate(sn_roi)}
            hdf1['roi1_ind'] = [self.roi_map[i] for i in hdf1.roi1.values]
            hdf1['roi2_ind'] = [self.roi_map[i] for i in hdf1.roi2.values]
        self.hk_mod_df = hdf1
        return

    def _df_sig_code(self, stat_val, p_val, thresh=0.05):
        h0 = []
        for i,j in zip(stat_val, p_val):
            if j < thresh:
                if i < 0:
                    h0.append(-1)
                elif i > 0:
                    h0.append(1)
            else:
                h0.append(0)
        return h0

    def get_hk_modulation_df(self, sig_only=True):
        """Method for returning pandas dataframe with data heat kernel value data from pre- and post-heat kernel modulation, in addition to data for making plots for these analyses.
        Parameters
        ----------
        sig_only : boolean
            specifies whether to query dataframe for significant rows/those that are pertinent to comparing and plotting pre- vs post-modulation heat kernel values only (default: True).
        Returns
        -------
        nbs_df : pandas dataframe
            dataframe containing results relating to heat kernel modulation analysess
        """
        if sig_only:
            if self.use_mcc:
                return self.hk_mod_df.query('qval < 0.05 or mod_qval < 0.05').query('h0 >= 0 or h1 >=0').copy()
            else:
                return self.hk_mod_df.query('pval < 0.05 or mod_pval < 0.05').query('h0 >= 0 or h1 >=0').copy()
        return self.hk_mod_df

    def make_adj_mat(self, val_map):
        N = len(self.roi_map)
        A = np.zeros((N, N))
        roi_ind = list(zip(self.hk_mod_df.roi1_ind.values,
                           self.hk_mod_df.roi2_ind.values))
        codes = list(zip(self.hk_mod_df.h0.values, self.hk_mod_df.h1.values))
        for i,j in zip(codes, roi_ind):
            if i in val_map.keys():
                A[j] = val_map[i]
        return A

    def avg_reg_xyz(self, xyz):
        df = pd.DataFrame(xyz, columns=['x', 'y', 'z'])
        df['name'] = self.roi_names
        df['region'] = self.roi_regions
        df = df.groupby('region').mean()
        N = len(self.roi_map)
        r_xyz = df.iloc[:,[0,1,2]].values
        sn_xyz = np.zeros((N,3))
        sn_names = []
        for i,j in enumerate(df.index.values):
            if j in self.roi_map.keys():
                sn_xyz[self.roi_map[j],:] = r_xyz[i,:]
                sn_names.append(j)
        return sn_xyz, sn_names

    def get_sn_xyz(self, xyz):
        return xyz[self.sn_ind,:], [self.roi_names[i] for i in self.sn_ind]
